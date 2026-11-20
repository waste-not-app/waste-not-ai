import copy
import hashlib
import importlib.util
import json
import zipfile
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image


@pytest.fixture
def general(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("general_focus", scripts / "prepare_general_focus.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def native(ident, split="train", phash=0, sha="image", boxes=None):
    row = {"source_id": ident, "source_split": split, "boxes": boxes or [[0, 0.5, 0.5, 0.5, 0.5]]}
    pixel = {"source_id": ident, "source_split": split, "phash": phash, "sha256": sha}
    return row, pixel


def test_fpi_duplicate_pixels_with_conflicting_annotations_are_both_quarantined(general):
    a, pa = native("a")
    b, pb = native("b", boxes=[[0, 0.4, 0.4, 0.2, 0.2]])
    reasons = general.inspect_fpi([a, b], [pa, pb], [])
    assert all("conflicting_annotations_for_identical_pixels" in reasons[general.key(row)] for row in (a, b))


def test_fpi_cross_split_and_protected_matches_are_quarantined(general):
    a, pa = native("a")
    b, pb = native("b", "test", phash=7, sha="other")
    reasons = general.inspect_fpi([a, b], [pa, pb], [{"phash": 1}])
    assert all("cross_publisher_split_duplicate" in reasons[general.key(row)] for row in (a, b))
    assert all("protected_dataset_near_duplicate" in reasons[general.key(row)] for row in (a, b))


def test_native_face_is_preserved_and_invalid_geometry_is_rejected(general):
    row, pixel = native("face", boxes=[[1, 0.5, 0.5, 0.2, 0.2]])
    original = copy.deepcopy(row)
    reasons = general.inspect_fpi([row], [pixel], [])
    assert "no_phone_annotation" in reasons[general.key(row)]
    assert row == original
    row["boxes"] = [[0, 0.99, 0.5, 0.2, 0.2]]
    assert "invalid_native_geometry" in general.inspect_fpi([row], [pixel], [])[general.key(row)]


def test_fpi_pixel_identity_collisions_are_not_silently_overwritten(general):
    a, pa = native("same")
    with pytest.raises(ValueError, match="identities"):
        general.inspect_fpi([a, a], [pa, pa], [])


def test_native_export_verifies_pixels_preserves_face_and_handles_relative_output(general, tmp_path, monkeypatch):
    import prepare_drinking_waste

    monkeypatch.chdir(tmp_path)
    raw = BytesIO()
    Image.new("RGB", (64, 64), "blue").save(raw, format="JPEG")
    payload = raw.getvalue()
    archive = tmp_path / "fpi.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("reorganized_dataset/train/images/a.jpg", payload)
        z.writestr("reorganized_dataset/train/labels/a.txt", "0 .5 .5 .5 .5 1 .5 .2 .2 .2\n")
    checksum = general.digest(archive)
    monkeypatch.setattr(general, "FPI_ARCHIVE_SHA256", checksum)
    monkeypatch.setattr(general, "FPI_IMAGE_COUNT", 1)
    monkeypatch.setattr(prepare_drinking_waste, "perceptual_hash", lambda image: 99)
    pixels = tmp_path / "pixels.json"
    pixels.write_text(
        json.dumps(
            {
                "archive_sha256": checksum,
                "records": [
                    {
                        "source_id": "a",
                        "source_split": "train",
                        "sha256": hashlib.sha256(payload).hexdigest(),
                        "phash": 99,
                        "width": 64,
                        "height": 64,
                    }
                ],
            }
        )
    )
    result = general.stage_fpi(archive, pixels, {"pixel_cache_sha256": general.digest(pixels)}, [], Path("review"), 1)
    assert result["candidate_images"] == 1 and result["combined_label_lines"] == 1
    assert result["approved_training_images"] == 0
    manifest = json.loads((tmp_path / "review/manifest.json").read_text())
    assert manifest["labels"] == ["phone", "face"]
    assert manifest["records"][0]["boxes"][1][0] == 1
    assert not (tmp_path / "review/data.yaml").exists()
    assert len((tmp_path / "review/native-labels/a.txt").read_text().splitlines()) == 2
    with pytest.raises(ValueError, match="fresh output"):
        general.stage_fpi(archive, pixels, {"pixel_cache_sha256": general.digest(pixels)}, [], Path("review"), 1)


def test_coco_acquisition_keeps_person_annotations_and_quarantines_duplicates(general, tmp_path, monkeypatch):
    import prepare_drinking_waste

    monkeypatch.chdir(tmp_path)
    raw = BytesIO()
    Image.new("RGB", (32, 32), "red").save(raw, format="JPEG")
    payload = raw.getvalue()

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def raise_for_status(self):
            pass

        def iter_bytes(self):
            yield payload

    class Client(Response):
        def __init__(self, **kwargs):
            pass

        def stream(self, method, url):
            assert url.startswith("https://s3.amazonaws.com/images.cocodataset.org/train2017/")
            return Response()

    monkeypatch.setattr(general.httpx, "Client", Client)
    monkeypatch.setattr(prepare_drinking_waste, "perceptual_hash", lambda image: 999)
    row = {"source_id": "1", "width": 32, "height": 32, "boxes": [[67, 0.5, 0.5, 0.3, 0.3], [0, 0.5, 0.5, 0.9, 0.9]]}
    plan = {
        "labels": general.LABELS,
        "usage": "REVIEW_ONLY_NO_TRAINING",
        "annotation_archive_sha256": "snapshot",
        "selected": [row, {**row, "source_id": "2"}],
    }
    result = general.acquire_coco(plan, [], Path("review"))
    assert result["candidate_images"] == result["rejected_images"] == 1
    manifest = json.loads((tmp_path / "review/manifest.json").read_text())
    assert manifest["records"][0]["boxes"] == row["boxes"]
    assert manifest["records"][0]["holdout_eligible"] is False
    resumed = general.acquire_coco(plan, [], Path("review"), resume=True)
    assert resumed["candidate_images"] == resumed["rejected_images"] == 1
    (tmp_path / "review/images/COCO-1.jpg").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        general.acquire_coco(plan, [], Path("review"), resume=True)


def test_binary_rounding_is_accepted_without_clipping_actual_bad_boxes(general):
    good = [67, 0.27912499999999996, 0.9682656250000001, 0.2827083333333333, 0.06346875]
    original = good.copy()
    general.validate_boxes([good], 80)
    assert good == original
    with pytest.raises(ValueError, match="edges"):
        general.validate_boxes([[67, 0.5, 0.99, 0.2, 0.1]], 80)


@pytest.fixture
def intake(general, tmp_path, monkeypatch):
    import prepare_drinking_waste

    image = tmp_path / "new.png"
    Image.new("RGB", (20, 20), "green").save(image)
    monkeypatch.setattr(prepare_drinking_waste, "perceptual_hash", lambda image: 123456)
    row = {
        "id": "phone1",
        "image": image.name,
        "source": "FPI-Det",
        "split": "train",
        "group": "object1",
        "session_id": "session1",
        "phash": 123456,
        "prepared_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
        "boxes": [[67, 0.5, 0.5, 0.5, 0.5], [0, 0.5, 0.5, 0.9, 0.9]],
        "annotation_scope": "COCO80_COMPLETE",
        "reannotated_from_native": True,
        "view_tags": ["phone_rear", "handheld"],
        "rights": {
            "accepted": True,
            "status": "RELEASE_ELIGIBLE",
            "source_page": "source",
            "license_url": "license",
            "attribution": "creator",
        },
        "visual_review": {
            "decision": "APPROVED",
            "reviewer": "independent human",
            "reviewer_type": "human",
            "reviewed_at": "2026-11-17",
            "all_objects_annotated": True,
            "labels": general.LABELS,
        },
    }
    return {"labels": general.LABELS, "records": [row]}


def test_general_intake_retains_all_80_class_ids_and_other_objects(general, intake, tmp_path):
    assert len(general.LABELS) == 80 and general.LABELS[67] == "cell phone" and general.LABELS[65] == "remote"
    result = general.audit_intake({"labels": general.LABELS, "records": []}, intake, tmp_path, [])
    assert result["accepted_new_images"] == 1
    assert result["new_support"]["train"]["cell phone"]["instances"] == 1
    assert result["new_support"]["train"]["remote"]["count_deficit"] == 150
    assert result["status"] == "INCOMPLETE_DATA_REVIEW"
    assert intake["records"][0]["boxes"][1][0] == 0


@pytest.mark.parametrize(
    "change",
    [
        lambda row: row.update(annotation_scope="NATIVE_PHONE_FACE"),
        lambda row: row.update(reannotated_from_native=False),
        lambda row: row.update(view_tags=[]),
        lambda row: row["visual_review"].update(reviewer_type="agent"),
        lambda row: row.update(boxes=[[80, 0.5, 0.5, 0.5, 0.5]]),
    ],
)
def test_native_or_unreviewed_intake_never_becomes_training_ready(general, intake, tmp_path, change):
    change(intake["records"][0])
    result = general.audit_intake({"labels": general.LABELS, "records": []}, intake, tmp_path, [])
    assert result["status"] == "INCOMPLETE_DATA_REVIEW"
    assert result["accepted_new_images"] == 0 and result["rejected"]


def test_wrong_taxonomy_is_rejected(general, intake, tmp_path):
    intake["labels"] = ["phone", "face"]
    with pytest.raises(ValueError, match="taxonomy"):
        general.audit_intake({"labels": general.LABELS, "records": []}, intake, tmp_path, [])


@pytest.fixture
def diagnostic(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("phone_diagnostic", scripts / "evaluate_phone_remote.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def predictions():
    sources, records = [], []
    for ident, truth, predicted in [("a", "cell phone", "remote"), ("b", "remote", "cell phone")]:
        sources.append(
            {
                "id": ident,
                "sha256": ident,
                "evaluated_classes": ["cell phone", "remote"],
                "truth": [{"label": truth, "bbox": [0, 0, 1, 1]}],
            }
        )
        records.append(
            {
                "id": ident,
                "image_sha256": ident,
                "profile": "standard",
                "model": "coco11s",
                "objects": [
                    {"raw_class": predicted, "bbox": [0, 0, 1, 1], "supported_for_review": True, "confidence": 0.8}
                ],
            }
        )
    return {"records": sources}, {
        "manifest_sha256": "manifest",
        "protocol": {"input_profiles": {"standard": 1}},
        "models": {"coco11s": {}},
        "records": records,
    }


def test_bidirectional_confusion_is_measured_separately_from_class_metrics(diagnostic, predictions):
    manifest, report = predictions
    result = diagnostic.diagnose(manifest, report, "manifest")
    scores = result["results"]["standard"]["coco11s"]
    assert scores["directions"]["cell phone -> remote"]["count"] == 1
    assert scores["directions"]["remote -> cell phone"]["rate"] == 1
    assert scores["per_class"]["cell phone"]["recall"] == 0
    assert result["status"] == "LEGACY_DIAGNOSIS_NOT_NEW_HOLDOUT"


@pytest.mark.parametrize(
    "change",
    [
        lambda report: report.update(manifest_sha256="other"),
        lambda report: report["records"].append(report["records"][0]),
        lambda report: report["records"].pop(),
        lambda report: report["records"][0].update(image_sha256="changed"),
    ],
)
def test_diagnostic_rejects_stale_duplicate_or_incomplete_predictions(diagnostic, predictions, change):
    manifest, report = predictions
    change(report)
    with pytest.raises(ValueError):
        diagnostic.diagnose(manifest, report, "manifest")
