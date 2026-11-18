import hashlib
import json
from io import BytesIO

import pytest
from PIL import Image

from wastenot_ai.camera_benchmark import aggregate, browser_input, load_manifest, score_image

BBOX = [0.1, 0.1, 0.8, 0.9]


def prediction(label, confidence=0.9, bbox=BBOX, reviewable=True):
    return {"raw_class": label, "confidence": confidence, "bbox": bbox, "supported_for_review": reviewable}


def test_matching_counts_duplicates_once_and_keeps_false_negatives_for_unsupported_classes():
    truth = [{"label": "bottle", "bbox": BBOX}, {"label": "cardboard", "bbox": [0.8, 0.8, 0.99, 0.99]}]
    result = score_image([prediction("bottle"), prediction("bottle", 0.8)], truth, ["bottle", "cardboard"])
    assert result["per_class"]["bottle"] == {"tp": 1, "fp": 1, "fn": 0}
    assert result["per_class"]["cardboard"] == {"tp": 0, "fp": 0, "fn": 1}
    assert result["same_class_duplicate_pairs"] == 1


def test_wrong_label_counts_miss_and_false_positive_with_separate_geometric_diagnostic():
    result = score_image([prediction("remote")], [{"label": "cell phone", "bbox": BBOX}], ["cell phone", "remote"])
    assert result["per_class"]["remote"]["fp"] == 1
    assert result["per_class"]["cell phone"]["fn"] == 1
    assert result["confusion"] == [{"truth": "cell phone", "predicted": "remote", "count": 1}]


def test_annotation_scope_excludes_unlabelled_classes_but_preserves_overlapping_wrong_label():
    result = score_image([prediction("refrigerator")], [{"label": "cardboard", "bbox": BBOX}], ["cardboard"])
    assert result["scored_candidates"] == 0 and result["unscored_candidates"] == 1
    assert result["per_class"]["cardboard"] == {"tp": 0, "fp": 0, "fn": 1}
    assert result["confusion"][0]["predicted"] == "refrigerator"


def test_low_confidence_unreviewable_predictions_do_not_rescue_camera_misses():
    result = score_image(
        [prediction("cell phone", 0.49, reviewable=False)], [{"label": "cell phone", "bbox": BBOX}], ["cell phone"]
    )
    assert result["per_class"]["cell phone"]["fn"] == 1
    assert result["confusion"][0]["predicted"] == "__missed__"


def test_aggregate_is_micro_weighted_and_undefined_precision_stays_null():
    first = score_image([prediction("bottle")], [{"label": "bottle", "bbox": BBOX}], ["bottle"])
    second = score_image([], [{"label": "cardboard", "bbox": BBOX}], ["cardboard"])
    result = aggregate([{"score": first}, {"score": second}], ["bottle", "cardboard"])
    assert result["overall"]["precision"] == 1 and result["overall"]["recall"] == 0.5
    assert result["per_class"]["cardboard"]["precision"] is None
    assert result["per_class"]["cardboard"]["recall"] == 0


def test_empty_target_false_positives_respect_annotation_scope():
    result = score_image([prediction("remote"), prediction("can")], [], ["remote", "cell phone"])
    assert result["false_positive_on_empty"] and result["scored_candidates"] == 1
    assert result["per_class"]["remote"]["fp"] == 1


def test_manifest_freezes_hashes_and_rejects_duplicate_images(tmp_path):
    image = tmp_path / "image.jpg"
    Image.new("RGB", (100, 100)).save(image)
    record = {
        "id": "one",
        "image": "image.jpg",
        "sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
        "evaluated_classes": ["bottle"],
        "truth": [{"label": "bottle", "bbox": BBOX}],
    }
    manifest = {"usage": "EVALUATION_ONLY", "target_classes": ["bottle"], "records": [record]}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    assert load_manifest(path, tmp_path)["records"][0]["id"] == "one"
    manifest["records"].append(dict(record, id="two"))
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Duplicate"):
        load_manifest(path, tmp_path)
    manifest["records"] = [dict(record, sha256="0" * 64)]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="checksum"):
        load_manifest(path, tmp_path)


def test_browser_replay_resize_keeps_geometry_and_dimming_is_deterministic(tmp_path):
    path = tmp_path / "wide.png"
    Image.new("RGB", (3200, 1800), (200, 200, 200)).save(path)
    normal, dim = browser_input(path), browser_input(path, 0.55)
    assert dim == browser_input(path, 0.55)
    with Image.open(BytesIO(normal)) as first, Image.open(BytesIO(dim)) as second:
        assert first.size == second.size == (1600, 900)
        assert first.getpixel((100, 100))[0] > second.getpixel((100, 100))[0]


def test_legacy_benchmark_paths_resolve_without_changing_frozen_manifest(tmp_path):
    image = tmp_path / "data/local/image.jpg"
    image.parent.mkdir(parents=True)
    Image.new("RGB", (10, 10)).save(image)
    record = {
        "id": "legacy",
        "image": "ai/data/local/image.jpg",
        "sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
        "evaluated_classes": ["bottle"],
        "truth": [],
    }
    manifest = {"usage": "EVALUATION_ONLY", "target_classes": ["bottle"], "records": [record]}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    original = path.read_bytes()
    assert load_manifest(path, tmp_path)["records"][0]["image"] == record["image"]
    assert path.read_bytes() == original
    manifest["records"][0]["image"] = "ai/../../outside.jpg"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="escapes"):
        load_manifest(path, tmp_path)
