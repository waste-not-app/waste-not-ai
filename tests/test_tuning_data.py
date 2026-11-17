import copy
import hashlib
import importlib.util
import json
from pathlib import Path

from PIL import Image
import pytest


@pytest.fixture
def tuning(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("tuning_data", scripts / "prepare_tuning_data.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def intake(tuning, tmp_path, monkeypatch):
    import prepare_drinking_waste
    image = tmp_path / "photo.png"
    # Real pixel evidence, deliberately independent of platform PRNG versions.
    source = Image.new("RGB", (64, 64))
    source.putdata([(i % 256, i // 16 % 256, i * 7 % 256) for i in range(4096)])
    # OpenCV is an optional research dependency, absent from locked service CI.
    phash = 123456789
    monkeypatch.setattr(prepare_drinking_waste, "perceptual_hash", lambda image: phash)
    source.save(image)
    row = {"id": "new", "image": image.name, "split": "val", "group": "object1",
           "session_id": "session1", "prepared_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
           "phash": phash, "boxes": [[5, .5, .5, .5, .5]],
           "rights": {"accepted": True, "status": "RELEASE_ELIGIBLE", "license_url": "https://example.org/license",
                      "source_page": "https://example.org/photo", "attribution": "Fixture creator"},
           "visual_review": {"decision": "APPROVED", "reviewer": "Independent reviewer",
                             "reviewer_type": "human", "reviewed_at": "2026-11-17",
                             "all_objects_annotated": True, "labels": tuning.LABELS}}
    return {"labels": tuning.LABELS, "records": [row]}


def test_legacy_objects_do_not_count_as_new_holdout(tuning, tmp_path):
    base = {"labels": tuning.LABELS, "records": [{"split": "test", "group": "old", "boxes": [[5,.5,.5,.2,.2]]}]}
    result = tuning.readiness(base, {"labels": tuning.LABELS, "records": []}, tmp_path)
    assert result["legacy_instances"]["test"]["cardboard"] == 1
    assert result["new_support"]["test"]["cardboard"]["count_deficit"] == 50
    assert result["status"] == "INCOMPLETE_DATA_REVIEW"


@pytest.mark.parametrize("change,reason", [
    (lambda r: r["rights"].update(status="EXPERIMENTAL_NONCOMMERCIAL_RESEARCH"), "rights"),
    (lambda r: r["visual_review"].update(reviewer_type="agent"), "human review"),
    (lambda r: r["visual_review"].update(all_objects_annotated=False), "human review"),
    (lambda r: r.update(prepared_sha256="wrong"), "checksum"),
    (lambda r: r.update(phash=0), "pHash mismatch"),
    (lambda r: r.update(boxes=[[5, .9, .5, .5, .5]]), "edges"),
    (lambda r: r.update(boxes=[[5, float("nan"), .5, .1, .1]]), "nonfinite"),
    (lambda r: r.update(boxes=[]), "verified-negative"),
])
def test_intake_rejects_unreviewed_restricted_or_corrupt_data(tuning, intake, tmp_path, change, reason):
    change(intake["records"][0])
    result = tuning.readiness({"labels": tuning.LABELS, "records": []}, intake, tmp_path)
    assert result["accepted_new_images"] == 0
    assert reason in result["rejected"][0]["reasons"][0]


def test_protected_benchmark_cannot_be_reintroduced(tuning, intake, tmp_path):
    row = intake["records"][0]
    result = tuning.readiness({"labels": tuning.LABELS, "records": []}, intake, tmp_path,
                              [{"sha256": row["prepared_sha256"]}])
    assert result["accepted_new_images"] == 0
    assert "protected" in result["rejected"][0]["reasons"][0]


def test_protected_near_duplicate_cannot_be_reintroduced(tuning, intake, tmp_path):
    row = intake["records"][0]
    result = tuning.readiness({"labels": tuning.LABELS, "records": []}, intake, tmp_path,
                              [{"phash": row["phash"] ^ 7}])
    assert "Near-duplicate" in result["rejected"][0]["reasons"][0]


def test_valid_review_counts_support_without_claiming_readiness(tuning, intake, tmp_path):
    result = tuning.readiness({"labels": tuning.LABELS, "records": []}, intake, tmp_path)
    assert result["accepted_new_images"] == 1
    assert result["new_support"]["val"]["cardboard"]["instances"] == 1
    assert result["status"] == "INCOMPLETE_DATA_REVIEW"


def test_restore_preserves_manifest_labels_and_active_sampling(tuning, tmp_path):
    source, target = tmp_path / "original", tmp_path / "restored"
    (source / "images/train").mkdir(parents=True)
    (source / "labels/train").mkdir(parents=True)
    rows = []
    for name in ("a", "b"):
        image = source / f"images/train/{name}.jpg"
        Image.new("RGB", (10, 10)).save(image)
        (source / f"labels/train/{name}.txt").write_text("5 .5 .5 .5 .5\n")
        rows.append({"image": f"images/train/{name}.jpg", "split": "train",
                     "prepared_sha256": tuning.digest(image), "boxes": [[5,.5,.5,.5,.5]]})
    manifest = {"labels": tuning.LABELS, "records": rows}
    (source / "manifest.json").write_text(json.dumps(manifest))
    (source / "train.txt").write_text("C:/old/ai/images/train/a.jpg\n")
    (source / "train-full.txt").write_text("C:/old/ai/images/train/a.jpg\nC:/old/ai/images/train/b.jpg\n")
    for split in ("val", "test"):
        (source / f"{split}.txt").write_text("")
    (source / "data.yaml").write_text(json.dumps({"path": "C:/old/ai", "names": tuning.LABELS,
                                                 "train": "train.txt", "val": "val.txt", "test": "test.txt"}))
    before = (source / "train.txt").read_bytes()
    result = tuning.restore(source, target)
    assert result["source_manifest_sha256"] == result["restored_manifest_sha256"]
    assert (source / "train.txt").read_bytes() == before
    assert (target / "train.txt").read_text() == (target / "images/train/a.jpg").as_posix() + "\n"
    assert len((target / "train-full.txt").read_text().splitlines()) == 2
    with pytest.raises(ValueError, match="new output"):
        tuning.restore(source, target)


def test_path_traversal_is_rejected(tuning, tmp_path):
    with pytest.raises(ValueError, match="escapes"):
        tuning.inside(tmp_path, "../outside.jpg")


def test_same_session_cannot_cross_splits(tuning, intake, tmp_path, monkeypatch):
    # Exercise the split policy independently from the already-tested pixel hash calculation.
    import prepare_drinking_waste
    first = intake["records"][0]
    other = copy.deepcopy(first)
    other.update(id="second", image="second.png", split="test", group="other-object", phash=first["phash"] ^ ((1 << 30)-1))
    image = tmp_path / "second.png"
    Image.new("RGB", (64,64), "red").save(image)
    other["prepared_sha256"] = tuning.digest(image)
    hashes = iter([first["phash"], other["phash"]])
    monkeypatch.setattr(prepare_drinking_waste, "perceptual_hash", lambda image: next(hashes))
    intake["records"].append(other)
    result = tuning.readiness({"labels": tuning.LABELS, "records": []}, intake, tmp_path)
    assert result["accepted_new_images"] == 0
    assert len(result["rejected"]) == 2


def test_review_queue_preserves_deferred_decisions_without_auto_approval(tuning, tmp_path):
    cache = tmp_path / "cache"
    (cache / "images").mkdir(parents=True)
    image = cache / "images/TACO-1.jpg"
    Image.new("RGB", (32, 32), "blue").save(image)
    row = {"source": "TACO", "source_id": "1", "boxes": [[5,.5,.5,.5,.5]],
           "raw_sha256": tuning.digest(image), "rights": {"accepted": True}}
    tuning.write(cache / "acquisition.json", {"records": [row]})
    tuning.write(cache / "review.json", [{"source_id": "1", "decision": "DEFERRED", "reason": "ambiguous"}])
    output = tmp_path / "review"
    result = tuning.stage_review(cache, {"records": []}, output)
    assert result["candidates"] == 1
    assert result["approved_training_images"] == result["new_holdout_images"] == 0
    queue = tuning.read(output / "review-queue.json")
    assert queue["records"][0]["prior_reviews"][0]["decision"] == "DEFERRED"
    assert queue["records"][0]["holdout_eligible"] is False
    assert (output / "contact-01.jpg").exists()
    with pytest.raises(ValueError, match="already exists"):
        tuning.stage_review(cache, {"records": []}, output)


def test_review_queue_checks_bytes_before_export(tuning, tmp_path):
    cache = tmp_path / "cache"
    (cache / "images").mkdir(parents=True)
    (cache / "images/TACO-1.jpg").write_bytes(b"corrupt")
    tuning.write(cache / "acquisition.json", {"records": [
        {"source": "TACO", "source_id": "1", "raw_sha256": "wrong"}]})
    with pytest.raises(ValueError, match="checksum"):
        tuning.stage_review(cache, {"records": []}, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_builder_refuses_incomplete_data_before_writing(tuning, intake, tmp_path):
    base = tmp_path / "baseline"
    base.mkdir()
    tuning.write(base / "manifest.json", {"labels": tuning.LABELS, "records": []})
    tuning.write(tmp_path / "intake.json", intake)
    with pytest.raises(ValueError, match="incomplete"):
        tuning.build(base, tmp_path / "intake.json", [], tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_builder_preserves_train_sampling_and_separates_legacy_test(tuning, intake, tmp_path, monkeypatch):
    base = tmp_path / "baseline"
    rows = []
    for split, name in [("train", "active"), ("train", "inactive"), ("test", "legacy-test")]:
        image = base / f"images/{split}/{name}.png"
        image.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (16,16), "yellow").save(image)
        label = base / f"labels/{split}/{name}.txt"
        label.parent.mkdir(parents=True, exist_ok=True)
        label.write_text("5 .5 .5 .5 .5\n")
        rows.append({"source": "Legacy", "image": image.relative_to(base).as_posix(), "split": split,
                     "prepared_sha256": tuning.digest(image), "boxes": [[5,.5,.5,.5,.5]]})
    tuning.write(base / "manifest.json", {"labels": tuning.LABELS, "records": rows})
    (base / "train.txt").write_text((base / "images/train/active.png").as_posix()+"\n")
    (base / "DATASET_NOTICES.md").write_text("Legacy attribution\n")
    tuning.write(tmp_path / "intake.json", intake)
    # Export behavior is tested separately from the quota gate above.
    monkeypatch.setattr(tuning, "readiness", lambda *args, **kwargs: {"status": "READY_FOR_TRAINING_EXPERIMENT"})
    output = tmp_path / "output"
    tuning.build(base, tmp_path / "intake.json", [], output)
    result = tuning.read(output / "manifest.json")
    assert result["selected_train_images"] == 1
    assert len((output / "train-full.txt").read_text().splitlines()) == 2
    assert not (output / "test.txt").read_text()
    assert all("legacy-test" not in r["image"] for r in result["records"])
    assert result["active_instances_per_split"]["val"]["cardboard"] == 1
    assert "Legacy attribution" in (output / "DATASET_NOTICES.md").read_text()
