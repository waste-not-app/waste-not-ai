"""Protect held-out data and prevent unverified acquisitions entering training."""

import importlib.util
import json
import urllib.error
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


@pytest.fixture
def expansion(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("dataset_expansion", SCRIPTS / "expand_broad_waste.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def record(ident, phash, split=None, group=None, classes=(3,)):
    row = {
        "source": "TACO",
        "source_id": ident,
        "raw_sha256": ident,
        "group": group or ident,
        "phash": phash,
        "boxes": [[cls, 0.5, 0.5, 0.2, 0.2] for cls in classes],
    }
    if split:
        row["split"] = split
    return row


def test_transitive_near_duplicates_cannot_bridge_training_and_holdout(expansion):
    # A--new--B is connected, although A and B are not direct near duplicates.
    old = [record("train", 0, "train"), record("holdout", 1023, "test")]
    additions = [record("bridge", 31), record("independent", 2**40 - 1)]
    assigned, rejected = expansion.split_new_records(old, additions)
    assert {r["source_id"] for r in assigned} == {"independent"}
    assert rejected == [{"source_id": "bridge", "reason": "bridges_locked_splits"}]
    assert old[0]["split"] == "train" and old[1]["split"] == "test"


def test_duplicate_and_proxy_groups_respect_old_holdout(expansion):
    old = [record("original", 0, "val", group="session")]
    additions = [record("new", 2**30 - 1, group="session"), record("original", 0)]
    assigned, rejected = expansion.split_new_records(old, additions)
    assert [(r["source_id"], r["split"]) for r in assigned] == [("new", "val")]
    assert rejected[0]["reason"] == "exact_duplicate"


def test_balancing_preserves_rare_scenes_and_all_cooccurring_boxes(expansion):
    rows = [record(str(i), i, "train", classes=(0,)) for i in range(20)]
    rare = record("rare", 999, "train", classes=(0, 3, 8))
    rows += [rare, record("holdout", 500, "test", classes=(3,))]
    selected = expansion.balanced_train(rows, cap=3)
    assert rare in selected and rare["boxes"] == [
        [0, 0.5, 0.5, 0.2, 0.2],
        [3, 0.5, 0.5, 0.2, 0.2],
        [8, 0.5, 0.5, 0.2, 0.2],
    ]
    assert len(selected) == 3
    assert len({r["source_id"] for r in selected}) == 3
    assert all(r["split"] == "train" for r in selected)


def test_planning_rejects_bad_boxes_without_erasing_other_targets(expansion):
    base = {"labels": expansion.LABELS, "records": [], "rejected": [{"source": "TACO", "source_id": "3"}]}
    images = [
        {
            "id": i,
            "width": 100,
            "height": 100,
            "file_name": f"batch_1/{i:06}.jpg",
            "flickr_url": "https://example.org/image.jpg",
        }
        for i in (1, 2, 3)
    ]
    annotations = [
        {"image_id": 1, "category_id": 40, "bbox": [10, 10, 20, 20]},
        {"image_id": 1, "category_id": 5, "bbox": [40, 40, 20, 20]},
        {"image_id": 2, "category_id": 40, "bbox": [95, 95, 20, 20]},
        {"image_id": 3, "category_id": 40, "bbox": [10, 10, 20, 20]},
    ]
    plan = expansion.make_plan(base, {"images": images, "annotations": annotations}, 300)
    assert plan["candidate_images"] == 1
    assert {b[0] for b in plan["candidates"][0]["boxes"]} == {0, 3}
    assert plan["invalid_annotation_images"][0]["source_id"] == "2"


@pytest.mark.parametrize(
    "error",
    [
        urllib.error.URLError("DNS unavailable"),
        urllib.error.HTTPError("https://www.flickr.com", 429, "Too Many Requests", {}, None),
    ],
)
def test_network_failure_is_checkpointed_without_retrying_or_claiming_data(expansion, monkeypatch, tmp_path, error):
    source = tmp_path / "source"
    source.mkdir()
    (source / "taco-rights.json").write_text("[]")
    monkeypatch.setattr(expansion, "SOURCE", source)
    calls = []

    def unavailable(url):
        calls.append(url)
        raise error

    monkeypatch.setattr(expansion, "check_rights", unavailable)
    plan = {
        "baseline_manifest_sha256": "base",
        "source_metadata_sha256": "source",
        "candidates": [{"source_id": str(i), "original_url": "https://example.org"} for i in range(3)],
    }
    state = expansion.acquire(plan, tmp_path / "cache", 100)
    assert len(calls) == 1 and state["records"] == [] and state["status"] == "BLOCKED_NETWORK"
    assert json.loads((tmp_path / "cache/acquisition.json").read_text())["status"] == "BLOCKED_NETWORK"


def test_stale_cache_cannot_mix_two_source_snapshots(expansion, tmp_path):
    (tmp_path / "acquisition.json").write_text(
        json.dumps({"records": [], "rights": {}, "baseline_manifest_sha256": "old"})
    )
    with pytest.raises(ValueError, match="snapshot changed"):
        expansion.acquire({"baseline_manifest_sha256": "new"}, tmp_path, 1)


def test_acquisition_checkpoint_reports_running_until_traversal_finishes(expansion, monkeypatch, tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "taco-rights.json").write_text("[]")
    monkeypatch.setattr(expansion, "SOURCE", source)
    cache = tmp_path / "cache"
    observed = []

    def reject_without_download(url):
        checkpoint = cache / "acquisition.json"
        if checkpoint.exists():
            observed.append(json.loads(checkpoint.read_text()))
        return {"accepted": False, "reason": "fixture_incompatible_license"}

    monkeypatch.setattr(expansion, "check_rights", reject_without_download)
    state = expansion.acquire(
        {
            "baseline_manifest_sha256": "base",
            "source_metadata_sha256": "source",
            "candidates": [{"source_id": str(i), "original_url": "https://example.org"} for i in range(2)],
        },
        cache,
        100,
    )
    assert observed[0]["status"] == "RUNNING" and observed[0]["requests_this_run"] == 1
    assert state["status"] == "COMPLETE" and state["records"] == []


def test_unreviewed_download_does_not_create_dataset(expansion, monkeypatch, tmp_path):
    base_path, source_path = tmp_path / "base", tmp_path / "source"
    base_path.mkdir()
    source_path.mkdir()
    (base_path / "manifest.json").write_text("{}")
    (source_path / "taco.json").write_text("{}")
    monkeypatch.setattr(expansion, "BASE", base_path)
    monkeypatch.setattr(expansion, "SOURCE", source_path)
    state = {
        "baseline_manifest_sha256": expansion.digest(base_path / "manifest.json"),
        "source_metadata_sha256": expansion.digest(source_path / "taco.json"),
        "records": [{"source_id": "1", "visual_review": "PENDING"}],
    }
    output = tmp_path / "v3"
    with pytest.raises(ValueError, match="No new"):
        expansion.build_dataset({"records": []}, state, [], output, 600)
    assert not output.exists() and not (tmp_path / "v3.building").exists()


@pytest.mark.parametrize("external_source", [False, True])
def test_build_writes_real_images_labels_and_preserves_old_holdout(expansion, monkeypatch, tmp_path, external_source):
    import prepare_drinking_waste
    import yaml
    from PIL import Image

    base_path, source_path = tmp_path / "base", tmp_path / "source"
    base_path.mkdir()
    source_path.mkdir()
    (base_path / "manifest.json").write_text("{}")
    (base_path / "DATASET_NOTICES.md").write_text("Baseline attribution\n")
    (source_path / "taco.json").write_text("{}")
    metadata = source_path / "taco.json"
    options = {}
    if external_source:
        metadata = source_path / "external-tree.json"
        metadata.write_text('{"source": "external-publisher"}')
        (source_path / "taco.json").unlink()
        options = {"source_metadata_path": metadata, "source_notice": "External publisher attribution"}
    old_image = base_path / "images/test/old.jpg"
    old_image.parent.mkdir(parents=True)
    Image.new("RGB", (100, 100), "red").save(old_image)
    new_image = tmp_path / "new.jpg"
    Image.new("RGB", (100, 100), "blue").save(new_image)
    monkeypatch.setattr(expansion, "BASE", base_path)
    monkeypatch.setattr(expansion, "SOURCE", source_path)
    monkeypatch.setattr(prepare_drinking_waste, "perceptual_hash", lambda _: 2**40 - 1)
    old = {
        **record("old", 0, "test", classes=(0,)),
        "image": "images/test/old.jpg",
        "prepared_sha256": expansion.digest(old_image),
    }
    new = {
        **record("new", 2**40 - 1, classes=(3, 8)),
        "cached_path": str(new_image),
        "raw_sha256": expansion.digest(new_image),
        "rights": {"accepted": True, "license_url": "https://creativecommons.org/licenses/by/2.0/"},
    }
    if external_source:
        new["source"] = "ExternalPublisher"
    state = {
        "baseline_manifest_sha256": expansion.digest(base_path / "manifest.json"),
        "source_metadata_sha256": expansion.digest(metadata),
        "records": [new],
    }
    review = [
        {
            "source_id": "new",
            "decision": "APPROVED",
            "raw_sha256": new["raw_sha256"],
            "reviewer": "synthetic-test-fixture",
            "reviewed_at": "2026-11-07",
        }
    ]
    output = tmp_path / "v3"
    result = expansion.build_dataset({"records": [old]}, state, review, output, 600, **options)
    assert json.loads((output / "review-snapshot.json").read_text()) == review
    assert json.loads((output / "acquisition-snapshot.json").read_text()) == state
    assert result["new_images"] == 1 and result["active_instances_per_split"]["train"]["plastic_bag"] == 1
    assert (output / "images/test/old.jpg").read_bytes() == old_image.read_bytes()
    assert (output / f"labels/train/{new['source']}-new.txt").read_text().count("\n") == 2
    if external_source:
        assert "External publisher attribution" in (output / "DATASET_NOTICES.md").read_text()
    assert (output / "test.txt").read_text().strip() == (output / "images/test/old.jpg").as_posix()
    config = yaml.safe_load((output / "data.yaml").read_text())
    assert config["train"] == "train.txt" and config["path"] == output.as_posix()
    with pytest.raises(ValueError, match="Output"):
        expansion.build_dataset({"records": [old]}, state, review, output, 600, **options)


def test_browser_import_checks_identity_before_writing_and_resumes_idempotently(expansion, tmp_path):
    from PIL import Image

    image = tmp_path / "download.jpg"
    Image.new("RGB", (100, 100), "red").save(image)
    candidate = {
        "source_id": "123",
        "source": "TACO",
        "original_url": "https://farm66.staticflickr.com/65535/123456_abcd_o.jpg",
        "width": 100,
        "height": 100,
    }
    plan = {"baseline_manifest_sha256": "base", "source_metadata_sha256": "source", "candidates": [candidate]}
    evidence = {
        "source_id": "123",
        "local_path": str(image),
        "source_page": "https://www.flickr.com/photos/example/123456/",
        "creator_page": "https://www.flickr.com/photos/example/",
        "author": "test-fixture",
        "checked_at": "2026-11-07",
        "license_url": "https://creativecommons.org/licenses/by/2.0/",
        "download_url": "https://live.staticflickr.com/65535/123456_abcd_c.jpg",
    }
    cache = tmp_path / "cache"
    with pytest.raises(ValueError, match="identity"):
        expansion.import_browser_downloads(
            plan, cache, [{**evidence, "source_page": "https://www.flickr.com/photos/example/999/"}]
        )
    assert not cache.exists()
    state = expansion.import_browser_downloads(plan, cache, [evidence])
    assert len(state["records"]) == 1 and state["records"][0]["visual_review"] == "PENDING"
    assert (cache / "images/TACO-123.jpg").read_bytes() == image.read_bytes()
    resumed = expansion.import_browser_downloads(plan, cache, [evidence])
    assert len(resumed["records"]) == 1 and resumed["browser_images_imported_last_batch"] == 0
    Image.new("RGB", (100, 100), "blue").save(image)
    with pytest.raises(ValueError, match="different browser image"):
        expansion.import_browser_downloads(plan, cache, [evidence])
