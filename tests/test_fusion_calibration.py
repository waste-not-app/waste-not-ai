from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from wastenot_ai.detector import CombinedDetector
from wastenot_ai.fusion import NATIVE_LABELS, FusionPolicy, fuse
from wastenot_ai.providers import FixtureProvider
from wastenot_ai.recommendation_worker import create_recommendation_app
from wastenot_ai.resolution_benchmark import FOCUS
from wastenot_ai.wire import internal_detection
from wastenot_ai.worker import create_app


def row(source, label, confidence, bbox=(0, 0, 1, 1)):
    return {"class_id": NATIVE_LABELS[source].index(label), "raw_class": label, "confidence": confidence, "bbox": bbox}


def test_fusion_preserves_other_labels_and_adjacent_objects_with_deterministic_ties():
    native = {
        "general": [row("general", "bottle", 0.8), row("general", "cell phone", 0.4), row("general", "remote", 0.8)],
        "broad": [row("broad", "bottle", 0.8), row("broad", "bottle", 0.8, (0, 0, 0.1, 0.1))],
    }
    policy = FusionPolicy(thresholds=(("cell phone", 0.4),))
    kept = fuse(native, policy)
    assert [source for source, raw in kept if raw.raw_class == "bottle"] == ["broad", "broad"]
    assert [raw.raw_class for _, raw in kept] == ["bottle", "bottle", "remote", "cell phone"]
    assert kept[-1][1].confidence == 0.4


@pytest.mark.parametrize("iou, count", [(0.55, 1), (0.65, 1), (0.75, 2)])
def test_exact_merge_boundary_is_inclusive(iou, count):
    predictions = {"broad": [row("broad", "bottle", 0.9)], "general": [row("general", "bottle", 0.8, (0, 0, 0.65, 1))]}
    assert len(fuse(predictions, FusionPolicy(iou))) == count


def test_calibrated_runtime_and_cache_share_reviewability_and_empty_fallback():
    def component(source, rows):
        labels = dict(enumerate(NATIVE_LABELS[source]))

        def detect(_, *, confidence_floor):
            assert confidence_floor == 0.05
            return {"objects": rows, "image": {"width": 100, "height": 80}}

        return SimpleNamespace(
            labels=labels,
            artifact={"model_version": source, "sha256": "a" * 64, "input_shape": "square", "preprocessing": "test"},
            detect=detect,
        )

    native = {"broad": [], "general": [row("general", "cell phone", 0.4), row("general", "remote", 0.8)]}
    broad, general = (component(source, rows) for source, rows in native.items())
    instance = CombinedDetector(broad, general, policy=FusionPolicy(thresholds=(("cell phone", 0.4),)))
    actual = instance.detect(b"image")
    assert actual["objects"] == instance.review_predictions(native)["objects"]
    assert all(item["supported_for_review"] and item["ambiguous_with"] for item in actual["objects"])
    assert len(internal_detection(actual).model.class_names) == 88
    assert instance.review_predictions({"broad": [], "general": []})["status"] == "MANUAL_REQUIRED"

    token = "test-credential-only-000000"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "image/jpeg"}
    with TestClient(create_app(instance, token=token)) as service:
        response = service.post("/internal/v1/detect", headers=headers, content=b"image")
        assert response.status_code == 200
        data = response.json()["data"]
        capabilities = service.get("/internal/v1/capabilities", headers=headers).json()["data"]
    assert capabilities["mapping_version"] == data["mapping_version"]
    assert capabilities["class_thresholds"]["cell phone"]["lower"] == 0.4
    phone = next(obj for obj in data["objects"] if obj["raw_class"] == "cell phone")
    selection = {
        "model_version": data["model_version"],
        "model_sha256": data["model"]["sha256"],
        "mapping_version": data["mapping_version"],
        "class_names": data["model"]["class_names"],
        "object": {field: phone[field] for field in ("class_id", "raw_class", "confidence", "bbox")},
    }
    payload = {
        "selection": selection,
        "context": {"object_key": "cell_phone", "category": "ELECTRONIC", "recycling_access": "YES"},
    }
    with TestClient(
        create_recommendation_app(
            token=token, provider=FixtureProvider(), guides=[], calibrated_model=instance.artifact
        )
    ) as service:
        response = service.post(
            "/internal/v1/recommendations", json=payload, headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 200
        selection["model_sha256"] = "0" * 64
        assert (
            service.post(
                "/internal/v1/recommendations", json=payload, headers={"Authorization": f"Bearer {token}"}
            ).status_code
            == 422
        )


@pytest.fixture
def workflows(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    import calibrate_combined
    import evaluate_tuning_holdout

    return calibrate_combined, evaluate_tuning_holdout


def test_cache_rejects_stale_preprocessing(workflows, tmp_path):
    calibrate, _ = workflows
    path = tmp_path / "cache.json"
    calibrate.write(path, {"identity": {"floor": 0.5}, "records": []})
    with pytest.raises(ValueError, match="identity"):
        calibrate.collect(None, [], {"floor": 0.05}, path)


def test_threshold_search_uses_supported_groups_and_refines_at_one_percent(workflows):
    calibrate, _ = workflows

    def score(policy):
        value = dict(policy.thresholds).get("cardboard", 0.5)
        per_class = {label: {"precision": 0.6, "recall": 0.6, "f1": 0.6, "support": 50} for label in FOCUS}
        per_class["cardboard"]["f1"] = 0.8 - abs(value - 0.41)
        if value == 0.5:
            per_class["cardboard"]["f1"] = 0.6
        return {"per_class": per_class, "overall": {"f1": 0.6}}

    policy, report, comparison = calibrate.select(
        score(FusionPolicy()),
        score,
        {"cardboard": {"instances": 50, "groups": 20}, "paper": {"instances": 70, "groups": 19}},
        FOCUS,
    )
    assert dict(policy.thresholds) == {"cardboard": 0.41}
    assert policy.merge_iou == 0.65 and set(comparison) == {0.55, 0.65, 0.75}
    assert report["paper"]["status"] == "KEEP_CONTROL_LOW_SUPPORT"


def test_holdout_ledger_rejects_renamed_or_near_duplicate_images(workflows, tmp_path):
    _, holdout = workflows
    manifest = {"records": [{"sha256": "first", "group": "object1", "session_id": "session1", "phash": 3}]}
    holdout.claim_holdout(tmp_path, manifest, {"manifest_sha256": "a" * 64})
    manifest["records"][0].update(sha256="other", group="other", session_id="other", phash=7)
    with pytest.raises(ValueError, match="previous evaluation"):
        holdout.claim_holdout(tmp_path, manifest, {"manifest_sha256": "b" * 64})


def test_paired_group_interval_and_budget_keep_failed_candidate_unpromoted(workflows):
    _, holdout = workflows
    old = {"per_class": {label: {"tp": 0, "fp": 0, "fn": 1} for label in FOCUS}}
    new = {"per_class": {label: {"tp": 1, "fp": 0, "fn": 0} for label in FOCUS}}
    interval = holdout.confidence_interval([("one", old, new), ("two", old, new)], {"one": 1, "two": 1}, draws=20)
    assert interval == [1.0, 1.0]
    a, b = (
        {
            "per_class": {label: holdout.metrics(row) for label, row in value["per_class"].items()},
            "overall": {"f1": float(index)},
            "confusion": [],
        }
        for index, value in enumerate((old, new))
    )
    small = ({"support": 0, "f1": None},) * 2
    decision = holdout.release_decision(a, b, dict.fromkeys(FOCUS, (50, 20)), interval, small, small, p95_ratio=1.26)
    assert decision["status"] == "REJECTED_KEEP_BASELINE" and not decision["promoted"]
