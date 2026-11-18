import json
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from wastenot_ai.catalog import detection_capabilities, preserve_detected_risk, review_detections
from wastenot_ai.contracts import RawDetection
from wastenot_ai.demo import create_demo_app
from wastenot_ai.providers import FixtureProvider
from wastenot_ai.taxonomy import BROAD_LABELS, COCO_LABELS
from wastenot_ai.wire import DecisionInput, internal_detection


class NativeDetector:
    def __init__(self, names, objects, marker):
        self.labels = dict(enumerate(names))
        self.artifact = {
            "model_version": "fixture-" + marker,
            "sha256": marker * 64,
            "labels": self.labels,
            "input_shape": "square",
        }
        self.objects, self.calls = objects, 0

    def detect(self, image):
        self.calls += 1
        return {
            **review_detections(self.objects, self.labels),
            "model": self.artifact,
            "image": {"width": 200, "height": 100},
            "timing_ms": {"total": 1},
        }


def raw(names, label, score=0.9, bbox=(0.1, 0.1, 0.8, 0.9)):
    return RawDetection(class_id=names.index(label), raw_class=label, confidence=score, bbox=bbox)


def test_camera_demo_keeps_general_objects_when_waste_checkpoint_is_selected(monkeypatch):
    from wastenot_ai import detector

    waste = NativeDetector(BROAD_LABELS, [], "a")
    general = NativeDetector(COCO_LABELS, [raw(COCO_LABELS, "cell phone")], "b")
    monkeypatch.setenv("AI_MODEL_PATH", "waste.pt")
    monkeypatch.setenv("AI_GENERAL_MODEL_PATH", "general.pt")
    monkeypatch.setattr(detector, "YoloDetector", lambda path, *args: waste if path.name == "waste.pt" else general)
    providers = {"gemini": FixtureProvider(), "fixture": FixtureProvider()}
    with TestClient(create_demo_app(providers=providers, demo_guides=True), base_url="http://127.0.0.1") as client:
        import re

        html = client.get("/").text
        headers = {"X-WasteNot-Demo": re.search(r'name="wastenot-demo" content="([^"]+)"', html)[1]}
        config = client.get("/demo/api/config", headers=headers).json()
        assert config["supported_object_labels"].get("cell_phone") == "cell phone", (
            "Selecting v3 removed phone coverage"
        )
        result = client.post(
            "/demo/api/detect",
            content=b"camera-frame",
            headers={**headers, "Content-Type": "image/jpeg", "X-Client-Request-Id": str(uuid4())},
        )
        assert result.status_code == 200
        objects = result.json()["data"]["objects"]
        assert any(obj["raw_class"] == "cell phone" and obj["supported_for_review"] for obj in objects)
        assert waste.calls == general.calls == 1


def combine(waste_objects=(), general_objects=()):
    from wastenot_ai.detector import CombinedDetector

    return CombinedDetector(
        NativeDetector(BROAD_LABELS, list(waste_objects), "a"), NativeDetector(COCO_LABELS, list(general_objects), "b")
    )


def test_combined_mapping_keeps_cardboard_and_general_electronics_with_native_risk_ids():
    from wastenot_ai.contracts import Context

    detector = combine(
        [raw(BROAD_LABELS, "cardboard"), raw(BROAD_LABELS, "can", bbox=(0.81, 0.1, 0.99, 0.9))],
        [raw(COCO_LABELS, "cell phone", bbox=(0.01, 0.01, 0.09, 0.09))],
    )
    result = detector.detect(b"frame")
    objects = {obj["raw_class"]: obj for obj in result["objects"]}
    assert all(obj["supported_for_review"] for obj in objects.values())
    assert objects["cardboard"]["suggested_object_keys"] == ["cardboard"]
    assert objects["cardboard"]["confidence_band"] == "UNCERTAIN"
    assert objects["cell phone"]["class_id"] == 67
    assert objects["can"]["class_id"] != BROAD_LABELS.index("can")
    context = Context.model_validate_json(json.dumps({"object_key": "cell_phone", "category": "ELECTRONIC"}))
    assert preserve_detected_risk(context, objects["cell phone"]).risk_type == "ELECTRICAL"
    internal_detection(result)
    capability = detection_capabilities(detector.labels)
    assert capability["supported_object_labels"]["cardboard"] == "cardboard"
    assert capability["supported_object_labels"]["cell_phone"] == "cell phone"


def test_duplicate_boxes_merge_but_conflicting_cardboard_fridge_hypotheses_remain_visible():
    detector = combine(
        [raw(BROAD_LABELS, "bottle", 0.8), raw(BROAD_LABELS, "cardboard", 0.85)],
        [raw(COCO_LABELS, "bottle", 0.9), raw(COCO_LABELS, "refrigerator", 0.95)],
    )
    result = detector.detect(b"frame")
    assert len(result["objects"]) == 3
    assert [obj["raw_class"] for obj in result["objects"]].count("bottle") == 1
    assert all(obj["confidence_band"] == "UNCERTAIN" and obj["ambiguous_with"] for obj in result["objects"])
    internal_detection(result)


def test_combined_selection_replays_same_mapping_in_recommendation_service():
    from wastenot_ai.decisions import Decisions
    from wastenot_ai.recommendation import load_guides

    detector = combine(general_objects=[raw(COCO_LABELS, "cell phone")])
    result = internal_detection(detector.detect(b"frame"))
    selection = {
        "model_version": result.model_version,
        "model_sha256": result.model.sha256,
        "mapping_version": result.mapping_version,
        "class_names": result.model.class_names,
        "object": {key: getattr(result.objects[0], key) for key in ("class_id", "raw_class", "confidence", "bbox")},
    }
    value = DecisionInput.model_validate_json(
        json.dumps(
            {
                "selection": selection,
                "context": {"object_key": "cell_phone", "category": "ELECTRONIC", "recycling_access": "YES"},
            }
        )
    )
    response = Decisions(FixtureProvider(), load_guides(), demo_guides=True).evaluate(value)
    assert response.recommended_method == "RECYCLE" and response.special_handling


def test_combined_provenance_changes_with_component_or_preprocessing_and_rejects_wrong_taxonomy():
    from wastenot_ai.detector import CombinedDetector

    detector = combine()
    changed = NativeDetector(COCO_LABELS, [], "c")
    assert CombinedDetector(detector.waste, changed).artifact["sha256"] != detector.artifact["sha256"]
    changed.artifact = dict(detector.general.artifact, input_shape="adaptive")
    assert CombinedDetector(detector.waste, changed).artifact["sha256"] != detector.artifact["sha256"]
    with pytest.raises(ValueError, match="Broad Waste"):
        CombinedDetector(detector.general, detector.waste)


def test_combined_result_caps_fifty_boxes_and_does_not_invent_detections():
    empty = combine().detect(b"empty")
    assert empty["objects"] == [] and empty["status"] == "MANUAL_REQUIRED"
    # Non-overlapping boxes must survive until the wire contract's max-50 boundary.
    boxes = [raw(COCO_LABELS, "cell phone", bbox=(i / 100, 0, (i + 0.5) / 100, 0.01)) for i in range(60)]
    result = combine(general_objects=boxes).detect(b"many")
    assert len(result["objects"]) == 50
    internal_detection(result)
