import importlib.util
import json
from pathlib import Path

import pytest

from wastenot_ai.catalog import detection_capabilities, review_detections
from wastenot_ai.contracts import RawDetection
from wastenot_ai.decisions import DecisionError, Decisions
from wastenot_ai.providers import FixtureProvider
from wastenot_ai.recommendation import load_guides
from wastenot_ai.taxonomy import BROAD_LABELS, COMBINED_LABELS
from wastenot_ai.wire import DecisionInput

V4_SHA = "3fc8b7a0baf192b6a88554ce83b24c3ac4ffbd4551e5ee2109a81b68a608462f"
COMBINED_SHA = "0b06663cfd8de5027adfae2f759320f8f92b6eab7a65ceb6366197fcd3d846c3"


@pytest.fixture
def calibration(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location(
        "cardboard_calibration", scripts / "evaluate_cardboard_calibration.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "names,version,checksum",
    [
        (BROAD_LABELS, "candidate-3fc8b7a0baf1-square640", V4_SHA),
        (COMBINED_LABELS, "broad-coco-0b06663cfd8d", COMBINED_SHA),
    ],
)
def test_candidate_replay_boundary_is_consistent_with_capabilities_and_private_decisions(
    names, version, checksum, calibration, monkeypatch
):
    monkeypatch.setattr("wastenot_ai.catalog.review_profile", calibration.candidate_profile)
    labels = dict(enumerate(names))
    identity = {"model_version": version, "model_sha256": checksum}
    capabilities = detection_capabilities(labels, **identity)
    assert capabilities["class_thresholds"]["cardboard"]["lower"] == 0.65
    decisions = Decisions(FixtureProvider(), load_guides(), demo_guides=True)
    for score, selectable in ((0.64, False), (0.65, True)):
        obj = RawDetection(
            class_id=names.index("cardboard"), raw_class="cardboard", confidence=score, bbox=(0.1, 0.1, 0.8, 0.9)
        )
        result = review_detections([obj], labels, **identity)
        assert result["objects"][0]["supported_for_review"] is selectable
        assert result["objects"][0]["confidence_band"] == ("UNCERTAIN" if selectable else "MANUAL_REQUIRED")
        selection = {
            **identity,
            "mapping_version": result["mapping_version"],
            "class_names": list(names),
            "object": obj.model_dump(mode="json"),
        }
        value = DecisionInput.model_validate_json(
            json.dumps({"selection": selection, "context": {"object_key": "cardboard", "category": "PAPER"}})
        )
        if selectable:
            decisions.evaluate(value)
        else:
            with pytest.raises(DecisionError, match="OBJECT_REQUIRES_MANUAL"):
                decisions.evaluate(value)


def test_calibration_cannot_apply_to_another_checkpoint_or_input_shape(calibration, monkeypatch):
    monkeypatch.setattr("wastenot_ai.catalog.review_profile", calibration.candidate_profile)
    labels = dict(enumerate(BROAD_LABELS))
    for version, checksum in (
        ("candidate-3fc8b7a0baf1", V4_SHA),
        ("candidate-3fc8b7a0baf1-square640", "f" * 64),
        ("candidate-cd8db14b189a-square640", "cd8db14b189a8195278694aac07bbf4ae972467c663c1f6aebb72fc5b7bc71f0"),
    ):
        profile = detection_capabilities(labels, model_version=version, model_sha256=checksum)
        assert profile["class_thresholds"]["cardboard"]["lower"] == 0.5


def test_rejected_threshold_remains_isolated_from_runtime(calibration):
    assert calibration.read(calibration.CANDIDATE)["status"] == "REJECTED_ON_TEST"
    for names, version, checksum in (
        (BROAD_LABELS, "candidate-3fc8b7a0baf1-square640", V4_SHA),
        (COMBINED_LABELS, "broad-coco-0b06663cfd8d", COMBINED_SHA),
    ):
        result = detection_capabilities(dict(enumerate(names)), model_version=version, model_sha256=checksum)
        assert result["class_thresholds"]["cardboard"]["lower"] == 0.5
