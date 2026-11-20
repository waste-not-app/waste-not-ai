import sys
from io import BytesIO
from types import SimpleNamespace

import pytest
from PIL import Image

from wastenot_ai import detector
from wastenot_ai.resolution_benchmark import (
    FOCUS,
    assess,
    benchmark_stage,
    cache_identity,
    distribution,
    prepare_inputs,
    process_memory,
    score_sizes,
    select_resolution,
)
from wastenot_ai.taxonomy import BROAD_LABELS, COCO_LABELS
from wastenot_ai.wire import internal_detection


@pytest.fixture
def sdk(monkeypatch, tmp_path):
    calls = []

    class Model:
        def __init__(self, path, **kwargs):
            self.names = dict(enumerate(BROAD_LABELS if str(path).endswith("waste.pt") else COCO_LABELS))

        def predict(self, image, **kwargs):
            calls.append((image.size, kwargs))
            cls = 5 if len(self.names) == 10 else 67
            box = SimpleNamespace(
                cls=SimpleNamespace(item=lambda: cls),
                conf=SimpleNamespace(item=lambda: 0.9),
                xyxyn=[SimpleNamespace(tolist=lambda: [0.1, 0.2, 0.7, 0.8])],
            )
            return [SimpleNamespace(orig_shape=(image.height, image.width), boxes=[box])]

    monkeypatch.setitem(
        sys.modules, "ultralytics", SimpleNamespace(YOLO=Model, settings=SimpleNamespace(update=lambda value: None))
    )
    monkeypatch.setattr(detector.importlib.metadata, "version", lambda name: "8.4.183")
    for filename in ("waste.pt", "general.pt"):
        (tmp_path / filename).write_bytes(filename.encode())
    return tmp_path, calls


@pytest.mark.parametrize("size", [640, 768, 960])
@pytest.mark.parametrize("original", [(1200, 400), (400, 1200)])
def test_resolution_preserves_original_frame_bbox_and_wire_contract(sdk, size, original):
    root, calls = sdk
    instance = detector.load_detector(root / "waste.pt", "cpu", "square", root / "general.pt", input_size=size)
    raw = BytesIO()
    Image.new("RGB", original, "white").save(raw, format="PNG")
    result = instance.detect(raw.getvalue())
    assert result["image"] == {"width": original[0], "height": original[1]}
    assert all(row["bbox"] == pytest.approx([0.1, 0.2, 0.7, 0.8]) for row in result["objects"])
    assert all(options["imgsz"] == size and options["batch"] == 1 and not options["rect"] for _, options in calls)
    assert len(internal_detection(result).model.class_names) == 88


def test_sdk_coordinate_context_mismatch_is_rejected(sdk):
    root, _ = sdk
    instance = detector.YoloDetector(root / "waste.pt", input_shape="square", input_size=768)
    instance.model.predict = lambda image, **kw: [SimpleNamespace(orig_shape=(768, 768), boxes=None)]
    raw = BytesIO()
    Image.new("RGB", (1200, 400)).save(raw, format="PNG")
    with pytest.raises(ValueError, match="decoded image"):
        instance.detect(raw.getvalue())


def test_sdk_boxes_collapsed_by_clipping_do_not_abort_low_confidence_inference(sdk):
    root, _ = sdk
    instance = detector.YoloDetector(root / "waste.pt", input_shape="square")
    valid = [0.1, 0.2, 0.9, 1.0]

    def predict(image, **options):
        assert options["conf"] == 0.05
        boxes = [
            SimpleNamespace(
                cls=SimpleNamespace(item=lambda: 0),
                conf=SimpleNamespace(item=lambda: 0.08),
                xyxyn=[SimpleNamespace(tolist=lambda coords=coords: coords)],
            )
            for coords in ([0.49, 1.0, 1.0, 1.0], valid, [0.0, 0.2, 0.0, 0.8])
        ]
        return [SimpleNamespace(orig_shape=(image.height, image.width), boxes=boxes)]

    instance.model.predict = predict
    raw = BytesIO()
    Image.new("RGB", (1200, 400)).save(raw, format="PNG")
    result = instance.detect(raw.getvalue(), confidence_floor=0.05)
    assert [obj["bbox"] for obj in result["objects"]] == [valid]
    assert result["image"] == {"width": 1200, "height": 400}


def test_resolution_cannot_be_mutated_without_rebuilding_profile_identity(sdk):
    root, _ = sdk
    instance = detector.YoloDetector(root / "waste.pt", input_shape="square", input_size=768)
    with pytest.raises(AttributeError):
        instance.input_size = 960
    assert instance.input_size == instance.artifact["input_size"] == 768


@pytest.mark.parametrize("size", [True, 0, 641, 1024, 768.0])
def test_invalid_resolutions_fail_before_model_load(tmp_path, size):
    with pytest.raises(ValueError, match="Input size"):
        detector.YoloDetector(tmp_path / "missing.pt", input_size=size)


def test_released_640_identity_is_preserved_and_nondefault_sizes_change_it():
    def component(names, sha):
        return SimpleNamespace(
            labels=dict(enumerate(names)),
            artifact={
                "sha256": sha,
                "model_version": f"candidate-{sha[:12]}-square640",
                "input_shape": "square",
                "input_size": 640,
            },
        )

    waste = component(BROAD_LABELS, "3fc8b7a0baf192b6a88554ce83b24c3ac4ffbd4551e5ee2109a81b68a608462f")
    general = component(COCO_LABELS, "85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5")
    base = detector.CombinedDetector(waste, general).artifact
    assert base["model_version"] == "broad-coco-b0f633193403"
    general.artifact["input_size"] = 768
    assert detector.CombinedDetector(waste, general).artifact["sha256"] != base["sha256"]


def test_exif_is_oriented_before_coordinate_context_is_checked(sdk):
    root, calls = sdk
    image = Image.new("RGB", (120, 80), "white")
    exif = Image.Exif()
    exif[274] = 6
    raw = BytesIO()
    image.save(raw, format="JPEG", exif=exif)
    result = detector.YoloDetector(root / "waste.pt", input_size=768).detect(raw.getvalue())
    assert result["image"] == {"width": 80, "height": 120}
    assert calls[0][0] == (80, 120)


def prediction(bbox=(0.1, 0.1, 0.8, 0.8)):
    return {"raw_class": "bottle", "bbox": list(bbox), "confidence": 0.9, "supported_for_review": True}


def test_warmups_are_excluded_and_models_interleave_on_identical_bytes():
    calls = []

    class Fake:
        def __init__(self, size):
            self.size = size
            self.artifact = {"size": size}

        def detect(self, raw):
            calls.append((self.size, raw))
            return {"image": {"width": 100, "height": 100}, "objects": [prediction()]}

    item = {
        "content": b"same input",
        "input_sha256": "input",
        "source_size": (100, 100),
        "transmitted_size": (100, 100),
        "profile": "browser_jpeg90",
        "source": {
            "id": "one",
            "truth": [{"label": "bottle", "bbox": [0.1, 0.1, 0.8, 0.8]}],
            "evaluated_classes": ["bottle"],
        },
    }
    times = iter(v for i in range(6) for v in (float(i), i + 0.01))
    stage = benchmark_stage(
        {640: Fake(640), 768: Fake(768)},
        [item],
        ["bottle"],
        clock=lambda: next(times),
        memory=lambda: {"rss_bytes": 1024, "method": "fixture"},
    )
    assert len(calls) == 16  # ten warmups and six timed calls
    assert [size for size, raw in calls[10:]] == [640, 768, 768, 640, 640, 768]
    assert {raw for _, raw in calls} == {b"same input"}
    for row in stage["results"]["browser_jpeg90"].values():
        assert row["latency"]["samples"] == 3 and row["latency"]["p95_ms"] == pytest.approx(10)
        assert row["overall"]["recall"] == 1


def test_object_size_tp_uses_gt_bucket_even_when_prediction_crosses_boundary():
    truth = [{"label": "bottle", "bbox": [0, 0, 0.1, 0.09]}]
    counts = score_sizes([prediction((0, 0, 0.1, 0.11))], truth, ["bottle"])
    assert counts["small"]["tp"] == 1 and counts["medium"]["fp"] == 0


def summary(f1=0.5, p95=100):
    row = {"support": 100, "precision": f1, "recall": f1, "f1": f1}
    return {
        "overall": dict(row),
        "latency": {"p95_ms": p95},
        "per_class": {label: dict(row) for label in FOCUS},
        "by_object_size": {name: dict(row) for name in ("small", "medium", "large")},
    }


def test_cpu_budget_and_quality_guard_resolution_selection():
    good = {"results": {"browser_jpeg90": {"640": summary(), "768": summary(0.6, 120)}}}
    result = select_resolution(good)
    assert result["selected_input_size"] == 768 and not result["promoted"]
    good["results"]["browser_jpeg90"]["768"]["latency"]["p95_ms"] = 126
    assert select_resolution(good)["selected_input_size"] == 640


def test_sparse_data_and_dim_regressions_do_not_select_larger_input():
    stage = {"results": {"browser_jpeg90": {"640": summary(), "768": summary(0.6, 120)}}}
    stage["results"]["simulated_dim055"] = {"640": summary(), "768": summary(0.4, 120)}
    assert select_resolution(stage)["selected_input_size"] == 640
    stage["results"].pop("simulated_dim055")
    stage["results"]["browser_jpeg90"]["640"]["per_class"]["paper"]["support"] = 49
    stage["results"]["browser_jpeg90"]["768"]["per_class"]["paper"]["support"] = 49
    assert select_resolution(stage)["selected_input_size"] == 640


@pytest.mark.parametrize("field", ["latency", "quality"])
def test_nonfinite_benchmark_metrics_are_rejected(field):
    candidate = summary(0.6, 120)
    if field == "latency":
        candidate["latency"]["p95_ms"] = float("nan")
    else:
        candidate["overall"]["f1"] = float("nan")
    with pytest.raises(ValueError):
        assess(summary(), candidate)


def test_cache_identity_covers_model_profile_input_and_environment():
    inputs = [{"source": {"id": "one"}, "profile": "standard", "input_sha256": "same"}]
    a = cache_identity("manifest", {"size": 640}, inputs, {"threads": 8})
    assert a != cache_identity("manifest", {"size": 768}, inputs, {"threads": 8})
    assert a != cache_identity("manifest", {"size": 640}, inputs, {"threads": 4})


def test_input_audit_records_resize_before_model_inference(tmp_path):
    image = tmp_path / "wide.png"
    Image.new("RGB", (3200, 1800)).save(image)
    inputs = prepare_inputs({"records": [{"id": "one", "image": "wide.png"}]}, tmp_path, {"browser_jpeg90": 1})
    assert inputs[0]["source_size"] == (3200, 1800) and inputs[0]["transmitted_size"] == (1600, 900)


def test_android_replay_uses_power_of_two_sampling_and_separate_input_hash(tmp_path):
    image = tmp_path / "camera.png"
    Image.new("RGB", (2400, 1800), (80, 120, 160)).save(image)
    inputs = prepare_inputs({"records": [{"id": "one", "image": "camera.png"}]}, tmp_path)
    rows = {row["profile"]: row for row in inputs}
    assert rows["browser_jpeg90"]["transmitted_size"] == (1600, 1200)
    assert rows["android_jpeg85_gateway90"]["transmitted_size"] == (1200, 900)
    assert len({row["input_sha256"] for row in inputs}) == 3


def test_native_memory_and_percentiles_are_explicit():
    memory = process_memory()
    assert memory["rss_bytes"] is None or memory["rss_bytes"] > 0
    assert distribution([1, 2, 3])["p95_ms"] == pytest.approx(2.9)
