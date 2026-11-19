import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
from time import perf_counter

from .catalog import review_detections
from .contracts import RawDetection
from .fusion import FusionPolicy, fuse
from .media import decode_image
from .taxonomy import BROAD_LABELS, COCO_LABELS, COMBINED_LABELS


class YoloDetector:
    def __init__(self, model_path: Path, device: str = "cpu", input_shape: str = "adaptive", input_size: int = 640):
        if input_shape not in {"adaptive", "square"}:
            raise ValueError("Input shape must be adaptive or square.")
        if isinstance(input_size, bool) or not isinstance(input_size, int) or input_size not in {640, 768, 960}:
            raise ValueError("Input size must be 640, 768 or 960")
        if not model_path.is_file():
            raise ValueError("Model not found. Run fetch-assets first.")
        # Configure before importing Ultralytics: no telemetry or automatic requirements install.
        os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(".cache-ai/ultralytics").resolve()))
        os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache-ai/matplotlib").resolve()))
        os.environ["YOLO_AUTOINSTALL"] = "false"
        from ultralytics import YOLO, settings

        settings.update({"sync": False})
        self.model = YOLO(str(model_path), task="detect")
        self.device = device
        self.input_shape = input_shape
        self._input_size = input_size
        self.labels = {int(k): v for k, v in self.model.names.items()}
        checksum = hashlib.sha256(model_path.read_bytes()).hexdigest()
        try:
            runtime_version = importlib.metadata.version("ultralytics")
        except importlib.metadata.PackageNotFoundError:
            runtime_version = importlib.metadata.version("ultralytics-opencv-headless")
        self.artifact = {
            "model_version": f"candidate-{checksum[:12]}"
            + (
                f"-square{input_size}"
                if input_shape == "square"
                else f"-adaptive{input_size}"
                if input_size != 640
                else ""
            ),
            "sha256": checksum,
            "labels": self.labels,
            "ultralytics_version": runtime_version,
            "device": device,
            "input_size": input_size,
            "input_shape": input_shape,
            "preprocessing": "EXIF transpose, RGB, Ultralytics letterbox",
            "license": "Ultralytics AGPL-3.0 / Enterprise; see provenance documentation",
            "scope": "drinking-waste-four-class-candidate"
            if self.labels == {0: "aluminium_can", 1: "glass_bottle", 2: "hdpe_bottle", 3: "plastic_bottle"}
            else "checkpoint-native-labels",
        }

    @property
    def input_size(self):
        return self._input_size

    def detect(self, image_bytes: bytes, *, confidence_floor=0.25) -> dict:
        started = perf_counter()
        image = decode_image(image_bytes)
        decoded_at = perf_counter()
        try:
            result = self.model.predict(
                image,
                imgsz=getattr(self, "input_size", 640),
                batch=1,
                conf=confidence_floor,
                device=self.device,
                rect=self.input_shape != "square",
                iou=0.7,
                agnostic_nms=False,
                verbose=False,
                save=False,
                max_det=50,
            )[0]
            if hasattr(result, "orig_shape") and tuple(result.orig_shape) != (image.height, image.width):
                raise ValueError("Prediction coordinates do not refer to the decoded image")
            objects = []
            if result.boxes is not None:
                for box in result.boxes:
                    class_id = int(box.cls.item())
                    objects.append(
                        RawDetection(
                            class_id=class_id,
                            raw_class=self.labels[class_id],
                            confidence=float(box.conf.item()),
                            bbox=tuple(float(x) for x in box.xyxyn[0].tolist()),
                        )
                    )
            return {
                **review_detections(
                    objects,
                    self.labels,
                    model_version=self.artifact["model_version"],
                    model_sha256=self.artifact["sha256"],
                ),
                "model": self.artifact,
                "image": {"width": image.width, "height": image.height},
                "timing_ms": {
                    "preprocess": round((decoded_at - started) * 1000, 2),
                    "inference_and_postprocess": round((perf_counter() - decoded_at) * 1000, 2),
                    "total": round((perf_counter() - started) * 1000, 2),
                },
            }
        finally:
            image.close()


class CombinedDetector:
    """Waste and general checkpoints with one versioned label space for review."""

    def __init__(self, waste, general, *, policy=None):
        if waste.labels != dict(enumerate(BROAD_LABELS)) or general.labels != dict(enumerate(COCO_LABELS)):
            raise ValueError("The companion model requires complete Broad Waste and COCO checkpoints.")
        self.waste, self.general = waste, general
        self._policy = policy
        self.labels = dict(enumerate(COMBINED_LABELS))
        components = [
            {key: detector.artifact[key] for key in ("model_version", "sha256", "input_shape")}
            for detector in (waste, general)
        ]
        # Preserve the released 640 bundle identity; nondefault resolutions are separately bound.
        for component, detector in zip(components, (waste, general), strict=True):
            if detector.artifact.get("input_size", 640) != 640:
                component["input_size"] = detector.artifact["input_size"]
        identity = {"components": components, "merge": "same-class-iou065-max50-v1", "labels": COMBINED_LABELS}
        if policy is not None:
            identity["merge"] = f"same-class-iou{round(policy.merge_iou * 100):03d}-max50-v1"
            identity["calibration"] = {
                "thresholds": policy.thresholds,
                "confidence_floor": 0.05,
                "nms_iou": 0.7,
                "preprocessing": [d.artifact["preprocessing"] for d in (waste, general)],
            }
        checksum = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        self.artifact = {
            "model_version": f"broad-coco-{checksum[:12]}",
            "sha256": checksum,
            "labels": self.labels,
            "scope": "broad-waste-plus-coco-experimental",
            "components": components,
            "merge_policy": identity["merge"],
        }
        if policy is not None:
            self.artifact["calibration"] = identity["calibration"]

    def native_predictions(self, image_bytes):
        waste = self.waste.detect(image_bytes, confidence_floor=0.05)
        general = self.general.detect(image_bytes, confidence_floor=0.05)
        if waste["image"] != general["image"]:
            raise ValueError("Component image dimensions disagree")
        return {"broad": waste["objects"], "general": general["objects"]}, waste["image"]

    def review_predictions(self, predictions):
        policy = self._policy or FusionPolicy()
        kept = fuse(predictions, policy)
        return review_detections(
            [raw for _, raw in kept],
            self.labels,
            model_version=self.artifact["model_version"],
            model_sha256=self.artifact["sha256"],
            lower_thresholds=dict(policy.thresholds),
        )

    def detect(self, image_bytes: bytes) -> dict:
        started = perf_counter()
        if self._policy is None:
            waste = self.waste.detect(image_bytes)
            general = self.general.detect(image_bytes)
            if waste["image"] != general["image"]:
                raise ValueError("Component image dimensions disagree")
            predictions, dimensions = {"broad": waste["objects"], "general": general["objects"]}, waste["image"]
        else:
            predictions, dimensions = self.native_predictions(image_bytes)
        return {
            **self.review_predictions(predictions),
            "model": self.artifact,
            "image": dimensions,
            "timing_ms": {"total": round((perf_counter() - started) * 1000, 2)},
        }


def load_detector(
    model_path: Path, device="cpu", input_shape="adaptive", general_model_path: Path | None = None, *, input_size=640
):
    primary = YoloDetector(model_path, device, input_shape, input_size)
    if general_model_path is None:
        return primary
    general = YoloDetector(general_model_path, device, input_shape, input_size)
    return CombinedDetector(primary, general)
