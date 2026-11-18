import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
from time import perf_counter

from .catalog import review_detections
from .contracts import RawDetection
from .detection_quality import box_iou
from .media import decode_image
from .taxonomy import BROAD_LABELS, COCO_LABELS, COMBINED_LABELS


class YoloDetector:
    def __init__(self, model_path: Path, device: str = "cpu", input_shape: str = "adaptive"):
        if input_shape not in {"adaptive", "square"}:
            raise ValueError("Input shape harus adaptive atau square.")
        if not model_path.is_file():
            raise ValueError("Model tidak ditemukan. Jalankan perintah fetch-assets terlebih dahulu.")
        # Configure before importing Ultralytics: no telemetry or automatic requirements install.
        os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(".cache-ai/ultralytics").resolve()))
        os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache-ai/matplotlib").resolve()))
        os.environ["YOLO_AUTOINSTALL"] = "false"
        from ultralytics import YOLO, settings
        settings.update({"sync": False})
        self.model = YOLO(str(model_path), task="detect")
        self.device = device
        self.input_shape = input_shape
        self.labels = {int(k): v for k, v in self.model.names.items()}
        checksum = hashlib.sha256(model_path.read_bytes()).hexdigest()
        try:
            runtime_version = importlib.metadata.version("ultralytics")
        except importlib.metadata.PackageNotFoundError:
            runtime_version = importlib.metadata.version("ultralytics-opencv-headless")
        self.artifact = {
            "model_version": f"candidate-{checksum[:12]}" + ("-square640" if input_shape == "square" else ""),
            "sha256": checksum,
            "labels": self.labels, "ultralytics_version": runtime_version,
            "device": device, "input_size": 640,
            "input_shape": input_shape,
            "preprocessing": "EXIF transpose, RGB, Ultralytics letterbox",
            "license": "Ultralytics AGPL-3.0 / Enterprise; see provenance documentation",
            "scope": "drinking-waste-four-class-candidate" if self.labels == {
                0: "aluminium_can", 1: "glass_bottle", 2: "hdpe_bottle", 3: "plastic_bottle"
            } else "checkpoint-native-labels",
        }

    def detect(self, image_bytes: bytes) -> dict:
        started = perf_counter()
        image = decode_image(image_bytes)
        decoded_at = perf_counter()
        try:
            result = self.model.predict(image, imgsz=640, conf=0.25, device=self.device,
                                        rect=self.input_shape != "square",
                                        iou=.7, agnostic_nms=False, verbose=False, save=False, max_det=50)[0]
            objects = []
            if result.boxes is not None:
                for box in result.boxes:
                    class_id = int(box.cls.item())
                    objects.append(RawDetection(
                        class_id=class_id, raw_class=self.labels[class_id],
                        confidence=float(box.conf.item()),
                        bbox=tuple(float(x) for x in box.xyxyn[0].tolist()),
                    ))
            return {
                **review_detections(objects, self.labels, model_version=self.artifact["model_version"],
                                    model_sha256=self.artifact["sha256"]),
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
    def __init__(self, waste, general):
        if waste.labels != dict(enumerate(BROAD_LABELS)) or general.labels != dict(enumerate(COCO_LABELS)):
            raise ValueError("Model pendamping memerlukan checkpoint Broad Waste dan COCO lengkap.")
        self.waste, self.general = waste, general
        self.labels = dict(enumerate(COMBINED_LABELS))
        components = [{key: detector.artifact[key] for key in ("model_version", "sha256", "input_shape")}
                      for detector in (waste, general)]
        identity = {"components": components, "merge": "same-class-iou065-max50-v1", "labels": COMBINED_LABELS}
        checksum = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        self.artifact = {"model_version": f"broad-coco-{checksum[:12]}", "sha256": checksum,
                         "labels": self.labels, "scope": "broad-waste-plus-coco-experimental",
                         "components": components, "merge_policy": identity["merge"]}

    def detect(self, image_bytes: bytes) -> dict:
        started = perf_counter()
        waste = self.waste.detect(image_bytes)
        general = self.general.detect(image_bytes)
        if waste["image"] != general["image"]:
            raise ValueError("Dimensi gambar kedua checkpoint tidak cocok.")
        # Only equivalent labels compete here; conflicting classes remain visible.
        objects = []
        for obj in sorted(waste["objects"] + general["objects"], key=lambda obj: obj["confidence"], reverse=True):
            candidate = RawDetection(class_id=COMBINED_LABELS.index(obj["raw_class"]), raw_class=obj["raw_class"],
                                     confidence=obj["confidence"], bbox=tuple(obj["bbox"]))
            if not any(other.class_id == candidate.class_id and box_iou(other.bbox, candidate.bbox) >= .65
                       for other in objects):
                objects.append(candidate)
            if len(objects) == 50:
                break
        return {**review_detections(objects, self.labels, model_version=self.artifact["model_version"],
                                    model_sha256=self.artifact["sha256"]),
                "model": self.artifact, "image": waste["image"],
                "timing_ms": {"total": round((perf_counter() - started) * 1000, 2)}}


def load_detector(model_path: Path, device="cpu", input_shape="adaptive", general_model_path: Path | None = None):
    primary = YoloDetector(model_path, device, input_shape)
    if general_model_path is None:
        return primary
    general = YoloDetector(general_model_path, device, input_shape)
    return CombinedDetector(primary, general)
