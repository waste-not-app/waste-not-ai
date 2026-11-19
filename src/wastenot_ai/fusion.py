"""Shared runtime and replay rules for combining native detector predictions."""

from dataclasses import dataclass

from .contracts import RawDetection
from .detection_quality import box_iou
from .taxonomy import BROAD_LABELS, COCO_LABELS, COMBINED_LABELS

LABEL_IDS = {label: index for index, label in enumerate(COMBINED_LABELS)}
NATIVE_LABELS = {"broad": BROAD_LABELS, "general": COCO_LABELS}
MAX_DETECTIONS = 50
MERGE_OPTIONS = (0.55, 0.65, 0.75)


@dataclass(frozen=True)
class FusionPolicy:
    merge_iou: float = 0.65
    thresholds: tuple[tuple[str, float], ...] = ()

    def __post_init__(self):
        if self.merge_iou not in MERGE_OPTIONS:
            raise ValueError("Unregistered fusion IoU")
        thresholds = dict(self.thresholds)
        if len(thresholds) != len(self.thresholds) or any(
            label not in LABEL_IDS or type(value) not in (float, int) or not 0.1 <= value <= 0.9
            for label, value in self.thresholds
        ):
            raise ValueError("Invalid or duplicate class threshold")
        object.__setattr__(self, "thresholds", tuple(sorted(thresholds.items())))


def fuse(predictions, policy=FusionPolicy(), *, floor=0.25):
    thresholds = dict(policy.thresholds)
    candidates = []
    for source, rows in predictions.items():
        for row in rows:
            raw = row if isinstance(row, RawDetection) else (
                RawDetection(class_id=row["class_id"], raw_class=row["raw_class"], confidence=row["confidence"],
                             bbox=tuple(row["bbox"]))
            )
            if raw.class_id >= len(NATIVE_LABELS[source]) or NATIVE_LABELS[source][raw.class_id] != raw.raw_class:
                raise ValueError("Native class provenance mismatch")
            if raw.confidence >= thresholds.get(raw.raw_class, floor):
                candidates.append((source, raw))
    # Match legacy priority for equal scores: Broad first, then native predictor order.
    candidates.sort(key=lambda item: (-item[1].confidence, item[0]))
    kept = []
    for source, raw in candidates:
        if any(other.raw_class == raw.raw_class and box_iou(other.bbox, raw.bbox) >= policy.merge_iou
               for _, other in kept):
            continue
        kept.append((source, raw.model_copy(update={"class_id": LABEL_IDS[raw.raw_class]})))
        if len(kept) == MAX_DETECTIONS:
            break
    return kept
