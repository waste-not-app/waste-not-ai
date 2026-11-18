"""Operational detection scoring for a frozen, partially annotated image collection."""

import hashlib
import json
import math
from collections import Counter
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageEnhance

from .detection_quality import box_iou


def repository_path(root: Path, relative: str) -> Path:
    """Resolve frozen monorepo paths after extraction into the AI repository."""
    path = Path(relative)
    if path.parts and path.parts[0] == "ai":
        path = Path(*path.parts[1:])
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("Benchmark image escapes repository root")
    return resolved


def load_manifest(path: Path, root: Path):
    manifest = json.loads(path.read_text("utf-8"))
    if manifest.get("usage") != "EVALUATION_ONLY" or not manifest.get("records"):
        raise ValueError("A nonempty evaluation-only manifest is required")
    labels = manifest["target_classes"]
    if not labels or len(set(labels)) != len(labels):
        raise ValueError("Target classes must be unique")
    identifiers, hashes = set(), set()
    for row in manifest["records"]:
        image = repository_path(root, row["image"])
        if row["id"] in identifiers or row["sha256"] in hashes:
            raise ValueError("Duplicate benchmark image or ID")
        if hashlib.sha256(image.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError("Benchmark image checksum mismatch")
        scope = row["evaluated_classes"]
        if not scope or not set(scope).issubset(labels) or len(scope) != len(set(scope)):
            raise ValueError("Invalid annotation coverage")
        for truth in row["truth"]:
            bbox = truth["bbox"]
            if truth["label"] not in scope or len(bbox) != 4 or not all(math.isfinite(x) for x in bbox):
                raise ValueError("Invalid ground truth")
            x1, y1, x2, y2 = bbox
            if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
                raise ValueError("Ground-truth box outside normalized image")
        identifiers.add(row["id"])
        hashes.add(row["sha256"])
    return manifest


def browser_input(path: Path, brightness: float = 1.0) -> bytes:
    with Image.open(path) as source:
        if source.getexif().get(274, 1) != 1:
            raise ValueError("Normalize image orientation and annotations before benchmarking")
        image = source.convert("RGB")
    ratio = min(1, 1600 / max(image.size))
    size = tuple(max(1, math.floor(side * ratio + 0.5)) for side in image.size)
    if image.size != size:
        resized = image.resize(size, Image.Resampling.BILINEAR)
        image.close()
        image = resized
    if brightness != 1:
        adjusted = ImageEnhance.Brightness(image).enhance(brightness)
        image.close()
        image = adjusted
    content = BytesIO()
    image.save(content, format="JPEG", quality=90)
    image.close()
    return content.getvalue()


def score_image(objects, truth, evaluated_classes, iou_threshold=0.5):
    """Class-aware confidence-ordered matching; every detection/truth matches once."""
    selected = [obj for obj in objects if obj["supported_for_review"]]
    scoped = [obj for obj in selected if obj["raw_class"] in evaluated_classes]
    per_class = {label: {"tp": 0, "fp": 0, "fn": 0} for label in evaluated_classes}
    unmatched = set(range(len(truth)))
    errors = []
    for obj in sorted(scoped, key=lambda obj: -obj["confidence"]):
        options = [index for index in unmatched if truth[index]["label"] == obj["raw_class"]]
        best = max(options, key=lambda index: box_iou(obj["bbox"], truth[index]["bbox"]), default=None)
        if best is not None and box_iou(obj["bbox"], truth[best]["bbox"]) >= iou_threshold:
            per_class[obj["raw_class"]]["tp"] += 1
            unmatched.remove(best)
        else:
            per_class[obj["raw_class"]]["fp"] += 1
            errors.append({"kind": "false_positive", "predicted": obj["raw_class"], "score": obj["confidence"]})
    for index in sorted(unmatched):
        label = truth[index]["label"]
        per_class[label]["fn"] += 1
        errors.append({"kind": "false_negative", "truth": label})

    edges = [
        (box_iou(obj["bbox"], target["bbox"]), obj["confidence"], i, j)
        for i, target in enumerate(truth)
        for j, obj in enumerate(selected)
        if box_iou(obj["bbox"], target["bbox"]) >= iou_threshold
    ]
    matched_truth, matched_predictions = set(), set()
    confusion = Counter()
    for overlap, confidence, i, j in sorted(edges, reverse=True):
        if i in matched_truth or j in matched_predictions:
            continue
        matched_truth.add(i)
        matched_predictions.add(j)
        expected, predicted = truth[i]["label"], selected[j]["raw_class"]
        confusion[(expected, predicted)] += 1
        if expected != predicted:
            errors.append(
                {"kind": "wrong_label", "truth": expected, "predicted": predicted, "iou": overlap, "score": confidence}
            )
    for i, target in enumerate(truth):
        if i not in matched_truth:
            confusion[(target["label"], "__missed__")] += 1

    duplicate_pairs = 0
    conflicting_pairs = 0
    for i, first in enumerate(selected):
        for second in selected[i + 1 :]:
            if box_iou(first["bbox"], second["bbox"]) >= 0.65:
                if first["raw_class"] == second["raw_class"]:
                    duplicate_pairs += 1
                else:
                    conflicting_pairs += 1
    return {
        "per_class": per_class,
        "errors": errors,
        "confusion": [
            {"truth": expected, "predicted": predicted, "count": count}
            for (expected, predicted), count in sorted(confusion.items())
        ],
        "same_class_duplicate_pairs": duplicate_pairs,
        "conflicting_label_pairs": conflicting_pairs,
        "reviewable_candidates": len(selected),
        "scored_candidates": len(scoped),
        "unscored_candidates": len(selected) - len(scoped),
        "empty_target_image": not truth,
        "false_positive_on_empty": not truth and bool(scoped),
    }


def metrics(counts):
    tp, fp, fn = (counts[key] for key in ("tp", "fp", "fn"))
    return {
        **counts,
        "support": tp + fn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
    }


def aggregate(rows, target_classes):
    per_class = {}
    for label in target_classes:
        counts = {
            key: sum(row["score"]["per_class"].get(label, {}).get(key, 0) for row in rows) for key in ("tp", "fp", "fn")
        }
        per_class[label] = metrics(counts)
    overall = metrics({key: sum(row[key] for row in per_class.values()) for key in ("tp", "fp", "fn")})
    confusion = Counter()
    for row in rows:
        for pair in row["score"]["confusion"]:
            confusion[(pair["truth"], pair["predicted"])] += pair["count"]
    return {
        "images": len(rows),
        "overall": overall,
        "per_class": per_class,
        "wrong_label_instances": sum(
            pair["count"]
            for row in rows
            for pair in row["score"]["confusion"]
            if pair["predicted"] not in {pair["truth"], "__missed__"}
        ),
        "no_overlapping_candidate_instances": sum(confusion[(label, "__missed__")] for label in target_classes),
        "same_class_duplicate_pairs": sum(row["score"]["same_class_duplicate_pairs"] for row in rows),
        "conflicting_label_pairs": sum(row["score"]["conflicting_label_pairs"] for row in rows),
        "images_without_reviewable_candidate": sum(row["score"]["reviewable_candidates"] == 0 for row in rows),
        "empty_target_images": sum(row["score"]["empty_target_image"] for row in rows),
        "images_with_false_positive_on_empty": sum(row["score"]["false_positive_on_empty"] for row in rows),
        "confusion": [
            {"truth": expected, "predicted": predicted, "count": count}
            for (expected, predicted), count in sorted(confusion.items())
        ],
    }
