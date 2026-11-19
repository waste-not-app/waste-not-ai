"""Fit combined-detector policy on validation predictions and freeze it for final evaluation."""

import argparse
import importlib.metadata
import json
from collections import Counter
from pathlib import Path

from benchmark_resolutions import ROOT, freeze_components, validation_manifest
from prepare_external_focus import phash_neighbors
from training_experiments import sha, write

from wastenot_ai.camera_benchmark import aggregate, load_manifest, score_image
from wastenot_ai.detector import CombinedDetector, YoloDetector
from wastenot_ai.fusion import MERGE_OPTIONS, FusionPolicy, fuse
from wastenot_ai.resolution_benchmark import FOCUS, prepare_inputs

PROTOCOL = {"split": "val", "floor": 0.05, "native_nms_iou": 0.7, "native_max_det": 50,
            "match_iou": 0.5, "merge_max_det": 50, "preprocessing": "browser-jpeg90-long1600-exif-rgb-square"}
BOOTSTRAP_DRAWS = 1000


def component_args(parser):
    parser.add_argument("--threads", type=int, default=8)
    for branch in ("broad", "general"):
        for field in ("data", "weights", "receipt"):
            parser.add_argument(f"--{branch}-{field}", required=True, type=Path)


def specs_from(args):
    return {branch: tuple(getattr(args, f"{branch}_{field}").resolve() for field in ("data", "weights", "receipt"))
            for branch in ("broad", "general")}


def make_detector(weights, size, policy=None):
    return CombinedDetector(*(YoloDetector(path, "cpu", "square", size) for path in weights), policy=policy)


def runtime_identity():
    import torch

    version = importlib.metadata.version("ultralytics")
    if version != "8.4.183":
        raise ValueError("Use pinned Ultralytics 8.4.183")
    return {"ultralytics": version, "torch": torch.__version__, "threads": torch.get_num_threads(),
            "pillow": importlib.metadata.version("Pillow"),
            "code": {path: sha(ROOT / path) for path in (
                "scripts/calibrate_combined.py", "scripts/evaluate_tuning_holdout.py", "examples/tuning-evaluation-protocol.json",
                "src/wastenot_ai/fusion.py", "src/wastenot_ai/detector.py", "src/wastenot_ai/catalog.py",
                "src/wastenot_ai/camera_benchmark.py", "src/wastenot_ai/detection_quality.py", "src/wastenot_ai/data/catalog.json",
                "src/wastenot_ai/resolution_benchmark.py", "src/wastenot_ai/taxonomy.py", "src/wastenot_ai/media.py")}}


def checked_resolution(report, frozen, manifest_path):
    if (report.get("status") != "VALIDATION_EXPERIMENT_PENDING_FINAL_REVIEW" or report.get("holdout_opened") is not False
            or report.get("promoted") is not False or report["frozen_components"] != frozen
            or report["manifest_sha256"] != sha(manifest_path)):
        raise ValueError("Resolution report does not match the frozen components and validation manifest")
    size = report["selection"]["selected_input_size"]
    if size not in {640, 768, 960}:
        raise ValueError("Unsupported resolution selection")
    return size


def collect(detector, inputs, identity, path):
    if path.exists():
        cache = json.loads(path.read_text("utf-8"))
        if cache["identity"] != identity or [r["id"] for r in cache["records"]] != [i["source"]["id"] for i in inputs]:
            raise ValueError("Prediction cache identity mismatch; use a new cache file")
        return cache["records"]
    records = []
    for item in inputs:
        native, image = detector.native_predictions(item["content"])
        if image != dict(zip(("width", "height"), item["transmitted_size"], strict=True)):
            raise ValueError("Prediction coordinates reference a different frame")
        records.append({"id": item["source"]["id"], "native": {
            source: [{key: row[key] for key in ("class_id", "raw_class", "confidence", "bbox")} for row in rows]
            for source, rows in native.items()}})
    temporary = path.with_suffix(".tmp")
    write(temporary, {"identity": identity, "records": records})
    temporary.replace(path)
    return records


def score(detector, manifest, cache):
    return aggregate([
        {"score": score_image(detector.review_predictions(row["native"])["objects"], source["truth"], source["evaluated_classes"])}
        for source, row in zip(manifest["records"], cache, strict=True)
    ], manifest["target_classes"])


def per_class_support(manifest):
    # A multi-source photo links its groups. Count connected groups only once.
    parents = {}

    def find(group):
        parents.setdefault(group, group)
        while parents[group] != group:
            group = parents[group]
        return group

    for row in manifest["records"]:
        for group in row.get("source_groups", []):
            parents[find(group)] = find(row["group"])
    rows = manifest["records"]
    for left, right in phash_neighbors(rows):
        parents[find(rows[left]["group"])] = find(rows[right]["group"])
    counts, groups = Counter(), {}
    for row in manifest["records"]:
        for truth in row["truth"]:
            counts[truth["label"]] += 1
            groups.setdefault(truth["label"], set()).add(find(row["group"]))
    return {label: {"instances": count, "groups": len(groups[label])} for label, count in counts.items()}


def metric(row, name):
    return row[name] or 0.0


def select(control, evaluate, support, labels):
    thresholds, tuning = {}, {}
    for label in labels:
        old = control["per_class"][label]
        evidence = support.get(label, {})
        if evidence.get("instances", 0) < 50 or evidence.get("groups", 0) < 20:
            tuning[label] = {"status": "KEEP_CONTROL_LOW_SUPPORT", **evidence}
            continue
        trials = {}

        def trial(value):
            if value not in trials:
                trials[value] = evaluate(FusionPolicy(thresholds=tuple({**thresholds, label: value}.items())))["per_class"][label]

        def rank(values):
            eligible = [value for value in values if all(metric(trials[value], key) >= metric(old, key) - .02 - 1e-12
                                                       for key in ("precision", "recall"))]
            return max(eligible, key=lambda value: (metric(trials[value], "f1"), -abs(value - .5), -value), default=None)

        for value in range(10, 91, 5):
            trial(value / 100)
        center = rank(trials)
        if center is not None:
            for value in range(max(10, round(center * 100) - 4), min(90, round(center * 100) + 4) + 1):
                trial(value / 100)
        winner = rank(trials)
        improved = winner is not None and metric(trials[winner], "f1") > metric(old, "f1") + 1e-12
        if improved:
            thresholds[label] = winner
        tuning[label] = {"status": "SELECTED" if improved else "KEEP_CONTROL", "lower": winner if improved else .5,
                         "trials": trials, **evidence}
    comparison = {iou: evaluate(FusionPolicy(iou, tuple(thresholds.items()))) for iou in MERGE_OPTIONS}
    viable = []
    for iou, result in comparison.items():
        if metric(result["overall"], "f1") < metric(control["overall"], "f1") - 1e-12:
            continue
        if any(metric(result["per_class"][label], key) < metric(control["per_class"][label], key) - .02 - 1e-12
               for label in labels if control["per_class"][label]["support"] >= 50 for key in ("precision", "recall")):
            continue
        gain = sum(metric(result["per_class"][label], "f1") - metric(control["per_class"][label], "f1") for label in FOCUS)
        if gain > 1e-12:
            viable.append((gain, -abs(iou - .65), iou))
    winner = max(viable, default=None)
    return (FusionPolicy(winner[-1], tuple(thresholds.items())) if winner else None), tuning, comparison


def fusion_diagnostics(cache, policy):
    shared = Counter()
    conflicts = Counter()
    pairs = ({"cell phone", "remote"}, {"cardboard", "paper", "drink_carton"}, {"can", "bottle"})
    from wastenot_ai.detection_quality import box_iou

    for row in cache:
        kept = fuse(row["native"], policy)
        for source, raw in kept:
            if raw.raw_class in {"bottle", "cup"}:
                shared[(source, raw.raw_class)] += 1
        for index, (_, a) in enumerate(kept):
            for _, b in kept[index + 1:]:
                labels = {a.raw_class, b.raw_class}
                if len(labels) == 2 and any(labels <= pair for pair in pairs) and box_iou(a.bbox, b.bbox) >= .65:
                    conflicts[" / ".join(sorted(labels))] += 1
    return {"kept_shared_classes_by_source": [{"source": s, "label": label, "count": n} for (s, label), n in sorted(shared.items())],
            "overlapping_conflicting_labels": dict(conflicts)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    component_args(parser)
    for field in ("resolution-report", "validation", "cache", "output"):
        parser.add_argument(f"--{field}", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists() or args.threads < 1:
        raise ValueError("Use a fresh output and positive thread count")
    specs = specs_from(args)
    frozen = freeze_components(specs)
    resolution = json.loads(args.resolution_report.read_text("utf-8"))
    size = checked_resolution(resolution, frozen, args.validation)
    manifest = load_manifest(args.validation, ROOT, allowed_usage=("VALIDATION_ONLY",))
    if manifest != validation_manifest(specs):
        raise ValueError("Validation manifest differs from the reviewed dataset snapshot")
    inputs = prepare_inputs(manifest, ROOT, {"browser_jpeg90": 1.0})
    import torch

    torch.set_num_threads(args.threads)
    weights = tuple(specs[branch][1] for branch in ("broad", "general"))
    detector = make_detector(weights, size)
    identity = {"protocol": PROTOCOL, "runtime": runtime_identity(), "artifact": detector.artifact,
                "manifest_sha256": sha(args.validation), "inputs_sha256": [item["input_sha256"] for item in inputs]}
    # Normalize integer label keys and tuples for a byte-independent JSON round trip.
    identity = json.loads(json.dumps(identity))
    cache = collect(detector, inputs, identity, args.cache)
    control = score(detector, manifest, cache)
    support = per_class_support(manifest)
    policy, tuning, comparison = select(control, lambda value: score(CombinedDetector(detector.waste, detector.general, policy=value),
                                                                     manifest, cache), support, manifest["target_classes"])
    released = json.loads((ROOT / "model.json").read_text("utf-8"))
    candidate = CombinedDetector(detector.waste, detector.general, policy=policy) if policy else detector
    if not policy and candidate.artifact["model_version"] == released["model_version"]:
        candidate = None
    if frozen != freeze_components(specs) or sha(args.validation) != identity["manifest_sha256"]:
        raise ValueError("Validation inputs changed during calibration")
    write(args.output, {"status": "FROZEN_FOR_HOLDOUT" if candidate else "KEEP_BASELINE_VALIDATION", "promoted": False,
                        "holdout_opened": False, "frozen_components": frozen, "input_size": size,
                        "identity": identity, "cache_sha256": sha(args.cache), "resolution_report_sha256": sha(args.resolution_report),
                        "validation_support": support, "control": control, "threshold_trials": tuning, "merge_trials": comparison,
                        "diagnostics": {str(iou): fusion_diagnostics(cache, FusionPolicy(iou, policy.thresholds if policy else ()))
                                        for iou in MERGE_OPTIONS},
                        "policy": {"merge_iou": policy.merge_iou, "thresholds": policy.thresholds} if policy else None,
                        "candidate_artifact": candidate.artifact if candidate else None, "bootstrap_draws": BOOTSTRAP_DRAWS})


if __name__ == "__main__":
    main()
