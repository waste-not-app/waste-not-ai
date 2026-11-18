"""Replay the v4 review threshold on validation, held-out test and frozen camera predictions."""
import argparse
import json
from unittest.mock import patch

from train_broad_waste import ROOT, evaluate_fixed, sha

from wastenot_ai.camera_benchmark import aggregate, load_manifest, repository_path, score_image
from wastenot_ai.catalog import review_detections
from wastenot_ai.contracts import RawDetection
from wastenot_ai.detection_quality import review_profile
from wastenot_ai.taxonomy import BROAD_LABELS, COMBINED_LABELS

DATA = ROOT / "data/local/broad-waste-v4"
RUN = ROOT / "artifacts/training/broad-waste-v4-focus-yolo11s"
CAMERA = ROOT / "evaluations/camera-replay-v1-v4-focus-comparison.json"
TEST_CACHE = ROOT / "artifacts/cardboard-v4-test-predictions.json"
CANDIDATE = ROOT / "evaluations/cardboard-v4-threshold-candidate.json"


def read(path):
    return json.loads(path.read_text("utf-8"))


def candidate_profile(labels, model_version=None, model_sha256=None):
    profile = review_profile(labels, model_version, model_sha256)
    for key, names in (("broad", BROAD_LABELS), ("combined", COMBINED_LABELS)):
        if labels != dict(enumerate(names)):
            continue
        candidate = read(CANDIDATE)
        artifact = candidate["artifacts"][key]
        if model_version == artifact["model_version"] and model_sha256 == artifact["sha256"]:
            profile["id"] = candidate["id"] + ("-combined" if key == "combined" else "")
            profile["classes"].update(candidate["classes"])
    return profile


def reviewed(objects, labels, artifact=None):
    raw = [RawDetection(class_id=obj["class_id"], raw_class=labels[obj["class_id"]],
                        confidence=obj["confidence"], bbox=tuple(obj["bbox"])) for obj in objects]
    identity = {} if artifact is None else {"model_version": artifact["model_version"], "model_sha256": artifact["sha256"]}
    if artifact is None:
        return review_detections(raw, labels)
    with patch("wastenot_ai.catalog.review_profile", side_effect=candidate_profile):
        return review_detections(raw, labels, **identity)


def score_dataset(cached, artifact):
    labels = dict(enumerate(BROAD_LABELS))
    rows = {"before": [], "after": []}
    for record in cached["records"]:
        truth = [{"label": labels[obj["class_id"]], "bbox": obj["bbox"]} for obj in record["truth"]]
        for mode, identity in (("before", None), ("after", artifact)):
            objects = reviewed(record["predictions"], labels, identity)["objects"]
            rows[mode].append({"score": score_image(objects, truth, list(BROAD_LABELS))})
    return {mode: aggregate(records, list(BROAD_LABELS)) for mode, records in rows.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collect-test", action="store_true", help="Collect fixed test predictions once from the exact v4 checkpoint")
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    camera = read(CAMERA)
    benchmark_path = repository_path(ROOT, camera["manifest"])
    benchmark = load_manifest(benchmark_path, ROOT)
    if sha(benchmark_path) != camera["manifest_sha256"]:
        raise ValueError("Frozen benchmark identity changed")
    training = read(RUN / "report.json")
    artifact = camera["models"]["v4"]
    if sha(RUN / "weights/best.pt") != artifact["sha256"] or sha(DATA / "manifest.json") != training["manifest_sha256"]:
        raise ValueError("Checkpoint or dataset differs from the recorded v4 experiment")

    if args.collect_test:
        if TEST_CACHE.exists():
            raise ValueError("Test prediction snapshot exists. Replay it without --collect-test")
        import torch
        from ultralytics import YOLO

        torch.set_num_threads(8)
        model = YOLO(str(RUN / "weights/best.pt"))
        labels = list(BROAD_LABELS)
        if model.names != dict(enumerate(labels)):
            raise ValueError("Checkpoint taxonomy mismatch")
        predictions = evaluate_fixed(model, DATA, labels, args.device,
                                     dict(enumerate(range(len(labels)))), split="test")
        predictions["model_sha256"] = artifact["sha256"]
        predictions["manifest_sha256"] = sha(DATA / "manifest.json")
        predictions["split_sha256"] = sha(DATA / "test.txt")
        TEST_CACHE.write_text(json.dumps(predictions, indent=2) + "\n", encoding="utf-8")
    test = read(TEST_CACHE)
    if (test["model_sha256"] != artifact["sha256"] or test["manifest_sha256"] != sha(DATA / "manifest.json")
            or test["split_sha256"] != sha(DATA / "test.txt")):
        raise ValueError("Test prediction snapshot identity mismatch")

    camera_truth = {row["id"]: row for row in benchmark["records"]}
    replay = {}
    for profile in camera["results"]:
        replay[profile] = {}
        for model in ("v4", "combined"):
            model_artifact = camera["models"][model]
            labels = {int(key): label for key, label in model_artifact["labels"].items()}
            rows = {"before": [], "after": []}
            for row in camera["records"]:
                if row["profile"] != profile or row["model"] != model:
                    continue
                source = camera_truth[row["id"]]
                before = reviewed(row["objects"], labels)["objects"]
                before_score = score_image(before, source["truth"], source["evaluated_classes"])
                if before_score != row["score"]:
                    raise ValueError("Cached camera scores cannot be reproduced")
                after = reviewed(row["objects"], labels, model_artifact)["objects"]
                rows["before"].append({"score": before_score})
                rows["after"].append({"score": score_image(after, source["truth"], source["evaluated_classes"])})
            replay[profile][model] = {mode: aggregate(records, camera["target_classes"]) for mode, records in rows.items()}

    report = {"kind": "validation_selected_cardboard_review_threshold", "old_threshold": .5, "new_threshold": .65,
              "selection": "0.65 chosen using v4 validation before evaluating held-out test and cached camera replay",
              "decision": "REJECTED_ON_TEST", "active_cardboard_threshold": .5,
              "validation": score_dataset(read(RUN / "candidate-fixed-val.json"), artifact),
              "test": score_dataset(test, artifact), "camera_replay": replay,
              "source_sha256": {"camera_report": sha(CAMERA), "benchmark": sha(benchmark_path),
                                "validation_predictions": sha(RUN / "candidate-fixed-val.json"),
                                "test_predictions": sha(TEST_CACHE), "manifest": sha(DATA / "manifest.json"),
                                "candidate_profile": sha(CANDIDATE)},
              "limitations": ["Review threshold changes candidate eligibility. Model weights and probability scores are identical.",
                              "Predictions below 0.5 were not retained by the fixed validation/test cache, which is sufficient for this upward threshold change.",
                              "Camera predictions are replayed through the actual review policy. Inference latency is not remeasured.",
                              "The public replay has already been inspected during development. Physical-camera generalization is unverified.",
                              "This policy does not resolve phone versus remote classification."]}
    destination = ROOT / "evaluations/cardboard-v4-calibration.json"
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({split: {mode: value["per_class"]["cardboard"] for mode, value in report[split].items()}
                      for split in ("validation", "test")}, indent=2))
    print(json.dumps({profile: {mode: value["overall"] for mode, value in modes["combined"].items()}
                      for profile, modes in replay.items()}, indent=2))


if __name__ == "__main__":
    main()
