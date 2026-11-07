"""Compare a completed candidate against its previous checkpoint on the same data."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".cache-ai/ultralytics"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache-ai/matplotlib"))
os.environ["YOLO_AUTOINSTALL"] = "false"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--previous", required=True, type=Path)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    report = json.loads(args.report.read_text("utf-8"))
    data = ROOT / "data/local" / report["dataset"]
    manifest = json.loads((data / "manifest.json").read_text("utf-8"))
    if sha(data / "manifest.json") != report["manifest_sha256"]:
        raise ValueError("Dataset changed after candidate evaluation")
    if sha(args.previous) != report["comparison_model_sha256"]:
        raise ValueError("Previous checkpoint differs from the fixed-validation comparison")
    for split, checksum in report["split_list_sha256"].items():
        if sha(data / (split + ".txt")) != checksum:
            raise ValueError("Split list changed after candidate evaluation")
    candidate = report["fixed_validation"]["candidate"]
    previous = report["fixed_validation"]["previous_broad"]
    if candidate["protocol"] != previous["protocol"] or candidate["mapped_classes"] != previous["mapped_classes"]:
        raise ValueError("Fixed-validation protocols differ")
    from ultralytics import YOLO, settings
    settings.update({"sync": False})
    model = YOLO(str(args.previous.resolve()))
    if model.names != dict(enumerate(manifest["labels"])):
        raise ValueError("Previous checkpoint taxonomy differs")
    run = ROOT / report["weights"]
    run = run.parent.parent
    test = model.val(data=str(data / "data.yaml"), split="test", device=args.device, workers=0,
                     batch=16, imgsz=640, rect=False, conf=.001, iou=.7, max_det=50, plots=False,
                     project=str(run), name="previous-test-ap")
    history = list(csv.DictReader((run / "results.csv").open(encoding="utf-8")))
    best = max(history, key=lambda row: float(row["metrics/mAP50-95(B)"]))
    gates = {"bottle": (.85, .60), "can": (.80, .65), "cup": (.80, .40)}
    rows = {}
    for label in manifest["labels"]:
        old, new = previous["per_class"][label], candidate["per_class"][label]
        rows[label] = {"previous": old, "candidate": new,
                       "validation_support": new["tp"] + new["fn"]}
        if label in gates:
            p, r = gates[label]
            rows[label]["common_usability_gate"] = {"precision_min": p, "recall_min": r,
                "passed": (new["precision"] or 0) >= p and (new["recall"] or 0) >= r}
    summary = {"dataset": report["dataset"], "manifest_sha256": report["manifest_sha256"],
               "split_list_sha256": report["split_list_sha256"],
               "candidate_sha256": report["sha256"], "previous_sha256": report["comparison_model_sha256"],
               "training": {"epochs_completed": len(history), "best_epoch": int(float(best["epoch"])),
                            "last_csv_time_seconds": float(history[-1]["time"])},
               "test_protocol": {"split": "test", "imgsz": 640, "rect": False, "conf": .001, "iou": .7, "max_det": 50},
               "candidate_test_metrics": report["test_metrics"],
               "previous_test_metrics": {k: float(v) for k, v in test.results_dict.items()},
               "fixed_validation_protocol": candidate["protocol"], "per_class_validation": rows,
               "promoted": False, "limitations": report["limitations"] + [
                   "Both test evaluations use the same images and ten-class taxonomy; do not compare old nine-class aggregate AP.",
                   "Common-class usability gates are not a release gate; sparse classes and local-camera generalization remain unproven."]}
    target = args.report.with_name(args.report.stem + "-comparison.json")
    target.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(target), "training": summary["training"],
                      "candidate_test_metrics": summary["candidate_test_metrics"],
                      "previous_test_metrics": summary["previous_test_metrics"]}), flush=True)


if __name__ == "__main__":
    main()
