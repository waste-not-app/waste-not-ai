"""Fine-tune and compare the native COCO detector on the reviewed focus dataset."""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
from time import perf_counter

from train_broad_waste import ROOT, evaluate_fixed, sha
from wastenot_ai.taxonomy import COCO_LABELS


def validate_dataset(data, expected_labels):
    import yaml

    manifest = json.loads((data / "manifest.json").read_text("utf-8"))
    config = yaml.safe_load((data / "data.yaml").read_text("utf-8"))
    if manifest["labels"] != list(expected_labels) or config["names"] != dict(enumerate(expected_labels)):
        raise ValueError("Focus dataset must retain the complete native COCO taxonomy")
    if Path(config["path"]).resolve() != data or any(config.get(split) != split + ".txt" for split in ("train", "val", "test")):
        raise ValueError("Dataset YAML must reference the reviewed dataset and its split lists")
    for split in ("train", "val", "test"):
        listed = (data / (split + ".txt")).read_text("utf-8").splitlines()
        expected = {(data / row["image"]).as_posix() for row in manifest["records"] if row["split"] == split}
        if not listed or len(listed) != len(set(listed)) or set(listed) != expected:
            raise ValueError(f"Dataset {split} list differs from the reviewed manifest")
    return manifest


def ap_summary(model, data, device, destination, name):
    metrics = model.val(data=str(data / "data.yaml"), split="test", device=device,
                        workers=0, batch=16, imgsz=640, rect=False, conf=.001,
                        iou=.7, max_det=50, plots=True, project=str(destination), name=name)
    classes = []
    for index, cls in enumerate(metrics.box.ap_class_index):
        precision, recall, ap50, ap = metrics.box.class_result(index)
        classes.append({"name": COCO_LABELS[int(cls)], "precision": float(precision),
                        "recall": float(recall), "map50": float(ap50), "map50_95": float(ap)})
    return {"metrics": {key: float(value) for key, value in metrics.results_dict.items()}, "per_class": classes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/local/general-confusion-v1")
    parser.add_argument("--initial-model", type=Path, default=ROOT / "artifacts/models/yolo11n.pt")
    parser.add_argument("--name", default="general-confusion-v1-yolo11n")
    parser.add_argument("--device", default="0")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr0", type=float, default=.0001)
    parser.add_argument("--freeze", type=int, default=10)
    parser.add_argument("--warmup-bias-lr", type=float, default=.1)
    parser.add_argument("--warmup-epochs", type=float, default=3)
    parser.add_argument("--evaluate-only", type=Path)
    args = parser.parse_args()

    import torch
    from ultralytics import YOLO, settings

    settings.update({"sync": False, "weights_dir": str(ROOT / "artifacts/models")})
    torch.set_num_threads(8)
    if args.device != "cpu" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable")
    if Path(args.name).name != args.name:
        raise ValueError("Run name must be a single directory name")
    data = args.data.resolve()
    manifest = validate_dataset(data, COCO_LABELS)
    initial = args.initial_model.resolve()
    baseline = YOLO(str(initial))
    if baseline.names != dict(enumerate(COCO_LABELS)):
        raise ValueError("Initial checkpoint must use the native 80 COCO labels")
    input_hashes = {"manifest": sha(data / "manifest.json"), "initial_model": sha(initial),
                    **{split: sha(data / (split + ".txt")) for split in ("train", "val", "test")}}
    run = ROOT / "artifacts/training" / args.name
    if args.evaluate_only:
        best = args.evaluate_only.resolve()
        run.mkdir(parents=True, exist_ok=True)
    else:
        if run.exists():
            raise ValueError("Run exists. Choose a fresh --name")
        started = perf_counter()

        def progress(trainer):
            snapshot = {"epoch_completed": trainer.epoch + 1, "max_epochs": args.epochs,
                        "elapsed_seconds": perf_counter() - started,
                        "metrics": {key: float(value) for key, value in trainer.metrics.items()}}
            path = Path(trainer.save_dir) / "progress.json"
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")
            temporary.replace(path)

        baseline.add_callback("on_fit_epoch_end", progress)
        baseline.train(data=str(data / "data.yaml"), device=args.device, imgsz=640, batch=16,
                       workers=4, epochs=args.epochs, project=str(run.parent), name=run.name,
                       optimizer="AdamW", lr0=args.lr0, lrf=.01, cos_lr=True, freeze=args.freeze,
                       warmup_bias_lr=args.warmup_bias_lr, warmup_epochs=args.warmup_epochs,
                       patience=10, close_mosaic=6, mosaic=.5, degrees=15, flipud=0, scale=.35,
                       seed=42, deterministic=True, amp=True, cache=False, plots=True,
                       save=True, save_period=5, verbose=False)
        best = Path(baseline.trainer.save_dir) / "weights/best.pt"

    candidate = YOLO(str(best))
    baseline = YOLO(str(initial))
    if candidate.names != dict(enumerate(COCO_LABELS)):
        raise ValueError("Candidate taxonomy mismatch")
    mapping = dict(enumerate(range(len(COCO_LABELS))))
    comparison = {}
    for name, model in (("baseline", baseline), ("candidate", candidate)):
        comparison[name] = {"test_ap": ap_summary(model, data, args.device, run, name + "-test-ap"),
                            "fixed_validation": evaluate_fixed(model, data, COCO_LABELS, args.device, mapping)}

    if input_hashes != {"manifest": sha(data / "manifest.json"), "initial_model": sha(initial),
                        **{split: sha(data / (split + ".txt")) for split in ("train", "val", "test")}}:
        raise ValueError("Dataset or baseline changed during training/evaluation")
    report = {"kind": "experimental_general_focus_fine_tuning", "created_at": datetime.now(timezone.utc).isoformat(),
              "weights": best.relative_to(ROOT).as_posix(), "sha256": sha(best),
              "dataset": data.name, "input_sha256": input_hashes, "labels": list(COCO_LABELS),
              "images_per_split": manifest["images_per_split"],
              "instances_per_split": manifest["instances_per_split"],
              "source_images": dict(Counter(row["source"] for row in manifest["records"])),
              "negative_images": manifest["negative_images"], "comparison": comparison,
              "environment": {"python": platform.python_version(), "torch": torch.__version__,
                              "ultralytics": importlib.metadata.version("ultralytics"), "cuda": torch.version.cuda,
                              "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"},
              "selection": "Best validation mAP50:95 checkpoint from the training run",
              "limitations": manifest["limitations"], "promoted": False}
    history_path = best.parent.parent / "results.csv"
    if history_path.exists():
        history = list(csv.DictReader(history_path.open(encoding="utf-8")))
        best_epoch = max(history, key=lambda row: float(row["metrics/mAP50-95(B)"]))
        report["training"] = {"epochs_completed": len(history), "best_epoch": int(float(best_epoch["epoch"])),
                              "last_csv_time_seconds": float(history[-1]["time"])}
    (run / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    shutil.copyfile(data / "DATASET_NOTICES.md", run / "DATASET_NOTICES.md")
    destination = ROOT / "evaluations" / (args.name + "-comparison.json")
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(destination), "sha256": report["sha256"],
                      "test_metrics": {name: result["test_ap"]["metrics"] for name, result in comparison.items()}}))


if __name__ == "__main__":
    main()
