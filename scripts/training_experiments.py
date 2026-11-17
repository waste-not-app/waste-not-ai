"""Validation-only, reproducible YOLO experiments; final holdout evaluation belongs to issue #12."""
import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
from time import perf_counter

import yaml

from prepare_tuning_data import inside, validate_boxes
from wastenot_ai.taxonomy import BROAD_LABELS, COCO_LABELS

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT/".cache-ai/ultralytics"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT/".cache-ai/matplotlib"))
os.environ["YOLO_AUTOINSTALL"] = "false"
FOCUS = {"broad": ("cardboard", "plastic_wrapper", "paper", "drink_carton"),
         "general": ("cell phone", "remote")}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 30
    imgsz: int = 640
    batch: int = 16
    workers: int = 4
    patience: int = 10
    seed: int = 42
    freeze: int = 10
    lr0: float = .0001
    lrf: float = .01
    nbs: int = 64
    warmup_epochs: float = 3.0
    warmup_bias_lr: float = .1
    close_mosaic: int = 6
    mosaic: float = .5
    degrees: float = 15.0
    scale: float = .35
    hsv_h: float = .015
    hsv_s: float = .7
    hsv_v: float = .4
    flipud: float = 0.0
    fliplr: float = .5
    translate: float = .1

    def validate(self):
        values = asdict(self)
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in values.values()):
            raise ValueError("Training parameters must be finite numbers")
        for name in ("epochs", "imgsz", "batch", "workers", "patience", "seed", "freeze", "nbs", "close_mosaic"):
            if not isinstance(values[name], int):
                raise ValueError(f"{name} must be an integer")
        if min(self.epochs, self.batch, self.patience, self.nbs) < 1 or self.imgsz < 32 or self.imgsz % 32:
            raise ValueError("Positive epochs/batch/patience/nbs and stride-32 imgsz required")
        if min(self.workers, self.seed, self.freeze, self.close_mosaic) < 0 or self.close_mosaic > self.epochs:
            raise ValueError("Invalid worker/seed/freeze/mosaic-close range")
        if not (0 < self.lr0 <= 1 and 0 < self.lrf <= 1 and self.warmup_epochs >= 0 and self.warmup_bias_lr >= 0):
            raise ValueError("Invalid learning rate or warmup")
        for name in ("mosaic", "scale", "hsv_h", "hsv_s", "hsv_v", "flipud", "fliplr", "translate"):
            if not 0 <= values[name] <= 1:
                raise ValueError(f"{name} must lie in [0,1]")
        if not 0 <= self.degrees <= 180:
            raise ValueError("Invalid rotation")
        return self

    def kwargs(self):
        return {**asdict(self.validate()), "optimizer": "AdamW", "cos_lr": True,
                "rect": False, "deterministic": True, "amp": True, "cache": False,
                "plots": True, "save": True, "save_period": 5, "verbose": False,
                "mixup": 0.0, "cutmix": 0.0, "copy_paste": 0.0, "bgr": 0.0, "multi_scale": 0.0}


def validate_dataset(data, labels, *, balanced=False):
    data = data.resolve()
    manifest = json.loads((data / "manifest.json").read_text("utf-8"))
    config = yaml.safe_load((data / "data.yaml").read_text("utf-8"))
    names = config.get("names")
    if names != dict(enumerate(labels)) or manifest["labels"] != list(labels):
        raise ValueError("Dataset must retain its registered taxonomy (native COCO taxonomy for general)")
    if Path(config["path"]).resolve() != data or any(config.get(s) != s+".txt" for s in ("train", "val")):
        raise ValueError("Dataset YAML must reference its audited train/val lists")
    # Never open test.txt, test images or test labels in this workflow.
    for split in ("train", "val"):
        lines = (data / (split+".txt")).read_text("utf-8").splitlines()
        expected = {(data / r["image"]).as_posix() for r in manifest["records"] if r["split"] == split}
        listed = set(lines)
        if not lines or len(lines) != len(listed) or not listed <= expected:
            raise ValueError(f"Dataset {split} list differs from manifest")
        if split != "train" or not balanced:
            if listed != expected:
                raise ValueError(f"Dataset {split} list differs from manifest")
        elif len(listed) != manifest.get("selected_train_images"):
            raise ValueError("Balanced train membership/count mismatch")
    return manifest


def dataset_snapshot(data, labels, balanced=False):
    manifest = validate_dataset(data, labels, balanced=balanced)
    evidence = json.loads((data / "readiness.json").read_text("utf-8"))
    if (evidence.get("status") != "READY_FOR_TRAINING_EXPERIMENT"
            or evidence.get("dataset_manifest_sha256") != sha(data / "manifest.json")
            or evidence.get("intake_sha256") != manifest.get("intake_sha256")
            or not manifest.get("intake_sha256")):
        raise ValueError("Dataset review/build evidence is missing, incomplete or stale")
    groups, hashes = {}, {}
    for row in manifest["records"]:
        if not row.get("group") or "phash" not in row or row["split"] not in {"train", "val", "test"}:
            raise ValueError("Dataset group/split/pHash evidence is incomplete")
        if groups.setdefault(row["group"], row["split"]) != row["split"]:
            raise ValueError("Capture group crosses splits")
        if hashes.setdefault(row["prepared_sha256"], row["split"]) != row["split"]:
            raise ValueError("Identical image crosses splits")
    from prepare_external_focus import phash_neighbors
    for i,j in phash_neighbors(manifest["records"]):
        if manifest["records"][i]["split"] != manifest["records"][j]["split"]:
            raise ValueError("Near-duplicate metadata crosses splits")
    active = {line for s in ("train", "val") for line in (data / (s+".txt")).read_text("utf-8").splitlines()}
    pixels, labels_digest = hashlib.sha256(), hashlib.sha256()
    for row in sorted(manifest["records"], key=lambda r: r["image"]):
        if row["split"] not in {"train", "val"} or (data / row["image"]).as_posix() not in active:
            continue
        image = inside(data, row["image"])
        if image.parent != data / "images" / row["split"] or sha(image) != row["prepared_sha256"]:
            raise ValueError("Training/validation image identity mismatch")
        label = inside(data, f"labels/{row['split']}/{image.with_suffix('.txt').name}")
        boxes = [list(map(float, line.split())) for line in label.read_text("utf-8").splitlines() if line.strip()]
        if len(boxes) != len(row["boxes"]) or any(len(a) != 5 or len(b) != 5 or
            any(not math.isfinite(x) or not math.isfinite(y) or abs(x-y) > 1e-7 for x,y in zip(a,b)) for a,b in zip(boxes,row["boxes"])):
            raise ValueError("Labels differ from reviewed manifest")
        validate_boxes(boxes, len(labels), edge_tolerance=.002 if row.get("source") == "DrinkingWasteV2" else 1e-12)
        if not boxes and not row.get("visual_review", {}).get("general_negative"):
            raise ValueError("Empty annotation is not a reviewed negative")
        pixels.update((row["image"]+":"+row["prepared_sha256"]+"\n").encode())
        labels_digest.update((row["image"]+":"+sha(label)+"\n").encode())
    return {"manifest": sha(data/"manifest.json"), "readiness": sha(data/"readiness.json"),
            "dataset_yaml": sha(data/"data.yaml"),
            **{s: sha(data/(s+".txt")) for s in ("train", "val")},
            "active_pixels": pixels.hexdigest(), "active_labels": labels_digest.hexdigest()}


def checkpoint_callbacks(run, started):
    def save(trainer):
        fitness, best = trainer.fitness, trainer.best_fitness
        if fitness is not None and best is not None and math.isfinite(float(fitness)) and fitness == best:
            write(run / "best-checkpoint-event.json", {
                "epoch_one_based": int(trainer.epoch)+1, "fitness": float(fitness),
                "rule": "trainer.fitness == trainer.best_fitness at on_model_save",
                "pre_strip_sha256": sha(Path(trainer.best))})

    def progress(trainer):
        batch, accumulate = int(trainer.batch_size), int(trainer.accumulate)
        snapshot = {"epoch_one_based": int(trainer.epoch)+1, "elapsed_seconds": perf_counter()-started,
                    "batch_size": batch, "accumulate": accumulate, "effective_batch": batch*accumulate,
                    "metrics": {k: float(v) for k,v in trainer.metrics.items() if math.isfinite(float(v))}}
        write(run / "progress.json", snapshot)
        with (run / "epochs.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(snapshot, allow_nan=False)+"\n")
        if hasattr(trainer, "train_loader") and not (run/"augmentation-pipeline.txt").exists():
            (run/"augmentation-pipeline.txt").write_text(repr(trainer.train_loader.dataset.transforms), encoding="utf-8")
    return save, progress


def ap_summary(model, config_path, device, run, name, config, labels):
    metrics = model.val(data=str(config_path), split="val", device=device, workers=config.workers,
                        batch=config.batch, imgsz=config.imgsz, rect=False, conf=.001, iou=.7,
                        max_det=50, plots=False, project=str(run), name=name)
    classes = []
    for index, cls in enumerate(metrics.box.ap_class_index):
        p, r, ap50, ap = metrics.box.class_result(index)
        classes.append({"name": labels[int(cls)], "precision": float(p), "recall": float(r),
                        "map50": float(ap50), "map50_95": float(ap)})
    return {"split": "val", "metrics": {k: float(v) for k,v in metrics.results_dict.items()}, "per_class": classes}


def experiment(args, branch, root=ROOT, model_factory=None):
    labels = list(BROAD_LABELS if branch == "broad" else COCO_LABELS)
    config = TrainingConfig(**{k: getattr(args, k) for k in asdict(TrainingConfig())}).validate()
    stage = getattr(args, "stage", "extra")
    if stage == "B0" and not args.evaluate_only:
        raise ValueError("B0 evaluates the baseline without training")
    if stage in {"B1", "B2", "B3"} and args.evaluate_only:
        raise ValueError("B1/B2/B3 must train; use B0 or extra to evaluate existing weights")
    if stage in {"B1", "B2"} and (config.freeze != (10 if stage == "B1" else 0) or config.lr0 != .0001 or config.seed != 42):
        raise ValueError("B1/B2 require their declared freeze/lr0/seed controls")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.name):
        raise ValueError("Run name must be a single safe directory name")
    if not re.fullmatch(r"cpu|[0-9]+", args.device):
        raise ValueError("Controlled experiments support cpu or one CUDA device index")
    run = root / "artifacts/training" / args.name
    if run.exists():
        raise ValueError("Run exists; choose a fresh name")
    data, initial = args.data.resolve(), args.initial_model.resolve()
    identity = {"branch": branch, "stage": stage, "dataset": data.name, "labels": labels, "config": asdict(config),
                "evaluation_split": "val", "holdout_opened": False, "promoted": False}
    try:
        inputs = dataset_snapshot(data, labels, balanced=branch == "broad")
        if sha(initial) != args.expected_initial_sha256:
            raise ValueError("Initial checkpoint differs from the pinned experiment baseline")
        if args.evaluate_only:
            sha(args.evaluate_only.resolve())  # Reject missing artifacts before model imports/download attempts.
        if stage == "B0" and sha(args.evaluate_only.resolve()) != sha(initial):
            raise ValueError("B0 must evaluate the pinned initial checkpoint")
        if stage == "B3":
            selection_path = getattr(args, "b3_selection", None)
            if selection_path is None:
                raise ValueError("B3 requires a completed B1/B2 validation selection")
            selection = json.loads(selection_path.read_text("utf-8"))
            if (selection.get("status") != "STABILITY_PENDING" or selection.get("branch") != branch
                    or not {"B1", "B2"} <= set(selection.get("source_stages", []))
                    or selection.get("input_sha256") != {**inputs, "initial_model": sha(initial)}
                    or asdict(config) != {**selection["selected_config"], "lr0": .00003}):
                raise ValueError("B3 selection/config is stale or differs from the B1/B2 winner")
    except (OSError, ValueError, KeyError) as error:
        if args.plan_only:
            return {**identity, "status": "DEFERRED_DATA_OR_BASELINE",
                    "reason": "Required local data/model artifact unavailable" if isinstance(error, OSError) else str(error)}
        raise ValueError(f"Experiment prerequisites failed: {error}") from error
    identity["input_sha256"] = {**inputs, "initial_model": sha(initial)}
    if args.plan_only:
        return {**identity, "status": "READY_NOT_EXECUTED"}
    if importlib.metadata.version("ultralytics") != "8.4.183":
        raise ValueError("Use pinned Ultralytics 8.4.183 for controlled experiments")
    import torch
    if args.device != "cpu" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable")
    torch.set_num_threads(8)
    if model_factory is None:
        from ultralytics import YOLO, settings
        settings.update({"sync": False, "weights_dir": str(root/"artifacts/models")})
        model_factory = YOLO
    baseline = model_factory(str(initial))
    if baseline.names != dict(enumerate(labels)):
        raise ValueError("Initial checkpoint taxonomy mismatch")
    if config.freeze >= len(baseline.model.model):
        raise ValueError("Freeze exceeds the model structure or freezes every layer")
    run.mkdir(parents=True)
    write(run / "run-config.json", {**identity, "status": "STARTED", "train_kwargs": config.kwargs()})
    # Omit test entirely, including from Ultralytics training/final-val YAML resolution.
    selection_yaml = run / "selection-data.yaml"
    selection_yaml.write_text(yaml.safe_dump({"path": data.as_posix(), "names": dict(enumerate(labels)),
        "train": (data/"train.txt").as_posix(), "val": (data/"val.txt").as_posix()}), encoding="utf-8")
    started = perf_counter()
    try:
        if args.evaluate_only:
            best = args.evaluate_only.resolve()
            training = {"status": "EVALUATION_ONLY", "best_epoch": None}
        else:
            save, progress = checkpoint_callbacks(run, started)
            baseline.add_callback("on_model_save", save)
            baseline.add_callback("on_fit_epoch_end", progress)
            baseline.train(data=str(selection_yaml), device=args.device, project=str(run.parent), name=run.name,
                           exist_ok=True, **config.kwargs())
            best = Path(baseline.trainer.save_dir) / "weights/best.pt"
            if best.parent.parent.resolve() != run.resolve():
                raise ValueError("Trainer wrote to an unexpected run directory")
            event = json.loads((run/"best-checkpoint-event.json").read_text("utf-8"))
            with (run/"results.csv").open(encoding="utf-8") as stream:
                epochs_completed = len(list(csv.DictReader(stream)))
            if not 1 <= event["epoch_one_based"] <= epochs_completed:
                raise ValueError("Checkpoint event is inconsistent with the training history")
            training = {"status": "TRAINED", "best_epoch": event["epoch_one_based"],
                        "epochs_completed": epochs_completed,
                        "checkpoint_selection": event, "effective_batch_trace": "epochs.jsonl",
                        "curves": "results.csv", "curves_sha256": sha(run/"results.csv"),
                        "resolved_train_arguments_sha256": sha(run/"args.yaml"),
                        "final_checkpoint_sha256": sha(best)}
        candidate = model_factory(str(best))
        candidate_hash = sha(best)
        if candidate.names != dict(enumerate(labels)):
            raise ValueError("Candidate checkpoint taxonomy mismatch")
        from train_broad_waste import evaluate_fixed
        results = {}
        for name, model in (("baseline", model_factory(str(initial))), ("candidate", candidate)):
            if name == "candidate" and sha(best) == sha(initial):
                results[name] = results["baseline"]
                continue
            fixed = evaluate_fixed(model, data, labels, args.device, dict(enumerate(range(len(labels)))),
                                   imgsz=config.imgsz)
            write(run / (name+"-fixed-val.json"), fixed)
            results[name] = {"validation_ap": ap_summary(model, selection_yaml, args.device, run, name+"-val-ap", config, labels),
                             "fixed_validation": {k:v for k,v in fixed.items() if k != "records"}}
        if (inputs != dataset_snapshot(data, labels, balanced=branch == "broad")
                or sha(initial) != identity["input_sha256"]["initial_model"] or sha(best) != candidate_hash):
            raise ValueError("Dataset or initial checkpoint changed during experiment")
        report = {**identity, "status": "COMPLETED_VALIDATION_ONLY", "candidate_sha256": sha(best),
                  "candidate_path": str(best), "comparison": results, "training": training,
                  "created_at": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": perf_counter()-started,
                  "environment": {"python": platform.python_version(), "torch": torch.__version__,
                      "ultralytics": importlib.metadata.version("ultralytics"), "cuda": torch.version.cuda,
                      "device_requested": args.device,
                      "device_name": "cpu" if args.device == "cpu" else torch.cuda.get_device_name(int(args.device)),
                      "torch_threads": torch.get_num_threads()},
                  "limitations": ["Validation selection only; no final holdout measurement or automatic promotion.",
                                  "mAP is reported per component, never averaged into combined-detector mAP."]}
        epochs_path = run/"epochs.jsonl"
        observed = [json.loads(line)["batch_size"] for line in epochs_path.read_text().splitlines()] if epochs_path.exists() else []
        report["comparison_eligible"] = all(batch == config.batch for batch in observed)
        report["observed_batch_sizes"] = sorted(set(observed))
        report["environment"]["optional_packages"] = {}
        for package in ("Pillow", "numpy", "opencv-python", "opencv-python-headless", "albumentations"):
            try:
                report["environment"]["optional_packages"][package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                report["environment"]["optional_packages"][package] = None
        if not report["comparison_eligible"]:
            report["limitations"].append("Trainer changed the batch size; rerun matched controls before comparison.")
        write(run / "report.json", report)
        shutil.copyfile(data/"DATASET_NOTICES.md", run/"DATASET_NOTICES.md")
        return report
    except Exception as error:
        write(run / "failure.json", {"status": "FAILED", "error_type": type(error).__name__, "reason": str(error)})
        raise


def cli(branch, root=ROOT):
    registry = json.loads((root/"model.json").read_text("utf-8"))
    artifact = registry if branch == "broad" else registry["general_model"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=root/"data/local"/("broad-waste-issue8-v5" if branch == "broad" else "general-focus-issue9-v2"))
    parser.add_argument("--initial-model", type=Path, default=root/"artifacts/verified-model-bundle"/artifact["filename"])
    parser.add_argument("--expected-initial-sha256", default=artifact["sha256"])
    parser.add_argument("--evaluate-only", type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--stage", choices=("B0", "B1", "B2", "B3", "extra"), default="extra")
    parser.add_argument("--b3-selection", type=Path)
    for name, default in asdict(TrainingConfig()).items():
        parser.add_argument("--"+name.replace("_", "-"), type=type(default), default=default)
    args = parser.parse_args()
    report = experiment(args, branch, root)
    print(json.dumps(report, allow_nan=False))
