"""Compare v3, COCO and their real combined adapter on identical frozen camera inputs."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np

from wastenot_ai.camera_benchmark import aggregate, browser_input, load_manifest, repository_path, score_image
from wastenot_ai.detector import CombinedDetector, YoloDetector

ROOT = Path(__file__).resolve().parents[1]
PROFILES = {"browser_jpeg90": 1.0, "simulated_dim055": .55}


def latency(values):
    return {"samples": len(values), "median_ms": float(np.median(values)),
            "p95_ms": float(np.percentile(values, 95)), "mean_ms": float(np.mean(values))}


def percent(value):
    return "â€”" if value is None else f"{100 * value:.1f}%"


def write_summary(report, destination):
    lines = ["# Benchmark replay kamera v1", "", f"Tanggal UTC {report['created_at']}", "",
             f"{report['images']} foto dasar, {len(report['results'])} profil input, {report['protocol']['timing_repeats']} pengulangan waktu per input/model.", "",
             "Matching IoU 0,50 pada kandidat yang bisa dipilih di demo. Precision/recall hanya menghitung kelas yang dianotasi oleh sumber pada setiap foto.", "",
             "| Profil | Model | Precision | Recall | F1 | TP | FP | FN | Median ms | P95 ms |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for profile, modes in report["results"].items():
        for name, result in modes.items():
            overall, timing = result["overall"], result["latency"]
            lines.append(f"| {profile} | {name} | {percent(overall['precision'])} | {percent(overall['recall'])} | {percent(overall['f1'])} | {overall['tp']} | {overall['fp']} | {overall['fn']} | {timing['median_ms']:.1f} | {timing['p95_ms']:.1f} |")
    baseline = report["results"]["browser_jpeg90"]
    names = list(baseline)
    lines += ["", "## Per kelas pada browser JPEG90", "",
              "| Kelas | GT | " + " | ".join(name + " P / R" for name in names) + " |",
              "|---|---:|" + "---:|" * len(names)]
    for label in report["target_classes"]:
        cells = []
        for name in names:
            row = baseline[name]["per_class"][label]
            cells.append(f"{percent(row['precision'])} / {percent(row['recall'])}")
        support = baseline[names[0]]["per_class"][label]["support"]
        lines.append(f"| {label} | {support} | " + " | ".join(cells) + " |")
    lines += ["", "## Diagnosis pada browser JPEG90", "",
              "| Model | Salah label pada lokasi GT | GT tanpa kandidat bertumpuk | Pasangan kotak kelas sama | Pasangan label bertentangan |", "|---|---:|---:|---:|---:|"]
    for name, result in baseline.items():
        lines.append(f"| {name} | {result['wrong_label_instances']} | {result['no_overlapping_candidate_instances']} | {result['same_class_duplicate_pairs']} | {result['conflicting_label_pairs']} |")
    lines += ["", "## Pasangan salah label yang paling sering", "",
              "| Model | Ground truth | Prediksi | Jumlah |", "|---|---|---|---:|"]
    for name, result in baseline.items():
        mistakes = [pair for pair in result["confusion"] if pair["predicted"] not in {pair["truth"], "__missed__"}]
        for pair in sorted(mistakes, key=lambda pair: (-pair["count"], pair["truth"], pair["predicted"]))[:5]:
            lines.append(f"| {name} | {pair['truth']} | {pair['predicted']} | {pair['count']} |")
    lines += ["", "## Batas interpretasi", ""]
    lines.extend("- " + item for item in report["limitations"])
    lines += ["", "Input redup merupakan transformasi sintetis dari foto yang sama. Jumlah foto independen tetap sama dengan jumlah foto dasar.",
              "", "Kesalahan per foto dan keluaran semua model tersedia di JSON serta CSV pendamping.", ""]
    destination.write_text("\n".join(lines), encoding="utf-8")


def write_plot(report, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    baseline = report["results"]["browser_jpeg90"]
    labels = report["target_classes"]
    figure, axes = plt.subplots(2, 1, figsize=(13, 9), gridspec_kw={"height_ratios": [2, 1]})
    names = list(baseline)
    colors = dict(zip(names, ("#195B3D", "#B07D23", "#476FA3")))
    x = np.arange(len(labels))
    for index, (name, result) in enumerate(baseline.items()):
        values = [100 * (result["per_class"][label]["recall"] or 0) for label in labels]
        axes[0].bar(x + (index - 1) * .25, values, width=.25, label=name, color=colors[name])
    axes[0].set_xticks(x, [f"{label}\n(n={baseline[names[0]]['per_class'][label]['support']})" for label in labels], rotation=25, ha="right")
    axes[0].set_ylabel("Recall (%)")
    axes[0].set_ylim(0, 105)
    support = baseline[names[0]]["overall"]["support"]
    axes[0].set_title(f"{report['images']} foto, {support} anotasi objek, JPEG90, IoU 0.50")
    axes[0].legend()
    names = list(baseline)
    timing = [baseline[name]["latency"]["median_ms"] for name in names]
    axes[1].barh(names, timing, color=[colors[name] for name in names])
    for i, value in enumerate(timing):
        axes[1].text(value, i, f"  {value:.0f} ms", va="center")
    axes[1].set_xlim(0, max(timing) * 1.2)
    device = report["environment"]["device"]
    axes[1].set_xlabel(f"Median waktu detector.detect pada {device} (ms)")
    repeats = report["protocol"]["timing_repeats"]
    axes[1].set_title(f"{repeats} pengulangan tiap foto setelah warmup")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "benchmarks/camera-replay-v1.json")
    parser.add_argument("--v3", type=Path, default=ROOT / "artifacts/training/broad-waste-v3-batch2-yolo11s/weights/best.pt")
    parser.add_argument("--coco", type=Path, default=ROOT / "artifacts/models/yolo11n.pt")
    parser.add_argument("--waste-name", default="v3")
    parser.add_argument("--general-name", default="coco")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, default=ROOT / "evaluations/camera-replay-v1-comparison.json")
    args = parser.parse_args()
    if args.repeats < 1:
        raise ValueError("Timing repeats must be positive")
    if len({args.waste_name, args.general_name, "combined"}) != 3:
        raise ValueError("Model names must be distinct and reserve 'combined' for the merged adapter")
    manifest = load_manifest(args.manifest, ROOT)
    manifest_sha256 = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    load_started = perf_counter()
    waste = YoloDetector(args.v3, args.device, "square")
    general = YoloDetector(args.coco, args.device, "square")
    detectors = {args.waste_name: waste, args.general_name: general, "combined": CombinedDetector(waste, general)}
    load_seconds = perf_counter() - load_started
    warmup = browser_input(repository_path(ROOT, manifest["records"][0]["image"]))
    for detector in detectors.values():
        for _ in range(2):
            detector.detect(warmup)
    records = []
    names = list(detectors)
    for index, source in enumerate(manifest["records"]):
        for profile, brightness in PROFILES.items():
            content = browser_input(repository_path(ROOT, source["image"]), brightness)
            outputs, elapsed = {}, {name: [] for name in names}
            for repetition in range(args.repeats):
                offset = (index + repetition) % len(names)
                for name in names[offset:] + names[:offset]:
                    started = perf_counter()
                    result = detectors[name].detect(content)
                    elapsed[name].append((perf_counter() - started) * 1000)
                    if repetition == 0:
                        outputs[name] = result
                    elif result["objects"] != outputs[name]["objects"]:
                        raise ValueError("Repeated inference changed predictions. Investigate determinism before comparing.")
            for name in names:
                objects = outputs[name]["objects"]
                records.append({"id": source["id"], "source": source["source"], "group": source["group"],
                    "model": name, "profile": profile, "image_sha256": source["sha256"],
                    "input_sha256": hashlib.sha256(content).hexdigest(),
                    "timing_ms": elapsed[name], "objects": objects,
                    "score": score_image(objects, source["truth"], source["evaluated_classes"])})
        if (index + 1) % 10 == 0 or index + 1 == len(manifest["records"]):
            print(f"Compared {index + 1}/{len(manifest['records'])} images", flush=True)
    results = {}
    for profile in PROFILES:
        results[profile] = {}
        for name in names:
            selected = [row for row in records if row["model"] == name and row["profile"] == profile]
            summary = aggregate(selected, manifest["target_classes"])
            summary["latency"] = latency([value for row in selected for value in row["timing_ms"]])
            summary["by_source"] = {source: aggregate([row for row in selected if row["source"] == source], manifest["target_classes"])
                                    for source in sorted({row["source"] for row in selected})}
            results[profile][name] = summary
    import torch
    report = {"version": "camera-replay-v1", "created_at": datetime.now(timezone.utc).isoformat(),
        "manifest": args.manifest.resolve().relative_to(ROOT).as_posix(),
        "manifest_sha256": manifest_sha256,
        "images": len(manifest["records"]), "target_classes": manifest["target_classes"],
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
            "processor": platform.processor(), "torch": torch.__version__, "torch_threads": torch.get_num_threads(),
            "ultralytics": importlib.metadata.version("ultralytics"), "device": args.device},
        "protocol": {"input_shape": "square640", "input_profiles": PROFILES,
            "browser_preprocessing": "long side <=1600, JPEG quality90, Pillow bilinear approximation to browser canvas",
            "prediction_confidence_floor": .25, "review_confidence_floor": .5, "iou_matching": .5,
            "duplicate_pair_iou": .65, "timing_repeats": args.repeats, "warmup_calls_per_mode": 2,
            "timing": "wall-clock detector.detect including decoding, inference, mapping and merging; excludes model load and JPEG preparation",
            "model_order": "rotated per image and repetition; sequential inference on same device",
            "load_seconds_both_checkpoints": load_seconds},
        "models": {name: detector.artifact for name, detector in detectors.items()},
        "results": results, "records": records,
        "limitations": manifest["limitations"] + [
            "Publisher boxes were screened on contact sheets; an independent human annotation review is pending.",
            "Metrics use operational review thresholds and IoU 0.50; these scores are precision/recall, rather than mAP.",
            "Predictions outside each image's annotation scope are excluded from precision/recall and retained for overlap diagnostics.",
            "Wrong-label diagnostics use independent class-agnostic geometric matching and can coexist with class-aware TP/FP/FN.",
            "Latency reflects this machine and detector path. Camera acquisition, browser upload and HTTP overhead are outside the timer."]}
    if hashlib.sha256(args.manifest.read_bytes()).hexdigest() != manifest_sha256:
        raise ValueError("Benchmark manifest changed during inference")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_summary(report, args.output.with_suffix(".md"))
    write_plot(report, args.output.with_suffix(".png"))
    with args.output.with_name(args.output.stem + "-errors.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_id", "source", "profile", "model", "kind", "truth", "predicted", "score", "iou"])
        writer.writeheader()
        for row in records:
            for error in row["score"]["errors"]:
                writer.writerow({"image_id": row["id"], "source": row["source"], "profile": row["profile"], "model": row["model"], **error})
    print(json.dumps({profile: {name: {"overall": result["overall"], "latency": result["latency"]}
                               for name, result in modes.items()} for profile, modes in results.items()}, indent=2))


if __name__ == "__main__":
    main()
