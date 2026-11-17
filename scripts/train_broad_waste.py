"""Train/evaluate a separate broad detector; never silently replace the active checkpoint."""
from collections import Counter
import statistics
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".cache-ai/ultralytics"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache-ai/matplotlib"))
os.environ["YOLO_AUTOINSTALL"] = "false"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def overlap(a, b):
    inter = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/union if union > 0 else 0


def evaluate_fixed(model, dataset, labels, device, mapping, split="val", *, imgsz=640):
    counts = {i: Counter(tp=0, fp=0, fn=0) for i in range(len(labels))}
    by_source = {}
    timings, records = [], []
    paths = [Path(line) for line in (dataset / (split + ".txt")).read_text("utf-8").splitlines() if line]
    for path in sorted(paths):
        truth = []
        for line in (dataset / "labels" / split / path.with_suffix(".txt").name).read_text().splitlines():
            c, x, y, w, h = map(float, line.split())
            if int(c) in mapping.values():
                truth.append({"class_id": int(c), "bbox": [x-w/2, y-h/2, x+w/2, y+h/2]})
        start = perf_counter()
        result = model.predict(str(path), device=device, imgsz=imgsz, rect=False, conf=.5,
                               iou=.7, max_det=50, agnostic_nms=False, verbose=False)[0]
        timings.append((perf_counter()-start)*1000)
        predictions = [{"class_id": mapping[int(b.cls.item())], "confidence": float(b.conf.item()),
                        "bbox": b.xyxyn[0].tolist()} for b in result.boxes if int(b.cls.item()) in mapping]
        unmatched = set(range(len(truth)))
        per_image = {i: Counter(tp=0, fp=0, fn=0) for i in mapping.values()}
        for p in sorted(predictions, key=lambda x: -x["confidence"]):
            options = [i for i in unmatched if truth[i]["class_id"] == p["class_id"]]
            best = max(options, key=lambda i: overlap(p["bbox"], truth[i]["bbox"]), default=None)
            key = "fp"
            if best is not None and overlap(p["bbox"], truth[best]["bbox"]) >= .5:
                key = "tp"
                unmatched.remove(best)
            per_image[p["class_id"]][key] += 1
        for i in unmatched:
            per_image[truth[i]["class_id"]]["fn"] += 1
        source = path.name.split("-")[0]
        by_source.setdefault(source, {i: Counter(tp=0, fp=0, fn=0) for i in mapping.values()})
        for cls, count in per_image.items():
            counts[cls].update(count)
            by_source[source][cls].update(count)
        records.append({"image": path.name, "truth": truth, "predictions": predictions})
    def summarize(count):
        tp, fp, fn = count["tp"], count["fp"], count["fn"]
        return {**count, "precision": tp/(tp+fp) if tp+fp else None,
                "recall": tp/(tp+fn) if tp+fn else None,
                "support": tp+fn, "f1": 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0}
    from wastenot_ai.camera_benchmark import score_image
    confusion = Counter()
    for record in records:
        objects = [{"raw_class": labels[p["class_id"]], "bbox": p["bbox"], "confidence": p["confidence"],
                    "supported_for_review": True} for p in record["predictions"]]
        truth = [{"label": labels[g["class_id"]], "bbox": g["bbox"]} for g in record["truth"]]
        for pair in score_image(objects, truth, [labels[i] for i in mapping.values()])["confusion"]:
            confusion[(pair["truth"], pair["predicted"])] += pair["count"]
    samples = sorted(timings[1:] or timings)
    p95 = statistics.quantiles(samples, n=100, method="inclusive")[94] if len(samples) > 1 else samples[0] if samples else None
    return {"protocol": {"split": split, "confidence": .5, "matching_iou": .5, "nms_iou": .7,
                          "imgsz": imgsz, "rect": False, "max_det": 50},
            "mapped_classes": [labels[i] for i in sorted(set(mapping.values()))],
            "per_class": {labels[i]: summarize(c) for i, c in counts.items() if i in mapping.values()},
            "per_source": {s: {labels[i]: summarize(c) for i, c in rows.items()} for s, rows in by_source.items()},
            "confusion": [{"truth": a, "predicted": b, "count": n} for (a,b),n in sorted(confusion.items())],
            "latency_p50_ms": statistics.median(samples) if samples else None,
            "latency_p95_ms": p95, "records": records}



def main():
    from training_experiments import cli
    cli("broad", ROOT)


if __name__ == "__main__":
    main()
