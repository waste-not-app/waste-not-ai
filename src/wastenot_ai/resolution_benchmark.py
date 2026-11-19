"""Matched-input, interleaved resolution experiments; no model loading or activation here."""
from collections import Counter
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
import statistics
from time import perf_counter

from PIL import Image

from .camera_benchmark import aggregate, browser_input, metrics, repository_path, score_image
from .detection_quality import box_iou

SIZES = (640, 768, 960)
FOCUS = ("cardboard", "plastic_wrapper", "paper", "drink_carton", "cell phone", "remote")
PROFILES = {"browser_jpeg90": 1.0, "simulated_dim055": .55, "android_jpeg85_gateway90": 1.0}


def distribution(values):
    if not values or any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError("Nonempty finite nonnegative timing samples required")
    return {"samples": len(values), "p50_ms": statistics.median(values),
            "p95_ms": statistics.quantiles(values, n=100, method="inclusive")[94] if len(values)>1 else values[0]}


def process_memory():
    """Native process RSS, not Python-only tracemalloc. Shared model memory is explicit."""
    import os
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("faults", wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in ("peak", "rss", "peak_pool", "pool", "peak_nonpool",
                                                    "nonpool", "pagefile", "peak_pagefile")]
        data = Counters()
        data.cb = ctypes.sizeof(data)
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(data), data.cb):
            raise OSError("Cannot measure native process memory")
        return {"rss_bytes": data.rss, "peak_rss_bytes": data.peak, "method": "GetProcessMemoryInfo"}
    path = Path("/proc/self/status")
    if path.exists():
        fields = dict(line.split(":",1) for line in path.read_text().splitlines() if ":" in line)
        return {"rss_bytes": int(fields["VmRSS"].split()[0])*1024,
                "peak_rss_bytes": int(fields["VmHWM"].split()[0])*1024, "method": "proc-self-status"}
    return {"rss_bytes": None, "peak_rss_bytes": None, "method": "unavailable"}


def size_bucket(bbox):
    area = (bbox[2]-bbox[0])*(bbox[3]-bbox[1])
    return "small" if area < .01 else "medium" if area < .1 else "large"


def android_input(path):
    """Approximate BitmapFactory power-of-two sampling + JPEG85 + Go JPEG90.

    Pillow codecs/resampling are not pixel-identical to Android and Go.
    """
    with Image.open(path) as original:
        if original.getexif().get(274,1) != 1:
            raise ValueError("Normalize source orientation and annotation before replay")
        image = original.convert("RGB")
    sample = 1
    while max(image.size)//sample > 1600:
        sample *= 2
    if sample > 1:
        resized = image.resize(tuple(max(1,v//sample) for v in image.size),Image.Resampling.BOX)
        image.close()
        image = resized
    first = BytesIO()
    image.save(first,format="JPEG",quality=85)
    image.close()
    second = BytesIO()
    with Image.open(BytesIO(first.getvalue())) as cached:
        cached.save(second,format="JPEG",quality=90)
    return second.getvalue()


def score_sizes(objects, truth, scope):
    counts = {name: Counter(tp=0,fp=0,fn=0) for name in ("small","medium","large")}
    unmatched = set(range(len(truth)))
    for obj in sorted((o for o in objects if o["supported_for_review"] and o["raw_class"] in scope),
                      key=lambda o:-o["confidence"]):
        options = [i for i in unmatched if truth[i]["label"] == obj["raw_class"]]
        best = max(options,key=lambda i:box_iou(obj["bbox"],truth[i]["bbox"]),default=None)
        if best is not None and box_iou(obj["bbox"],truth[best]["bbox"]) >= .5:
            counts[size_bucket(truth[best]["bbox"])]["tp"] += 1
            unmatched.remove(best)
        else:
            counts[size_bucket(obj["bbox"])]["fp"] += 1
    for i in unmatched:
        counts[size_bucket(truth[i]["bbox"])]["fn"] += 1
    return counts


def prepare_inputs(manifest, root, profiles=PROFILES):
    inputs = []
    for row in manifest["records"]:
        source = repository_path(root,row["image"])
        with Image.open(source) as image:
            original = image.size
        for profile, brightness in profiles.items():
            content = android_input(source) if profile == "android_jpeg85_gateway90" else browser_input(source,brightness)
            with Image.open(BytesIO(content)) as image:
                transmitted = image.size
            inputs.append({"source":row,"profile":profile,"content":content,
                           "input_sha256":hashlib.sha256(content).hexdigest(),
                           "source_size":original,"transmitted_size":transmitted})
    if not inputs:
        raise ValueError("No validation inputs")
    return inputs


def benchmark_stage(detectors, inputs, labels, *, repeats=3, warmup=5, clock=perf_counter, memory=process_memory,
                    comparison_names=None):
    sizes = sorted(detectors)
    allowed = list(comparison_names) == sizes if comparison_names else sizes in ([640,768],[640,768,960])
    if not allowed or repeats < 3 or warmup < 5:
        raise ValueError("Compare 640/768 first, >=3 repeats and >=5 excluded warmups")
    for detector in detectors.values():
        for _ in range(warmup):
            detector.detect(inputs[0]["content"])
    records = []
    for index, item in enumerate(inputs):
        outputs, times = {}, {size:[] for size in sizes}
        resident = {size:[] for size in sizes}
        for repeat in range(repeats):
            offset = (index+repeat)%len(sizes)
            for size in sizes[offset:]+sizes[:offset]:
                started = clock()
                result = detectors[size].detect(item["content"])
                elapsed = (clock()-started)*1000
                if result["image"] != dict(zip(("width","height"),item["transmitted_size"])):
                    raise ValueError("Detector coordinates reference another image size")
                if repeat and result["objects"] != outputs[size]["objects"]:
                    raise ValueError("Predictions changed between timing repetitions")
                outputs[size] = result
                times[size].append(elapsed)
                resident[size].append(memory())
        row = item["source"]
        for size in sizes:
            objects = outputs[size]["objects"]
            records.append({"id":row["id"],"profile":item["profile"],"size":size,
                "input_sha256":item["input_sha256"],"source_size":item["source_size"],
                "transmitted_size":item["transmitted_size"],"timing_ms":times[size],"memory":resident[size],
                "objects":objects,"score":score_image(objects,row["truth"],row["evaluated_classes"]),
                "size_counts":score_sizes(objects,row["truth"],row["evaluated_classes"])})
    results = {}
    for profile in sorted({item["profile"] for item in inputs}):
        results[profile] = {}
        for size in sizes:
            selected = [r for r in records if r["size"] == size and r["profile"] == profile]
            summary = aggregate(selected,labels)
            summary["latency"] = distribution([t for r in selected for t in r["timing_ms"]])
            summary["by_object_size"] = {bucket:metrics({key:sum(r["size_counts"][bucket][key] for r in selected)
                                                        for key in ("tp","fp","fn")}) for bucket in ("small","medium","large")}
            summary["process_memory"] = [sample for r in selected for sample in r["memory"]]
            results[profile][str(size)] = summary
    return {"results":results,"records":records,"models":{str(s):d.artifact for s,d in detectors.items()},
            "protocol":{"warmup_calls":warmup,"repeats":repeats,"batch":1,"device":"cpu",
                "model_order":"rotated per image/profile/repetition", "timing_scope":"detector.detect wall clock; excludes HTTP and input encoding",
                "memory_scope":"native RSS of this process with all compared profiles resident; not isolated per-model allocation",
                "object_size":"normalized GT area <.01 / <.1 / >=.1; unmatched FP bucketed by predicted area"}}


def assess(base, candidate):
    for row in (base,candidate):
        if not math.isfinite(row["latency"]["p95_ms"]) or row["latency"]["p95_ms"] <= 0:
            raise ValueError("Invalid p95 measurement")
        for metric_row in [row["overall"],*row["per_class"].values(),*row["by_object_size"].values()]:
            for key in ("precision","recall","f1"):
                value = metric_row.get(key)
                if value is not None and (not math.isfinite(value) or not 0 <= value <= 1):
                    raise ValueError("Invalid quality metric")
    reasons = []
    if base["latency"]["p95_ms"] <= 0 or candidate["latency"]["p95_ms"] > 1.25*base["latency"]["p95_ms"]:
        reasons.append("cpu_p95_budget")
    for label in FOCUS:
        if base["per_class"].get(label,{}).get("support",0) < 50:
            reasons.append("insufficient_support_"+label)
    gain = 0.0
    for label, old in base["per_class"].items():
        new = candidate["per_class"][label]
        if old["support"] != new["support"]:
            raise ValueError("Resolution comparisons have different ground truth support")
        if old["support"] >= 50 and (new["f1"] or 0) < (old["f1"] or 0)-.02-1e-12:
            reasons.append("class_regression_"+label)
        if label in FOCUS:
            gain += ((new["f1"] or 0)-(old["f1"] or 0))/len(FOCUS)
    if (candidate["overall"]["f1"] or 0) < (base["overall"]["f1"] or 0)-1e-12:
        reasons.append("overall_f1_regression")
    if gain <= 0:
        reasons.append("no_focus_quality_gain")
    for bucket in ("small","medium","large"):
        old,new = base["by_object_size"][bucket],candidate["by_object_size"][bucket]
        if old["support"] >= 50 and (new["f1"] or 0) < (old["f1"] or 0)-.02-1e-12:
            reasons.append("object_size_regression_"+bucket)
    return {"accepted_for_further_evaluation":not reasons,"reasons":reasons,"focus_macro_f1_gain":gain}


def select_resolution(stage):
    rows = stage["results"]["browser_jpeg90"]
    decisions = {size:assess(rows["640"],row) for size,row in rows.items() if size != "640"}
    for size,decision in decisions.items():
        for profile,variants in stage["results"].items():
            if profile == "browser_jpeg90":
                continue
            check = assess(variants["640"],variants[size])
            decision["reasons"].extend(profile+":"+reason for reason in check["reasons"] if reason != "no_focus_quality_gain")
        decision["accepted_for_further_evaluation"] = not decision["reasons"]
    eligible = [(value["focus_macro_f1_gain"],int(size)) for size,value in decisions.items() if value["accepted_for_further_evaluation"]]
    chosen = sorted(eligible,key=lambda pair:(-pair[0],pair[1]))[0][1] if eligible else 640
    return {"selected_input_size":chosen,"decisions":decisions,"promoted":False,"holdout_opened":False,
            "next_step":"calibration and final review required; no active worker/model config changed"}


def cache_identity(manifest_sha, artifacts, inputs, environment):
    value = {"manifest_sha256":manifest_sha,"models":artifacts,"environment":environment,
             "inputs":[(i["source"]["id"],i["profile"],i["input_sha256"]) for i in inputs],
             "protocol":"square-resolution-v2-conf025-review050-nms070-max50-browser90-android85gateway90"}
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
