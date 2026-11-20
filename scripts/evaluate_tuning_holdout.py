"""Evaluate a frozen candidate once on an independent reviewed holdout."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from random import Random

from benchmark_resolutions import ROOT, freeze_components
from calibrate_combined import BOOTSTRAP_DRAWS, component_args, make_detector, metric, runtime_identity, specs_from
from PIL import Image
from prepare_drinking_waste import perceptual_hash
from prepare_external_focus import phash_neighbors
from training_experiments import sha, write

from wastenot_ai.camera_benchmark import aggregate, load_manifest, metrics, repository_path
from wastenot_ai.fusion import FusionPolicy
from wastenot_ai.resolution_benchmark import FOCUS, benchmark_stage, prepare_inputs
from wastenot_ai.taxonomy import COMBINED_LABELS
from wastenot_ai.wire import internal_detection


def verify_holdout(manifest, protected, root):
    if manifest["target_classes"] != list(COMBINED_LABELS):
        raise ValueError("Holdout label map must match the combined detector")
    hashes = {row.get("prepared_sha256", row.get("sha256")) for row in protected}
    groups = {group for row in protected for group in [row.get("group"), *row.get("source_groups", [])]}
    sessions = {row.get("session_id") for row in protected}
    phashes = [row["phash"] for row in protected if "phash" in row]
    records = manifest["records"]
    for row in records:
        rights = row.get("rights", {})
        review = row.get("review", {})
        if (
            row.get("split") != "test"
            or row.get("holdout_eligible") is not True
            or not row.get("group")
            or not row.get("session_id")
            or rights.get("status") != "RELEASE_ELIGIBLE"
            or any(not rights.get(key) for key in ("source_page", "license_url", "attribution"))
            or review.get("reviewer_type") != "human"
            or not review.get("reviewer")
            or not review.get("reviewed_at")
            or review.get("complete_in_scope") is not True
        ):
            raise ValueError("Holdout requires eligibility, group/session, rights and complete scoped review evidence")
        if row["sha256"] in hashes or row["group"] in groups or row["session_id"] in sessions:
            raise ValueError("Holdout reuses protected image, object or session")
        with Image.open(repository_path(root, row["image"])) as image:
            if image.getexif().get(274, 1) != 1 or perceptual_hash(image.convert("RGB")) != row["phash"]:
                raise ValueError("Holdout orientation or pHash mismatch")
        if any((row["phash"] ^ value).bit_count() <= 5 for value in phashes):
            raise ValueError("Near-duplicate protected image in holdout")
    parents = list(range(len(records)))

    def find(index):
        while parents[index] != index:
            index = parents[index]
        return index

    for field in ("group", "session_id"):
        seen = {}
        for index, row in enumerate(records):
            parents[find(index)] = find(seen.setdefault(row[field], index))
    for left, right in phash_neighbors(records):
        parents[find(left)] = find(right)
    return {row["id"]: find(index) for index, row in enumerate(records)}


def claim_holdout(ledger, manifest, identity):
    """Claim the photos before inference. A failed run is still an opened holdout."""
    ledger.mkdir(parents=True, exist_ok=True)
    lock = ledger / "write.lock"
    lock.mkdir()
    try:
        hashes = {row["sha256"] for row in manifest["records"]}
        groups = {row["group"] for row in manifest["records"]}
        sessions = {row["session_id"] for row in manifest["records"]}
        phashes = {row["phash"] for row in manifest["records"]}
        for path in ledger.glob("*.json"):
            previous = json.loads(path.read_text("utf-8"))
            if (
                hashes.intersection(previous["images"])
                or groups.intersection(previous["groups"])
                or sessions.intersection(previous["sessions"])
                or any((a ^ b).bit_count() <= 5 for a in phashes for b in previous["phashes"])
            ):
                raise ValueError("Holdout overlaps a previous evaluation")
        temporary = ledger / "claim.tmp"
        write(
            temporary,
            {
                **identity,
                "images": sorted(hashes),
                "groups": sorted(groups),
                "sessions": sorted(sessions),
                "phashes": sorted(phashes),
            },
        )
        temporary.replace(ledger / (identity["manifest_sha256"] + ".json"))
    finally:
        lock.rmdir()


def confidence_interval(paired, groups, draws=BOOTSTRAP_DRAWS):
    bins = defaultdict(lambda: [Counter(), Counter()])
    for image, base, candidate in paired:
        for index, result in enumerate((base, candidate)):
            for label in FOCUS:
                bins[groups[image]][index].update(
                    {(label, key): value for key, value in result["per_class"].get(label, {}).items()}
                )
    population = list(bins.values())
    random = Random(42)
    gains = []
    for _ in range(draws):
        totals = [Counter(), Counter()]
        for base, candidate in random.choices(population, k=len(population)):
            totals[0].update(base)
            totals[1].update(candidate)
        gains.append(
            sum(
                metric(metrics({key: totals[1][label, key] for key in ("tp", "fp", "fn")}), "f1")
                - metric(metrics({key: totals[0][label, key] for key in ("tp", "fp", "fn")}), "f1")
                for label in FOCUS
            )
            / len(FOCUS)
        )
    gains.sort()
    return [gains[int(0.025 * (draws - 1))], gains[int(0.975 * (draws - 1))]]


def release_decision(base, new, support, interval, small, dim, *, p95_ratio):
    insufficient = [label for label in FOCUS if support[label][0] < 50 or support[label][1] < 20]
    gain = sum(metric(new["per_class"][label], "f1") - metric(base["per_class"][label], "f1") for label in FOCUS) / len(
        FOCUS
    )
    failures = []
    if gain < 0.05 - 1e-12:
        failures.append("focus_macro_f1_gain_below_five_points")
    if metric(new["overall"], "f1") < metric(base["overall"], "f1") - 1e-12:
        failures.append("micro_f1_regression")
    for label, old in base["per_class"].items():
        for field in ("precision", "recall") if label in FOCUS else ("f1",) if old["support"] >= 50 else ():
            if metric(new["per_class"][label], field) < metric(old, field) - 0.02 - 1e-12:
                failures.append(f"regression:{label}:{field}")
    for truth, predicted in (("cell phone", "remote"), ("remote", "cell phone")):
        old_count, new_count = (
            sum(row["count"] for row in report["confusion"] if row["truth"] == truth and row["predicted"] == predicted)
            for report in (base, new)
        )
        if new_count > old_count:
            failures.append(f"confusion_regression:{truth}:{predicted}")
    for name, condition in (("small", small), ("real_dim", dim)):
        old, current = condition
        if old["support"] >= 50 and metric(current, "f1") < metric(old, "f1") - 0.02 - 1e-12:
            failures.append(f"regression:{name}")
    if p95_ratio > 1.25:
        failures.append("cpu_p95_budget")
    return {
        "status": "INSUFFICIENT_EVIDENCE"
        if insufficient or interval[0] <= 0
        else "REJECTED_KEEP_BASELINE"
        if failures
        else "MODEL_GATES_PASSED_PENDING_RELEASE_INTEGRATION",
        "focus_macro_f1_gain": gain,
        "confidence_interval_95": interval,
        "insufficient_classes": insufficient,
        "failures": failures,
        "p95_ratio": p95_ratio,
        "promoted": False,
        "holdout_opened": True,
    }


def evaluate(detectors, inputs, manifest, groups, repeats, warmup):
    # Same interleaved measurement for both arms, including when both use 640.
    stage = benchmark_stage(
        detectors,
        inputs,
        manifest["target_classes"],
        repeats=repeats,
        warmup=warmup,
        comparison_names=("baseline", "candidate"),
    )
    by_arm = stage["results"]["browser_jpeg90"]
    index = {(row["id"], row["size"]): row["score"] for row in stage["records"]}
    paired = [(row["id"], index[row["id"], "baseline"], index[row["id"], "candidate"]) for row in manifest["records"]]
    support = {}
    for label in FOCUS:
        rows = [(row["id"], sum(obj["label"] == label for obj in row["truth"])) for row in manifest["records"]]
        support[label] = (sum(count for _, count in rows), len({groups[ident] for ident, count in rows if count}))
    dim_ids = {row["id"] for row in manifest["records"] if row.get("condition") == "real_low_light"}
    base, candidate = by_arm["baseline"], by_arm["candidate"]
    small = (base["by_object_size"]["small"], candidate["by_object_size"]["small"])
    dim = [
        aggregate([{"score": row[index]} for row in paired if row[0] in dim_ids], manifest["target_classes"])["overall"]
        for index in (1, 2)
    ]
    interval = confidence_interval(paired, groups)
    decision = release_decision(
        base,
        candidate,
        support,
        interval,
        small,
        dim,
        p95_ratio=candidate["latency"]["p95_ms"] / base["latency"]["p95_ms"],
    )
    return {
        "decision": decision,
        "support_instances_and_groups": support,
        "conditions": {"real_low_light": dim},
        "measurement": stage,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    component_args(parser)
    for name in ("freeze", "holdout", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--holdout-sha256", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()
    if args.output.exists() or args.threads < 1 or args.repeats < 3 or args.warmup < 5:
        raise ValueError("Require fresh output, positive threads, >=3 repeats and >=5 warmups")
    specs = specs_from(args)
    frozen = freeze_components(specs)
    freeze = json.loads(args.freeze.read_text("utf-8"))
    if (
        freeze.get("status") != "FROZEN_FOR_HOLDOUT"
        or freeze["frozen_components"] != frozen
        or freeze.get("holdout_opened") is not False
        or freeze.get("promoted") is not False
        or sha(args.holdout) != args.holdout_sha256
    ):
        raise ValueError("Frozen candidate or reserved holdout identity mismatch")
    import torch

    torch.set_num_threads(args.threads)
    if freeze["identity"]["runtime"] != runtime_identity():
        raise ValueError("Frozen calibration runtime differs")
    settings = freeze["policy"]
    policy = FusionPolicy(settings["merge_iou"], tuple(map(tuple, settings["thresholds"]))) if settings else None
    candidate = make_detector(tuple(specs[branch][1] for branch in ("broad", "general")), freeze["input_size"], policy)
    if json.loads(json.dumps(candidate.artifact)) != freeze["candidate_artifact"]:
        raise ValueError("Frozen candidate identity mismatch")
    registry = json.loads((ROOT / "model.json").read_text("utf-8"))
    artifacts = (registry, registry["general_model"])
    baseline_weights = tuple(ROOT / "artifacts/verified-model-bundle" / artifact["filename"] for artifact in artifacts)
    if any(sha(path) != artifact["sha256"] for path, artifact in zip(baseline_weights, artifacts, strict=True)):
        raise ValueError("Baseline weight checksum mismatch")
    baseline = make_detector(baseline_weights, 640)
    if baseline.artifact["model_version"] != registry["model_version"]:
        raise ValueError("Baseline bundle version mismatch")
    protected = json.loads((ROOT / "benchmarks/camera-replay-v1.json").read_text("utf-8"))["records"]
    for name in ("broad-waste-v4", "general-confusion-v1"):
        legacy = ROOT / "data/local" / name / "manifest.json"
        protected.extend(json.loads(legacy.read_text("utf-8"))["records"])
    for data, _, _ in specs.values():
        protected.extend(
            row for row in json.loads((data / "manifest.json").read_text("utf-8"))["records"] if row["split"] != "test"
        )
    manifest = load_manifest(args.holdout, ROOT, allowed_usage=("FINAL_HOLDOUT",))
    groups = verify_holdout(manifest, protected, ROOT)
    inputs = prepare_inputs(manifest, ROOT, {"browser_jpeg90": 1.0})
    identity = {"manifest_sha256": args.holdout_sha256, "freeze_sha256": sha(args.freeze), "holdout_opened": True}
    claim_holdout(ROOT / "artifacts/holdout-ledger", manifest, identity)
    for detector in (baseline, candidate):
        internal_detection(detector.detect(inputs[0]["content"]))
    result = evaluate(
        {"baseline": baseline, "candidate": candidate}, inputs, manifest, groups, args.repeats, args.warmup
    )
    if freeze_components(specs) != frozen or sha(args.holdout) != args.holdout_sha256:
        raise ValueError("Frozen inputs changed during holdout evaluation")
    write(args.output, {**identity, **result, "release_integration": "PENDING_PAIRED_WORKER_AND_ROLLBACK_VERIFICATION"})


if __name__ == "__main__":
    main()
