"""Plan B0-B3 without training and select candidates using validation reports only."""
import argparse
from dataclasses import asdict, replace
import json
import math
from pathlib import Path

from training_experiments import ROOT, FOCUS, TrainingConfig, experiment, write


def plan(root=ROOT):
    registry = json.loads((root/"model.json").read_text("utf-8"))
    entries = []
    for branch, dataset, artifact in (
        ("broad", "broad-waste-issue8-v5", registry),
        ("general", "general-focus-issue9-v2", registry["general_model"]),
    ):
        base = TrainingConfig()
        for run_id, config in (("B0", base), ("B1", base), ("B2", replace(base, freeze=0)),
                               ("B3", replace(base, lr0=.00003))):
            options = argparse.Namespace(**asdict(config), data=root/"data/local"/dataset,
                initial_model=root/"artifacts/verified-model-bundle"/artifact["filename"],
                expected_initial_sha256=artifact["sha256"], name=f"issue10-{branch}-{run_id.lower()}-seed42",
                evaluate_only=root/"artifacts/verified-model-bundle"/artifact["filename"] if run_id == "B0" else None,
                plan_only=True, device="0", stage=run_id, b3_selection=None)
            preflight = experiment(options, branch, root)
            entries.append({"branch": branch, "run_id": run_id, "name": options.name,
                            "config": asdict(config), "initial_sha256": artifact["sha256"],
                            "dataset": dataset, "kind": "EVALUATE_BASELINE" if run_id == "B0" else "TRAIN",
                            "preflight": preflight, "freeze_from": "validation winner B1/B2" if run_id == "B3" else None,
                            "requires": ["B1", "B2"] if run_id == "B3" else ["B0"] if run_id != "B0" else []})
    return {"status": "PLAN_ONLY_NO_EXPERIMENT_EXECUTED", "runs": entries,
            "initial_training_budget": 6, "selection_split": "val", "holdout_opened": False,
            "augmentation_followup": {"mosaic": .2, "degrees": 5.0, "scale": .2},
            "stability_seeds": [17,42,73],
            "notes": ["B3 freeze in this plan is provisional; resolve it from B1/B2 validation selection before execution.",
                      "Additional augmentation/seed runs are outside the initial six-run budget.",
                      "Data review is deferred by the user; no unreviewed photo is used for training."]}


def choose(reports):
    if not reports:
        raise ValueError("At least one completed validation report is required")
    reference = reports[0]
    branch = reference["branch"]
    if branch not in FOCUS:
        raise ValueError("Unknown model branch")
    signatures = []
    decisions = []
    eligible = []
    for report in reports:
        if (report.get("status") != "COMPLETED_VALIDATION_ONLY" or report.get("evaluation_split") != "val"
                or report.get("holdout_opened") is not False or report.get("branch") != branch):
            raise ValueError("Selection accepts completed validation-only reports from one branch")
        signature = {"inputs": report["input_sha256"],
                     "controlled": {k:report["config"][k] for k in ("imgsz", "batch", "nbs", "seed")}}
        signatures.append(signature)
        if signature != signatures[0]:
            raise ValueError("Dataset, initial model or controlled experiment parameters differ")
        comparison = report["comparison"]
        a, b = comparison["baseline"]["fixed_validation"], comparison["candidate"]["fixed_validation"]
        if a["protocol"] != b["protocol"] or a["protocol"]["split"] != "val":
            raise ValueError("Validation protocols differ")
        if any(comparison[name]["validation_ap"]["split"] != "val" for name in ("baseline", "candidate")):
            raise ValueError("Test mAP cannot select a candidate")
        reasons = []
        if not report.get("comparison_eligible", False):
            reasons.append("batch_changed_or_comparison_not_certified")
        for label, old in a["per_class"].items():
            new = b["per_class"][label]
            if old["support"] != new["support"]:
                raise ValueError("Validation ground-truth supports differ")
            for item in (old, new):
                for metric in ("precision", "recall", "f1"):
                    value = item[metric]
                    if value is not None and (not math.isfinite(value) or not 0 <= value <= 1):
                        raise ValueError("Nonfinite or invalid validation metric")
            if label in FOCUS[branch]:
                for metric in ("precision", "recall"):
                    if (new[metric] or 0) < (old[metric] or 0)-.02-1e-12:
                        reasons.append(f"{label}_{metric}_regression")
            elif old["support"] >= 50 and new["f1"] < old["f1"]-.02-1e-12:
                reasons.append(f"{label}_control_regression")
        for label in FOCUS[branch]:
            if a["per_class"][label]["support"] < 50:
                reasons.append(f"{label}_insufficient_support")
        if branch == "general":
            for expected, predicted in (("cell phone", "remote"), ("remote", "cell phone")):
                counts = [sum(p["count"] for p in row["confusion"] if p["truth"] == expected and p["predicted"] == predicted)
                          for row in (a,b)]
                if counts[1] > counts[0]:
                    reasons.append(f"{expected}_to_{predicted}_regression")
        gain = sum(b["per_class"][label]["f1"]-a["per_class"][label]["f1"] for label in FOCUS[branch])/len(FOCUS[branch])
        if gain <= 0:
            reasons.append("no_focus_macro_f1_gain")
        decision = {"candidate_sha256": report["candidate_sha256"], "gain": gain, "reasons": reasons,
                    "status": "REJECTED_VALIDATION" if reasons else "ELIGIBLE_FOR_FURTHER_EVALUATION"}
        decisions.append(decision)
        if not reasons:
            eligible.append((gain, report["candidate_sha256"], report))
    winner = sorted(eligible, key=lambda r: (-r[0],r[1]))[0][2] if eligible else None
    return {"status": "STABILITY_PENDING" if winner else "KEEP_BASELINE", "branch": branch,
            "input_sha256": reference["input_sha256"], "source_stages": sorted({r.get("stage", "extra") for r in reports}),
            "selected_candidate_sha256": winner["candidate_sha256"] if winner else None,
            "selected_config": winner["config"] if winner else None,
            "decisions": decisions, "promoted": False, "holdout_opened": False,
            "required_seed_checks": [17,42,73] if winner else [],
            "next_b3_freeze": winner["config"]["freeze"] if winner else None}


def stability(selection, reports):
    if selection.get("status") != "STABILITY_PENDING":
        raise ValueError("Choose a validation candidate before checking seed stability")
    by_seed = {r["config"]["seed"]: r for r in reports}
    if len(by_seed) != len(reports) or set(by_seed) != {17,42,73}:
        raise ValueError("Stability requires exactly one report for each seed 17/42/73")
    if by_seed[42]["candidate_sha256"] != selection["selected_candidate_sha256"]:
        raise ValueError("Seed 42 must be the frozen primary candidate report")
    decisions = []
    for seed, report in sorted(by_seed.items()):
        if (report["branch"] != selection["branch"] or report["input_sha256"] != selection["input_sha256"]
                or report["config"] != {**selection["selected_config"], "seed": seed}
                or report.get("training", {}).get("status") != "TRAINED"):
            raise ValueError("Seed run data/config/training provenance differs from the selected experiment")
        decision = choose([report])["decisions"][0]
        decisions.append({"seed": seed, **decision})
    passed = all(not row["reasons"] for row in decisions)
    return {"status": "VALIDATION_STABLE_PENDING_RESOLUTION_REVIEW" if passed else "KEEP_BASELINE_UNSTABLE",
            "branch": selection["branch"], "selected_candidate_sha256": selection["selected_candidate_sha256"] if passed else None,
            "seed_results": decisions, "mean_focus_f1_gain": sum(r["gain"] for r in decisions)/3,
            "minimum_focus_f1_gain": min(r["gain"] for r in decisions), "promoted": False, "holdout_opened": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    planned = commands.add_parser("plan")
    planned.add_argument("--output", required=True, type=Path)
    selected = commands.add_parser("choose")
    selected.add_argument("--report", action="append", required=True, type=Path)
    selected.add_argument("--output", required=True, type=Path)
    stable = commands.add_parser("stability")
    stable.add_argument("--selection", required=True, type=Path)
    stable.add_argument("--report", action="append", required=True, type=Path)
    stable.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "plan":
        result = plan()
    else:
        reports = [json.loads(p.read_text("utf-8")) for p in args.report]
        result = choose(reports) if args.command == "choose" else stability(json.loads(args.selection.read_text("utf-8")), reports)
    write(args.output, result)
    print(json.dumps({"output": str(args.output), "status": result["status"]}))


if __name__ == "__main__":
    main()
