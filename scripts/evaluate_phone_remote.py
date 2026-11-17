"""Replay frozen phone/remote diagnostics from verified existing predictions, without inference."""
import argparse
import hashlib
import json
from pathlib import Path

from wastenot_ai.camera_benchmark import aggregate, score_image

FOCUS = ("cell phone", "remote")


def diagnose(manifest, report, manifest_sha):
    if report["manifest_sha256"] != manifest_sha:
        raise ValueError("Prediction cache belongs to another benchmark")
    sources = {r["id"]: r for r in manifest["records"]}
    if len(sources) != len(manifest["records"]):
        raise ValueError("Duplicate source identities")
    selected = {ident for ident, row in sources.items() if set(FOCUS) & set(row["evaluated_classes"])}
    profiles, model_names = report["protocol"]["input_profiles"], set(report["models"])
    scored = {}
    for row in report["records"]:
        source = sources.get(row["id"])
        if source is None or row["image_sha256"] != source["sha256"]:
            raise ValueError("Prediction/source image identity mismatch")
        identity = (row["profile"], row["model"], row["id"])
        if identity in scored or row["profile"] not in profiles or row["model"] not in model_names:
            raise ValueError("Duplicate or unexpected cached prediction")
        scored[identity] = {**row, "score": score_image(row["objects"], source["truth"], source["evaluated_classes"])}
    expected = {(p, m, ident) for p in profiles for m in model_names for ident in sources}
    if scored.keys() != expected:
        raise ValueError("Prediction cache is incomplete")
    results = {}
    for profile in profiles:
        results[profile] = {}
        for model in sorted(model_names):
            rows = [scored[(profile, model, ident)] for ident in sorted(selected)]
            summary = aggregate(rows, FOCUS)
            directions = {}
            for a, b in (("cell phone", "remote"), ("remote", "cell phone")):
                wrong = sum(p["count"] for p in summary["confusion"] if p["truth"] == a and p["predicted"] == b)
                support = summary["per_class"][a]["support"]
                directions[f"{a} -> {b}"] = {"count": wrong, "support": support,
                                             "rate": wrong/support if support else None}
            results[profile][model] = {"per_class": summary["per_class"], "directions": directions}
    return {"status": "LEGACY_DIAGNOSIS_NOT_NEW_HOLDOUT", "images": len(selected),
            "image_ids": sorted(selected), "manifest_sha256": manifest_sha,
            "models": report["models"], "results": results,
            "limitations": ["Subset includes images with annotation coverage for phone or remote, including negatives.",
                            "Wrong-label geometry and class-aware precision/recall are separate diagnostics.",
                            "No new inference, tuning or holdout selection was performed.",
                            "Rear-view capture tags need independent review; absence of tags is not evidence of front view."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    raw = args.manifest.read_bytes()
    result = diagnose(json.loads(raw), json.loads(args.predictions.read_text("utf-8")), hashlib.sha256(raw).hexdigest())
    release = json.loads((Path(__file__).resolve().parents[1] / "model.json").read_text("utf-8"))
    expected_sha = release["general_model"]["sha256"]
    if not any(model.get("sha256") == expected_sha for model in result["models"].values()):
        raise ValueError("Prediction cache does not contain the registered baseline general model")
    result["baseline_general_model_sha256"] = expected_sha
    result["prediction_cache_sha256"] = hashlib.sha256(args.predictions.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"images": result["images"], "status": result["status"]}))


if __name__ == "__main__":
    main()
