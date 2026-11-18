"""Exercise handling invariants across the full catalog without calling a live provider."""
import argparse
import json
from collections import Counter
from itertools import product
from pathlib import Path

from wastenot_ai.catalog import load_catalog
from wastenot_ai.contracts import Category, Condition, Context
from wastenot_ai.providers import FixtureProvider
from wastenot_ai.recommendation import load_guides, recommend
from wastenot_ai.sources import load_sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/recommendation-matrix.json"))
    args = parser.parse_args()
    guides, catalog, registry = load_guides(), load_catalog(), load_sources()
    counts, failures = Counter(), []
    cases = 0
    risks = (None, "UNKNOWN", "SHARP", "CHEMICAL", "BIOLOGICAL", "ELECTRICAL", "OTHER")
    for item, risk, integrity, access, goal, soiling in product(catalog["items"], risks,
            ("INTACT", "DAMAGED", "UNKNOWN"), ("YES", "NO", "UNKNOWN"),
            ("HANDLING", "PREVENTION"), ("UNKNOWN", "HAZARDOUS")):
        value = Context(object_key=item["object_key"], category=Category(item["category"]),
            condition=Condition(cleanliness="CLEAN", integrity=integrity, reusable="YES", soiling=soiling),
            reuse_intention=True, risk_flag=risk is not None, risk_type=risk or "UNKNOWN",
            recommendation_goal=goal, recycling_access=access)
        cases += 1
        result = recommend(value, FixtureProvider(), guides, demo_guides=True)
        counts[result.status] += 1
        checks = {
            "steps_and_sources_present": bool(result.steps and result.source_urls),
            "sources_registered": all(url in registry for url in result.source_urls),
            "no_recycle_without_access": not (access == "NO" and result.recommended_method == "RECYCLE"),
            "no_reuse_when_damaged": not (integrity == "DAMAGED" and result.recommended_method == "REUSE"),
            "no_ordinary_handling_for_contamination": not (goal == "HANDLING" and
                (risk in {"CHEMICAL", "BIOLOGICAL", "OTHER"} or soiling == "HAZARDOUS") and result.recommended_method),
            "electrical_risk_keeps_its_route": not (goal == "HANDLING" and risk == "ELECTRICAL" and
                item["category"] != "ELECTRONIC" and result.recommended_method),
            "biological_guidance_retained": not (result.status != "READY" and risk == "BIOLOGICAL" and
                "https://www.who.int/news-room/fact-sheets/detail/health-care-waste" not in result.source_urls),
        }
        failed = [key for key, passed in checks.items() if not passed]
        if failed:
            failures.append({"object_key": item["object_key"], "risk": risk, "integrity": integrity,
                "access": access, "goal": goal, "soiling": soiling, "checks": failed})
    report = {"cases": cases, "catalog_items": len(catalog["items"]), "status_counts": dict(counts),
              "failures": failures, "provider": "fixture", "live_provider_calls": 0,
              "scope": "Policy invariants, not visual detection accuracy or expert review of every item."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "failures"} | {"failure_count": len(failures)}))
    return bool(failures)


if __name__ == "__main__":
    raise SystemExit(main())
