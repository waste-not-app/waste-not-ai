"""Build the recommendation guide bundle from the backend's reviewed waste guide articles.

Usage: python scripts/build_reviewed_guides.py ../waste-not-backend/guides/core-guides-v1.json
"""

import argparse
import json
from datetime import date, timedelta
from pathlib import Path

OUTPUT = Path(__file__).resolve().parents[1] / "guides/core-reviewed-v1.json"
VALIDITY_DAYS = 181

PRODUCE = ["banana", "apple", "orange", "broccoli", "carrot"]
LEFTOVERS = ["sandwich", "hot_dog", "pizza", "donut", "cake"]
GLASSWARE = ["wine_glass", "glass_cup", "glass_bowl", "glass_vase", "other_glass"]
SMALL_ELECTRONICS = ["cell_phone", "laptop", "mouse", "remote", "keyboard", "clock", "hair_drier"]

CLEAN = {"cleanliness": "CLEAN"}
CLEAN_INTACT = {"cleanliness": "CLEAN", "integrity": "INTACT"}
FOOD_CONTAINER = {"cleanliness": "CLEAN", "integrity": "INTACT", "former_contents": "FOOD_DRINK"}
BROKEN = {"integrity": "DAMAGED"}


def rule(article, method, requires=None, *, steps=None, text=None, risk=False):
    return {
        "article": article,
        "method": method,
        "requires": requires or {},
        "steps": steps,
        "text": text,
        "risk_compatible": risk,
    }


def mapping():
    rules = {
        "pet_bottle": [rule("pet-plastic-bottles", "RECYCLE", FOOD_CONTAINER)],
        "hdpe_bottle": [rule("hdpe-containers", "RECYCLE", FOOD_CONTAINER)],
        "cardboard": [rule("clean-cardboard-boxes", "RECYCLE", CLEAN_INTACT)],
        "paper": [rule("office-paper-and-newspapers", "RECYCLE", CLEAN)],
        "book": [
            rule("office-paper-and-newspapers", "RECYCLE", CLEAN),
            rule(
                "office-paper-and-newspapers",
                "REUSE",
                steps=[
                    "Consider donating readable books to a school, library, or community group before recycling them."
                ],
                text="A readable book can be passed on to a school, library, or community group before you think about recycling it.",
            ),
        ],
        "glass_bottle": [
            rule("intact-glass-bottles-and-jars", "RECYCLE", CLEAN_INTACT),
            rule(
                "intact-glass-bottles-and-jars",
                "REUSE",
                steps=["Empty the bottle or jar and rinse off any food or drink residue."],
                text="An intact glass bottle or jar can be cleaned and reused as a container.",
            ),
            rule("broken-glass-and-sharp-shards", "TRASH", BROKEN, risk=True),
        ],
        "aluminium_can": [rule("aluminium-drink-cans", "RECYCLE", FOOD_CONTAINER)],
        "steel_can": [
            rule("food-cans-and-clean-foil", "RECYCLE", {"cleanliness": "CLEAN", "former_contents": "FOOD_DRINK"})
        ],
        "plastic_bag": [
            rule("plastic-bags-and-wrap", "REDUCE"),
            rule(
                "plastic-bags-and-wrap",
                "REUSE",
                steps=["If you already have clean bags, reuse them as bin liners or for wrapping."],
                text="A clean plastic bag you already have can be reused as a bin liner or for wrapping.",
            ),
            rule(
                "plastic-bags-and-wrap",
                "RECYCLE",
                CLEAN,
                steps=["Ask whether a local store or waste bank collects clean plastic film separately."],
                text="Clean plastic film can go to a collector that takes it separately. It doesn't belong in mixed recycling by default.",
            ),
        ],
        "plastic_wrapper": [
            rule("plastic-bags-and-wrap", "REDUCE"),
            rule(
                "plastic-bags-and-wrap",
                "RECYCLE",
                CLEAN,
                steps=["Ask whether a local store or waste bank collects clean plastic film separately."],
                text="Clean plastic film can go to a collector that takes it separately. It doesn't belong in mixed recycling by default.",
            ),
        ],
        "battery": [rule("used-household-batteries", "RECYCLE", risk=True)],
        "toothbrush": [rule("manual-plastic-toothbrushes", "TRASH")],
    }
    for key in PRODUCE:
        rules[key] = [rule("food-scraps-and-composting", "RECYCLE"), rule("preventing-household-food-waste", "REDUCE")]
    for key in LEFTOVERS:
        rules[key] = [rule("preventing-household-food-waste", "REDUCE")]
    for key in GLASSWARE:
        rules[key] = [rule("broken-glass-and-sharp-shards", "TRASH", BROKEN, risk=True)]
    for key in SMALL_ELECTRONICS:
        rules[key] = [rule("small-electronics-and-cables", "RECYCLE", risk=True)]
    return rules


def build(articles, catalog):
    by_slug = {a["slug"]: a for a in articles}
    items = {item["object_key"]: item for item in catalog}
    guides = []
    for object_key, rules in mapping().items():
        item = items[object_key]
        used = [by_slug[r["article"]] for r in rules]
        for article in used:
            if article["publication_status"] != "PUBLISHED" or not article.get("reviewer"):
                raise ValueError(f"Article is not published and reviewed: {article['slug']}")
        reviewed = max(date.fromisoformat(a["reviewed_at"]) for a in used)
        reviewers = sorted({a["reviewer"] for a in used})
        primary = used[0]
        evidence = []
        for r in rules:
            article = by_slug[r["article"]]
            evidence.append(
                {
                    "id": f"{object_key}-{article['slug']}-{r['method'].lower()}",
                    "method": r["method"],
                    "text": r["text"] or article["summary"],
                    "requires": r["requires"],
                    "source_urls": [article["source_url"]],
                    "steps": r["steps"] or article["steps"],
                    "risk_compatible": r["risk_compatible"] or r["method"] == "REDUCE",
                }
            )
        cautions = list(dict.fromkeys(c for a in used for c in a["cautions"]))
        caution = ""
        for text in cautions:
            candidate = f"{caution} {text}".strip()
            if len(candidate) > 500:
                break
            caution = candidate
        slugs = ", ".join(dict.fromkeys(a["slug"] for a in used))
        guides.append(
            {
                "id": f"core-{object_key}",
                "version": 1,
                "object_key": object_key,
                "category": item["category"],
                "publication_status": "PUBLISHED",
                "reviewed_at": reviewed.isoformat(),
                "reviewer": f"{' and '.join(reviewers)} (backend guide articles)",
                "valid_until": (reviewed + timedelta(days=VALIDITY_DAYS)).isoformat(),
                "source_urls": list(dict.fromkeys(a["source_url"] for a in used)),
                "scope_note": (
                    f"Built from the reviewed backend articles: {slugs}. General handling advice, "
                    "not proof that a local facility accepts the item."
                ),
                "special_handling": bool(item["special_handling"]),
                "steps": primary["steps"],
                "caution": caution,
                "evidence": evidence,
            }
        )
    return guides


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("articles", type=Path, help="Backend core-guides JSON file")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from wastenot_ai.catalog import load_catalog

    guides = build(json.loads(args.articles.read_text("utf-8")), load_catalog()["items"])
    args.output.write_text(json.dumps(guides, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(guides)} guides to {args.output}")


if __name__ == "__main__":
    main()
