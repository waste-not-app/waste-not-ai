# Recommendation Guides

The recommendation worker only gives advice for objects that have a published, reviewed guide that hasn't expired. This folder holds the bundles an operator can install.

| Bundle | Objects | Review | Valid until |
|---|---|---|---|
| `core-reviewed-v1.json` | 34 | Alberd, 16 November 2026, through the backend guide articles | 16 May 2027 |
| `pilot-source-reviewed-v6.json` | 4 | Automated source review on 4 November 2026, not a human safety review | 4 December 2026 |

Use `core-reviewed-v1.json`. The v6 pilot bundle stays here as history and stops working on 4 December 2026, when the worker starts rejecting it as expired.

## Where the core bundle comes from

`core-reviewed-v1.json` is generated from the 16 articles in [`waste-not-backend/guides/core-guides-v1.json`](https://github.com/waste-not-app/waste-not-backend/blob/main/guides/core-guides-v1.json). Those are the same articles the app shows on the Guide screen, so the Guide screen and the scan advice now say the same thing. Every evidence entry reuses an article's summary, steps, cautions, and source, and is labelled with the article it came from.

[`scripts/build_reviewed_guides.py`](../scripts/build_reviewed_guides.py) holds the mapping from articles to catalog objects and the conditions each method needs. Regenerate the bundle whenever the backend articles change:

```sh
python scripts/build_reviewed_guides.py ../waste-not-backend/guides/core-guides-v1.json
```

The script refuses articles that aren't published or have no reviewer. Each guide's review date is the newest article it uses, and it stays valid for 181 days after that.

## What the core bundle covers

| Objects | Article | Methods |
|---|---|---|
| PET and HDPE food or drink bottles, aluminium cans | Plastic bottles, HDPE containers, aluminium cans | Recycle once the item is clean, intact, held food or drink, and a collector accepts it |
| Steel food cans | Food cans and clean foil | Recycle once clean and confirmed as a food can |
| Cardboard, paper | Cardboard boxes, office paper and newspapers | Recycle when clean and dry |
| Paper books | Office paper and newspapers | Pass on readable books, otherwise recycle |
| Glass bottles | Intact bottles and jars, broken glass | Recycle or reuse when intact; protected residual waste when broken |
| Glass cups, bowls, vases, and other glass | Broken glass | Protected residual waste only when broken. Intact glassware isn't covered |
| Plastic bags and wrap | Plastic bags and wrap | Reduce, reuse a clean bag, or a film collector |
| Fruit and vegetables | Food scraps and composting, preventing food waste | Compost with access to composting; prevention advice |
| Cooked leftovers | Preventing food waste | Prevention advice only, because the composting article excludes meat, dairy, and oil |
| Batteries | Used household batteries | Specialist collection point |
| Phones, laptops, mice, remotes, keyboards, clocks, hair dryers | Small electronics and cables | Official e-waste collector |
| Manual toothbrushes | Manual plastic toothbrushes | Residual waste, or a take-back program if there is one |

Objects that no article clearly covers, such as coated cups, drink cartons, foam containers, large appliances, and furniture, still fall back to manual entry. Leaking batteries and items with a chemical history are never sent through the regular routes.

## Review status

Alberd reviewed the article text on 16 November 2026. The mapping from articles to catalog objects in the build script is new, so it should get one review pass alongside the English wording of the articles. Approval covers the advice text, not a legal review, a facility inspection, or a guarantee that a local collector accepts the item.

## Installation

The production worker reads its bundle from a file the operator manages outside the release directory. CD never overwrites it.

1. Copy `core-reviewed-v1.json` to a versioned path that only the recommendation service can read, and record its checksum.
2. Point `AI_GUIDES_PATH` in `/etc/wastenot/ai-recommendation.env` at that path, then restart `wastenot-decisions`. The worker validates the schema, status, and validity window on load and refuses to start on a bad file.
3. Keep `AI_ALLOW_PROTOTYPE=false` and the real Gemini provider.
4. Send one request through the public API to confirm advice comes back.

Slow or invalid provider responses still return `UNAVAILABLE`. The server never invents a method or saves anything automatically.
