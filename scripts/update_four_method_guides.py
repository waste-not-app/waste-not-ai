"""Idempotently author the draft evidence for the four-method demo policy."""

import json
from pathlib import Path

root = Path(__file__).resolve().parents[1] / "src/wastenot_ai/data"
prevention_url = "https://www.epa.gov/climate-change/what-you-can-do-about-climate-change-waste"
trash_url = "https://www.cornwall.gov.uk/rubbish-recycling-and-waste/recycling/recycling-items/toothbrushes/"
glass_url = "https://www.cornwall.gov.uk/rubbish-recycling-and-waste/recycling/recycling-items/broken-glass/"
sharp_url = (
    "https://www.wandsworth.gov.uk/rubbish-and-recycling/recycle-and-reuse/waste-and-recycling-a-z/kitchen-knives/"
)
ceramic_url = "https://www.islington.gov.uk/recycling-and-rubbish/recycling/what-we-pick-up-as-household-rubbish"
wash_url = "https://www.epa.gov/greenerproducts/identifying-sustainable-food-service-and-food-service-ware"
local_reuse_url = (
    "https://banjarnegarakab.go.id/2026/06/11/banjarnegara-terapkan-aturan-baru-kelola-sampah-wajib-dari-hulu/"
)


def prevention(identifier, food=False):
    return {
        "id": identifier,
        "method": "REDUCE",
        "requires": {},
        "risk_compatible": True,
        "text": (
            "To avoid leftovers next time, plan portions and buy only what you need. "
            "This advice doesn't decide whether the food you have now is safe to eat or how to throw it away."
        )
        if food
        else (
            "To avoid the next piece of waste, skip purchases you don't need and choose durable products "
            "or less packaging. This is advice for next time, not for getting rid of the item you have now."
        ),
        "source_urls": [prevention_url],
        "steps": ["Plan what you need before you shop and avoid buying too much."]
        if food
        else ["Next time you shop, buy only what you need and look for durable products or less packaging."],
    }


templates = json.loads((root / "guide_templates.json").read_text("utf-8"))
for key, guide in templates.items():
    if key in {"electronics", "sharp", "toothbrush"}:
        continue
    guide["evidence"] = [e for e in guide["evidence"] if e["method"] != "REDUCE"]
    guide["evidence"].append(prevention("prevent-waste-v2", food=key == "food"))
    if key == "food":
        guide["caution"] = (
            "A photo can't tell whether food is safe to eat. REDUCE is about preventing the next waste. Handle current scraps through your local organic waste service."
        )
templates["toothbrush"] = {
    "source_urls": [trash_url],
    "steps": [
        "Confirm it's a manual plastic toothbrush with no battery or electrical parts.",
        "Check for a specialist collection program before choosing residual waste under local rules.",
    ],
    "caution": "Cornwall, UK reference. These are not Indonesian acceptance rules. Electric or battery toothbrushes need specialist electronics handling.",
    "evidence": [
        prevention("prevent-waste-v2"),
        {
            "id": "manual-plastic-residue-v2",
            "method": "TRASH",
            "text": "A worn-out manual plastic toothbrush with no specialist recycling option can go to residual waste under local rules. Cornwall does not include toothbrushes in regular recycling collections.",
            "requires": {"reusable": "NO", "recycling_access": "NO"},
            "source_urls": [trash_url],
            "steps": [
                "Make sure it's a manual plastic toothbrush without a battery, and not medical or hazardous waste.",
                "If there's no specialist recycling program, use the residual waste route your local collector sets.",
            ],
        },
    ],
}


def broken_glass(identifier):
    return {
        "id": identifier,
        "method": "TRASH",
        "risk_compatible": True,
        "requires": {"category": "GLASS", "integrity": "DAMAGED"},
        "text": "Broken glass goes to Trash as residual waste, wrapped so it can't cut the people who collect it. Cornwall's guidance puts broken glass in residual waste. Check your local collector's rules.",
        "source_urls": [glass_url],
        "steps": [
            "Don't reuse cracked or broken glass, and avoid touching sharp edges.",
            "Wrap it or put it in a sturdy container labelled as broken glass. Ask the collection staff for help if it isn't safe to handle.",
            "Put it out with residual waste as your local collector instructs. Don't mix loose shards with regular recycling.",
        ],
    }


tableware = templates["tableware"]
tableware["evidence"] = [e for e in tableware["evidence"] if not e["id"].endswith(("-v3", "-v4"))]
tableware["source_urls"] = [wash_url, local_reuse_url]
tableware["caution"] = (
    "Wash it before using it again. Normal washing doesn't deal with chemical or biological contamination. Don't reuse cracked or broken glass or ceramics."
)
tableware["evidence"][0].update(
    text="Clean, intact, reusable tableware that's still in good shape can go back into use. You said you want to keep using it.",
    source_urls=[wash_url, local_reuse_url],
    steps=[
        "Check again that it's clean and has no cracks or sharp edges.",
        "Use it for its intended purpose and follow the material or product instructions. Keeping usable tableware in service avoids extra waste.",
    ],
)
for soil, label, verb in [("FOOD_DRINK", "food or drink residue", "washes off"), ("DUST", "dust", "cleans off")]:
    tableware["evidence"].append(
        {
            "id": f"tableware-wash-{soil.lower()}-reuse-v4",
            "method": "REUSE",
            "risk_compatible": False,
            "requires": {
                "cleanliness": "DIRTY",
                "integrity": "INTACT",
                "reusable": "YES",
                "reuse_intention": True,
                "cleanable": "YES",
                "soiling": soil,
            },
            "text": f"This item is intact and usable, and you confirmed the dirt is ordinary {label} that {verb}. Reuse it after washing and checking it's clean.",
            "source_urls": [wash_url, local_reuse_url],
            "steps": [
                "Remove the dirt, then wash, rinse, and dry it as the material or product instructions say before using it again.",
                "Check its condition again after washing. If you find cracks, damage, or signs of hazardous contamination, stop and update the item's condition.",
                "Reuse it for its intended purpose once it's clean and dry.",
            ],
        }
    )
tableware["evidence"].append(broken_glass("broken-glass-trash-v3"))
for state, requirements in [
    ("damaged", {"integrity": "DAMAGED"}),
    ("unusable", {"integrity": "INTACT", "reusable": "NO"}),
]:
    tableware["evidence"].append(
        {
            "id": f"ceramic-{state}-trash-v3",
            "method": "TRASH",
            "risk_compatible": state == "damaged",
            "requires": {"category": "OTHER", **requirements},
            "text": "Damaged or unusable ceramics go to Trash through your local residual waste route, not into glass bottle recycling. Islington's guidance puts ceramic tableware in household waste.",
            "source_urls": [ceramic_url],
            "steps": [
                "Don't reuse ceramic tableware that's damaged or no longer usable.",
                "Cover sharp parts with wrapping or a sturdy container, and ask local collection staff how to hand it over.",
            ],
        }
    )
templates["electronics"]["evidence"] = [
    prevention("prevent-waste-v2"),
    {
        "id": "specialist-electronics-recycle-v3",
        "method": "RECYCLE",
        "risk_compatible": True,
        "requires": {},
        "text": "Electronics go to Recycle through a manufacturer take-back program or an e-waste handler that accepts this type of device, so the materials can be recovered. This doesn't mean it can go in a regular recycling bin.",
        "source_urls": templates["electronics"]["source_urls"],
        "steps": [
            "Contact the manufacturer or an e-waste handler to confirm they accept it and how to hand it over.",
            "Protect your personal data if you can do it safely. Don't take apart batteries or damaged devices yourself.",
            "Hand it over through a specialist service. Devices with batteries never go in the regular trash or recycling.",
        ],
    },
]
templates["sharp"].update(
    source_urls=[sharp_url],
    evidence=[
        prevention("prevent-waste-v2"),
        {
            "id": "knife-specialist-recycle-v3",
            "method": "RECYCLE",
            "risk_compatible": True,
            "requires": {"object_key": "knife"},
            "text": "A metal knife can go to Recycle through a metal handler that accepts sharp items. Wandsworth runs a dedicated scrap metal route. Confirm acceptance and packaging with your local collector.",
            "source_urls": [sharp_url],
            "steps": [
                "Contact a metal handler to confirm they accept knives, and follow how they want it handed over.",
                "Cover the blade so it can't hurt collection staff. Don't put a loose knife in the regular recycling bin.",
            ],
        },
    ],
)
templates["sharp"]["evidence"].append(
    {
        "id": "scissors-specialist-recycle-v3",
        "method": "RECYCLE",
        "risk_compatible": True,
        "requires": {"object_key": "scissors"},
        "text": "Metal scissors go to Recycle through a collection center that accepts scissors. Auckland runs such a route. Check that yours accepts them, and cover the blades as local staff instruct.",
        "source_urls": [
            "https://www.aucklandcouncil.govt.nz/en/rubbish-recycling/get-rid-unwanted-items/organisation/331.html"
        ],
        "steps": [
            "Contact a collection center that accepts metal scissors and ask how to package them.",
            "Cover the blades so they can't hurt collection staff. Don't put loose scissors in the regular recycling bin.",
        ],
    }
)

guides = json.loads((root / "guides.json").read_text("utf-8"))
for guide in guides:
    if guide["version"] >= 5:
        continue  # Pilot rules are maintained in guides.json after the policy-05 review.
    guide["version"] = 4
    guide["evidence"] = [e for e in guide["evidence"] if e["method"] != "REDUCE"]
    guide["evidence"].append(prevention(guide["object_key"] + "-prevent-waste-v2"))
    if guide["object_key"] == "glass_bottle":
        guide["evidence"] = [e for e in guide["evidence"] if e["id"] != "glass-broken-trash-v3"]
        guide["evidence"].append(broken_glass("glass-broken-trash-v3"))
    if guide["object_key"] == "battery":
        guide["evidence"] = [
            prevention("battery-prevent-waste-v2"),
            {
                "id": "battery-specialist-recycle-v3",
                "method": "RECYCLE",
                "risk_compatible": True,
                "requires": {},
                "text": "Used batteries go to Recycle through a battery collector or specialist waste handler that accepts this type, so valuable materials can be recovered. Don't mix them into household trash or a regular recycling bin.",
                "source_urls": guide["source_urls"],
                "steps": [
                    "Contact a battery collector or a specialist waste handler to confirm they accept this type of battery.",
                    "If the battery is damaged, leaking, or swollen, ask the manufacturer or the collector how to handle it, and don't take it apart.",
                    "Hand it over through the specialist service. Don't burn it or put it in the regular trash or recycling.",
                ],
            },
        ]
(root / "guide_templates.json").write_text(json.dumps(templates, ensure_ascii=False, indent=2) + "\n", "utf-8")
(root / "guides.json").write_text(json.dumps(guides, ensure_ascii=False, indent=2) + "\n", "utf-8")
