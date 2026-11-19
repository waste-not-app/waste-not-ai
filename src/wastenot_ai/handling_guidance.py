"""Useful first steps when a specific handling method cannot yet be validated."""

COMMON = "https://www.epa.gov/recycle/how-do-i-recycle-common-recyclables"
HAZARDOUS = "https://www.epa.gov/hw/household-hazardous-waste-hhw"
SHARP = "https://www.cornwall.gov.uk/rubbish-recycling-and-waste/recycling/recycling-items/broken-glass/"
BIOLOGICAL = "https://www.who.int/news-room/fact-sheets/detail/health-care-waste"
BATTERIES = "https://www.epa.gov/recycle/used-household-batteries"
ELECTRONICS = "https://www.epa.gov/recycle/electronics-donation-and-recycling"


def in_sentence(label):
    return label[:1].lower() + label[1:] if label[1:2].islower() else label


def handling_guidance(context, item, *, risk):
    chemical = (
        context.risk_type == "CHEMICAL"
        or context.condition.soiling == "HAZARDOUS"
        or context.condition.former_contents == "CHEMICAL"
    )
    if chemical and context.risk_type != "BIOLOGICAL":
        return {
            "reason": "There are signs of hazardous material. Follow these interim steps while you confirm a specialist handler.",
            "steps": [
                "Follow the instructions on the product label, and keep the original container and label if it is safe to do so.",
                "Don't mix the contents, pour them down a drain, or put them in the regular trash or recycling.",
                "Contact your local environment agency or hazardous waste handler. Tell them the product type and the container's condition, and ask how to package and hand it over.",
            ],
            "source_urls": [HAZARDOUS],
        }
    if context.risk_type == "BIOLOGICAL":
        return {
            "reason": "Items that may be biologically contaminated need a specialist route. Here are the first steps before handing it over.",
            "steps": [
                "Avoid direct contact with body fluids, contaminated material, and sharp parts.",
                "Keep it out of general waste as staff instruct. Don't try to reuse or treat it yourself.",
                "Contact a health facility or licensed medical waste handler about containers, collection, and hand-over.",
                *(
                    [
                        "If chemicals are involved too, don't mix the contents or pour them down a drain, and tell the handler about both risks."
                    ]
                    if chemical
                    else []
                ),
            ],
            "source_urls": [BIOLOGICAL, *([HAZARDOUS] if chemical else [])],
        }
    if context.risk_type == "ELECTRICAL" or context.category == "ELECTRONIC":
        return {
            "reason": "Electronics and batteries need their type checked and a specialist collector. Here are some first steps.",
            "steps": [
                "Check the label and the device and battery type without opening or puncturing anything.",
                "Never put lithium batteries, or devices that contain them, in the trash or general recycling.",
                "Find an official electronics or battery collection point, or a manufacturer program. Confirm they accept your item in its condition and ask how to package it.",
            ],
            "source_urls": [BATTERIES, ELECTRONICS],
        }
    if context.risk_type == "SHARP" or (context.category == "GLASS" and context.condition.integrity == "DAMAGED"):
        return {
            "reason": "There is a risk of shards or sharp edges. Check the item and follow these handling steps before deciding where it goes.",
            "steps": [
                "Don't reuse cracked or broken items, and don't touch sharp edges with bare hands.",
                "Wrap broken glass in a sturdy container and label it so collection staff know what is inside.",
                "Tell your local collector what the item is and that it is sharp before you hand it over. What they accept and where it goes follow local rules.",
            ],
            "source_urls": [SHARP, COMMON],
        }
    if risk:
        return {
            "reason": "The risk needs more detail. While you check, follow these first steps and ask a suitable handler for advice.",
            "steps": [
                "Check the label and the type of risk without opening the item, mixing contents, or trying to treat it.",
                "Don't reuse a risky item or mix it with general waste until you know what the hazard is.",
                "Contact your local waste handler or environment agency and describe the item and any known risks.",
            ],
            "source_urls": [HAZARDOUS],
        }
    material = {
        "PLASTIC": "plastik",
        "PAPER": "kertas/kardus",
        "GLASS": "kaca",
        "METAL": "logam",
        "ORGANIC": "bahan organik",
    }.get(context.category, "bahan barang")
    return {
        "reason": f"We can't choose a method for this {in_sentence(item['label'])} yet. Here is some starting advice based on what we know.",
        "steps": [
            f"Check the {material} type, its condition, and what it held. If you don't know, don't guess from the photo.",
            "Consider reusing it only once you've confirmed it works, is clean, and is safe.",
            "Ask your local collector whether they accept this item in its condition, then follow their sorting and hand-over instructions.",
            "If there are signs of a battery or electrics, sharp edges, chemicals, or biological contamination, mark the risk type to get specialist advice.",
        ],
        "source_urls": [COMMON],
    }
