import json

import pytest

from wastenot_ai.catalog import load_catalog
from wastenot_ai.contracts import Context
from wastenot_ai.providers import FixtureProvider
from wastenot_ai.recommendation import ProviderError, load_guides, recommend


def context(**updates):
    return Context.model_validate_json(json.dumps({"object_key": "glass_bottle", "category": "GLASS",
        "condition": {"cleanliness": "CLEAN", "integrity": "INTACT", "reusable": "YES"},
        "reuse_intention": True, "recycling_access": "YES", **updates}))


def test_screenshot_sharp_risk_with_confirmed_intact_glass_has_a_method_and_warning():
    value = context(risk_flag=True, risk_type="SHARP", recommendation_goal="HANDLING")
    result = recommend(value, FixtureProvider(), load_guides(), demo_guides=True)
    assert result.status == "READY" and result.recommended_method == "RECYCLE"
    assert result.special_handling and result.caution and result.steps and result.source_urls
    assert value.condition.integrity == "INTACT" and value.risk_type == "SHARP"


@pytest.mark.parametrize("risk", [None, "UNKNOWN", "SHARP", "CHEMICAL", "BIOLOGICAL", "ELECTRICAL", "OTHER"])
def test_every_catalog_item_has_handling_steps_even_when_no_method_can_be_selected(risk):
    guides = load_guides()
    for item in load_catalog()["items"]:
        result = recommend(Context.model_validate_json(json.dumps({"object_key": item["object_key"],
            "category": item["category"], "risk_flag": risk is not None, "risk_type": risk or "UNKNOWN"})),
            FixtureProvider(), guides, demo_guides=True)
        assert result.steps and result.reason, (item["object_key"], risk, result.reason_code)
        assert result.source_urls, item["object_key"]


def test_provider_failure_uses_eligible_rules_and_preserves_failure_provenance():
    class Offline(FixtureProvider):
        def generate(self, payload):
            raise ProviderError("LLM_TIMEOUT")
    result = recommend(context(risk_flag=False), Offline(), load_guides(), demo_guides=True)
    assert result.status == 'READY' and result.reason_code == 'LLM_TIMEOUT'
    assert result.provider == 'rules' and result.recommended_method in {'REUSE', 'RECYCLE', 'TRASH'}
    assert result.steps and result.source_urls and result.evidence_ids


@pytest.mark.parametrize("category", ["ORGANIC", "PLASTIC", "PAPER", "GLASS", "METAL", "ELECTRONIC", "OTHER"])
def test_items_outside_named_catalog_have_explicit_manual_material_choices(category):
    result = recommend(Context.model_validate_json(json.dumps({"object_key": "other_" + category.lower(),
        "category": category})), FixtureProvider(), load_guides(), demo_guides=True)
    assert result.steps and result.source_urls and result.recommended_method == 'RECYCLE'
    assert result.recommendation_basis == 'CONDITIONAL' and 'recycling_access' in result.conditions_to_verify


def test_no_reviewed_guide_still_returns_independent_general_handling_guidance():
    result = recommend(context(), FixtureProvider(), [], demo_guides=False)
    assert result.status == "UNAVAILABLE" and result.reason_code == "NO_REVIEWED_GUIDE"
    assert result.steps and result.source_urls and not result.guide_refs
