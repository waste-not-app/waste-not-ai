import json

from wastenot_ai.contracts import Context
from wastenot_ai.providers import FixtureProvider
from wastenot_ai.recommendation import load_guides, recommend


def test_broken_glass_with_electrical_risk_cannot_get_ordinary_glass_trash():
    context = Context.model_validate_json(json.dumps({"object_key": "glass_bottle", "category": "GLASS",
        "condition": {"integrity": "DAMAGED"}, "risk_flag": True, "risk_type": "ELECTRICAL"}))
    result = recommend(context, FixtureProvider(), load_guides(), demo_guides=True)
    assert result.recommended_method is None, "Electrical risk was matched to unrelated broken-glass Trash evidence"
    assert result.steps and any("baterai" in step.lower() for step in result.steps)


def test_explicit_biological_risk_with_hazardous_soiling_keeps_biological_guidance():
    context = Context.model_validate_json(json.dumps({"object_key": "ceramic_cup", "category": "OTHER",
        "condition": {"cleanliness": "DIRTY", "soiling": "HAZARDOUS"},
        "risk_flag": True, "risk_type": "BIOLOGICAL"}))
    result = recommend(context, FixtureProvider(), load_guides(), demo_guides=True)
    assert "https://www.who.int/news-room/fact-sheets/detail/health-care-waste" in result.source_urls
    assert any("medis" in step for step in result.steps)


def test_sharp_glass_without_confirmed_receiver_does_not_recommend_reuse_or_unconditional_recycle():
    for access in ("NO", "UNKNOWN"):
        context = Context.model_validate_json(json.dumps({"object_key": "glass_bottle", "category": "GLASS",
            "condition": {"integrity": "INTACT", "cleanliness": "CLEAN", "reusable": "YES"},
            "reuse_intention": True, "risk_type": "SHARP", "recycling_access": access}))
        result = recommend(context, FixtureProvider(), load_guides(), demo_guides=True)
        assert result.recommended_method != "REUSE" and result.steps
        if access == "NO":
            assert result.recommended_method is None
        else:
            assert result.status == "READY" and result.recommendation_basis == "CONDITIONAL"
            assert "recycling_access" in result.conditions_to_verify
