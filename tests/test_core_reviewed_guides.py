import json
from datetime import date
from pathlib import Path

import pytest

from wastenot_ai.catalog import load_catalog
from wastenot_ai.contracts import Context
from wastenot_ai.providers import FixtureProvider
from wastenot_ai.recommendation import is_reviewed, load_guides, recommend

BUNDLE = Path(__file__).parents[1] / "guides/core-reviewed-v1.json"
GUIDES = load_guides(BUNDLE)
TODAY = date(2026, 11, 20)


class ContractProvider(FixtureProvider):
    name = "contract-test"


def advise(object_key, condition=None, **extra):
    category = next(g.category for g in GUIDES if g.object_key == object_key)
    context = Context.model_validate(
        {"object_key": object_key, "category": category, "condition": condition or {}, **extra}
    )
    return recommend(context, ContractProvider(), GUIDES, today=TODAY)


def test_every_guide_is_published_reviewed_and_time_bounded():
    catalog = {item["object_key"]: item for item in load_catalog()["items"]}
    assert len(GUIDES) == 34
    for guide in GUIDES:
        assert guide.publication_status == "PUBLISHED"
        assert guide.reviewer == "Alberd (backend guide articles)"
        assert guide.category == catalog[guide.object_key]["category"]
        assert guide.special_handling == catalog[guide.object_key]["special_handling"]
        assert is_reviewed(guide, date(2026, 11, 16))
        assert is_reviewed(guide, date(2027, 5, 16))
        assert not is_reviewed(guide, date(2027, 5, 17))


def test_bundle_cites_only_backend_article_sources():
    raw = json.loads(BUNDLE.read_text("utf-8"))
    for guide in raw:
        assert guide["scope_note"].startswith("Built from the reviewed backend articles")
        for evidence in guide["evidence"]:
            assert evidence["source_urls"] and set(evidence["source_urls"]) <= set(guide["source_urls"])


@pytest.mark.parametrize("object_key", ["pet_bottle", "hdpe_bottle", "aluminium_can"])
def test_food_and_drink_containers_recycle_once_confirmed(object_key):
    result = advise(
        object_key,
        {"cleanliness": "CLEAN", "integrity": "INTACT", "former_contents": "FOOD_DRINK"},
        recycling_access="YES",
    )
    assert result.status == "READY" and result.recommended_method == "RECYCLE"
    assert result.recommendation_basis == "CONFIRMED" and result.prototype_only is False


def test_unknown_former_contents_never_becomes_recycle_advice():
    result = advise("hdpe_bottle", {"cleanliness": "CLEAN", "integrity": "INTACT"}, recycling_access="YES")
    assert result.status == "NEEDS_CONTEXT"
    assert result.recommended_method is None
    assert "condition.former_contents" in result.missing_context


def test_broken_glass_bottle_goes_to_protected_residual_waste():
    result = advise("glass_bottle", {"integrity": "DAMAGED"})
    assert result.status == "READY" and result.recommended_method == "TRASH"
    assert result.special_handling is True
    assert any("thick gloves" in step for step in result.steps)


def test_intact_clean_glass_bottle_recycles_with_collector_access():
    result = advise("glass_bottle", {"cleanliness": "CLEAN", "integrity": "INTACT"}, recycling_access="YES")
    assert result.status == "READY" and result.recommended_method == "RECYCLE"


def test_intact_glassware_is_not_covered_by_the_bottle_article():
    result = advise("glass_cup", {"cleanliness": "CLEAN", "integrity": "INTACT"}, recycling_access="YES")
    assert result.status != "READY"
    assert advise("glass_cup", {"integrity": "DAMAGED"}).recommended_method == "TRASH"


@pytest.mark.parametrize("object_key", ["battery", "cell_phone", "laptop"])
def test_batteries_and_small_electronics_use_a_specialist_route(object_key):
    result = advise(object_key, recycling_access="YES")
    assert result.status == "READY" and result.recommended_method == "RECYCLE"
    assert result.special_handling is True


def test_leaking_battery_is_not_covered_by_regular_collection():
    result = advise("battery", risk_type="CHEMICAL", recycling_access="YES")
    assert result.status == "UNAVAILABLE" and result.reason_code == "RISK_NOT_COVERED"


def test_fresh_produce_composts_and_leftovers_only_get_prevention_advice():
    assert advise("banana", recycling_access="YES").recommended_method == "RECYCLE"
    assert advise("pizza", recycling_access="YES").status != "READY"
    prevention = advise("pizza", recommendation_goal="PREVENTION")
    assert prevention.status == "READY" and prevention.recommended_method == "REDUCE"
    assert prevention.recommendation_scope == "PREVENTION"


def test_plastic_bags_reuse_or_go_to_a_film_collector():
    reuse = advise(
        "plastic_bag",
        {"cleanliness": "CLEAN", "integrity": "INTACT", "reusable": "YES"},
        reuse_intention=True,
        recycling_access="NO",
    )
    assert reuse.recommended_method == "REUSE"
    film = advise("plastic_wrapper", {"cleanliness": "CLEAN"}, recycling_access="YES")
    assert film.recommended_method == "RECYCLE"


def test_manual_toothbrush_goes_to_residual_waste():
    assert advise("toothbrush").recommended_method == "TRASH"
