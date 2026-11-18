import json
from datetime import date
from pathlib import Path

import pytest

from wastenot_ai.catalog import load_catalog
from wastenot_ai.contracts import Context
from wastenot_ai.providers import RulesProvider
from wastenot_ai.recommendation import ProviderError, load_guides, recommend


class OfflineGemini:
    name = 'gemini'

    def __init__(self, code='LLM_UNAVAILABLE'):
        self.code = code
        self.calls = 0

    def generate(self, payload):
        self.calls += 1
        raise ProviderError(self.code)


def ordinary_context(item, **changes):
    value = {
        'object_key': item['object_key'],
        'category': item['category'],
        'condition': {
            'cleanliness': 'CLEAN',
            'integrity': 'INTACT',
            'reusable': 'YES',
            'former_contents': 'FOOD_DRINK',
        },
        'reuse_intention': False,
        'recycling_access': 'UNKNOWN',
    }
    return Context.model_validate_json(json.dumps({**value, **changes}))


@pytest.mark.parametrize('provider', [RulesProvider(), OfflineGemini()])
def test_all_85_catalog_items_return_sourced_methods_for_normal_confirmed_conditions(provider):
    guides = load_guides()
    for item in load_catalog()['items']:
        context = ordinary_context(item)
        before = context.model_dump()
        result = recommend(context, provider, guides, demo_guides=True)
        assert result.status == 'READY', (item['object_key'], result.reason_code)
        assert result.recommended_method in {'REUSE', 'RECYCLE', 'TRASH'}
        assert result.provider == 'rules' and result.steps and result.source_urls and result.evidence_ids
        assert context.model_dump() == before
        assert result.recommendation_basis == 'CONDITIONAL'
        assert 'recycling_access' in result.conditions_to_verify


@pytest.mark.parametrize(
    'code', ['LLM_TIMEOUT', 'LLM_RATE_LIMITED', 'LLM_UNAVAILABLE', 'LLM_NOT_CONFIGURED', 'DEMO_HOURLY_LIMIT']
)
def test_screenshot_plastic_cup_returns_conditional_recycle_when_gemini_fails(code):
    item = next(item for item in load_catalog()['items'] if item['object_key'] == 'plastic_cup')
    result = recommend(ordinary_context(item), OfflineGemini(code), load_guides(), demo_guides=True)
    assert result.status == 'READY' and result.recommended_method == 'RECYCLE'
    assert result.provider == 'rules' and result.reason_code == code
    assert result.recommendation_basis == 'CONDITIONAL'
    assert result.conditions_to_verify == ['recycling_access']


def test_rule_priority_uses_confirmed_reuse_and_confirmed_residue():
    item = next(item for item in load_catalog()['items'] if item['object_key'] == 'book')
    reuse = recommend(ordinary_context(item, reuse_intention=True), RulesProvider(), load_guides(), demo_guides=True)
    assert reuse.recommended_method == 'REUSE' and reuse.recommendation_basis == 'CONFIRMED'
    trash = recommend(ordinary_context(item, recycling_access='NO'), RulesProvider(), load_guides(), demo_guides=True)
    assert trash.recommended_method == 'TRASH' and trash.recommendation_basis == 'CONFIRMED'


def test_strict_context_still_uses_direct_routes_with_all_requirements_confirmed():
    item = next(item for item in load_catalog()['items'] if item['object_key'] == 'book')
    result = recommend(
        ordinary_context(item, recycling_access='NO', allow_conditional=False),
        RulesProvider(),
        load_guides(),
        demo_guides=True,
    )
    assert result.status == 'READY' and result.recommended_method == 'TRASH'
    assert result.recommendation_basis == 'CONFIRMED'
    unknown = recommend(
        ordinary_context(item, allow_conditional=False), RulesProvider(), load_guides(), demo_guides=True
    )
    assert unknown.status == 'NEEDS_CONTEXT' and unknown.recommended_method is None


@pytest.mark.parametrize('key', ['battery', 'other_electronic'])
def test_unspecified_battery_types_are_not_given_generic_device_reuse(key):
    item = next(item for item in load_catalog()['items'] if item['object_key'] == key)
    result = recommend(ordinary_context(item, reuse_intention=True), RulesProvider(), load_guides(), demo_guides=True)
    assert result.recommended_method == 'RECYCLE' and result.special_handling


@pytest.mark.parametrize('risk', ['CHEMICAL', 'BIOLOGICAL', 'OTHER', 'UNKNOWN'])
def test_special_risk_requires_a_supported_route_without_an_ordinary_default(risk):
    item = next(item for item in load_catalog()['items'] if item['object_key'] == 'plastic_cup')
    provider = OfflineGemini()
    result = recommend(
        ordinary_context(item, risk_flag=True, risk_type=risk), provider, load_guides(), demo_guides=True
    )
    assert result.status in {'NEEDS_CONTEXT', 'UNAVAILABLE'}
    assert result.recommended_method is None and result.special_handling and result.steps
    assert provider.calls == 0


def test_direct_routes_do_not_publish_or_modify_a_reviewed_production_guide(tmp_path):
    source = next(guide for guide in load_guides() if guide.object_key == 'cardboard')
    published = source.model_copy(
        update={
            'publication_status': 'PUBLISHED',
            'reviewer': 'Test reviewer',
            'reviewed_at': '2026-11-01',
            'valid_until': '2026-12-01',
        }
    )
    path = tmp_path / 'guides.json'
    path.write_text(json.dumps([published.model_dump(mode='json')]))
    assert load_guides(path)[0] == published


def test_gemini_outage_uses_only_the_published_production_evidence():
    guides = load_guides(Path(__file__).resolve().parents[1] / 'guides/pilot-source-reviewed-v6.json')
    item = next(item for item in load_catalog()['items'] if item['object_key'] == 'pet_bottle')
    result = recommend(
        ordinary_context(item, recycling_access='YES'), OfflineGemini(), guides, today=date(2026, 11, 13)
    )
    assert result.status == 'READY' and result.recommended_method == 'RECYCLE'
    assert result.provider == 'rules' and not result.prototype_only and result.reason_code == 'LLM_UNAVAILABLE'
    assert all('-direct09-' not in identifier for identifier in result.evidence_ids)
    expired = recommend(ordinary_context(item, recycling_access='YES'), OfflineGemini(), guides, today=date(2027, 1, 1))
    assert expired.status == 'UNAVAILABLE' and expired.recommended_method is None
