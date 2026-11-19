import json
from datetime import date
from importlib.resources import files
from pathlib import Path
from typing import Protocol

from pydantic import ValidationError

from .catalog import confirm_context, load_catalog
from .contracts import Context, Evidence, Guide, ModelSuggestion, Recommendation
from .guide_audit import validate_guides
from .handling_guidance import handling_guidance, in_sentence
from .sources import describe_sources


class ProviderError(Exception):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


class Provider(Protocol):
    name: str

    def generate(self, payload: dict) -> str: ...


def load_guides(path: Path | None = None) -> list[Guide]:
    text = path.read_text("utf-8") if path else files("wastenot_ai").joinpath("data/guides.json").read_text("utf-8")
    guides = [Guide.model_validate_json(json.dumps(x)) for x in json.loads(text)]
    if path is None:
        templates = json.loads(files("wastenot_ai").joinpath("data/guide_templates.json").read_text("utf-8"))
        for item in load_catalog()["items"]:
            if item["object_key"] in {g.object_key for g in guides} or not item.get("guide_template"):
                continue
            template = templates[item["guide_template"]]
            evidence = [{**e, "id": f"{item['object_key']}-{e['id']}"} for e in template["evidence"]]
            value = {
                **template,
                "evidence": evidence,
                "id": f"demo-{item['object_key']}",
                "version": 4,
                "object_key": item["object_key"],
                "category": item["category"],
                "publication_status": "DRAFT",
                "reviewed_at": None,
                "reviewer": None,
                "valid_until": None,
                "scope_note": "Draf demo berbasis prinsip umum; perlu review penerapan lokal Indonesia.",
                "special_handling": item["special_handling"],
            }
            guides.append(Guide.model_validate_json(json.dumps(value)))
        routes = json.loads(files('wastenot_ai').joinpath('data/handling_routes.json').read_text('utf-8'))
        items = {item['object_key']: item for item in load_catalog()['items']}
        for index, guide in enumerate(guides):
            item = items[guide.object_key]
            if guide.publication_status != 'DRAFT':
                continue
            route = (
                'organic'
                if guide.category == 'ORGANIC'
                else 'electronic'
                if guide.category == 'ELECTRONIC'
                else 'ordinary'
            )
            if route == 'ordinary' and (item['special_handling'] or guide.special_handling):
                continue
            additions = []
            for value in routes[route]:
                if value['method'] == 'REUSE' and route == 'electronic' and item.get('guide_template') != 'electronics':
                    continue
                if (
                    value['method'] == 'REUSE'
                    and route == 'ordinary'
                    and item.get('guide_template') not in {'reuse', 'tableware', 'book'}
                ):
                    continue
                requirements = dict(value['requires'])
                if value['method'] == 'RECYCLE' and route == 'ordinary':
                    requirements['cleanliness'] = 'CLEAN'
                    if guide.object_key in {'pet_bottle', 'hdpe_bottle', 'glass_bottle', 'aluminium_can', 'cardboard'}:
                        requirements['integrity'] = 'INTACT'
                if guide.object_key == 'hdpe_bottle' or item.get('guide_template') == 'broad_food_container':
                    requirements['former_contents'] = 'FOOD_DRINK'
                additions.append(
                    Evidence.model_validate(
                        {**value, 'requires': requirements, 'id': f"{guide.object_key}-direct09-{value['id']}"}
                    )
                )
            guides[index] = guide.model_copy(
                update={'evidence': [*guide.evidence, *additions], 'version': guide.version + 1}
            )
    if len({x.object_key for x in guides}) != len(guides):
        raise ValueError("Duplicate guide for object_key.")
    for guide in guides:
        ids = [x.id for x in guide.evidence]
        if len(set(ids)) != len(ids):
            raise ValueError("Evidence ID duplikat.")
    validate_guides(guides)
    return guides


def is_reviewed(guide: Guide, today: date) -> bool:
    if guide.publication_status != "PUBLISHED" or not guide.reviewer or not guide.reviewed_at or not guide.valid_until:
        return False
    try:
        return date.fromisoformat(guide.reviewed_at) <= today <= date.fromisoformat(guide.valid_until)
    except ValueError:
        return False


def evidence_values(context):
    return {
        **context.condition.model_dump(),
        'reuse_intention': context.reuse_intention,
        'recycling_access': context.recycling_access,
        'category': context.category.value,
        'object_key': context.object_key,
        'risk_type': context.risk_type,
        'risk_flag': context.risk_flag,
    }


def evidence_requirements(evidence):
    defaults = {}
    if evidence.method == 'REUSE':
        defaults = {'cleanliness': 'CLEAN', 'integrity': 'INTACT', 'reusable': 'YES', 'reuse_intention': True}
    elif evidence.method == 'RECYCLE':
        return {**evidence.requires, 'recycling_access': evidence.requires.get('recycling_access', 'YES')}
    return {**defaults, **evidence.requires}


def eligible_evidence(context: Context, guide: Guide, *, risk: bool = False):
    values = evidence_values(context)
    washable = (
        context.condition.cleanliness == "DIRTY"
        and context.condition.cleanable == "YES"
        and context.condition.soiling in {"FOOD_DRINK", "DUST"}
    )
    reusable = (
        (context.condition.cleanliness == "CLEAN" or washable)
        and context.condition.integrity == "INTACT"
        and context.condition.reusable == "YES"
        and context.reuse_intention is True
    )
    hazardous = (
        context.risk_type in {"CHEMICAL", "BIOLOGICAL", "OTHER"}
        or context.condition.soiling == "HAZARDOUS"
        or context.condition.former_contents == "CHEMICAL"
    )
    return [
        e
        for e in guide.evidence
        if (e.method != "REUSE" or reusable)
        and (not risk or e.risk_compatible)
        and (context.recommendation_goal != "PREVENTION" or e.method == "REDUCE")
        and (context.recommendation_goal == "PREVENTION" or e.method != "REDUCE")
        and (not hazardous or e.method == "REDUCE")
        and (e.method != "RECYCLE" or context.recycling_access != "NO")
        and all(values.get(k) == v for k, v in evidence_requirements(e).items())
    ]


def missing_context(context: Context, guide: Guide, *, risk: bool) -> list[str]:
    values = evidence_values(context)
    options = []
    for evidence in guide.evidence:
        if evidence.method == "REDUCE" or (risk and not evidence.risk_compatible):
            continue
        if evidence.method == "RECYCLE" and context.recycling_access == "NO":
            continue
        requirements = evidence_requirements(evidence)
        if any(values.get(key) not in {None, "UNKNOWN", expected} for key, expected in requirements.items()):
            continue
        needed = [
            f"condition.{key}" if key in type(context.condition).model_fields else key
            for key in requirements
            if values.get(key) in {None, "UNKNOWN"}
        ]
        if needed:
            options.append(needed)
    return min(options, key=len) if options else []


def conditional_evidence(context: Context, guide: Guide, *, risk: bool):
    values = evidence_values(context)
    possible = []
    for evidence in guide.evidence:
        if evidence.method == "REDUCE" or (risk and not evidence.risk_compatible):
            continue
        requirements = evidence_requirements(evidence)
        if evidence.method == "RECYCLE" and context.recycling_access == "NO":
            continue
        if any(values.get(key) not in {None, "UNKNOWN", expected} for key, expected in requirements.items()):
            continue
        unknown = {key: expected for key, expected in requirements.items() if values.get(key) in {None, "UNKNOWN"}}
        if (
            "former_contents" in unknown
            or "risk_type" in unknown
            or unknown.get("integrity") == "DAMAGED"
            or (risk and context.risk_type == "SHARP" and "integrity" in unknown)
        ):
            continue
        if unknown:
            possible.append((evidence, unknown))
    return possible


def conditional_text(method, label, unknown):
    descriptions = {
        ("cleanliness", "CLEAN"): "Check that it's clean. This route needs a clean item.",
        ("cleanliness", "DIRTY"): "If it's dirty, check what the dirt is before following the washing steps.",
        ("integrity", "INTACT"): "Check for cracks and damage. Use this route only if the item is intact.",
        ("reusable", "YES"): "Make sure the item still works for its purpose.",
        ("reusable", "NO"): "Make sure the item really can't be used anymore before choosing residual waste.",
        ("reuse_intention", True): "Go ahead with reuse only if you want to keep using it.",
        ("reuse_intention", False): "Choose residual waste only after you've decided not to reuse it.",
        ("cleanable", "YES"): "Make sure the dirt washes off normally, following the material's instructions.",
        (
            "soiling",
            "FOOD_DRINK",
        ): "Make sure the dirt is only ordinary food or drink residue. Don't use this rule for hazardous contamination.",
        (
            "soiling",
            "DUST",
        ): "Make sure the dirt is only ordinary dust. Don't use this rule for hazardous contamination.",
        (
            "recycling_access",
            "YES",
        ): "Confirm a collector accepts this type of item in this condition before you hand it over.",
        (
            "recycling_access",
            "NO",
        ): "Look for a suitable collection program. Choose residual waste only if no collector named in the guide accepts it.",
    }
    labels = {"REUSE": "Reuse", "RECYCLE": "Recycle", "TRASH": "Trash"}
    actions = {
        "REUSE": "Reuse it only after you've confirmed it's clean, intact, usable, and something you want to keep using.",
        "RECYCLE": "Check the material and cleanliness, and make sure a collector accepts it before you hand it over.",
        "TRASH": "Don't throw it away yet. First make sure it meets the guide's conditions for residual waste.",
    }
    reason = f"Provisional advice: {labels[method]} for this {in_sentence(label)}. Some information is still missing. {actions[method]}"
    steps = list(
        dict.fromkeys(
            descriptions.get((key, expected), "Check the unknown conditions before acting.")
            for key, expected in unknown.items()
        )
    )
    return reason, steps


def selection_error(raw, evidence, allowed, risk):
    if raw.special_handling != bool(risk) or raw.recommended_method not in allowed:
        return 'INVALID_MODEL_OUTPUT'
    by_id = {row.id: row for row in evidence}
    if len(set(raw.evidence_ids)) != len(raw.evidence_ids) or any(
        identifier not in by_id for identifier in raw.evidence_ids
    ):
        return 'INVALID_EVIDENCE'
    if {by_id[identifier].method for identifier in raw.evidence_ids} != {raw.recommended_method}:
        return 'INVALID_EVIDENCE'
    return None


def recommend(
    context: Context, provider: Provider, guides: list[Guide], *, demo_guides: bool = False, today: date | None = None
) -> Recommendation:
    item = confirm_context(context)
    today = today or date.today()
    guide = next((g for g in guides if g.object_key == context.object_key and g.category == context.category), None)
    prototype = demo_guides or provider.name == "fixture"
    base = {"provider": provider.name, "prototype_only": prototype}

    risk = (
        item["special_handling"]
        or context.risk_flag
        or context.risk_type != "UNKNOWN"
        or context.condition.soiling == "HAZARDOUS"
        or context.condition.former_contents == "CHEMICAL"
        or (
            (context.category == "GLASS" or context.object_key.startswith("ceramic_"))
            and context.condition.integrity == "DAMAGED"
        )
        or (guide is not None and guide.special_handling)
    )
    permitted_guide = guide is not None and (
        is_reviewed(guide, today) or (demo_guides and guide.publication_status == "DRAFT")
    )
    caution = guide.caution if permitted_guide else ""
    if risk:
        caution = " ".join(
            filter(
                None,
                [
                    "There is a special risk. Follow the packaging instructions and use a suitable handler. Never dump hazardous items.",
                    caution,
                ],
            )
        )

    def unavailable(code: str, *, missing=None, reason=None):
        guidance = handling_guidance(context, item, risk=bool(risk))
        urls = guidance["source_urls"]
        return Recommendation(
            status="NEEDS_CONTEXT" if missing else "UNAVAILABLE",
            reason=reason or guidance["reason"],
            steps=guidance["steps"],
            reason_code=code,
            special_handling=bool(risk),
            caution=caution,
            guide_refs=[{"guide_id": guide.id, "version": guide.version}] if permitted_guide else [],
            source_urls=urls,
            source_details=describe_sources(urls),
            missing_context=missing or [],
            **base,
        )

    if not permitted_guide:
        return unavailable("NO_REVIEWED_GUIDE")
    if context.recommendation_goal != "PREVENTION" and context.risk_flag and context.risk_type == "UNKNOWN":
        return unavailable(
            "RISK_DETAILS_REQUIRED",
            missing=["risk_type"],
            reason="You marked a special risk but didn't say which kind. Choose the risk type so ordinary dirt isn't treated as hazardous contamination.",
        )
    if (
        context.recommendation_goal != "PREVENTION"
        and context.risk_type == "ELECTRICAL"
        and context.category != "ELECTRONIC"
    ):
        return unavailable(
            "RISK_NOT_COVERED",
            reason="Electrical or battery risks need a specialist electronics route, so the regular material guide doesn't apply. Check the item type and follow the specialist advice below.",
        )
    evidence = eligible_evidence(context, guide, risk=bool(risk))
    conditional = []
    hazardous = (
        context.risk_type in {"CHEMICAL", "BIOLOGICAL", "OTHER"}
        or context.condition.soiling == "HAZARDOUS"
        or context.condition.former_contents == "CHEMICAL"
    )
    if not evidence and context.allow_conditional and not hazardous and context.recommendation_goal != "PREVENTION":
        conditional = conditional_evidence(context, guide, risk=bool(risk))
        evidence = [row[0] for row in conditional]
    allowed = sorted({e.method for e in evidence})
    if not allowed:
        if (
            context.risk_type in {"CHEMICAL", "BIOLOGICAL", "OTHER"}
            or context.condition.soiling == "HAZARDOUS"
            or context.condition.former_contents == "CHEMICAL"
        ):
            return unavailable(
                "RISK_NOT_COVERED",
                reason="This item's guide doesn't cover the contamination you reported. Don't follow the regular washing advice or put it in residual waste. Ask a suitable waste handler for advice.",
            )
        missing = missing_context(context, guide, risk=bool(risk))
        if missing:
            return unavailable(
                "CONTEXT_REQUIRED",
                missing=missing,
                reason="There isn't enough information to choose how to handle this item yet. Fill in what you know about its condition. We won't swap in Reduce advice instead.",
            )
        return unavailable("RISK_NOT_COVERED" if risk else "NO_ALLOWED_METHOD")
    payload = {
        "object_key": context.object_key,
        "category": context.category.value,
        "condition": context.condition.model_dump(),
        "reuse_intention": context.reuse_intention,
        "recommendation_goal": context.recommendation_goal,
        "recycling_access": context.recycling_access,
        "special_handling": bool(risk),
        "risk_type": context.risk_type,
        "allowed_methods": allowed,
        "guide": {
            "id": guide.id,
            "version": guide.version,
            "scope_note": guide.scope_note,
            "evidence": [e.model_dump() for e in evidence],
            "caution": guide.caution,
        },
    }
    if conditional:
        payload.update(
            advice_basis="CONDITIONAL", unknown_requirements={e.id: requirements for e, requirements in conditional}
        )
    failure_code = None

    def rules_choice():
        from .providers import RulesProvider

        base['provider'] = 'rules'
        return ModelSuggestion.model_validate_json(RulesProvider().generate(payload))

    try:
        raw = ModelSuggestion.model_validate_json(provider.generate(payload))
    except ProviderError as exc:
        failure_code = exc.code
        raw = rules_choice()
    except (ValidationError, ValueError, TypeError):
        if provider.name != 'gemini':
            return unavailable('INVALID_MODEL_OUTPUT')
        failure_code = 'INVALID_MODEL_OUTPUT'
        raw = rules_choice()
    error = selection_error(raw, evidence, allowed, risk)
    if error and provider.name == 'gemini':
        failure_code = error
        raw = rules_choice()
        error = selection_error(raw, evidence, allowed, risk)
    if error:
        return unavailable(error)
    by_id = {e.id: e for e in evidence}
    cited = [by_id[x] for x in raw.evidence_ids]
    # Render public text from validated evidence; free model prose is never a safety authority.
    reason = " ".join(e.text for e in cited)
    verify = {}
    if conditional:
        by_id_unknown = {e.id: requirements for e, requirements in conditional}
        for e in cited:
            for key, expected in by_id_unknown[e.id].items():
                if key in verify and verify[key] != expected:
                    return unavailable("INVALID_EVIDENCE")
                verify[key] = expected
        reason, verification_steps = conditional_text(raw.recommended_method, item["label"], verify)
    if len(reason) > 500:
        return unavailable("INVALID_EVIDENCE")
    source_urls = list(dict.fromkeys(url for e in cited for url in (e.source_urls or guide.source_urls)))
    steps = list(dict.fromkeys(step for e in cited for step in (e.steps or guide.steps)))
    if conditional:
        steps = [*verification_steps, "Follow these steps only after the conditions above are met.", *steps]
        caution = " ".join(
            filter(None, ["Conditional advice. Anything left blank is not treated as confirmed.", caution])
        )
    return Recommendation(
        status="READY",
        recommended_method=raw.recommended_method,
        recommendation_basis="CONDITIONAL" if conditional else "CONFIRMED",
        conditions_to_verify=[
            f"condition.{key}" if key in type(context.condition).model_fields else key for key in verify
        ],
        recommendation_scope="PREVENTION" if raw.recommended_method == "REDUCE" else "HANDLING",
        reason=reason,
        special_handling=bool(risk),
        caution=caution,
        evidence_ids=raw.evidence_ids,
        source_urls=source_urls,
        source_details=describe_sources(source_urls),
        steps=steps,
        reason_code=failure_code,
        guide_refs=[{"guide_id": guide.id, "version": guide.version}],
        **base,
    )
