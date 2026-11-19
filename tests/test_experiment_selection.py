import copy
from dataclasses import asdict
from pathlib import Path

import pytest


@pytest.fixture
def manager(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    import manage_tuning_experiments

    return manage_tuning_experiments


def report(manager, candidate="candidate", gain=0.1):
    labels = list(manager.FOCUS["broad"]) + ["bottle"]
    old = {label: {"support": 100, "precision": 0.5, "recall": 0.5, "f1": 0.5} for label in labels}
    new = copy.deepcopy(old)
    for label in manager.FOCUS["broad"]:
        new[label].update(precision=0.5 + gain, recall=0.5 + gain, f1=0.5 + gain)
    comparison = {
        name: {
            "fixed_validation": {"protocol": {"split": "val", "confidence": 0.5}, "per_class": rows, "confusion": []},
            "validation_ap": {"split": "val"},
        }
        for name, rows in (("baseline", old), ("candidate", new))
    }
    return {
        "branch": "broad",
        "stage": "B1",
        "status": "COMPLETED_VALIDATION_ONLY",
        "evaluation_split": "val",
        "holdout_opened": False,
        "comparison_eligible": True,
        "input_sha256": {"manifest": "frozen", "initial_model": "same-initial"},
        "config": asdict(manager.TrainingConfig()),
        "candidate_sha256": candidate,
        "comparison": comparison,
    }


def test_one_winner_still_requires_seed_validation(manager):
    a, b = report(manager, "one", 0.1), report(manager, "two", 0.15)
    b["stage"], b["config"]["freeze"] = "B2", 0
    result = manager.choose([a, b])
    assert result["selected_candidate_sha256"] == "two" and result["next_b3_freeze"] == 0
    assert result["source_stages"] == ["B1", "B2"]
    assert result["status"] == "STABILITY_PENDING" and not result["promoted"]
    assert result["required_seed_checks"] == [17, 42, 73]


@pytest.mark.parametrize(
    "change",
    [
        lambda r: r.update(holdout_opened=True),
        lambda r: r.update(evaluation_split="test"),
        lambda r: r["comparison"]["candidate"]["validation_ap"].update(split="test"),
        lambda r: r["comparison"]["candidate"]["fixed_validation"]["per_class"]["cardboard"].update(f1=float("nan")),
    ],
)
def test_test_or_nonfinite_metrics_cannot_select(manager, change):
    r = report(manager)
    change(r)
    with pytest.raises(ValueError):
        manager.choose([r])


@pytest.mark.parametrize(
    "change",
    [
        lambda r: r["input_sha256"].update(initial_model="other"),
        lambda r: r["input_sha256"].update(manifest="other"),
        lambda r: r["config"].update(seed=17),
        lambda r: r["config"].update(batch=8),
    ],
)
def test_incomparable_runs_cannot_be_mixed(manager, change):
    a, b = report(manager), report(manager, "two")
    change(b)
    with pytest.raises(ValueError, match="differ"):
        manager.choose([a, b])


@pytest.mark.parametrize(
    "change",
    [
        lambda r: r.update(comparison_eligible=False),
        lambda r: r["comparison"]["candidate"]["fixed_validation"]["per_class"]["bottle"].update(f1=0.45),
        lambda r: r["comparison"]["candidate"]["fixed_validation"]["per_class"]["cardboard"].update(recall=0.45),
    ],
)
def test_regressions_keep_baseline(manager, change):
    r = report(manager)
    change(r)
    result = manager.choose([r])
    assert result["status"] == "KEEP_BASELINE" and result["selected_candidate_sha256"] is None


def test_no_improvement_keeps_baseline(manager):
    assert manager.choose([report(manager, gain=0)])["status"] == "KEEP_BASELINE"


def test_stability_requires_all_three_seeds_and_matching_configuration(manager):
    primary = report(manager)
    primary["training"] = {"status": "TRAINED"}
    selection = manager.choose([primary])
    runs = []
    for seed in (17, 42, 73):
        value = copy.deepcopy(primary)
        value["config"]["seed"] = seed
        runs.append(value)
    result = manager.stability(selection, runs)
    assert result["status"] == "VALIDATION_STABLE_PENDING_RESOLUTION_REVIEW"
    assert not result["promoted"] and not result["holdout_opened"]
    with pytest.raises(ValueError, match="exactly"):
        manager.stability(selection, runs[:2])
    runs[0]["config"]["batch"] = 8
    with pytest.raises(ValueError, match="differs"):
        manager.stability(selection, runs)
