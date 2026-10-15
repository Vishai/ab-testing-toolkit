"""End-to-end checks on the bundled datasets."""

import json

import pytest

from abkit import SCENARIOS, Settings, analyze
from abkit.pipeline import to_jsonable

FAST = Settings(peeking_sims=2000, uplift_permutations=50)


@pytest.fixture(scope="module")
def results():
    return {k: analyze(k, settings=FAST) for k in SCENARIOS}


def test_all_scenarios_run_and_serialise(results):
    for r in results.values():
        json.dumps(to_jsonable(r))  # must be JSON-safe
        assert r["n_sessions"] == 1200


def test_known_headline_numbers(results):
    p = results["travel_booking"]["primary"]
    assert (p["x_a"], p["n_a"], p["x_b"], p["n_b"]) == (32, 608, 59, 592)
    assert p["p_value"] == pytest.approx(0.0021, abs=1e-4)


def test_free_trial_is_held_for_srm(results):
    r = results["free_trial"]
    assert r["quality"]["srm"]["status"] == "warn"
    assert r["decision"]["verdict"] == "hold"


def test_clean_scenarios_ship(results):
    for k in ("online_education", "travel_booking", "charity_donations"):
        assert results[k]["decision"]["verdict"] == "ship", k


def test_charity_flags_inconclusive_revenue(results):
    caveats = " ".join(results["charity_donations"]["decision"]["caveats"])
    assert "revenue-per-visitor gain is not conclusive" in caveats


def test_uplift_uses_only_pre_treatment_features(results):
    for r in results.values():
        assert set(r["uplift_model"]["features"]) <= {"Device", "UserType"}
