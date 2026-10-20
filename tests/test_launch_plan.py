import numpy as np
import pandas as pd
import pytest
from scipy import stats

from abkit.launch_plan import (
    aggregate_readout,
    bayesian_assurance,
    detectable_lift,
    dunnett_critical_value,
    plan_launch,
    sample_size,
    simulated_familywise_error,
)
from abkit.power import sample_size_per_arm
from abkit.simulate import ARMS, MARKETS, ExperimentConfig


def test_dunnett_sits_between_unadjusted_and_bonferroni():
    z = dunnett_critical_value(3)
    assert z == pytest.approx(2.349, abs=0.002)
    assert stats.norm.ppf(0.975) < z < stats.norm.ppf(1 - 0.05 / 6)
    assert dunnett_critical_value(1) == pytest.approx(stats.norm.ppf(0.975))


def test_sample_size_matches_the_existing_two_arm_formula():
    z = stats.norm.ppf(0.975)
    assert sample_size(0.069, 0.10, z) == sample_size_per_arm(0.069, 0.10)
    n = sample_size(0.05, 0.2, z)
    assert detectable_lift(0.05, n, z) == pytest.approx(0.2, abs=0.002)
    with pytest.raises(ValueError):
        sample_size(0.9, 0.2, z)


def test_plan_drives_the_simulated_experiment():
    plan = plan_launch(0.069, ARMS[1:])
    config = ExperimentConfig()
    assert plan["n_per_arm"] == 28_723
    assert plan["planned_users"] == config.n_users == 117_000
    assert plan["runtime_days"] == config.enrollment_days == 22


def test_the_plan_holds_up_in_simulation():
    plan = plan_launch(0.069, ARMS[1:])
    assert all(0.77 < p < 0.83 for p in plan["simulated_power_at_mde"].values())
    assert plan["simulated_familywise_error"] == pytest.approx(0.05, abs=0.006)
    assert 0.7 < plan["bayesian_assurance"] < 0.9


def test_small_segments_cannot_detect_much_alone():
    plan = plan_launch(0.069, ARMS[1:], segment_shares={"US": MARKETS["US"],
                                                         "NZ": MARKETS["NZ"]})
    lifts = {row["segment"]: row["detectable_lift"] for row in plan["segments"]}
    assert lifts["US"] < 0.2
    assert lifts["NZ"] > 0.5


def test_familywise_error_inflates_without_correction():
    unadjusted = simulated_familywise_error(0.069, 5_000, 3, stats.norm.ppf(0.975))
    adjusted = simulated_familywise_error(0.069, 5_000, 3, dunnett_critical_value(3))
    assert unadjusted > 0.11
    assert adjusted == pytest.approx(0.05, abs=0.006)


def test_assurance_rises_with_sample_size_and_effect():
    small = bayesian_assurance(0.069, 2_000, 0.10, 0.05)
    big = bayesian_assurance(0.069, 30_000, 0.10, 0.05)
    null = bayesian_assurance(0.069, 30_000, 0.0, 0.001)
    assert small < big
    assert null < 0.08


def test_aggregate_readout_adjusts_for_several_offers():
    rng = np.random.default_rng(0)
    arms = rng.choice(["control", "a", "b", "c"], 40_000)
    rate = np.where(arms == "a", 0.08, 0.06)
    frame = pd.DataFrame({"arm": arms, "converted": (rng.random(40_000) < rate).astype(int)})
    rows = {row["offer"]: row for row in aggregate_readout(frame, ("control", "a", "b", "c"))}
    assert rows["a"]["significant"]
    assert rows["a"]["p_dunnett"] >= rows["a"]["p_value"]
    assert not rows["b"]["significant"] and not rows["c"]["significant"]
    assert rows["b"]["p_dunnett"] > rows["b"]["p_value"]
