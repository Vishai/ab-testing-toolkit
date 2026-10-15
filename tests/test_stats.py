"""Unit tests: each statistical routine is checked against an independent
reference (statsmodels / scipy / closed-form) or a property it must satisfy."""

import numpy as np
import pandas as pd
import pytest
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize, proportions_ztest

from abkit import bayes, power, quality, stats


def test_two_proportions_matches_statsmodels():
    r = stats.two_proportions(43, 601, 69, 599)
    z, p = proportions_ztest([69, 43], [599, 601])
    assert r["z"] == pytest.approx(z, rel=1e-9)
    assert r["p_value"] == pytest.approx(p, rel=1e-9)
    assert r["uplift"] == pytest.approx((69 / 599 - 43 / 601) / (43 / 601))


def test_ci_contains_estimate_and_excludes_zero_when_significant():
    r = stats.two_proportions(32, 608, 59, 592)
    lo, hi = r["abs_ci"]
    assert lo < r["abs_diff"] < hi
    assert r["significant"] and lo > 0
    rlo, rhi = r["uplift_ci"]
    assert rlo < r["uplift"] < rhi


def test_no_difference_gives_large_p():
    r = stats.two_proportions(50, 1000, 50, 1000)
    assert r["abs_diff"] == 0
    assert r["p_value"] == pytest.approx(1.0)


def test_srm_detects_imbalance():
    assert quality.srm_test(600, 600)["status"] == "pass"
    assert quality.srm_test(639, 561)["status"] == "warn"   # p ~ 0.024
    assert quality.srm_test(700, 500)["status"] == "fail"   # p < 0.001


def test_sample_size_matches_statsmodels():
    base, lift = 0.05, 0.20
    ours = power.sample_size_per_arm(base, lift)
    h = proportion_effectsize(base * (1 + lift), base)
    ref = NormalIndPower().solve_power(effect_size=h, alpha=0.05, power=0.8, ratio=1)
    # statsmodels also counts the (negligible) opposite tail, so allow +-1 session
    assert abs(ours - int(np.ceil(ref))) <= 1


def test_power_matches_statsmodels():
    h = proportion_effectsize(0.06, 0.05)
    ref = NormalIndPower().power(effect_size=abs(h), nobs1=8000, alpha=0.05)
    assert power.power_for(0.05, 0.06, 8000) == pytest.approx(ref, rel=1e-6)


def test_mde_round_trips_with_sample_size():
    base = 0.07
    n = power.sample_size_per_arm(base, 0.30)
    mde = power.mde_for_n(base, n)["mde_rel"]
    assert mde == pytest.approx(0.30, abs=0.005)


def test_peeking_inflates_and_correction_restores_error():
    sim = power.peeking_simulation(0.05, 2000, looks=10, sims=8000)
    assert sim["fixed_fpr"] == pytest.approx(0.05, abs=0.01)
    assert sim["naive_fpr"] > 0.12
    assert sim["corrected_fpr"] == pytest.approx(0.05, abs=0.005)
    assert sim["corrected_nominal_alpha"] < 0.05


def test_winners_curse_exaggerates_when_underpowered():
    wc = power.winners_curse(0.05, 0.20, 600, sims=20_000)
    assert wc["power"] < 0.3
    assert wc["exaggeration_ratio"] > 2


def test_bayes_symmetric_and_directional():
    same = bayes.beta_binomial(50, 1000, 50, 1000, draws=100_000)
    assert same["prob_b_better"] == pytest.approx(0.5, abs=0.01)
    better = bayes.beta_binomial(50, 1000, 90, 1000)
    assert better["prob_b_better"] > 0.99
    assert better["expected_loss_b"] < better["expected_loss_a"]


def _df(a, b):
    return pd.DataFrame({"group": ["A"] * len(a) + ["B"] * len(b), "x": list(a) + list(b)})


def test_non_inferiority_guardrail():
    rng = np.random.default_rng(0)
    ok = _df(rng.binomial(1, 0.30, 4000), rng.binomial(1, 0.28, 4000)).rename(columns={"x": "Bounce"})
    bad = _df(rng.binomial(1, 0.30, 4000), rng.binomial(1, 0.36, 4000)).rename(columns={"x": "Bounce"})
    assert stats.non_inferiority_proportion(ok, "Bounce", 0.02)["passed"]
    assert not stats.non_inferiority_proportion(bad, "Bounce", 0.02)["passed"]


def test_bootstrap_ci_brackets_difference():
    rng = np.random.default_rng(1)
    df = _df(rng.exponential(10, 500), rng.exponential(12, 500)).rename(columns={"x": "Revenue"})
    r = stats.bootstrap_mean_diff(df, "Revenue", n_boot=2000)
    assert r["abs_ci"][0] < r["abs_diff"] < r["abs_ci"][1]
