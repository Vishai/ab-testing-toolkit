import numpy as np
import pandas as pd
import pytest

from abkit import data
from abkit import hierarchical as h
from abkit.simulate import ExperimentConfig, analysis_frame, simulate

FACTORS = ("device", "tenure")
ARMS = ("control", "pct10_discount", "free_shipping", "bundle")
FAST = h.SamplerSettings(draws=400, tune=600, chains=2, seed=5)


@pytest.fixture(scope="module")
def experiment():
    return simulate(ExperimentConfig(n_users=40_000, seed=21))


@pytest.fixture(scope="module")
def fitted(experiment):
    frame = analysis_frame(experiment)
    cells = h.cells_from_frame(frame, FACTORS)
    return h.fit(cells, FACTORS, ARMS, sampler=FAST, prior_samples=200), frame


def test_fit_shapes_and_cell_table(fitted):
    fit, frame = fitted
    assert fit.rate_draws.shape == (800, 6, 4)
    assert fit.prior_rate_draws.shape[1:] == (6, 4)
    assert fit.cells[[f"users_{arm}" for arm in ARMS]].to_numpy().sum() == len(frame)
    assert np.all((fit.rate_draws > 0) & (fit.rate_draws < 1))


def test_sampler_converges(fitted):
    diagnostics = h.diagnostics(fitted[0])
    assert diagnostics["max_rhat"] < 1.02
    assert diagnostics["divergences"] <= 2
    assert diagnostics["min_ess_bulk"] > 100


def test_prior_and_posterior_predictive_checks(fitted):
    prior = h.prior_predictive_check(fitted[0])
    assert prior["plausible"]
    assert prior["control_rate_90"][0] < 0.07 < prior["control_rate_90"][1]
    ppc = h.posterior_predictive_check(fitted[0])
    assert ppc["passed"]
    assert ppc["cell_coverage"] >= 0.85
    assert all(0.02 < p < 0.98 for p in ppc["arm_total_p"].values())


def test_segment_effects_recover_the_truth(fitted, experiment):
    fit, frame = fitted
    table = h.segment_table(fit, by=["tenure"]).set_index(["tenure", "arm"])
    assert table.loc[("new", "pct10_discount"), "verdict"] == "positive"
    assert table.loc[("returning", "pct10_discount"), "rel_lift_mean"] < 0
    truth = experiment.truth.loc[frame.user_id].reset_index(drop=True)
    joined = frame[list(FACTORS)].join(truth).groupby(list(FACTORS)).mean().reset_index()
    merged = fit.cells.merge(joined, on=list(FACTORS))
    # One control cell in this seed came out low by chance, which makes every offer
    # look strong there in the raw data. The pooled estimate should still beat the
    # raw one for every offer and stay reasonably close to the truth.
    errors = []
    for arm in ARMS[1:]:
        true_lift = (merged[arm] / merged["control"] - 1).to_numpy()
        pooled = np.sqrt(np.mean((h.posterior_cell_lifts(fit, arm) - true_lift) ** 2))
        raw = np.sqrt(np.mean((h.raw_cell_lifts(fit, arm) - true_lift) ** 2))
        assert pooled < raw
        errors.append(pooled)
    assert np.mean(errors) < 0.15


def test_partial_pooling_beats_raw_estimates_in_small_cells(experiment):
    frame = analysis_frame(experiment)
    small = frame[frame.market.isin(["IE", "NZ", "SE", "ES", "IT", "NL"])]
    cells = h.cells_from_frame(small, ("market", "device", "tenure"))
    fit = h.fit(cells, ("market", "device", "tenure"), ARMS, sampler=FAST, prior_samples=50)
    truth = experiment.truth.loc[small.user_id].reset_index(drop=True)
    joined = (
        small[["market", "device", "tenure"]].reset_index(drop=True).join(truth)
        .groupby(["market", "device", "tenure"]).mean().reset_index()
    )
    merged = fit.cells.merge(joined, on=["market", "device", "tenure"])
    raw_errors, pooled_errors = [], []
    for arm in ARMS[1:]:
        true_lift = (merged[arm] / merged["control"] - 1).to_numpy()
        raw = h.raw_cell_lifts(fit, arm)
        ok = np.isfinite(raw)
        raw_errors.append(np.sqrt(np.mean((raw[ok] - true_lift[ok]) ** 2)))
        pooled_errors.append(np.sqrt(np.mean((h.posterior_cell_lifts(fit, arm) - true_lift) ** 2)))
    assert np.mean(pooled_errors) < 0.5 * np.mean(raw_errors)


def test_summaries_are_consistent(fitted):
    fit, _ = fitted
    summary = h.summarize_lift(fit, "free_shipping")
    low, high = summary["rel_lift_ci"]
    assert low < summary["rel_lift_median"] < high
    assert 0.0 <= summary["prob_in_rope"] <= 1.0
    assert summary["control_rate"] == pytest.approx(0.069, abs=0.01)
    mobile = (fit.cells.device == "mobile").to_numpy()
    control, treated, relative = h.lift_draws(fit, "free_shipping", mobile)
    assert relative.shape == (800,)
    assert np.allclose(relative, treated / control - 1)
    with pytest.raises(ValueError):
        h.lift_draws(fit, "bundle", np.zeros(len(fit.cells), dtype=bool))


@pytest.mark.parametrize(
    ("center", "spread", "expected"),
    [(0.10, 0.01, "positive"), (-0.10, 0.01, "negative"), (0.0, 0.002, "no_effect"),
     (0.02, 0.05, "uncertain")],
)
def test_rope_decision_rule(center, spread, expected):
    draws = np.random.default_rng(0).normal(center, spread, 20_000)
    assert h.judge(draws, rope=0.02) == expected


def test_session_adapter_and_real_dataset():
    sessions = data.load("online_education")
    cells = h.cells_from_sessions(sessions, ["Device", "UserType"])
    assert set(cells.arm) == {"control", "variation"}
    assert cells.users.sum() == len(sessions)
    assert cells.conversions.sum() == sessions.Conversion.sum()
    fit = h.fit(cells, ["Device", "UserType"], ["control", "variation"], sampler=FAST,
                prior_samples=50)
    summary = h.summarize_lift(fit, "variation")
    raw = sessions.groupby("group").Conversion.mean()
    assert 0 < summary["rel_lift_mean"] < raw["B"] / raw["A"] - 1


def test_prior_sensitivity_reports_every_prior():
    cells = pd.DataFrame(
        {"seg": ["a", "a", "b", "b"], "arm": ["control", "variation"] * 2,
         "users": [800, 800, 300, 300], "conversions": [56, 72, 20, 27]}
    )
    table = h.prior_sensitivity(cells, ["seg"], ["control", "variation"],
                                sampler=h.SamplerSettings(draws=200, tune=300, chains=2))
    assert list(table.prior) == ["default", "skeptical", "wide"]
    skeptical, wide = table.set_index("prior").loc[["skeptical", "wide"], "rel_lift_mean"]
    assert abs(skeptical) < abs(wide)


def test_diagnostics_cover_every_sampled_parameter(fitted):
    fit, _ = fitted
    checked = set(h.diagnostics(fit)["parameters_checked"])
    sampled = {v for v in fit.idata.posterior.data_vars if v not in {"rate", "baseline", "effect"}}
    assert checked == sampled
    assert {"effect_mean", "sd_effect_device", "z_effect_device_x_tenure"} <= checked


def test_finer_cells_are_summed_not_averaged(experiment):
    frame = analysis_frame(experiment)
    fine = h.cells_from_frame(frame, ("market", "device", "tenure"))
    fit = h.fit(fine, ("tenure",), ARMS, sampler=h.SamplerSettings(draws=100, tune=200, chains=1),
                prior_samples=20)
    users = fit.cells[[f"users_{arm}" for arm in ARMS]].to_numpy().sum()
    assert users == len(frame)
