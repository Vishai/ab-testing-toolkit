from datetime import datetime

import pandas as pd
import pytest

from abkit import integrity
from abkit.simulate import ExperimentConfig, simulate, with_defects

BASE = ExperimentConfig(n_users=40_000, seed=13)


def statuses(config: ExperimentConfig) -> dict[str, str]:
    report = integrity.run_integrity_checks(simulate(config))
    return {check["name"]: check["status"] for check in report["checks"]}


def failing(config: ExperimentConfig) -> set[str]:
    return {name for name, status in statuses(config).items() if status != "pass"}


def test_clean_experiment_passes_everything():
    report = integrity.run_integrity_checks(simulate(BASE))
    assert report["overall"] == "pass"
    assert len(report["checks"]) == 8


def test_arm_specific_exposure_loss_is_caught_at_exposure_not_assignment():
    result = statuses(with_defects(BASE, exposure_loss={"free_shipping": 0.05}))
    assert result["Sample ratio mismatch (assignment)"] == "pass"
    assert result["Sample ratio mismatch (exposure)"] == "fail"
    assert result["Missing exposures"] == "fail"


def test_uniform_exposure_loss_warns_without_being_differential():
    loss = {arm: 0.03 for arm in BASE.arms}
    report = integrity.run_integrity_checks(simulate(with_defects(BASE, exposure_loss=loss)))
    missing = report["details"]["missing_exposures"]
    assert missing["status"] == "warn"
    assert missing["differential_p"] > 0.001
    assert missing["missing_share"] == pytest.approx(0.03, abs=0.005)


def test_contamination_is_measured():
    report = integrity.run_integrity_checks(simulate(with_defects(BASE, contamination_share=0.03)))
    detail = report["details"]["contamination"]
    assert detail["status"] == "fail"
    assert detail["share"] == pytest.approx(0.03, abs=0.005)
    assert failing(with_defects(BASE, contamination_share=0.005)) == {"Contamination"}


def test_assignment_conflicts():
    assert failing(with_defects(BASE, duplicate_assignment_share=0.01)) == {"Assignment conflicts"}


def test_leaky_feature_is_flagged_and_excluded():
    report = integrity.run_integrity_checks(simulate(with_defects(BASE, leaky_feature=True)))
    leakage = report["details"]["leakage"]
    assert leakage["leaky_features"] == ["checkout_visits"]
    assert set(leakage["safe_features"]) == {"market", "device", "tenure"}
    row = next(r for r in leakage["features"] if r["feature"] == "checkout_visits")
    assert any("after assignment" in reason for reason in row["reasons"])


def test_declared_pre_treatment_feature_that_leaks_fails():
    experiment = simulate(with_defects(BASE, leaky_feature=True))
    first = experiment.assignments.drop_duplicates("user_id")
    converted = pd.Series(1, index=experiment.conversions.user_id.unique())
    result = integrity.leakage(first, experiment.users, converted,
                               pre_treatment=("market", "checkout_visits"))
    assert result["status"] == "fail"


def test_data_pulled_too_early():
    early = with_defects(BASE, analysis_at=datetime(2025, 9, 25))
    report = integrity.run_integrity_checks(simulate(early))
    delay = report["details"]["delayed_conversions"]
    assert delay["status"] == "warn"
    assert 0.3 < delay["immature_share"] < 0.7
    assert 0 < delay["expected_share_visible_for_immature"] < 1
    assert failing(early) == {"Delayed conversions"}


def test_conversion_window_shorter_than_the_lag_warns():
    short = with_defects(BASE, conversion_window_days=3, mean_conversion_delay_days=4.0)
    delay = integrity.run_integrity_checks(simulate(short))["details"]["delayed_conversions"]
    assert delay["p99_lag_days"] > 3
    assert delay["status"] == "warn"


def test_srm_helper_matches_scipy_and_names_the_worst_arm():
    result = integrity.srm(pd.Series({"control": 5_000, "variation": 4_700}),
                           {"control": 0.5, "variation": 0.5})
    assert result["status"] == "warn"
    assert result["p_value"] == pytest.approx(0.0023, abs=0.0005)
    assert result["most_off_arm"] in {"control", "variation"}


def test_lag_statistics_use_only_users_whose_window_closed():
    cutoff = datetime(2025, 9, 18)
    experiment = simulate(with_defects(BASE, analysis_at=cutoff))
    delay = integrity.run_integrity_checks(experiment)["details"]["delayed_conversions"]
    first = experiment.assignments.drop_duplicates("user_id").set_index("user_id")
    mature = first[first.assigned_at + pd.Timedelta(days=14) <= cutoff]
    conv = experiment.conversions.groupby("user_id").converted_at.min()
    joined = mature.join(conv, how="inner")
    lags = (joined.converted_at - joined.assigned_at).dt.total_seconds() / 86_400
    assert delay["median_lag_days"] == pytest.approx(lags.median())
