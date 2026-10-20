import numpy as np
import pandas as pd
import pytest

from abkit.launch_plan import aggregate_readout
from abkit.offer_uplift import (
    cross_fitted_predictions,
    evaluate_policies,
    ipw_value,
    policy_from_predictions,
    run_offer_uplift,
)
from abkit.simulate import (
    ARMS,
    SEGMENT_COLUMNS,
    ExperimentConfig,
    analysis_frame,
    simulate,
)


@pytest.fixture(scope="module")
def experiment():
    return simulate(ExperimentConfig(seed=7))


@pytest.fixture(scope="module")
def result(experiment):
    frame = analysis_frame(experiment)
    propensity = dict(zip(experiment.config.arms, experiment.config.allocation, strict=True))
    result = run_offer_uplift(frame, SEGMENT_COLUMNS, ARMS, propensity, truth=experiment.truth,
                              n_boot=300, n_perm=60)
    return result | {"frame": frame}


def test_targeting_finds_lift_the_aggregate_test_misses(result):
    policies = {row["policy"]: row for row in result["policies"]}
    targeted = policies["targeted"]
    assert targeted["rel_lift_ci"][0] > 0
    assert targeted["prob_beats_baseline"] > 0.99
    for arm in ARMS[1:]:
        single = policies[f"all_{arm}"]
        assert targeted["rel_lift_ci"][0] > single["rel_lift"]
    readout = aggregate_readout(result["frame"], ARMS)
    assert not any(row["significant"] for row in readout)


def test_targeting_is_close_to_the_oracle_in_truth(result):
    truth = result["truth"]
    values = truth["policy_conversion"]
    assert values["targeted"] > max(values[f"all_{arm}"] for arm in ARMS)
    assert truth["share_of_oracle_gain"] > 0.6
    assert all(c > 0.4 for c in truth["uplift_correlation"].values())


def test_each_offer_has_targeting_signal(result):
    significant = [row["offer"] for row in result["qini"] if row["perm_p_value"] < 0.05]
    assert {"pct10_discount", "bundle"} <= set(significant)


def test_responders_match_the_designed_effects(result):
    table = pd.DataFrame(result["responders"]).set_index(["market", "device", "tenure"])
    assert table.loc[("AU", "mobile", "returning"), "best_offer"] == "free_shipping"
    assert table.loc[("US", "desktop", "returning"), "best_offer"] == "bundle"
    new_users = table.xs("new", level="tenure")
    assert (new_users["uplift_pct10_discount"] > 0).mean() > 0.8


def test_predictions_are_out_of_fold_probabilities(experiment):
    frame = analysis_frame(experiment).sample(8_000, random_state=1).reset_index(drop=True)
    predictions = cross_fitted_predictions(frame, SEGMENT_COLUMNS, ARMS, folds=3)
    assert predictions.shape == (8_000, 4)
    assert np.all((predictions > 0) & (predictions < 1))


def test_policy_respects_min_gain():
    predictions = np.array([[0.10, 0.11, 0.09], [0.10, 0.20, 0.12], [0.10, 0.05, 0.06]])
    arms = ["control", "a", "b"]
    assert list(policy_from_predictions(predictions, arms)) == ["a", "a", "control"]
    assert list(policy_from_predictions(predictions, arms, min_gain=0.05)) == ["control", "a",
                                                                                "control"]


def test_ipw_matches_arm_means_for_fixed_policies():
    rng = np.random.default_rng(0)
    assigned = rng.choice(["control", "a"], 10_000, p=[0.7, 0.3])
    converted = (rng.random(10_000) < np.where(assigned == "a", 0.2, 0.1)).astype(int)
    propensity = {"control": 0.7, "a": 0.3}
    everyone_a = np.full(10_000, "a")
    assert ipw_value(converted, assigned, everyone_a, propensity) == pytest.approx(0.2, abs=0.02)
    frame = pd.DataFrame({"arm": assigned, "converted": converted})
    table = evaluate_policies(frame, {"all_control": np.full(10_000, "control"),
                                      "all_a": everyone_a}, propensity, n_boot=200)
    row = table.set_index("policy").loc["all_a"]
    assert row["rel_lift"] == pytest.approx(1.0, abs=0.25)
    assert row["rel_lift_ci"][0] > 0
