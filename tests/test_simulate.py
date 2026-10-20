from datetime import datetime

import numpy as np
import pytest

from abkit.simulate import (
    ARMS,
    MARKETS,
    ExperimentConfig,
    analysis_frame,
    cell_counts,
    simulate,
    true_conversion_probabilities,
    with_defects,
)

SMALL = ExperimentConfig(n_users=20_000, seed=3)


@pytest.fixture(scope="module")
def clean():
    return simulate(SMALL)


def test_tables_are_consistent(clean):
    assert len(clean.users) == len(clean.assignments) == len(clean.exposures) == 20_000
    assert clean.users.user_id.is_unique
    assert set(clean.assignments.arm) == set(ARMS)
    assert set(clean.conversions.user_id) <= set(clean.users.user_id)
    merged = clean.exposures.merge(clean.assignments, on=["user_id", "arm"])
    assert len(merged) == len(clean.exposures)
    assert (merged.exposed_at > merged.assigned_at).all()
    assert (clean.users.signup_date <= datetime(2025, 9, 1)).all()


def test_allocation_and_markets_follow_the_config(clean):
    shares = clean.assignments.arm.value_counts(normalize=True)
    assert np.allclose(shares.sort_index(), 0.25, atol=0.015)
    market_share = clean.users.market.value_counts(normalize=True)
    assert market_share["US"] == pytest.approx(MARKETS["US"], abs=0.015)


def test_simulation_is_deterministic():
    a, b = simulate(SMALL), simulate(SMALL)
    assert a.conversions.equals(b.conversions)
    assert a.assignments.equals(b.assignments)


def test_truth_matches_the_designed_effects(clean):
    truth = true_conversion_probabilities(clean.users)
    users = clean.users.set_index("user_id")
    new = users.tenure == "new"
    logit = np.log(truth / (1 - truth))
    lift = logit.sub(logit["control"], axis=0)
    assert np.allclose(lift.loc[new, "pct10_discount"], 0.25)
    assert np.allclose(lift.loc[~new, "pct10_discount"], -0.12)
    far_mobile = users.market.isin(["AU", "NZ", "IE", "CA"]) & (users.device == "mobile")
    assert np.allclose(lift.loc[far_mobile, "free_shipping"], 0.45)
    assert truth.to_numpy().min() > 0.02 and truth.to_numpy().max() < 0.2


def test_observed_rates_track_the_truth(clean):
    frame = analysis_frame(clean)
    truth = clean.truth.loc[frame.user_id]
    expected = truth.to_numpy()[np.arange(len(frame)), [ARMS.index(a) for a in frame.arm]]
    for arm in ARMS:
        mask = (frame.arm == arm).to_numpy()
        assert frame.converted[mask].mean() == pytest.approx(expected[mask].mean(), abs=0.01)


def test_analysis_frame_is_intent_to_treat_and_drops_immature_users():
    early = simulate(with_defects(SMALL, analysis_at=datetime(2025, 9, 20)))
    frame = analysis_frame(early)
    assert frame.mature.all()
    assert len(frame) < len(early.users)
    everyone = analysis_frame(early, require_mature=False)
    assert len(everyone) == len(early.users)
    assert (everyone.revenue[everyone.converted == 0] == 0).all()


def test_defects_show_up_in_the_logs():
    broken = simulate(
        with_defects(
            SMALL,
            exposure_loss={"free_shipping": 0.1},
            contamination_share=0.05,
            duplicate_assignment_share=0.02,
            leaky_feature=True,
        )
    )
    first_exposures = broken.exposures.drop_duplicates("user_id")
    assigned = broken.assignments.drop_duplicates("user_id")
    shipping = assigned[assigned.arm == "free_shipping"].user_id
    assert (~shipping.isin(first_exposures.user_id)).mean() == pytest.approx(0.1, abs=0.02)
    assert broken.exposures.user_id.duplicated().mean() == pytest.approx(0.05, abs=0.01)
    assert broken.assignments.user_id.duplicated().sum() > 0
    assert "checkout_visits" in broken.users


def test_cell_counts_cover_every_segment(clean):
    counts = cell_counts(analysis_frame(clean))
    assert counts.users.sum() == 20_000
    assert set(counts.columns) == {"market", "device", "tenure", "arm", "users", "conversions"}
    assert (counts.conversions <= counts.users).all()


def test_config_validation():
    with pytest.raises(ValueError):
        ExperimentConfig(allocation=(0.5, 0.5))
    with pytest.raises(ValueError):
        ExperimentConfig(arms=("bundle", "control"), allocation=(0.5, 0.5))
    with pytest.raises(ValueError):
        ExperimentConfig(exposure_loss={"mystery": 0.1})


def test_an_early_data_pull_contains_nothing_from_after_it():
    cutoff = datetime(2025, 9, 18)
    early = simulate(with_defects(SMALL, analysis_at=cutoff))
    assert (early.assignments.assigned_at <= cutoff).all()
    assert (early.exposures.exposed_at <= cutoff).all()
    assert (early.conversions.converted_at <= cutoff).all()
    assert set(early.users.user_id) == set(early.assignments.user_id)
    assert len(early.users) < SMALL.n_users
