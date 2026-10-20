"""Event-level simulator for a multi-offer experiment with known true effects.

The four bundled datasets are one row per session with no user ids or
timestamps, so they can't exercise the checks a real experimentation platform
needs (missing exposures, contamination, delayed conversions). This module
produces the raw tables such a platform would log:

  users        user_id, market, device, tenure (all fixed before assignment)
  assignments  user_id, arm, assigned_at
  exposures    user_id, arm, exposed_at   (the user actually saw the offer)
  conversions  user_id, converted_at, revenue

Every number is simulated, and the true conversion probability of every user
under every arm is known, so estimates can be checked against the truth.

The default world is built to show a common failure of aggregate A/B testing:
each offer works well for one kind of customer and slightly hurts others, so
the average effect is small, while the targeted effect is large.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

ARMS = ("control", "pct10_discount", "free_shipping", "bundle")

# Traffic share per market. The last few are deliberately small, so segment
# estimates there are noisy and benefit from partial pooling.
MARKETS = {
    "US": 0.30, "UK": 0.14, "DE": 0.12, "FR": 0.09, "CA": 0.08, "AU": 0.06,
    "NL": 0.05, "SE": 0.04, "ES": 0.04, "IT": 0.04, "IE": 0.02, "NZ": 0.02,
}
FAR_MARKETS = frozenset({"AU", "NZ", "IE", "CA"})  # shipping is slow and expensive
DEVICES = {"mobile": 0.58, "desktop": 0.34, "tablet": 0.08}
TENURES = {"new": 0.45, "returning": 0.55}
SEGMENT_COLUMNS = ("market", "device", "tenure")

AVERAGE_ORDER_VALUE = 62.0
OFFER_COST_SHARE = {"control": 0.0, "pct10_discount": 0.10, "free_shipping": 0.07, "bundle": 0.04}


@dataclass(frozen=True)
class ExperimentConfig:
    n_users: int = 117_000
    arms: tuple[str, ...] = ARMS
    allocation: tuple[float, ...] = (0.25, 0.25, 0.25, 0.25)
    start: datetime = datetime(2025, 9, 1)
    enrollment_days: int = 22
    conversion_window_days: int = 14
    mean_conversion_delay_days: float = 2.5
    analysis_at: datetime | None = None
    seed: int = 7
    # Defects that can be switched on to prove the trust checks catch them.
    exposure_loss: dict[str, float] = field(default_factory=dict)
    contamination_share: float = 0.0
    duplicate_assignment_share: float = 0.0
    leaky_feature: bool = False

    def __post_init__(self) -> None:
        if len(self.arms) != len(self.allocation):
            raise ValueError("arms and allocation must have the same length")
        if not np.isclose(sum(self.allocation), 1.0):
            raise ValueError("allocation must add up to 1")
        if self.n_users <= 0 or self.enrollment_days <= 0:
            raise ValueError("n_users and enrollment_days must be positive")
        if self.arms[0] != "control":
            raise ValueError("the first arm must be control")
        unknown = set(self.exposure_loss) - set(self.arms)
        if unknown:
            raise ValueError(f"exposure_loss names unknown arms: {sorted(unknown)}")

    @property
    def end_of_enrollment(self) -> datetime:
        return self.start + timedelta(days=self.enrollment_days)

    @property
    def cutoff(self) -> datetime:
        """When data is pulled. Defaults to after every user's conversion window closed."""
        return self.analysis_at or self.end_of_enrollment + timedelta(
            days=self.conversion_window_days
        )


@dataclass(frozen=True)
class Experiment:
    config: ExperimentConfig
    users: pd.DataFrame
    assignments: pd.DataFrame
    exposures: pd.DataFrame
    conversions: pd.DataFrame
    truth: pd.DataFrame  # true conversion probability of every user under every arm


def baseline_logit(users: pd.DataFrame) -> np.ndarray:
    market_effect = {
        "US": 0.05, "UK": 0.00, "DE": -0.10, "FR": -0.05, "CA": 0.10, "AU": -0.05,
        "NL": 0.05, "SE": 0.00, "ES": -0.15, "IT": -0.10, "IE": 0.10, "NZ": -0.05,
    }
    device_effect = {"mobile": -0.15, "desktop": 0.10, "tablet": 0.0}
    tenure_effect = {"new": -0.20, "returning": 0.30}
    return (
        -2.65
        + users["market"].map(market_effect).to_numpy()
        + users["device"].map(device_effect).to_numpy()
        + users["tenure"].map(tenure_effect).to_numpy()
    )


def treatment_logit(users: pd.DataFrame, arm: str) -> np.ndarray:
    """True change in conversion log-odds from each offer.

    - 10% discount wins new customers but slightly puts off returning ones,
      who were going to buy anyway and now wonder about the price.
    - Free shipping matters on mobile in markets where shipping is slow and
      costly, and does nothing much elsewhere.
    - The bundle suits returning desktop shoppers and adds friction for others.
    """
    new = users["tenure"].eq("new").to_numpy()
    mobile = users["device"].eq("mobile").to_numpy()
    desktop = users["device"].eq("desktop").to_numpy()
    far = users["market"].isin(FAR_MARKETS).to_numpy()
    if arm == "control":
        return np.zeros(len(users))
    if arm == "pct10_discount":
        return np.where(new, 0.25, -0.12)
    if arm == "free_shipping":
        return np.where(far & mobile, 0.45, -0.03)
    if arm == "bundle":
        return np.where(~new & desktop, 0.30, -0.05)
    raise ValueError(f"unknown arm: {arm}")


def true_conversion_probabilities(users: pd.DataFrame, arms=ARMS) -> pd.DataFrame:
    base = baseline_logit(users)
    return pd.DataFrame(
        {arm: 1.0 / (1.0 + np.exp(-(base + treatment_logit(users, arm)))) for arm in arms},
        index=users["user_id"],
    )


def _choice(rng: np.random.Generator, options: dict[str, float], size: int) -> np.ndarray:
    keys = list(options)
    return np.array(keys)[rng.choice(len(keys), size=size, p=list(options.values()))]


def simulate(config: ExperimentConfig | None = None) -> Experiment:
    cfg = config or ExperimentConfig()
    rng = np.random.default_rng(cfg.seed)
    n = cfg.n_users
    user_ids = np.array([f"u{index:07d}" for index in range(n)])

    users = pd.DataFrame(
        {
            "user_id": user_ids,
            "market": _choice(rng, MARKETS, n),
            "device": _choice(rng, DEVICES, n),
            "tenure": _choice(rng, TENURES, n),
        }
    )
    account_age = np.where(users["tenure"].eq("returning"), rng.integers(30, 900, n), 0)
    users["signup_date"] = pd.Timestamp(cfg.start) - pd.to_timedelta(account_age, unit="D")

    seconds = rng.uniform(0, cfg.enrollment_days * 86_400, n)
    assigned_at = pd.to_datetime(cfg.start) + pd.to_timedelta(np.sort(seconds), unit="s")
    arm_index = rng.choice(len(cfg.arms), size=n, p=list(cfg.allocation))
    arms = np.array(cfg.arms)[arm_index]
    assignments = pd.DataFrame({"user_id": user_ids, "arm": arms, "assigned_at": assigned_at})

    truth = true_conversion_probabilities(users, cfg.arms)
    probability = truth.to_numpy()[np.arange(n), arm_index]
    converted = rng.random(n) < probability
    delay = pd.to_timedelta(rng.exponential(cfg.mean_conversion_delay_days, n), unit="D")
    converted_at = assigned_at + delay
    # Every conversion that happened before the data pull is logged, even after the
    # conversion window. The analysis applies the window, the way a platform would.
    happened = converted & (converted_at <= pd.Timestamp(cfg.cutoff))
    discount = pd.Series(arms).map(OFFER_COST_SHARE).to_numpy()
    revenue = rng.lognormal(np.log(AVERAGE_ORDER_VALUE), 0.45, n) * (1 - discount)
    conversions = pd.DataFrame(
        {
            "user_id": user_ids[happened],
            "converted_at": converted_at[happened],
            "revenue": revenue[happened].round(2),
        }
    )

    # Exposure is logged a few seconds after assignment, when the offer renders.
    exposed_at = assigned_at + pd.to_timedelta(rng.uniform(0.5, 8.0, n), unit="s")
    keep = np.ones(n, dtype=bool)
    for arm, loss in cfg.exposure_loss.items():
        in_arm = arms == arm
        keep &= ~(in_arm & (rng.random(n) < loss))
    exposures = pd.DataFrame(
        {"user_id": user_ids[keep], "arm": arms[keep], "exposed_at": exposed_at[keep]}
    )

    if cfg.contamination_share > 0:
        crossed = rng.random(n) < cfg.contamination_share
        other = np.array(cfg.arms)[(arm_index + rng.integers(1, len(cfg.arms), n)) % len(cfg.arms)]
        extra = pd.DataFrame(
            {
                "user_id": user_ids[crossed],
                "arm": other[crossed],
                "exposed_at": exposed_at[crossed] + pd.to_timedelta(2, unit="h"),
            }
        )
        exposures = pd.concat([exposures, extra], ignore_index=True)

    if cfg.duplicate_assignment_share > 0:
        repeated = rng.random(n) < cfg.duplicate_assignment_share
        again = assignments[repeated].copy()
        again["arm"] = np.array(cfg.arms)[rng.integers(0, len(cfg.arms), int(repeated.sum()))]
        again["assigned_at"] = again["assigned_at"] + pd.to_timedelta(1, unit="D")
        assignments = pd.concat([assignments, again], ignore_index=True)

    if cfg.leaky_feature:
        # Recorded after assignment: how many times the user opened the checkout
        # page. Converters open it more, and the offer itself changes it.
        users = users.copy()
        users["checkout_visits"] = rng.poisson(0.6 + 2.2 * converted + 0.4 * (arm_index > 0))

    exposures = exposures.sort_values("exposed_at", ignore_index=True)
    assignments = assignments.sort_values("assigned_at", ignore_index=True)
    conversions = conversions.sort_values("converted_at", ignore_index=True)
    return Experiment(cfg, users, assignments, exposures, conversions, truth)


def with_defects(config: ExperimentConfig | None = None, **defects) -> ExperimentConfig:
    """Same experiment with defects switched on, for demos and tests."""
    return replace(config or ExperimentConfig(), **defects)


def analysis_frame(experiment: Experiment, require_mature: bool = True) -> pd.DataFrame:
    """One row per user with the arm they were assigned and whether they converted.

    Intent to treat: users count in the arm they were assigned to, whether or not
    the exposure was logged. A conversion counts only inside the conversion
    window. Users whose window had not closed when the data was pulled are
    dropped by default, because their conversions may still be on the way.
    """
    cfg = experiment.config
    first = experiment.assignments.sort_values("assigned_at").drop_duplicates("user_id")
    frame = first.merge(experiment.users, on="user_id", how="left")
    conv = experiment.conversions.groupby("user_id", as_index=False).agg(
        converted_at=("converted_at", "min"), revenue=("revenue", "sum")
    )
    frame = frame.merge(conv, on="user_id", how="left")
    window_end = frame["assigned_at"] + pd.Timedelta(days=cfg.conversion_window_days)
    frame["converted"] = (
        frame["converted_at"].notna() & (frame["converted_at"] <= window_end)
    ).astype(int)
    frame["revenue"] = frame["revenue"].where(frame["converted"] == 1, 0.0).fillna(0.0)
    frame["mature"] = window_end <= pd.Timestamp(cfg.cutoff)
    if require_mature:
        frame = frame[frame["mature"]]
    return frame.drop(columns=["converted_at"]).reset_index(drop=True)


def cell_counts(frame: pd.DataFrame, by=SEGMENT_COLUMNS) -> pd.DataFrame:
    """Users and conversions per segment cell and arm, the input to the Bayesian model."""
    return (
        frame.groupby([*by, "arm"], observed=True)["converted"]
        .agg(users="size", conversions="sum")
        .reset_index()
    )
