"""Power analysis before launch, for a test with several offers against one control.

Answers the questions to settle before any traffic is spent:
  - how many users each arm needs to detect the smallest lift worth acting on,
    with the false-positive rate controlled across all offers (Dunnett),
  - how long that takes at the expected traffic,
  - what a single segment could detect on its own (usually very little, which is
    why segment effects are estimated with a pooled hierarchical model),
  - how often a Bayesian read-out would end up confident (assurance),
  - and whether the plan's power holds up in simulation.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
from scipy import optimize, stats

from .power import _h


def _mvn(cov: np.ndarray):
    """Multivariate normal whose CDF is integrated with a fixed seed and tight tolerance.

    SciPy integrates the CDF with randomised quasi-Monte Carlo, so without a seed the
    same plan could come out one user per arm different from run to run.
    """
    return stats.multivariate_normal(
        mean=np.zeros(len(cov)), cov=cov, seed=0, maxpts=200_000, abseps=1e-8, releps=1e-8
    )


def dunnett_critical_value(n_offers: int, alpha: float = 0.05, control_ratio: float = 1.0) -> float:
    """Two-sided critical z for comparing `n_offers` arms with one shared control.

    With equal arm sizes every comparison shares the control, so the test
    statistics are correlated at 0.5. Using that correlation gives a slightly
    lower bar than Bonferroni for the same family-wise error rate.
    """
    if n_offers == 1:
        return float(stats.norm.ppf(1 - alpha / 2))
    rho = 1.0 / (1.0 + control_ratio)
    cov = np.full((n_offers, n_offers), rho)
    np.fill_diagonal(cov, 1.0)
    mvn = _mvn(cov)

    def coverage(c: float) -> float:
        return mvn.cdf(np.full(n_offers, c), lower_limit=np.full(n_offers, -c)) - (1 - alpha)

    return float(optimize.brentq(coverage, 1.0, 6.0))


def sample_size(baseline: float, mde_rel: float, z_crit: float, power: float = 0.8) -> int:
    """Users per arm for a two-proportion test at a given critical value (arcsine method)."""
    treated = baseline * (1 + mde_rel)
    if not 0 < treated < 1:
        raise ValueError("baseline * (1 + mde_rel) must be in (0, 1)")
    h = _h(baseline, treated)
    return math.ceil(2 * ((z_crit + stats.norm.ppf(power)) / h) ** 2)


def detectable_lift(baseline: float, n_per_arm: float, z_crit: float, power: float = 0.8) -> float:
    """Smallest relative lift detectable with `n_per_arm` users per arm."""
    h = (z_crit + stats.norm.ppf(power)) * math.sqrt(2 / n_per_arm)
    phi = min(2 * math.asin(math.sqrt(baseline)) + h, math.pi)
    return math.sin(phi / 2) ** 2 / baseline - 1


def bayesian_assurance(
    baseline: float,
    n_per_arm: int,
    true_lift_mean: float,
    true_lift_sd: float,
    threshold: float = 0.95,
    sims: int = 20_000,
    seed: int = 4,
) -> float:
    """Chance the posterior ends up with P(offer beats control) above `threshold`.

    The true lift isn't known before launch, so it's drawn from a design prior.
    Each simulated test uses a Beta(1, 1) posterior per arm (normal approximation
    of the difference, which is accurate at these sample sizes).
    """
    rng = np.random.default_rng(seed)
    lifts = rng.normal(true_lift_mean, true_lift_sd, sims)
    treated = np.clip(baseline * (1 + lifts), 1e-6, 1 - 1e-6)
    x_control = rng.binomial(n_per_arm, baseline, sims)
    x_treated = rng.binomial(n_per_arm, treated)
    a_c, b_c = 1 + x_control, 1 + n_per_arm - x_control
    a_t, b_t = 1 + x_treated, 1 + n_per_arm - x_treated
    mean_diff = a_t / (a_t + b_t) - a_c / (a_c + b_c)
    var = (a_t * b_t) / ((a_t + b_t) ** 2 * (a_t + b_t + 1)) + (a_c * b_c) / (
        (a_c + b_c) ** 2 * (a_c + b_c + 1)
    )
    prob_better = stats.norm.cdf(mean_diff / np.sqrt(var))
    return float((prob_better > threshold).mean())


def simulated_power(
    baseline: float,
    n_per_arm: int,
    lifts: Mapping[str, float],
    z_crit: float,
    sims: int = 5_000,
    seed: int = 8,
) -> dict[str, float]:
    """Share of simulated tests in which each offer clears the critical value."""
    rng = np.random.default_rng(seed)
    control = rng.binomial(n_per_arm, baseline, sims) / n_per_arm
    result = {}
    for arm, lift in lifts.items():
        treated = rng.binomial(n_per_arm, baseline * (1 + lift), sims) / n_per_arm
        pooled = (control + treated) / 2
        se = np.sqrt(pooled * (1 - pooled) * 2 / n_per_arm)
        result[arm] = float(((treated - control) / se > z_crit).mean())
    return result


def simulated_familywise_error(
    baseline: float,
    n_per_arm: int,
    n_offers: int,
    z_crit: float,
    sims: int = 20_000,
    seed: int = 6,
) -> float:
    """With no offer doing anything, how often does at least one look like a winner or loser?"""
    rng = np.random.default_rng(seed)
    control = rng.binomial(n_per_arm, baseline, sims) / n_per_arm
    treated = rng.binomial(n_per_arm, baseline, (n_offers, sims)) / n_per_arm
    pooled = (control + treated) / 2
    z = (treated - control) / np.sqrt(pooled * (1 - pooled) * 2 / n_per_arm)
    return float((np.abs(z) > z_crit).any(axis=0).mean())


def plan_launch(
    baseline: float,
    offers: tuple[str, ...],
    mde_rel: float = 0.10,
    alpha: float = 0.05,
    power: float = 0.80,
    daily_users: float = 5_500,
    buffer: float = 0.01,
    segment_shares: Mapping[str, float] | None = None,
    min_days: int = 14,
) -> dict:
    """Everything needed to sign off on a launch, in one dictionary.

    `buffer` adds users to cover expected losses (missing exposures, users whose
    conversion window hasn't closed when the data is pulled). The planned total is
    rounded up to the nearest thousand.
    """
    z_crit = dunnett_critical_value(len(offers), alpha)
    n_per_arm = sample_size(baseline, mde_rel, z_crit, power)
    arms = len(offers) + 1
    required = n_per_arm * arms
    planned = int(math.ceil(required * (1 + buffer) / 1_000) * 1_000)
    days = max(min_days, math.ceil(planned / daily_users))
    segments = []
    for name, share in (segment_shares or {}).items():
        per_arm = planned / arms * share
        segments.append(
            {
                "segment": name,
                "share": share,
                "users_per_arm": per_arm,
                "detectable_lift": detectable_lift(baseline, per_arm, stats.norm.ppf(0.975), power),
            }
        )
    return {
        "baseline": baseline,
        "offers": list(offers),
        "mde_rel": mde_rel,
        "alpha": alpha,
        "power": power,
        "correction": "Dunnett (many offers vs one control)",
        "critical_z": z_crit,
        "bonferroni_z": float(stats.norm.ppf(1 - alpha / (2 * len(offers)))),
        "n_per_arm": n_per_arm,
        "required_users": required,
        "planned_users": planned,
        "buffer": buffer,
        "daily_users": daily_users,
        "runtime_days": days,
        "segments": segments,
        "simulated_power_at_mde": simulated_power(
            baseline, n_per_arm, dict.fromkeys(offers, mde_rel), z_crit
        ),
        "simulated_familywise_error": simulated_familywise_error(
            baseline, n_per_arm, len(offers), z_crit
        ),
        "bayesian_assurance": bayesian_assurance(baseline, n_per_arm, mde_rel, mde_rel / 2),
    }


def aggregate_readout(frame, arms: tuple[str, ...], alpha: float = 0.05) -> list[dict]:
    """The aggregate A/B read-out the plan calls for: each offer vs control, Dunnett-adjusted.

    `frame` needs one row per user with `arm` and `converted`. The adjusted p-value
    is the chance that the largest of all the offers' |z| would be at least this
    big if no offer did anything.
    """
    control = frame.loc[frame["arm"] == arms[0], "converted"]
    k = len(arms) - 1
    cov = np.full((k, k), 0.5)
    np.fill_diagonal(cov, 1.0)
    mvn = _mvn(cov)
    rows = []
    for arm in arms[1:]:
        treated = frame.loc[frame["arm"] == arm, "converted"]
        p_c, p_t = control.mean(), treated.mean()
        pooled = (control.sum() + treated.sum()) / (len(control) + len(treated))
        se = math.sqrt(pooled * (1 - pooled) * (1 / len(control) + 1 / len(treated)))
        z = (p_t - p_c) / se
        p_raw = 2 * stats.norm.sf(abs(z))
        p_adjusted = 1 - mvn.cdf(np.full(k, abs(z)), lower_limit=np.full(k, -abs(z)))
        rows.append(
            {
                "offer": arm,
                "control_rate": float(p_c),
                "offer_rate": float(p_t),
                "rel_lift": float(p_t / p_c - 1),
                "z": float(z),
                "p_value": float(p_raw),
                "p_dunnett": float(min(max(p_adjusted, p_raw), 1.0)),
                "significant": bool(p_adjusted < alpha),
            }
        )
    return rows
