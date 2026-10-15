"""Experiment planning: sample size, MDE, runtime and the cost of peeking."""

from __future__ import annotations

import numpy as np
from scipy import stats


def _h(p1: float, p2: float) -> float:
    """Cohen's h for two proportions."""
    return abs(2 * np.arcsin(np.sqrt(p2)) - 2 * np.arcsin(np.sqrt(p1)))


def sample_size_per_arm(
    baseline: float, mde_rel: float, alpha: float = 0.05, power: float = 0.80
) -> int:
    """Sessions needed per arm to detect a relative lift of `mde_rel`.

    Uses the arcsine (Cohen's h) approximation for a two-sided two-proportion
    test, the same method as G*Power and statsmodels' NormalIndPower.
    """
    p2 = baseline * (1 + mde_rel)
    if not 0 < p2 < 1:
        raise ValueError("baseline * (1 + mde_rel) must be in (0, 1)")
    h = _h(baseline, p2)
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    # Var(2*arcsin(sqrt(p_hat))) ~= 1/n per arm, so the difference has variance 2/n
    return int(np.ceil(2 * ((z_a + z_b) / h) ** 2))


def mde_for_n(baseline: float, n_per_arm: int, alpha: float = 0.05, power: float = 0.80) -> dict:
    """Smallest true lift the test could reliably detect with `n_per_arm`."""
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    h = (z_a + z_b) * np.sqrt(2 / n_per_arm)
    phi = 2 * np.arcsin(np.sqrt(baseline)) + h
    p2 = float(np.sin(min(phi, np.pi) / 2) ** 2)
    return {"baseline": baseline, "mde_rate": p2, "mde_abs": p2 - baseline, "mde_rel": p2 / baseline - 1}


def power_for(
    baseline: float, treated: float, n_per_arm: int, alpha: float = 0.05
) -> float:
    """Power of a two-sided test for a given true pair of rates.

    Note: plugging the *observed* effect in here ("post-hoc power") just
    re-expresses the p-value. It is reported for design discussion only.
    """
    h = _h(baseline, treated)
    z_a = stats.norm.ppf(1 - alpha / 2)
    k = h * np.sqrt(n_per_arm / 2)
    return float(stats.norm.sf(z_a - k) + stats.norm.cdf(-z_a - k))


def winners_curse(
    baseline: float,
    true_rel: float,
    n_per_arm: int,
    alpha: float = 0.05,
    sims: int = 50_000,
    seed: int = 5,
) -> dict:
    """Type M (magnitude) error for an underpowered test - Gelman & Carlin (2014).

    If the true lift were `true_rel`, how much would a *significant* result
    overstate it on average? Small tests only reach significance when noise
    happens to push the estimate up, so their winning estimates are inflated.
    """
    rng = np.random.default_rng(seed)
    p_b = baseline * (1 + true_rel)
    xa = rng.binomial(n_per_arm, baseline, sims)
    xb = rng.binomial(n_per_arm, p_b, sims)
    pa, pb = xa / n_per_arm, xb / n_per_arm
    pool = (xa + xb) / (2 * n_per_arm)
    se = np.sqrt(pool * (1 - pool) * 2 / n_per_arm)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(se > 0, (pb - pa) / se, 0.0)
        est = np.where(pa > 0, pb / pa - 1, np.nan)
    sig_pos = (z > stats.norm.ppf(1 - alpha / 2))
    return {
        "assumed_true_lift": true_rel,
        "power": float(sig_pos.mean()),
        "mean_significant_estimate": float(np.nanmean(est[sig_pos])) if sig_pos.any() else None,
        "exaggeration_ratio": float(np.nanmean(est[sig_pos]) / true_rel) if sig_pos.any() else None,
    }


def runtime_days(n_per_arm: int, daily_sessions: float, arms: int = 2, min_days: int = 14) -> int:
    """Days to reach the target sample, never less than two full weekly cycles."""
    return int(max(min_days, np.ceil(n_per_arm * arms / daily_sessions)))


def peeking_simulation(
    baseline: float,
    n_per_arm: int,
    looks: int = 10,
    alpha: float = 0.05,
    sims: int = 20_000,
    seed: int = 7,
) -> dict:
    """Simulate A/A tests (no true effect) to show what peeking does.

    Compares three stopping rules over `looks` evenly spaced interim checks:
      fixed     - analyse once at the planned sample size
      naive     - stop at the first look where p < alpha
      corrected - stop at the first look where p < a calibrated, stricter
                  per-look threshold (a Pocock-style constant boundary found
                  by simulation) that holds the overall error at alpha.
    """
    rng = np.random.default_rng(seed)
    checkpoints = np.linspace(n_per_arm / looks, n_per_arm, looks).astype(int)
    increments = np.diff(np.concatenate([[0], checkpoints]))

    # cumulative conversions at each look for both arms (same true rate)
    ca = np.cumsum(rng.binomial(increments, baseline, size=(sims, looks)), axis=1)
    cb = np.cumsum(rng.binomial(increments, baseline, size=(sims, looks)), axis=1)
    n = checkpoints[None, :]
    pa, pb = ca / n, cb / n
    pool = (ca + cb) / (2 * n)
    se = np.sqrt(pool * (1 - pool) * 2 / n)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(se > 0, (pb - pa) / se, 0.0)
    absz = np.abs(z)
    max_z = absz.max(axis=1)

    z_crit = stats.norm.ppf(1 - alpha / 2)
    fixed_fpr = float((absz[:, -1] > z_crit).mean())
    naive_fpr = float((max_z > z_crit).mean())

    # calibrated constant boundary: the (1 - alpha) quantile of max |z|
    z_pocock = float(np.quantile(max_z, 1 - alpha))
    nominal_alpha = float(2 * stats.norm.sf(z_pocock))
    corrected_fpr = float((max_z > z_pocock).mean())

    # false-positive rate as a function of how many looks were taken
    by_looks = [
        {"looks": k, "fpr": float((absz[:, np.linspace(0, looks - 1, k).astype(int)].max(axis=1) > z_crit).mean())}
        for k in range(1, looks + 1)
    ]

    return {
        "baseline": baseline,
        "n_per_arm": n_per_arm,
        "looks": looks,
        "sims": sims,
        "alpha": alpha,
        "fixed_fpr": fixed_fpr,
        "naive_fpr": naive_fpr,
        "corrected_fpr": corrected_fpr,
        "corrected_nominal_alpha": nominal_alpha,
        "by_looks": by_looks,
    }
