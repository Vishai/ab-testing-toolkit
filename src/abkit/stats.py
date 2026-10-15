"""Frequentist analysis of A/B results.

Primary metric (a proportion) uses a pooled two-proportion z-test, with a Wald
CI for the absolute difference and a delta-method CI for relative uplift.
Fisher's exact test is reported alongside as a small-sample check.
Continuous metrics use Welch's t-test (no equal-variance assumption) and,
for heavily skewed revenue, a percentile bootstrap.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def _z(alpha: float) -> float:
    return stats.norm.ppf(1 - alpha / 2)


def two_proportions(x_a: int, n_a: int, x_b: int, n_b: int, alpha: float = 0.05) -> dict:
    """Compare two conversion rates.

    Returns rates, absolute difference with CI, relative uplift
    ((B - A) / A) with delta-method CI, pooled z-test p-value, Fisher exact
    p-value and Cohen's h effect size.
    """
    p_a, p_b = x_a / n_a, x_b / n_b
    diff = p_b - p_a

    # Pooled z-test (H0: p_a == p_b)
    p_pool = (x_a + x_b) / (n_a + n_b)
    se_pool = np.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
    z = diff / se_pool if se_pool > 0 else 0.0
    p_value = 2 * stats.norm.sf(abs(z))

    # Unpooled SE for the CI of the difference
    se_diff = np.sqrt(p_a * (1 - p_a) / n_a + p_b * (1 - p_b) / n_b)
    zc = _z(alpha)
    ci_diff = (diff - zc * se_diff, diff + zc * se_diff)

    # Relative uplift CI on the log-ratio scale (delta method)
    if x_a > 0 and x_b > 0:
        log_rr = np.log(p_b / p_a)
        se_log = np.sqrt((1 - p_a) / x_a + (1 - p_b) / x_b)
        ci_rel = (np.exp(log_rr - zc * se_log) - 1, np.exp(log_rr + zc * se_log) - 1)
    else:
        ci_rel = (np.nan, np.nan)

    fisher_p = stats.fisher_exact([[x_b, n_b - x_b], [x_a, n_a - x_a]])[1]
    cohens_h = 2 * np.arcsin(np.sqrt(p_b)) - 2 * np.arcsin(np.sqrt(p_a))

    return {
        "n_a": int(n_a),
        "n_b": int(n_b),
        "x_a": int(x_a),
        "x_b": int(x_b),
        "rate_a": p_a,
        "rate_b": p_b,
        "abs_diff": diff,
        "abs_ci": ci_diff,
        "uplift": diff / p_a if p_a > 0 else np.nan,
        "uplift_ci": ci_rel,
        "z": float(z),
        "p_value": float(p_value),
        "fisher_p": float(fisher_p),
        "cohens_h": float(cohens_h),
        "significant": bool(p_value < alpha),
        "alpha": alpha,
    }


def proportion_metric(df: pd.DataFrame, col: str, alpha: float = 0.05) -> dict:
    a = df.loc[df.group == "A", col]
    b = df.loc[df.group == "B", col]
    return two_proportions(int(a.sum()), len(a), int(b.sum()), len(b), alpha)


def welch(df: pd.DataFrame, col: str, alpha: float = 0.05) -> dict:
    """Welch's t-test for a continuous metric, with CI and Cohen's d."""
    a = df.loc[df.group == "A", col].astype(float).to_numpy()
    b = df.loc[df.group == "B", col].astype(float).to_numpy()
    res = stats.ttest_ind(b, a, equal_var=False)
    diff = b.mean() - a.mean()
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = np.sqrt(va + vb)
    dof = (va + vb) ** 2 / (va**2 / (len(a) - 1) + vb**2 / (len(b) - 1))
    tc = stats.t.ppf(1 - alpha / 2, dof)
    pooled_sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    mw = stats.mannwhitneyu(b, a, alternative="two-sided")
    return {
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "abs_diff": float(diff),
        "abs_ci": (float(diff - tc * se), float(diff + tc * se)),
        "uplift": float(diff / a.mean()) if a.mean() else np.nan,
        "p_value": float(res.pvalue),
        "mannwhitney_p": float(mw.pvalue),
        "cohens_d": float(diff / pooled_sd) if pooled_sd else 0.0,
        "significant": bool(res.pvalue < alpha),
    }


def bootstrap_mean_diff(
    df: pd.DataFrame,
    col: str,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    """Percentile bootstrap CI for difference and relative lift in a mean.

    Used for revenue per visitor, which is zero-inflated and right-skewed, so
    a t-interval's normality assumption is shaky at this sample size.
    """
    rng = np.random.default_rng(seed)
    a = df.loc[df.group == "A", col].astype(float).to_numpy()
    b = df.loc[df.group == "B", col].astype(float).to_numpy()
    ia = rng.integers(0, len(a), size=(n_boot, len(a)))
    ib = rng.integers(0, len(b), size=(n_boot, len(b)))
    ma = a[ia].mean(axis=1)
    mb = b[ib].mean(axis=1)
    diffs = mb - ma
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(ma > 0, diffs / ma, np.nan)
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    # two-sided bootstrap p-value: share of resamples on the other side of 0
    p_boot = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return {
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "abs_diff": float(b.mean() - a.mean()),
        "abs_ci": (float(np.percentile(diffs, lo)), float(np.percentile(diffs, hi))),
        "uplift": float((b.mean() - a.mean()) / a.mean()) if a.mean() else np.nan,
        "uplift_ci": (float(np.nanpercentile(rel, lo)), float(np.nanpercentile(rel, hi))),
        "p_value": float(max(p_boot, 1 / n_boot)),
        "n_boot": n_boot,
    }


def non_inferiority_proportion(
    df: pd.DataFrame, col: str, margin: float, alpha: float = 0.05, higher_is_worse: bool = True
) -> dict:
    """Guardrail test: is B no worse than A by more than `margin`?

    For a 'higher is worse' metric like bounce rate, B passes if the upper
    bound of the one-sided (1 - alpha) CI for (B - A) is below `margin`.
    """
    a = df.loc[df.group == "A", col]
    b = df.loc[df.group == "B", col]
    pa, pb = a.mean(), b.mean()
    diff = pb - pa
    se = np.sqrt(pa * (1 - pa) / len(a) + pb * (1 - pb) / len(b))
    zc = stats.norm.ppf(1 - alpha)
    if higher_is_worse:
        bound = diff + zc * se
        passed = bound < margin
        p_value = stats.norm.cdf((diff - margin) / se) if se else 0.0
    else:
        bound = diff - zc * se
        passed = bound > -margin
        p_value = stats.norm.sf((diff + margin) / se) if se else 0.0
    return {
        "rate_a": float(pa),
        "rate_b": float(pb),
        "abs_diff": float(diff),
        "one_sided_bound": float(bound),
        "margin": margin,
        "p_value": float(p_value),
        "passed": bool(passed),
    }


def adjusted_effect(df: pd.DataFrame, covariates, outcome: str = "Conversion") -> dict:
    """Robustness check: logistic regression of outcome on treatment + covariates.

    If randomisation worked, the adjusted treatment effect should agree with
    the raw one. A large shift would signal that a covariate imbalance is
    driving the headline result.
    """
    import statsmodels.formula.api as smf

    terms = " + ".join(f"C({c})" for c in covariates)
    formula = f"{outcome} ~ treatment" + (f" + {terms}" if terms else "")
    model = smf.logit(formula, data=df).fit(disp=False)
    coef = model.params["treatment"]
    lo, hi = model.conf_int().loc["treatment"]
    return {
        "formula": formula,
        "odds_ratio": float(np.exp(coef)),
        "or_ci": (float(np.exp(lo)), float(np.exp(hi))),
        "p_value": float(model.pvalues["treatment"]),
    }


def interaction_test(df: pd.DataFrame, covariate: str, outcome: str = "Conversion") -> dict:
    """Likelihood-ratio test: does the treatment effect differ across a covariate?

    This is the principled test for heterogeneity. Eyeballing per-segment
    p-values is not, because each segment is underpowered and many are tested.
    """
    import statsmodels.formula.api as smf

    base = smf.logit(f"{outcome} ~ treatment + C({covariate})", data=df).fit(disp=False)
    full = smf.logit(f"{outcome} ~ treatment * C({covariate})", data=df).fit(disp=False)
    lr = 2 * (full.llf - base.llf)
    dof = full.df_model - base.df_model
    return {
        "covariate": covariate,
        "lr_stat": float(lr),
        "df": int(dof),
        "p_value": float(stats.chi2.sf(lr, dof)),
    }
