"""Trust checks: run these before believing any A/B result.

A significant uplift means nothing if the experiment itself is broken. These
checks follow the practice described in Kohavi, Tang & Xu (2020),
*Trustworthy Online Controlled Experiments*, ch. 21 (Sample Ratio Mismatch)
and ch. 3 (Twyman's law: results that look too good deserve extra scrutiny).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy import stats

# Two-tier SRM thresholds. p < 0.001 is the common industry "stop" line;
# 0.001 <= p < 0.05 is flagged for investigation but not treated as fatal.
SRM_FAIL_P = 0.001
SRM_WARN_P = 0.05

SMD_WARN = 0.10  # standardized mean difference above this = notable imbalance


@dataclass
class Check:
    name: str
    status: str  # "pass" | "warn" | "fail"
    detail: str
    value: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def srm_test(counts_a: int, counts_b: int, expected_share_b: float = 0.5) -> dict:
    """Chi-square goodness-of-fit for the A/B traffic split."""
    n = counts_a + counts_b
    expected = np.array([1 - expected_share_b, expected_share_b]) * n
    chi2, p = stats.chisquare([counts_a, counts_b], f_exp=expected)
    if p < SRM_FAIL_P:
        status = "fail"
    elif p < SRM_WARN_P:
        status = "warn"
    else:
        status = "pass"
    return {
        "n_a": int(counts_a),
        "n_b": int(counts_b),
        "share_b": counts_b / n,
        "expected_share_b": expected_share_b,
        "chi2": float(chi2),
        "p_value": float(p),
        "status": status,
    }


def covariate_balance(df: pd.DataFrame, columns: list[str] | tuple[str, ...]) -> pd.DataFrame:
    """Per-level share in A vs B with standardized mean difference (SMD).

    Also returns a chi-square test of independence per covariate. For
    pre-treatment covariates, imbalance hints at broken randomisation; for
    post-treatment variables it is expected and simply means the variation
    changed behaviour.
    """
    rows = []
    for col in columns:
        ct = pd.crosstab(df[col], df["group"])
        p = stats.chi2_contingency(ct)[1] if ct.shape[0] > 1 else 1.0
        for level in ct.index:
            pa = (df.loc[df.group == "A", col] == level).mean()
            pb = (df.loc[df.group == "B", col] == level).mean()
            pooled_sd = np.sqrt((pa * (1 - pa) + pb * (1 - pb)) / 2)
            smd = (pb - pa) / pooled_sd if pooled_sd > 0 else 0.0
            rows.append(
                {
                    "variable": col,
                    "level": level,
                    "share_a": pa,
                    "share_b": pb,
                    "smd": smd,
                    "chi2_p": p,
                }
            )
    return pd.DataFrame(rows)


def run_checks(df: pd.DataFrame, covariates, exploratory=()) -> dict:
    """Run every trust check and return a structured report."""
    checks: list[Check] = []

    # 1. Sample ratio mismatch
    n_a = int((df.group == "A").sum())
    n_b = int((df.group == "B").sum())
    srm = srm_test(n_a, n_b)
    checks.append(
        Check(
            "Sample ratio mismatch",
            srm["status"],
            f"A={n_a:,} vs B={n_b:,} (B share {srm['share_b']:.1%}, expected 50%); "
            f"chi-square p={srm['p_value']:.3f}",
            srm["p_value"],
        )
    )

    # 2. Duplicate sessions
    dupes = int(df["SessionID"].duplicated().sum())
    checks.append(
        Check(
            "Duplicate sessions",
            "pass" if dupes == 0 else "fail",
            f"{dupes} duplicated SessionID values",
            dupes,
        )
    )

    # 3. Missing values
    nulls = int(df.isna().sum().sum())
    checks.append(
        Check("Missing values", "pass" if nulls == 0 else "warn", f"{nulls} null cells", nulls)
    )

    # 4. Tracking consistency: revenue must only appear on converting sessions
    rev_no_conv = int(((df.Revenue > 0) & (df.Conversion == 0)).sum())
    conv_no_rev = int(((df.Revenue <= 0) & (df.Conversion == 1)).sum())
    has_revenue = bool((df.Revenue > 0).any())
    if not has_revenue:
        detail = "No revenue recorded in this dataset (revenue checks skipped)"
        status = "pass"
    else:
        detail = (
            f"{rev_no_conv} sessions with revenue but no conversion; "
            f"{conv_no_rev} conversions with zero revenue"
        )
        status = "pass" if rev_no_conv == 0 and conv_no_rev == 0 else "warn"
    checks.append(Check("Revenue / conversion consistency", status, detail))

    # 5. Contradictory events: a bounced session cannot convert
    bounce_conv = int(((df.Bounce == 1) & (df.Conversion == 1)).sum())
    checks.append(
        Check(
            "Bounce / conversion consistency",
            "pass" if bounce_conv == 0 else "warn",
            f"{bounce_conv} sessions marked as both bounced and converted",
            bounce_conv,
        )
    )

    # 6. Plausible engagement times (bot-like or broken timers)
    t = df["TimeOnPage_sec"]
    implausible = int(((t <= 2) | (t > 3600)).sum())
    checks.append(
        Check(
            "Bot / timer plausibility",
            "pass" if implausible == 0 else "warn",
            f"{implausible} sessions with time on page <=2s or >1h "
            f"(range {t.min():.0f}-{t.max():.0f}s)",
            implausible,
        )
    )

    # 7. Covariate balance on pre-treatment attributes
    bal = covariate_balance(df, list(covariates))
    worst = bal.loc[bal.smd.abs().idxmax()] if len(bal) else None
    min_p = bal.chi2_p.min() if len(bal) else 1.0
    status = "pass"
    if worst is not None and (abs(worst.smd) > SMD_WARN or min_p < 0.05):
        status = "warn"
    checks.append(
        Check(
            "Pre-treatment covariate balance",
            status,
            (
                f"Largest imbalance: {worst.variable}={worst.level} "
                f"(A {worst.share_a:.1%} vs B {worst.share_b:.1%}, SMD {worst.smd:+.2f}); "
                f"min chi-square p={min_p:.3f}"
            )
            if worst is not None
            else "No covariates",
            float(min_p),
        )
    )

    # 8. Exploratory (possibly post-treatment) variables - informational
    exp_bal = covariate_balance(df, list(exploratory)) if exploratory else pd.DataFrame()
    if len(exp_bal):
        p = exp_bal.chi2_p.min()
        checks.append(
            Check(
                "Page mix shift (post-treatment)",
                "info" if p < 0.05 else "pass",
                (
                    f"Page mix differs between arms (p={p:.3f}). Expected if B changes "
                    "navigation; PageViewed is therefore not used as a covariate."
                )
                if p < 0.05
                else f"Page mix similar between arms (p={p:.3f})",
                float(p),
            )
        )

    worst_status = "pass"
    for c in checks:
        if c.status == "fail":
            worst_status = "fail"
            break
        if c.status == "warn":
            worst_status = "warn"

    return {
        "overall": worst_status,
        "srm": srm,
        "checks": [c.to_dict() for c in checks],
        "balance": bal.to_dict(orient="records"),
        "exploratory_balance": exp_bal.to_dict(orient="records"),
    }
