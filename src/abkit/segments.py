"""Segment analysis with multiple-testing control.

Slicing an experiment by device, user type and page produces many small
tests. At alpha = 0.05, a dozen segment tests will throw up a 'significant'
segment by chance about half the time. Every segment p-value is therefore
adjusted with Holm (family-wise error) and Benjamini-Hochberg (false
discovery rate), and heterogeneity is judged by a formal interaction test.
"""

from __future__ import annotations

import pandas as pd
from statsmodels.stats.multitest import multipletests

from .stats import interaction_test, two_proportions


def segment_effects(df: pd.DataFrame, variables, alpha: float = 0.05, kind: dict | None = None) -> pd.DataFrame:
    kind = kind or {}
    rows = []
    for var in variables:
        for level, sub in df.groupby(var):
            a = sub[sub.group == "A"].Conversion
            b = sub[sub.group == "B"].Conversion
            if len(a) == 0 or len(b) == 0:
                continue
            r = two_proportions(int(a.sum()), len(a), int(b.sum()), len(b), alpha)
            rows.append(
                {
                    "variable": var,
                    "level": level,
                    "kind": kind.get(var, "pre-treatment"),
                    "n_a": r["n_a"],
                    "n_b": r["n_b"],
                    "rate_a": r["rate_a"],
                    "rate_b": r["rate_b"],
                    "abs_diff": r["abs_diff"],
                    "abs_ci_lo": r["abs_ci"][0],
                    "abs_ci_hi": r["abs_ci"][1],
                    "uplift": r["uplift"],
                    "p_raw": r["fisher_p"],  # exact test: segments are small
                }
            )
    out = pd.DataFrame(rows)
    if len(out):
        out["p_holm"] = multipletests(out.p_raw, alpha=alpha, method="holm")[1]
        out["p_bh"] = multipletests(out.p_raw, alpha=alpha, method="fdr_bh")[1]
        out["sig_raw"] = out.p_raw < alpha
        out["sig_holm"] = out.p_holm < alpha
        out["direction_flip"] = out.abs_diff < 0
    return out


def heterogeneity(df: pd.DataFrame, covariates) -> list[dict]:
    """Formal treatment x covariate interaction tests (pre-treatment only)."""
    return [interaction_test(df, c) for c in covariates]
