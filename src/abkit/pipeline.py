"""End-to-end analysis of one scenario, returning a JSON-ready result dict."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import bayes, data, power, quality, segments, stats, uplift
from .decision import recommend
from .scenarios import SCENARIOS, Scenario, get


@dataclass
class Settings:
    alpha: float = 0.05
    power: float = 0.80
    # Planning target: the smallest lift worth detecting (relative).
    target_mde_rel: float = 0.20
    # Guardrail: bounce rate may not rise by more than this (absolute).
    bounce_margin: float = 0.02
    # Business assumptions - illustrative, clearly labelled in outputs.
    daily_sessions: int = 1_000
    monthly_sessions: int = 30_000
    peeking_looks: int = 10
    peeking_sims: int = 20_000
    uplift_permutations: int = 500


def _funnel(df: pd.DataFrame, sc: Scenario) -> list[dict]:
    rows = []
    for page in sc.funnel:
        sub = df[df.PageViewed == page]
        for g in ("A", "B"):
            s = sub[sub.group == g]
            rows.append(
                {
                    "page": page,
                    "group": g,
                    "sessions": int(len(s)),
                    "conversion": float(s.Conversion.mean()) if len(s) else None,
                    "bounce": float(s.Bounce.mean()) if len(s) else None,
                    "time_on_page": float(s.TimeOnPage_sec.mean()) if len(s) else None,
                }
            )
    return rows


def analyze(
    scenario: str | Scenario,
    df: pd.DataFrame | None = None,
    settings: Settings | None = None,
    data_dir: str | Path | None = None,
) -> dict:
    sc = get(scenario) if isinstance(scenario, str) else scenario
    st = settings or Settings()
    df = data.load(sc, data_dir) if df is None else df

    res: dict = {
        "scenario": {
            "key": sc.key,
            "title": sc.title,
            "problem": sc.problem,
            "hypothesis": sc.hypothesis,
            "change": sc.change,
            "funnel": list(sc.funnel),
            "revenue_label": sc.revenue_label,
        },
        "settings": asdict(st),
        "n_sessions": int(len(df)),
    }

    # 1. Trust
    res["quality"] = quality.run_checks(df, sc.covariates, sc.exploratory)

    # 2. Primary KPI
    res["primary"] = stats.proportion_metric(df, "Conversion", st.alpha)

    # 3. Secondary KPIs
    secondary = {"time_on_page": stats.welch(df, "TimeOnPage_sec", st.alpha)}
    if sc.has_revenue and (df.Revenue > 0).any():
        secondary["revenue_per_visitor"] = stats.bootstrap_mean_diff(df, "Revenue", alpha=st.alpha)
        # Average order value among converters: did B change *what* people buy?
        secondary["avg_order_value"] = stats.welch(df[df.Conversion == 1], "Revenue", st.alpha)
    res["secondary"] = secondary

    # 4. Guardrails
    res["guardrails"] = {
        "bounce": stats.non_inferiority_proportion(df, "Bounce", st.bounce_margin, st.alpha),
    }

    # 5. Robustness
    res["adjusted"] = stats.adjusted_effect(df, sc.covariates)

    # 6. Bayesian view
    p = res["primary"]
    res["bayes"] = bayes.beta_binomial(p["x_a"], p["n_a"], p["x_b"], p["n_b"])

    # 7. Power and planning
    baseline = p["rate_a"]
    n_arm = min(p["n_a"], p["n_b"])
    mde = power.mde_for_n(baseline, n_arm, st.alpha, st.power)
    n_target = power.sample_size_per_arm(baseline, st.target_mde_rel, st.alpha, st.power)
    n_observed = (
        power.sample_size_per_arm(baseline, p["uplift"], st.alpha, st.power)
        if 0 < p["uplift"] and baseline * (1 + p["uplift"]) < 1
        else None
    )
    curve = [
        {"mde_rel": m, "n_per_arm": power.sample_size_per_arm(baseline, m, st.alpha, st.power)}
        for m in np.round(np.arange(0.10, 1.01, 0.05), 2)
    ]
    res["power"] = {
        "baseline": baseline,
        "n_per_arm": n_arm,
        "mde_rel_at_n": mde["mde_rel"],
        "mde_abs_at_n": mde["mde_abs"],
        "target_mde_rel": st.target_mde_rel,
        "n_for_target": n_target,
        "runtime_days_for_target": power.runtime_days(n_target, st.daily_sessions),
        "n_for_observed_effect": n_observed,
        "power_curve": curve,
        "winners_curse": power.winners_curse(baseline, st.target_mde_rel, n_arm, st.alpha),
    }
    res["peeking"] = power.peeking_simulation(
        baseline, max(n_target, n_arm), looks=st.peeking_looks, alpha=st.alpha, sims=st.peeking_sims
    )

    # 8. Segments
    kinds = {c: "pre-treatment" for c in sc.covariates} | {c: "exploratory" for c in sc.exploratory}
    seg = segments.segment_effects(df, list(sc.covariates) + list(sc.exploratory), st.alpha, kinds)
    res["segments"] = seg.to_dict(orient="records")
    res["heterogeneity"] = segments.heterogeneity(df, sc.covariates)

    # 9. Uplift model
    res["uplift_model"] = uplift.fit_uplift(df, sc.covariates, n_perm=st.uplift_permutations)

    # 10. Funnel view
    res["funnel"] = _funnel(df, sc)

    # 11. Business impact (illustrative traffic assumption)
    impact = {"monthly_sessions_assumed": st.monthly_sessions}
    impact["extra_conversions_per_month"] = p["abs_diff"] * st.monthly_sessions
    impact["extra_conversions_ci"] = tuple(x * st.monthly_sessions for x in p["abs_ci"])
    if "revenue_per_visitor" in secondary:
        rpv = secondary["revenue_per_visitor"]
        impact["extra_revenue_per_month"] = rpv["abs_diff"] * st.monthly_sessions
        impact["extra_revenue_ci"] = tuple(x * st.monthly_sessions for x in rpv["abs_ci"])
    res["impact"] = impact

    # 12. Decision
    res["decision"] = recommend(res)
    return res


def analyze_all(settings: Settings | None = None, data_dir=None) -> dict[str, dict]:
    return {k: analyze(k, settings=settings, data_dir=data_dir) for k in SCENARIOS}


def to_jsonable(obj):
    """Recursively convert numpy / pandas / tuple values for json.dump."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return None if math.isnan(f) or math.isinf(f) else f
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    return obj


def save_json(results: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(to_jsonable(results), indent=2))
