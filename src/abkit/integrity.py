"""Trust checks on raw experiment logs.

`quality.py` checks one-row-per-session files. A live experimentation platform
has richer logs (assignments, exposures, conversions, each with user ids and
timestamps), and with them a longer list of ways an experiment quietly breaks:

  sample ratio mismatch   traffic split differs from the design, at assignment or exposure
  assignment conflicts    a user assigned twice, or to two different arms
  missing exposures       assigned users who never saw the experience, more in one arm
  contamination           users who saw more than one arm
  delayed conversions     conversions arriving after the window, or data pulled too early
  leakage                 "user features" that were really recorded after treatment

Every check returns pass, warn or fail with the numbers behind it.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

from .quality import SMD_WARN, SRM_FAIL_P, SRM_WARN_P, Check
from .simulate import Experiment

PRE_TREATMENT = ("market", "device", "tenure")


def _status_from_p(p: float) -> str:
    if p < SRM_FAIL_P:
        return "fail"
    if p < SRM_WARN_P:
        return "warn"
    return "pass"


def srm(counts: pd.Series, allocation: dict[str, float]) -> dict:
    """Chi-square goodness of fit of arm counts against the designed split."""
    arms = list(allocation)
    observed = np.array([int(counts.get(arm, 0)) for arm in arms])
    expected = np.array([allocation[arm] for arm in arms]) * observed.sum()
    chi2, p = stats.chisquare(observed, f_exp=expected)
    share = observed / observed.sum()
    worst = arms[int(np.argmax(np.abs(share - np.array(list(allocation.values())))))]
    return {
        "counts": dict(zip(arms, observed.tolist(), strict=True)),
        "chi2": float(chi2),
        "p_value": float(p),
        "most_off_arm": worst,
        "status": _status_from_p(p),
    }


def assignment_conflicts(assignments: pd.DataFrame) -> dict:
    per_user = assignments.groupby("user_id")["arm"].agg(["size", "nunique"])
    repeated = int((per_user["size"] > 1).sum())
    switched = int((per_user["nunique"] > 1).sum())
    share = switched / len(per_user)
    status = "pass" if switched == 0 else ("fail" if share > 0.01 else "warn")
    return {"users": len(per_user), "repeated": repeated, "switched_arms": switched,
            "switched_share": share, "status": status}


def missing_exposures(first_assignment: pd.DataFrame, exposures: pd.DataFrame) -> dict:
    """Share of assigned users with no exposure in their own arm, and whether it depends on arm.

    A uniform loss mostly costs power. A loss that differs by arm means any analysis
    that conditions on exposure compares different kinds of users, which is SRM
    by another name.
    """
    seen = exposures.merge(first_assignment[["user_id", "arm"]], on=["user_id", "arm"])
    exposed = first_assignment["user_id"].isin(seen["user_id"])
    table = pd.crosstab(first_assignment["arm"], exposed)
    table = table.reindex(columns=[False, True], fill_value=0)
    missing_rate = (table[False] / table.sum(axis=1)).to_dict()
    p = float(stats.chi2_contingency(table)[1]) if table[False].sum() > 0 else 1.0
    overall = float((~exposed).mean())
    if p < SRM_FAIL_P:
        status = "fail"
    elif p < SRM_WARN_P or overall > 0.02:
        status = "warn"
    else:
        status = "pass"
    orphans = int((~exposures["user_id"].isin(first_assignment["user_id"])).sum())
    return {"missing_share": overall, "missing_by_arm": missing_rate, "differential_p": p,
            "orphan_exposures": orphans, "status": status if orphans == 0 else "fail"}


def contamination(first_assignment: pd.DataFrame, exposures: pd.DataFrame) -> dict:
    """Users who were exposed to an arm other than the one they were assigned."""
    joined = exposures.merge(
        first_assignment[["user_id", "arm"]].rename(columns={"arm": "assigned"}), on="user_id"
    )
    crossed = joined.loc[joined["arm"] != joined["assigned"], "user_id"].unique()
    share = len(crossed) / len(first_assignment)
    by_arm = (
        first_assignment.assign(crossed=first_assignment["user_id"].isin(crossed))
        .groupby("arm")["crossed"].mean().to_dict()
    )
    status = "pass" if share < 0.002 else ("warn" if share < 0.02 else "fail")
    return {"contaminated_users": len(crossed), "share": share, "by_arm": by_arm,
            "status": status}


def delayed_conversions(
    first_assignment: pd.DataFrame,
    conversions: pd.DataFrame,
    window_days: float,
    cutoff: pd.Timestamp,
) -> dict:
    """How long conversions take, and whether the data was pulled before they arrived.

    Users assigned less than one window before the cutoff haven't had their full
    chance to convert. Counting them as non-converters biases every rate down,
    and biases comparisons whenever arms enrolled at different times.
    """
    window_end = first_assignment["assigned_at"] + pd.Timedelta(days=window_days)
    immature = window_end > cutoff
    # Lags are measured only on users whose window had closed. An immature user's
    # slow conversions haven't happened yet, so including them biases lags short.
    measured = first_assignment[~immature] if (~immature).any() else first_assignment
    first_conversion = conversions.groupby("user_id")["converted_at"].min()
    joined = measured.set_index("user_id").join(first_conversion, how="inner")
    lag_days = (joined["converted_at"] - joined["assigned_at"]).dt.total_seconds() / 86_400
    after_window = float((lag_days > window_days).mean()) if len(lag_days) else 0.0
    quantiles = lag_days.quantile([0.5, 0.9, 0.99]).to_dict() if len(lag_days) else {}
    p99 = quantiles.get(0.99, 0.0)

    # Expected share of eventual conversions already visible for immature users,
    # from the empirical lag distribution of mature users.
    mature_lags = np.sort(lag_days[lag_days <= window_days].to_numpy())
    elapsed = (cutoff - first_assignment.loc[immature, "assigned_at"]).dt.total_seconds() / 86_400
    if len(mature_lags) and immature.any():
        visible = np.searchsorted(mature_lags, elapsed.to_numpy(), side="right") / len(mature_lags)
        expected_visible = float(visible.mean())
    else:
        expected_visible = 1.0

    if p99 > window_days or immature.mean() > 0:
        status = "warn"
    else:
        status = "pass"
    return {
        "median_lag_days": float(quantiles.get(0.5, 0.0)),
        "p90_lag_days": float(quantiles.get(0.9, 0.0)),
        "p99_lag_days": float(p99),
        "window_days": window_days,
        "share_after_window": after_window,
        "immature_users": int(immature.sum()),
        "immature_share": float(immature.mean()),
        "expected_share_visible_for_immature": expected_visible,
        "status": status,
    }


def leakage(
    first_assignment: pd.DataFrame,
    users: pd.DataFrame,
    converted: pd.Series,
    pre_treatment: Sequence[str] = PRE_TREATMENT,
    auc_limit: float = 0.8,
) -> dict:
    """Find user columns that can't really be pre-treatment.

    Randomisation balances everything that existed before assignment, so a
    column whose distribution differs by arm was shaped by the treatment. A
    column that predicts conversion almost perfectly is usually the outcome in
    disguise. Either kind would leak the answer into an uplift model or a
    regression adjustment.
    """
    frame = first_assignment[["user_id", "arm"]].merge(users, on="user_id")
    frame["converted"] = frame["user_id"].map(converted).fillna(0).astype(int)
    candidates = [
        c for c in users.columns
        if c not in {"user_id", "signup_date"} and users[c].nunique() > 1
    ]
    rows = []
    for column in candidates:
        values = frame[column]
        if values.dtype.kind in "biuf" and values.nunique() > 10:
            groups = [values[frame["arm"] == arm] for arm in frame["arm"].unique()]
            balance_p = float(stats.kruskal(*groups).pvalue)
            score = values.rank().to_numpy()
        else:
            table = pd.crosstab(values, frame["arm"])
            balance_p = float(stats.chi2_contingency(table)[1])
            score = values.map(frame.groupby(column)["converted"].mean()).to_numpy()
        auc = float(roc_auc_score(frame["converted"], score)) if frame["converted"].nunique() > 1 else 0.5
        auc = max(auc, 1 - auc)
        threshold = 0.001 / max(len(candidates), 1)
        reasons = []
        if balance_p < threshold:
            reasons.append("differs by arm, so it was recorded after assignment")
        if auc > auc_limit:
            reasons.append(f"predicts conversion with AUC {auc:.2f}")
        rows.append({"feature": column, "declared_pre_treatment": column in pre_treatment,
                     "balance_p": balance_p, "outcome_auc": auc, "leaky": bool(reasons),
                     "reasons": reasons})
    leaky = [row["feature"] for row in rows if row["leaky"]]
    declared = [f for f in leaky if f in pre_treatment]
    status = "fail" if declared else ("warn" if leaky else "pass")
    return {"features": rows, "leaky_features": leaky, "safe_features": [
        row["feature"] for row in rows if not row["leaky"]], "status": status}


def covariate_balance(first_assignment: pd.DataFrame, users: pd.DataFrame,
                      columns: Sequence[str] = PRE_TREATMENT) -> dict:
    frame = first_assignment[["user_id", "arm"]].merge(users, on="user_id")
    worst = {"smd": 0.0}
    min_p = 1.0
    control = frame["arm"] == "control"
    for column in columns:
        table = pd.crosstab(frame[column], frame["arm"])
        min_p = min(min_p, float(stats.chi2_contingency(table)[1]))
        for arm in table.columns:
            if arm == "control":
                continue
            for level in table.index:
                pa = (frame.loc[control, column] == level).mean()
                pb = (frame.loc[frame["arm"] == arm, column] == level).mean()
                sd = np.sqrt((pa * (1 - pa) + pb * (1 - pb)) / 2)
                smd = (pb - pa) / sd if sd > 0 else 0.0
                if abs(smd) > abs(worst["smd"]):
                    worst = {"column": column, "level": level, "arm": arm, "smd": float(smd)}
    status = "warn" if abs(worst["smd"]) > SMD_WARN or min_p < 0.001 else "pass"
    return {"worst": worst, "min_p": min_p, "status": status}


def run_integrity_checks(experiment: Experiment) -> dict:
    cfg = experiment.config
    allocation = dict(zip(cfg.arms, cfg.allocation, strict=True))
    first = experiment.assignments.sort_values("assigned_at").drop_duplicates("user_id")
    first_exposure = experiment.exposures.merge(
        first[["user_id", "arm"]], on=["user_id", "arm"]
    ).drop_duplicates("user_id")
    window_end = first.set_index("user_id")["assigned_at"] + pd.Timedelta(
        days=cfg.conversion_window_days
    )
    first_conversion = experiment.conversions.groupby("user_id")["converted_at"].min()
    converted = (first_conversion <= window_end.reindex(first_conversion.index)).astype(int)

    results = {
        "srm_assignment": srm(first["arm"].value_counts(), allocation),
        "srm_exposure": srm(first_exposure["arm"].value_counts(), allocation),
        "assignment_conflicts": assignment_conflicts(experiment.assignments),
        "missing_exposures": missing_exposures(first, experiment.exposures),
        "contamination": contamination(first, experiment.exposures),
        "delayed_conversions": delayed_conversions(
            first, experiment.conversions, cfg.conversion_window_days, pd.Timestamp(cfg.cutoff)
        ),
        "covariate_balance": covariate_balance(first, experiment.users),
        "leakage": leakage(first, experiment.users, converted),
    }
    checks = [
        Check("Sample ratio mismatch (assignment)", results["srm_assignment"]["status"],
              _srm_detail(results["srm_assignment"]), results["srm_assignment"]["p_value"]),
        Check("Sample ratio mismatch (exposure)", results["srm_exposure"]["status"],
              _srm_detail(results["srm_exposure"]), results["srm_exposure"]["p_value"]),
        Check("Assignment conflicts", results["assignment_conflicts"]["status"],
              f"{results['assignment_conflicts']['switched_arms']:,} users assigned to more than "
              f"one arm ({results['assignment_conflicts']['switched_share']:.2%})"),
        Check("Missing exposures", results["missing_exposures"]["status"], _missing_detail(
            results["missing_exposures"])),
        Check("Contamination", results["contamination"]["status"],
              f"{results['contamination']['contaminated_users']:,} users saw another arm "
              f"({results['contamination']['share']:.2%})"),
        Check("Delayed conversions", results["delayed_conversions"]["status"],
              _delay_detail(results["delayed_conversions"])),
        Check("Pre-treatment covariate balance", results["covariate_balance"]["status"],
              f"largest SMD {results['covariate_balance']['worst']['smd']:+.3f}, "
              f"min chi-square p={results['covariate_balance']['min_p']:.3f}"),
        Check("Feature leakage", results["leakage"]["status"],
              "no leaky features" if not results["leakage"]["leaky_features"]
              else "leaky: " + ", ".join(results["leakage"]["leaky_features"])),
    ]
    order = {"pass": 0, "info": 0, "warn": 1, "fail": 2}
    overall = max((c.status for c in checks), key=order.__getitem__)
    return {"overall": overall, "checks": [c.to_dict() for c in checks], "details": results}


def _srm_detail(result: dict) -> str:
    counts = ", ".join(f"{arm} {n:,}" for arm, n in result["counts"].items())
    return f"{counts}; chi-square p={result['p_value']:.4f}"


def _missing_detail(result: dict) -> str:
    worst = max(result["missing_by_arm"], key=result["missing_by_arm"].get)
    return (
        f"{result['missing_share']:.2%} of assigned users have no exposure "
        f"(worst: {worst} {result['missing_by_arm'][worst]:.2%}); "
        f"differs by arm p={result['differential_p']:.4f}"
    )


def _delay_detail(result: dict) -> str:
    text = (
        f"median lag {result['median_lag_days']:.1f} days, p99 {result['p99_lag_days']:.1f} "
        f"vs a {result['window_days']:g}-day window"
    )
    if result["immature_users"]:
        text += (
            f"; {result['immature_share']:.1%} of users had not finished their window when the "
            f"data was pulled (only about {result['expected_share_visible_for_immature']:.0%} of "
            "their conversions visible yet), so they are left out of the analysis"
        )
    return text
