"""Uplift modelling: who is most persuaded by variation B?

A standard A/B test answers "is B better on average?". An uplift model
estimates the *conditional* treatment effect - how much B changes the
conversion probability for a given kind of visitor - so B (or a tailored
variant) can be targeted where it helps most.

Method: a T-learner (Kunzel et al., 2019). One classifier is fitted on
control sessions and one on variation sessions; predicted uplift is the
difference of their probabilities. Predictions are cross-fitted (each
session is scored by models that never saw it) and evaluated with a Qini
curve against a permutation baseline, so we can tell real targeting signal
from noise.

Only pre-treatment features are used. Variables that the variation itself can
change (page reached, time on page, bounce) would leak the outcome.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder


def _encoder(df: pd.DataFrame, features) -> OneHotEncoder:
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    enc.fit(df[list(features)])
    return enc


def _fit_t_learner(train: pd.DataFrame, features, enc, C: float):
    models = {}
    for t in (0, 1):
        part = train[train.treatment == t]
        m = LogisticRegression(C=C, max_iter=1000)
        m.fit(enc.transform(part[list(features)]), part.Conversion)
        models[t] = m
    return models


def _predict(models, X) -> np.ndarray:
    return models[1].predict_proba(X)[:, 1] - models[0].predict_proba(X)[:, 1]


def qini_curve(y: np.ndarray, t: np.ndarray, score: np.ndarray, points: int = 101) -> pd.DataFrame:
    """Cumulative incremental conversions when targeting by descending score."""
    order = np.argsort(-score, kind="stable")
    y, t = y[order], t[order]
    n = len(y)
    cum_t = np.cumsum(t)
    cum_c = np.cumsum(1 - t)
    cum_yt = np.cumsum(y * t)
    cum_yc = np.cumsum(y * (1 - t))
    with np.errstate(divide="ignore", invalid="ignore"):
        q = cum_yt - np.where(cum_c > 0, cum_yc * cum_t / cum_c, 0.0)
    idx = np.unique(np.linspace(0, n - 1, points).astype(int))
    frac = (idx + 1) / n
    return pd.DataFrame({"fraction": frac, "qini": q[idx], "random": q[-1] * frac})


def qini_coefficient(curve: pd.DataFrame) -> float:
    """Area between the model's Qini curve and random targeting (normalised by n=1)."""
    gain = curve.qini - curve.random
    return float(np.trapezoid(gain, curve.fraction))


def fit_uplift(
    df: pd.DataFrame,
    features=("Device", "UserType"),
    folds: int = 5,
    C: float = 1.0,
    n_perm: int = 500,
    seed: int = 3,
) -> dict:
    features = list(features)
    enc = _encoder(df, features)
    X_all = enc.transform(df[features])

    # cross-fitted uplift scores
    scores = np.zeros(len(df))
    strat = df.treatment.astype(str) + "_" + df.Conversion.astype(str)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for tr, te in skf.split(df, strat):
        models = _fit_t_learner(df.iloc[tr], features, enc, C)
        scores[te] = _predict(models, X_all[te])

    y = df.Conversion.to_numpy()
    t = df.treatment.to_numpy()
    curve = qini_curve(y, t, scores)
    q = qini_coefficient(curve)

    # permutation baseline: how good does a *random* ranking look?
    rng = np.random.default_rng(seed)
    perm = np.array([qini_coefficient(qini_curve(y, t, rng.permutation(scores))) for _ in range(n_perm)])
    p_perm = float((np.sum(perm >= q) + 1) / (n_perm + 1))

    # full-data model for the segment table (what we'd act on)
    models = _fit_t_learner(df, features, enc, C)
    cells = df[features].drop_duplicates().sort_values(features).reset_index(drop=True)
    Xc = enc.transform(cells)
    cells["p_control"] = models[0].predict_proba(Xc)[:, 1]
    cells["p_variation"] = models[1].predict_proba(Xc)[:, 1]
    cells["pred_uplift"] = cells.p_variation - cells.p_control

    obs = (
        df.groupby(features + ["group"]).Conversion.agg(["mean", "size"]).unstack("group").reset_index()
    )
    obs.columns = features + ["obs_rate_a", "obs_rate_b", "n_a", "n_b"]
    cells = cells.merge(obs, on=features, how="left")
    cells["obs_uplift"] = cells.obs_rate_b - cells.obs_rate_a
    cells = cells.sort_values("pred_uplift", ascending=False).reset_index(drop=True)

    # targeting simulation: treat only top-k% by predicted uplift
    top = []
    for share in (0.2, 0.4, 0.6, 0.8, 1.0):
        row = curve.iloc[(curve.fraction - share).abs().argmin()]
        top.append({"share_targeted": share, "incremental_conversions": float(row.qini),
                    "share_of_total_gain": float(row.qini / curve.qini.iloc[-1]) if curve.qini.iloc[-1] else np.nan})

    return {
        "features": features,
        "learner": "T-learner (L2 logistic regression), 5-fold cross-fitted",
        "qini_coefficient": q,
        "perm_mean": float(perm.mean()),
        "perm_p_value": p_perm,
        "signal": bool(p_perm < 0.05),
        "curve": curve.to_dict(orient="list"),
        "segments": cells.to_dict(orient="records"),
        "targeting": top,
    }
