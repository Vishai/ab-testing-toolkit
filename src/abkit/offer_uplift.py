"""Which users respond to which offer: multi-arm uplift modelling.

`uplift.py` asks whether one variation can be targeted. With several offers the
question becomes: for each user, which offer (if any) raises their chance of
converting the most, and does sending everyone their best offer beat what the
aggregate A/B test recommends?

Method
  1. Cross-fitted outcome models. Users are split into folds. For each fold, one
     gradient-boosted classifier per arm is trained on the other folds and
     predicts every arm's conversion probability for the held-out users. Only
     pre-treatment features are used.
  2. Each user's predicted uplift from offer k is p_k(x) - p_control(x).
  3. The targeted policy gives each user the arm with the highest prediction.
  4. Because assignment was randomised with known probabilities, a policy's
     conversion rate can be estimated from the experiment itself (self-normalised
     inverse propensity weighting): the weighted conversion rate of users whose
     random arm happened to match what the policy would have chosen, with weight
     1 / P(arm). Every user's decision comes from models that never saw them, and
     a paired bootstrap gives the interval.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold

from .uplift import qini_coefficient, qini_curve


def _encode(frame: pd.DataFrame, features: Sequence[str]) -> np.ndarray:
    return np.column_stack(
        [pd.Categorical(frame[f]).codes for f in features]
    ).astype(float)


def _model(seed: int, categorical: np.ndarray) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        categorical_features=categorical,
        max_iter=150,
        learning_rate=0.05,
        max_leaf_nodes=15,
        min_samples_leaf=60,
        l2_regularization=1.0,
        random_state=seed,
    )


def cross_fitted_predictions(
    frame: pd.DataFrame,
    features: Sequence[str],
    arms: Sequence[str],
    folds: int = 5,
    seed: int = 3,
) -> np.ndarray:
    """Out-of-fold P(convert | x, arm) for every user and every arm."""
    x = _encode(frame, features)
    categorical = np.ones(x.shape[1], dtype=bool)
    y = frame["converted"].to_numpy()
    arm = frame["arm"].to_numpy()
    predictions = np.zeros((len(frame), len(arms)))
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    strata = pd.Series(arm).astype(str) + "_" + pd.Series(y).astype(str)
    for train, test in splitter.split(x, strata):
        for k, name in enumerate(arms):
            rows = train[arm[train] == name]
            model = _model(seed, categorical)
            model.fit(x[rows], y[rows])
            predictions[test, k] = model.predict_proba(x[test])[:, 1]
    return predictions


def policy_from_predictions(predictions: np.ndarray, arms: Sequence[str],
                            min_gain: float = 0.0) -> np.ndarray:
    """Best arm per user; stick with control unless an offer beats it by `min_gain`."""
    best = predictions.argmax(axis=1)
    gain = predictions[np.arange(len(best)), best] - predictions[:, 0]
    best = np.where(gain > min_gain, best, 0)
    return np.asarray(arms)[best]


def ipw_value(converted: np.ndarray, assigned: np.ndarray, chosen: np.ndarray,
              propensity: dict[str, float]) -> float:
    """Conversion rate of a policy, estimated from a randomised experiment.

    Self-normalised inverse propensity weighting: among users whose random arm
    matched what the policy would choose, a weighted conversion rate. Dividing by
    the realised weight instead of n means chance imbalance in arm sizes cancels
    out, and a fixed policy ("everyone gets offer k") gets exactly arm k's
    observed rate.
    """
    weights = (assigned == chosen) / np.array([propensity[a] for a in assigned])
    return float((weights * converted).sum() / weights.sum())


def evaluate_policies(
    frame: pd.DataFrame,
    policies: dict[str, np.ndarray],
    propensity: dict[str, float],
    baseline: str = "all_control",
    n_boot: int = 500,
    seed: int = 9,
) -> pd.DataFrame:
    """Self-normalised IPW conversion rate of each policy and its lift over the baseline.

    Intervals come from a paired bootstrap: every policy is re-estimated on the same
    resampled users, so the lift interval reflects the comparison, not two
    independent noises.
    """
    y = frame["converted"].to_numpy()
    assigned = frame["arm"].to_numpy()
    inverse = 1.0 / np.array([propensity[a] for a in assigned])
    weights = {name: (assigned == chosen) * inverse for name, chosen in policies.items()}

    def value(name: str, idx: np.ndarray | slice) -> float:
        w = weights[name][idx]
        return float((w * y[idx]).sum() / w.sum())

    rng = np.random.default_rng(seed)
    n = len(y)
    boot = {name: np.empty(n_boot) for name in policies}
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        for name in policies:
            boot[name][b] = value(name, idx)
    base = value(baseline, slice(None))
    rows = []
    for name in policies:
        estimate = value(name, slice(None))
        lift = boot[name] - boot[baseline]
        relative = boot[name] / boot[baseline] - 1
        rows.append(
            {
                "policy": name,
                "conversion_rate": estimate,
                "abs_lift": estimate - base,
                "abs_lift_ci": (float(np.quantile(lift, 0.025)), float(np.quantile(lift, 0.975))),
                "rel_lift": estimate / base - 1,
                "rel_lift_ci": (
                    float(np.quantile(relative, 0.025)),
                    float(np.quantile(relative, 0.975)),
                ),
                "prob_beats_baseline": float((lift > 0).mean()),
            }
        )
    return pd.DataFrame(rows)


def offer_qini(frame: pd.DataFrame, predictions: np.ndarray, arms: Sequence[str],
               n_perm: int = 200, seed: int = 5) -> list[dict]:
    """Per offer: does ranking users by predicted uplift beat random ranking?"""
    rng = np.random.default_rng(seed)
    results = []
    for k, arm in enumerate(arms[1:], start=1):
        mask = frame["arm"].isin([arms[0], arm]).to_numpy()
        y = frame.loc[mask, "converted"].to_numpy()
        t = (frame.loc[mask, "arm"] == arm).to_numpy().astype(int)
        score = (predictions[mask, k] - predictions[mask, 0])
        curve = qini_curve(y, t, score)
        coefficient = qini_coefficient(curve)
        permuted = np.array([
            qini_coefficient(qini_curve(y, t, rng.permutation(score))) for _ in range(n_perm)
        ])
        results.append(
            {
                "offer": arm,
                "qini_coefficient": coefficient,
                "perm_p_value": float((np.sum(permuted >= coefficient) + 1) / (n_perm + 1)),
                "curve": curve.to_dict(orient="list"),
            }
        )
    return results


def responder_table(frame: pd.DataFrame, predictions: np.ndarray, arms: Sequence[str],
                    by: Sequence[str]) -> pd.DataFrame:
    """Average predicted uplift of each offer per segment, and the offer the policy picks."""
    uplift = pd.DataFrame(
        predictions[:, 1:] - predictions[:, [0]], columns=list(arms[1:]), index=frame.index
    )
    joined = frame[list(by)].join(uplift)
    joined["best_offer"] = policy_from_predictions(predictions, arms)
    table = joined.groupby(list(by)).agg(
        users=(arms[1], "size"),
        **{f"uplift_{arm}": (arm, "mean") for arm in arms[1:]},
        best_offer=("best_offer", lambda s: s.mode().iat[0]),
    )
    return table.reset_index()


def run_offer_uplift(
    frame: pd.DataFrame,
    features: Sequence[str],
    arms: Sequence[str],
    propensity: dict[str, float],
    truth: pd.DataFrame | None = None,
    folds: int = 5,
    n_boot: int = 500,
    n_perm: int = 200,
    seed: int = 3,
) -> dict:
    """Fit, build policies, and evaluate them on the experiment itself.

    Policies compared: everyone gets control; everyone gets the same offer (one
    per offer); and the targeted policy. With `truth` (simulation only) every
    policy is also scored against the users' real conversion probabilities.
    """
    predictions = cross_fitted_predictions(frame, features, arms, folds, seed)
    n = len(frame)
    policies = {"all_control": np.full(n, arms[0], dtype=object)}
    policies |= {f"all_{arm}": np.full(n, arm, dtype=object) for arm in arms[1:]}
    policies["targeted"] = policy_from_predictions(predictions, arms)
    evaluation = evaluate_policies(frame, policies, propensity, n_boot=n_boot, seed=seed)

    result = {
        "features": list(features),
        "learner": "gradient-boosted outcome model per arm, cross-fitted",
        "policies": evaluation.to_dict(orient="records"),
        "targeted_allocation": pd.Series(policies["targeted"]).value_counts(normalize=True)
        .to_dict(),
        "qini": offer_qini(frame, predictions, arms, n_perm=n_perm, seed=seed),
        "responders": responder_table(frame, predictions, arms, features).to_dict(
            orient="records"
        ),
    }
    if truth is not None:
        true = truth.loc[frame["user_id"]].reset_index(drop=True)
        index = np.arange(n)
        values = {
            name: float(true.to_numpy()[index, [list(arms).index(a) for a in chosen]].mean())
            for name, chosen in policies.items()
        }
        values["oracle"] = float(true[list(arms)].max(axis=1).mean())
        predicted = predictions[:, 1:] - predictions[:, [0]]
        actual = true[list(arms[1:])].to_numpy() - true[[arms[0]]].to_numpy()
        result["truth"] = {
            "policy_conversion": values,
            "uplift_correlation": {
                arm: float(np.corrcoef(predicted[:, k], actual[:, k])[0, 1])
                for k, arm in enumerate(arms[1:])
            },
            "share_of_oracle_gain": float(
                (values["targeted"] - values["all_control"])
                / (values["oracle"] - values["all_control"])
            ),
        }
    return result
