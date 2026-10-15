"""Bayesian read-out of a conversion test (Beta-Binomial model).

Answers the questions stakeholders actually ask:
  * What is the probability B is better than A?
  * If we ship B and we're wrong, how much conversion do we expect to lose?
"""

from __future__ import annotations

import numpy as np


def beta_binomial(
    x_a: int,
    n_a: int,
    x_b: int,
    n_b: int,
    prior: tuple[float, float] = (1.0, 1.0),
    draws: int = 200_000,
    seed: int = 11,
    loss_threshold: float = 0.001,
) -> dict:
    """Posterior comparison with a Beta(prior) on each arm's rate.

    The default Beta(1, 1) prior is uniform (weakly informative). Expected
    loss is the average conversion-rate shortfall from choosing an arm when
    the other is actually better; a common decision rule is to ship when the
    expected loss of B falls below a small 'threshold of caring'.
    """
    rng = np.random.default_rng(seed)
    a0, b0 = prior
    post_a = rng.beta(a0 + x_a, b0 + n_a - x_a, draws)
    post_b = rng.beta(a0 + x_b, b0 + n_b - x_b, draws)

    diff = post_b - post_a
    rel = diff / post_a
    loss_b = np.maximum(post_a - post_b, 0).mean()
    loss_a = np.maximum(post_b - post_a, 0).mean()

    # thin samples for plotting so results stay small when serialised
    grid = np.linspace(
        min(np.quantile(post_a, 0.0005), np.quantile(post_b, 0.0005)),
        max(np.quantile(post_a, 0.9995), np.quantile(post_b, 0.9995)),
        200,
    )
    from scipy import stats

    dens_a = stats.beta.pdf(grid, a0 + x_a, b0 + n_a - x_a)
    dens_b = stats.beta.pdf(grid, a0 + x_b, b0 + n_b - x_b)

    return {
        "prior": prior,
        "prob_b_better": float((diff > 0).mean()),
        "expected_loss_b": float(loss_b),
        "expected_loss_a": float(loss_a),
        "loss_threshold": loss_threshold,
        "ship_b": bool(loss_b < loss_threshold),
        "rel_uplift_median": float(np.median(rel)),
        "rel_uplift_ci95": (float(np.quantile(rel, 0.025)), float(np.quantile(rel, 0.975))),
        "abs_diff_ci95": (float(np.quantile(diff, 0.025)), float(np.quantile(diff, 0.975))),
        "prob_uplift_gt_10pct": float((rel > 0.10).mean()),
        "density": {"x": grid.tolist(), "a": dens_a.tolist(), "b": dens_b.tolist()},
    }
