"""Hierarchical Bayesian treatment effects with PyMC.

Splitting an experiment into many segments leaves most of them with a few
hundred users, where a raw lift estimate is mostly noise. This model estimates
every segment's effect at once and lets them share strength.

Data: conversions and users for each segment cell (for example market x device
x tenure) and each arm. On the log-odds scale,

    baseline_c       = a0 + sum_f a_f[level_f(c)] + e_c
    effect_k,c       = mu_k + sum_f b_k,f[level_f(c)]
                       + sum_(f,g) b_k,fg[level_f(c), level_g(c)] + d_k,c   (arm k vs control)
    conversions_k,c  ~ Binomial(users_k,c, sigmoid(baseline_c + effect_k,c))

Every term is drawn from a zero-centred normal whose scale is learned from the
data (non-centred parametrisation). The pairwise terms let an offer work for,
say, returning desktop shoppers only, without pretending it works for every
desktop shopper. A tiny cell's effect is pulled toward what
its market, its device and its tenure say, and toward the arm's overall mean,
by an amount that depends on how much evidence it has and on how much effects
actually vary across segments. Priors are weakly informative: they say a
typical offer moves log-odds by less than about 0.3 (about +/-35% relative)
and that baselines sit near the observed overall rate.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd

DEFAULT_ROPE = 0.02  # relative lifts within +/-2% count as "no practical difference"


@dataclass(frozen=True)
class Priors:
    effect_mean_sd: float = 0.3
    effect_factor_sd: float = 0.25
    effect_interaction_sd: float = 0.2
    effect_cell_sd: float = 0.1
    baseline_intercept_sd: float = 1.0
    baseline_factor_sd: float = 0.5
    baseline_cell_sd: float = 0.2


@dataclass(frozen=True)
class SamplerSettings:
    draws: int = 1_000
    tune: int = 1_500
    chains: int = 4
    target_accept: float = 0.99
    seed: int = 11
    backend: str = "auto"  # "numpyro", "pymc", or "auto" (numpyro when installed)


@dataclass
class HierarchicalFit:
    cells: pd.DataFrame
    factors: tuple[str, ...]
    arms: tuple[str, ...]
    idata: object
    rate_draws: np.ndarray  # (draws, cells, arms) posterior conversion probabilities
    prior_rate_draws: np.ndarray | None = None

    @property
    def control(self) -> str:
        return self.arms[0]


def _backend(name: str) -> str:
    if name != "auto":
        return name
    try:
        import numpyro  # noqa: F401
    except ImportError:
        return "pymc"
    return "numpyro"


def cells_from_frame(
    frame: pd.DataFrame,
    factors: Sequence[str],
    arm_column: str = "arm",
    outcome: str = "converted",
) -> pd.DataFrame:
    """Collapse user or session rows into one row per segment cell."""
    wide = (
        frame.groupby([*factors, arm_column], observed=True)[outcome]
        .agg(users="size", conversions="sum")
        .reset_index()
        .rename(columns={arm_column: "arm"})
    )
    return wide


def cells_from_sessions(sessions: pd.DataFrame, factors: Sequence[str]) -> pd.DataFrame:
    """Adapter for the bundled session datasets (group A/B, Conversion 0/1)."""
    frame = sessions.assign(
        arm=sessions["group"].map({"A": "control", "B": "variation"}),
        converted=sessions["Conversion"],
    )
    return cells_from_frame(frame, factors)


def _prepare(cells: pd.DataFrame, factors: Sequence[str], arms: Sequence[str]):
    table = cells.pivot_table(
        index=list(factors), columns="arm", values=["users", "conversions"], fill_value=0,
        observed=True, aggfunc="sum",
    )
    table = table.sort_index()
    users = np.column_stack([table["users"].get(arm, 0) for arm in arms]).astype(int)
    conversions = np.column_stack([table["conversions"].get(arm, 0) for arm in arms]).astype(int)
    index = table.index.to_frame(index=False)
    codes = {}
    levels = {}
    for factor in factors:
        categories = pd.Categorical(index[factor])
        codes[factor] = categories.codes.astype(int)
        levels[factor] = list(categories.categories)
    return index, users, conversions, codes, levels


def build_model(
    users: np.ndarray,
    conversions: np.ndarray,
    codes: dict[str, np.ndarray],
    levels: dict[str, list],
    priors: Priors,
    observed: bool = True,
):
    import pymc as pm

    n_cells, n_arms = users.shape
    pooled = (conversions.sum() + 0.5) / (users.sum() + 1.0)
    coords = {"cell": np.arange(n_cells), "arm": np.arange(n_arms), "treated": np.arange(1, n_arms)}
    coords |= {f"{factor}_level": levels[factor] for factor in codes}
    pairs = list(combinations(codes, 2))
    pair_codes = {}
    for first, second in pairs:
        width = len(levels[second])
        pair_codes[(first, second)] = codes[first] * width + codes[second]
        coords[f"{first}_x_{second}"] = np.arange(len(levels[first]) * width)

    with pm.Model(coords=coords) as model:
        a0 = pm.Normal("a0", np.log(pooled / (1 - pooled)), priors.baseline_intercept_sd)
        baseline = a0 * np.ones(n_cells)
        for factor, code in codes.items():
            sd = pm.HalfNormal(f"sd_base_{factor}", priors.baseline_factor_sd)
            z = pm.Normal(f"z_base_{factor}", 0.0, 1.0, dims=f"{factor}_level")
            baseline = baseline + sd * z[code]
        sd_cell = pm.HalfNormal("sd_base_cell", priors.baseline_cell_sd)
        z_cell = pm.Normal("z_base_cell", 0.0, 1.0, dims="cell")
        baseline = pm.Deterministic("baseline", baseline + sd_cell * z_cell, dims="cell")

        mu = pm.Normal("effect_mean", 0.0, priors.effect_mean_sd, dims="treated")
        effect = pm.math.ones((n_cells, 1)) * mu[None, :]
        for factor, code in codes.items():
            sd = pm.HalfNormal(f"sd_effect_{factor}", priors.effect_factor_sd, dims="treated")
            z = pm.Normal(f"z_effect_{factor}", 0.0, 1.0, dims=(f"{factor}_level", "treated"))
            effect = effect + (z * sd[None, :])[code]
        for (first, second), code in pair_codes.items():
            name = f"{first}_x_{second}"
            sd = pm.HalfNormal(f"sd_effect_{name}", priors.effect_interaction_sd, dims="treated")
            z = pm.Normal(f"z_effect_{name}", 0.0, 1.0, dims=(name, "treated"))
            effect = effect + (z * sd[None, :])[code]
        sd_effect_cell = pm.HalfNormal("sd_effect_cell", priors.effect_cell_sd, dims="treated")
        z_effect_cell = pm.Normal("z_effect_cell", 0.0, 1.0, dims=("cell", "treated"))
        effect = pm.Deterministic(
            "effect", effect + z_effect_cell * sd_effect_cell[None, :], dims=("cell", "treated")
        )

        logit = pm.math.concatenate([baseline[:, None], baseline[:, None] + effect], axis=1)
        rate = pm.Deterministic("rate", pm.math.sigmoid(logit), dims=("cell", "arm"))
        pm.Binomial(
            "conversions",
            n=users,
            p=rate,
            observed=conversions if observed else None,
            dims=("cell", "arm"),
        )
    return model


def fit(
    cells: pd.DataFrame,
    factors: Sequence[str],
    arms: Sequence[str],
    priors: Priors | None = None,
    sampler: SamplerSettings | None = None,
    prior_samples: int = 500,
) -> HierarchicalFit:
    """Fit the model, then draw prior and posterior predictive samples."""
    import pymc as pm

    priors = priors or Priors()
    sampler = sampler or SamplerSettings()
    arms = tuple(arms)
    factors = tuple(factors)
    index, users, conversions, codes, levels = _prepare(cells, factors, arms)
    model = build_model(users, conversions, codes, levels, priors)
    with model, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        idata = pm.sample(
            draws=sampler.draws,
            tune=sampler.tune,
            chains=sampler.chains,
            target_accept=sampler.target_accept,
            random_seed=sampler.seed,
            nuts_sampler=_backend(sampler.backend),
            progressbar=False,
            **({"cores": 1} if _backend(sampler.backend) == "pymc" else {}),
        )
        prior = pm.sample_prior_predictive(prior_samples, random_seed=sampler.seed)
        idata["prior"] = prior["prior"]
        idata["prior_predictive"] = prior["prior_predictive"]
        pm.sample_posterior_predictive(idata, random_seed=sampler.seed, extend_inferencedata=True,
                                       progressbar=False)

    rate = idata.posterior["rate"].stack(sample=("chain", "draw")).transpose("sample", "cell", "arm")
    prior_rate = idata.prior["rate"].stack(sample=("chain", "draw")).transpose("sample", "cell", "arm")
    index = index.assign(**{f"users_{arm}": users[:, k] for k, arm in enumerate(arms)})
    index = index.assign(
        **{f"conversions_{arm}": conversions[:, k] for k, arm in enumerate(arms)}
    )
    return HierarchicalFit(
        cells=index,
        factors=factors,
        arms=arms,
        idata=idata,
        rate_draws=rate.to_numpy(),
        prior_rate_draws=prior_rate.to_numpy(),
    )


# -- summaries ----------------------------------------------------------------


def _weights(fit: HierarchicalFit, mask: np.ndarray) -> np.ndarray:
    users = fit.cells[[f"users_{arm}" for arm in fit.arms]].to_numpy().sum(axis=1).astype(float)
    weights = np.where(mask, users, 0.0)
    if weights.sum() == 0:
        raise ValueError("no users in the requested segment")
    return weights / weights.sum()


def lift_draws(
    fit: HierarchicalFit,
    arm: str,
    mask: np.ndarray | None = None,
    draws: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Traffic-weighted control rate, arm rate and relative lift, one value per draw."""
    source = fit.rate_draws if draws is None else draws
    selected = np.ones(len(fit.cells), dtype=bool) if mask is None else np.asarray(mask)
    weights = _weights(fit, selected)
    control = source[:, :, 0] @ weights
    treated = source[:, :, fit.arms.index(arm)] @ weights
    return control, treated, treated / control - 1.0


def judge(relative_lift: np.ndarray, rope: float = DEFAULT_ROPE, threshold: float = 0.95) -> str:
    """Was the lift real? Decide with a region of practical equivalence.

    'positive' when at least `threshold` of the posterior says the lift beats +rope,
    'negative' when it says the lift is below -rope, 'no_effect' when that much of
    it sits inside the rope, and 'uncertain' otherwise.
    """
    if (relative_lift > rope).mean() >= threshold:
        return "positive"
    if (relative_lift < -rope).mean() >= threshold:
        return "negative"
    if (np.abs(relative_lift) <= rope).mean() >= threshold:
        return "no_effect"
    return "uncertain"


def summarize_lift(
    fit: HierarchicalFit,
    arm: str,
    mask: np.ndarray | None = None,
    rope: float = DEFAULT_ROPE,
    interval: float = 0.95,
) -> dict:
    control, treated, relative = lift_draws(fit, arm, mask)
    absolute = treated - control
    tail = (1 - interval) / 2
    return {
        "arm": arm,
        "control_rate": float(control.mean()),
        "arm_rate": float(treated.mean()),
        "abs_lift_mean": float(absolute.mean()),
        "rel_lift_mean": float(relative.mean()),
        "rel_lift_median": float(np.median(relative)),
        "rel_lift_ci": (float(np.quantile(relative, tail)), float(np.quantile(relative, 1 - tail))),
        "abs_lift_ci": (float(np.quantile(absolute, tail)), float(np.quantile(absolute, 1 - tail))),
        "prob_positive": float((relative > 0).mean()),
        "prob_above_rope": float((relative > rope).mean()),
        "prob_in_rope": float((np.abs(relative) <= rope).mean()),
        "rope": rope,
        "verdict": judge(relative, rope),
    }


def segment_table(
    fit: HierarchicalFit,
    by: Sequence[str] | None = None,
    rope: float = DEFAULT_ROPE,
) -> pd.DataFrame:
    """Posterior lift of every treated arm in every segment, next to the raw estimate."""
    group = list(by) if by is not None else list(fit.factors)
    rows = []
    keys = fit.cells[group].drop_duplicates().sort_values(group)
    for _, key in keys.iterrows():
        mask = np.logical_and.reduce([fit.cells[g].to_numpy() == key[g] for g in group])
        cells = fit.cells[mask]
        n_control = int(cells[f"users_{fit.control}"].sum())
        x_control = int(cells[f"conversions_{fit.control}"].sum())
        for arm in fit.arms[1:]:
            n_arm = int(cells[f"users_{arm}"].sum())
            x_arm = int(cells[f"conversions_{arm}"].sum())
            raw = (
                (x_arm / n_arm) / (x_control / n_control) - 1
                if n_arm and n_control and x_control
                else np.nan
            )
            summary = summarize_lift(fit, arm, mask, rope)
            rows.append(
                {
                    **{g: key[g] for g in group},
                    "arm": arm,
                    "users_control": n_control,
                    "users_arm": n_arm,
                    "raw_rel_lift": raw,
                    "rel_lift_mean": summary["rel_lift_mean"],
                    "ci_low": summary["rel_lift_ci"][0],
                    "ci_high": summary["rel_lift_ci"][1],
                    "prob_positive": summary["prob_positive"],
                    "verdict": summary["verdict"],
                }
            )
    return pd.DataFrame(rows)


def diagnostics(fit: HierarchicalFit) -> dict:
    import arviz as az

    # Check every sampled parameter by exact name. Deterministic transforms are left
    # out because they inherit their convergence from the parameters behind them.
    deterministic = {"rate", "baseline", "effect"}
    parameters = [name for name in fit.idata.posterior.data_vars if name not in deterministic]
    summary = az.summary(fit.idata, var_names=parameters, kind="diagnostics")
    divergences = int(fit.idata.sample_stats["diverging"].sum())
    max_rhat = float(summary["r_hat"].max())
    min_ess = float(summary["ess_bulk"].min())
    return {
        "max_rhat": max_rhat,
        "min_ess_bulk": min_ess,
        "divergences": divergences,
        "draws": int(fit.rate_draws.shape[0]),
        "parameters_checked": parameters,
        "converged": bool(max_rhat < 1.01 and min_ess > 400 and divergences == 0),
    }


def prior_predictive_check(fit: HierarchicalFit) -> dict:
    """Do the priors alone produce plausible conversion rates and lifts?"""
    draws = fit.prior_rate_draws
    if draws is None:
        raise ValueError("this fit has no prior draws")
    weights = _weights(fit, np.ones(len(fit.cells), dtype=bool))
    control = draws[:, :, 0] @ weights
    lifts = np.concatenate(
        [draws[:, :, k] @ weights / control - 1 for k in range(1, len(fit.arms))]
    )
    low, high = np.quantile(control, [0.05, 0.95])
    lift_low, lift_high = np.quantile(lifts, [0.05, 0.95])
    return {
        "control_rate_90": (float(low), float(high)),
        "rel_lift_90": (float(lift_low), float(lift_high)),
        "plausible": bool(high < 0.6 and abs(lift_low) < 1.0 and lift_high < 1.5),
    }


def posterior_predictive_check(fit: HierarchicalFit, interval: float = 0.9) -> dict:
    """Could the fitted model have produced the data we saw?

    Compares every cell's observed conversions with replicated datasets drawn from
    the posterior. Reports how many cells fall inside the predictive interval and a
    Bayesian p-value for a chi-square discrepancy, which should not be near 0 or 1.
    """
    replicated = (
        fit.idata.posterior_predictive["conversions"]
        .stack(sample=("chain", "draw"))
        .transpose("sample", "cell", "arm")
        .to_numpy()
    )
    observed = fit.cells[[f"conversions_{arm}" for arm in fit.arms]].to_numpy()
    users = fit.cells[[f"users_{arm}" for arm in fit.arms]].to_numpy()
    tail = (1 - interval) / 2
    low = np.quantile(replicated, tail, axis=0)
    high = np.quantile(replicated, 1 - tail, axis=0)
    covered = (observed >= low) & (observed <= high)
    present = users > 0

    expected = fit.rate_draws * users[None, :, :]
    variance = np.maximum(expected * (1 - fit.rate_draws), 1e-9)
    discrepancy_observed = (((observed[None] - expected) ** 2) / variance).sum(axis=(1, 2))
    discrepancy_replicated = (((replicated - expected) ** 2) / variance).sum(axis=(1, 2))
    p_value = float((discrepancy_replicated >= discrepancy_observed).mean())

    total_observed = observed.sum(axis=0)
    total_replicated = replicated.sum(axis=1)
    arm_p = {
        arm: float((total_replicated[:, k] >= total_observed[k]).mean())
        for k, arm in enumerate(fit.arms)
    }
    coverage = float(covered[present].mean())
    return {
        "interval": interval,
        "cell_coverage": coverage,
        "chi2_bayes_p": p_value,
        "arm_total_p": arm_p,
        "passed": bool(coverage >= interval - 0.05 and 0.05 < p_value < 0.95),
    }


def raw_cell_lifts(fit: HierarchicalFit, arm: str) -> np.ndarray:
    """No-pooling estimate: each cell's own observed lift (NaN where it can't be computed)."""
    control_rate = fit.cells[f"conversions_{fit.control}"] / fit.cells[f"users_{fit.control}"]
    arm_rate = fit.cells[f"conversions_{arm}"] / fit.cells[f"users_{arm}"]
    with np.errstate(divide="ignore", invalid="ignore"):
        return (arm_rate / control_rate - 1).to_numpy(dtype=float)


def posterior_cell_lifts(fit: HierarchicalFit, arm: str) -> np.ndarray:
    rates = fit.rate_draws
    return (rates[:, :, fit.arms.index(arm)] / rates[:, :, 0] - 1).mean(axis=0)


SENSITIVITY_PRIORS = {
    "default": Priors(),
    "skeptical": Priors(
        effect_mean_sd=0.1, effect_factor_sd=0.1, effect_interaction_sd=0.07, effect_cell_sd=0.05
    ),
    "wide": Priors(
        effect_mean_sd=1.0, effect_factor_sd=0.75, effect_interaction_sd=0.5, effect_cell_sd=0.3
    ),
}


def prior_sensitivity(
    cells: pd.DataFrame,
    factors: Sequence[str],
    arms: Sequence[str],
    priors: dict[str, Priors] | None = None,
    sampler: SamplerSettings | None = None,
    rope: float = DEFAULT_ROPE,
) -> pd.DataFrame:
    """Refit under skeptical and wide priors and see whether the conclusion moves.

    If an offer's verdict flips between priors, the data alone isn't strong enough
    to settle it and the prior is doing the work.
    """
    rows = []
    for name, prior in (priors or SENSITIVITY_PRIORS).items():
        result = fit(cells, factors, arms, priors=prior, sampler=sampler, prior_samples=50)
        for arm in result.arms[1:]:
            summary = summarize_lift(result, arm, rope=rope)
            rows.append(
                {
                    "prior": name,
                    "arm": arm,
                    "rel_lift_mean": summary["rel_lift_mean"],
                    "ci_low": summary["rel_lift_ci"][0],
                    "ci_high": summary["rel_lift_ci"][1],
                    "prob_positive": summary["prob_positive"],
                    "verdict": summary["verdict"],
                }
            )
    return pd.DataFrame(rows)
