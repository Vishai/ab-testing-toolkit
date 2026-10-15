"""Static figures (matplotlib) for the README and written report.

Palette: the two leading slots of a CVD-validated categorical palette
(Control = blue, Variation = orange), hairline grids, thin marks, and
direct labels used sparingly.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

A_COLOR = "#2a78d6"   # Control
B_COLOR = "#eb6834"   # Variation
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"
CRITICAL = "#d03b3b"
WARNING_INK = "#8a5a00"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "legend.frameon": False,
        "figure.dpi": 110,
        "savefig.dpi": 160,
        "savefig.bbox": "tight",
    }
)

NICE = {
    "online_education": "Online education",
    "travel_booking": "Travel booking",
    "charity_donations": "Charity donations",
    "free_trial": "Free trial (SaaS)",
}


def _subtitle(ax, text):
    ax.text(0, 1.02, text, transform=ax.transAxes, color=INK_2, fontsize=9, va="bottom")


def _save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def uplift_forest(results: dict, path) -> Path:
    """Relative uplift with 95% CI for every scenario, one row each."""
    keys = list(results)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.grid(axis="y", visible=False)
    for i, k in enumerate(keys[::-1]):
        p = results[k]["primary"]
        lo, hi = p["uplift_ci"]
        ax.plot([lo * 100, hi * 100], [i, i], color=B_COLOR, lw=2, solid_capstyle="round")
        ax.plot(p["uplift"] * 100, i, "o", ms=9, color=B_COLOR, mec=SURFACE, mew=2)
        verdict = results[k]["decision"]["verdict"]
        tag = {"ship": "Ship", "hold": "Hold - SRM", "extend": "Extend", "reject": "Keep A",
               "investigate": "Investigate"}[verdict]
        color = INK_2 if verdict == "ship" else WARNING_INK
        ax.text(hi * 100 + 6, i, f"{p['uplift']:+.0%}   p={p['p_value']:.3f}   {tag}",
                va="center", fontsize=9, color=color)
    ax.axvline(0, color=AXIS, lw=1)
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels([NICE.get(k, k) for k in keys[::-1]], color=INK)
    ax.set_xlabel("Relative uplift in conversion, B vs A (%)")
    ax.set_xlim(-20, max(results[k]["primary"]["uplift_ci"][1] for k in keys) * 100 + 90)
    ax.set_title("Variation B lifts conversion in all four tests", pad=22)
    _subtitle(ax, "Point estimate with 95% CI. Wide intervals = small samples; read the lower bound.")
    return _save(fig, path)


def funnel_bars(res: dict, path) -> Path:
    f = pd.DataFrame(res["funnel"])
    pages = res["scenario"]["funnel"]
    x = np.arange(len(pages))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.grid(axis="x", visible=False)
    for j, (g, c, lab) in enumerate((("A", A_COLOR, "A - Control"), ("B", B_COLOR, "B - Variation"))):
        vals = [f[(f.page == p) & (f.group == g)].conversion.iloc[0] * 100 for p in pages]
        ax.bar(x + (j - 0.5) * (w + 0.02), vals, width=w, color=c, label=lab, edgecolor=SURFACE, lw=0)
        if g == "B":
            for xi, v in zip(x, vals):
                ax.text(xi + (j - 0.5) * (w + 0.02), v + 0.3, f"{v:.1f}%", ha="center", fontsize=8, color=INK_2)
    ax.set_xticks(x)
    ax.set_xticklabels([p.replace("_", " ") for p in pages], color=INK)
    ax.set_ylabel("Conversion rate (%)")
    ax.legend(loc="upper left", ncols=2, bbox_to_anchor=(0, 1.0))
    ax.set_ylim(0, max(ax.get_ylim()[1], 5) * 1.15)
    ax.set_title(f"{res['scenario']['title']}: conversion by funnel page", pad=22)
    _subtitle(ax, "Sessions grouped by the page they were recorded on.")
    return _save(fig, path)


def posterior(res: dict, path) -> Path:
    d = res["bayes"]["density"]
    x = np.array(d["x"]) * 100
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.grid(axis="x", visible=False)
    ax.fill_between(x, d["a"], color=A_COLOR, alpha=0.18, lw=0)
    ax.plot(x, d["a"], color=A_COLOR, lw=2, label="A - Control")
    ax.fill_between(x, d["b"], color=B_COLOR, alpha=0.18, lw=0)
    ax.plot(x, d["b"], color=B_COLOR, lw=2, label="B - Variation")
    ax.set_yticks([])
    ax.set_xlabel("Conversion rate (%)")
    ax.legend(loc="upper right")
    b = res["bayes"]
    ax.set_title(f"P(B beats A) = {b['prob_b_better']:.1%}", pad=22)
    loss = b["expected_loss_b"] * 100
    loss_txt = "<0.001" if loss < 0.001 else f"{loss:.3f}"
    _subtitle(ax, f"Beta-Binomial posteriors, uniform prior. Expected loss if we ship B: "
                  f"{loss_txt} pp of conversion.")
    return _save(fig, path)


def segment_forest(res: dict, path) -> Path:
    s = pd.DataFrame(res["segments"])
    s = s.iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(s) + 1.4))
    ax.grid(axis="y", visible=False)
    for i, r in s.iterrows():
        color = B_COLOR if r.kind == "pre-treatment" else MUTED
        ax.plot([r.abs_ci_lo * 100, r.abs_ci_hi * 100], [i, i], color=color, lw=2, solid_capstyle="round")
        ax.plot(r.abs_diff * 100, i, "o", ms=8, color=color if r.sig_holm else SURFACE, mec=color, mew=2)
        # right-margin annotation, positioned in axes coordinates
        ax.text(1.01, i, f"Holm p={r.p_holm:.2f}", transform=ax.get_yaxis_transform(),
                ha="left", va="center", fontsize=8, color=INK_2)
    ax.axvline(0, color=AXIS, lw=1)
    ax.set_yticks(range(len(s)))
    ax.set_yticklabels([f"{r.variable}: {str(r.level).replace('_', ' ')}  (n={r.n_a + r.n_b})" for r in s.itertuples()],
                       color=INK, fontsize=9)
    ax.set_xlabel("Conversion difference, B - A (percentage points), 95% CI")
    ax.set_title("Segment effects after multiple-testing correction", pad=22)
    _subtitle(ax, "Filled = significant after Holm correction. Grey = page context (post-treatment, exploratory).")
    return _save(fig, path)


def qini(res: dict, path) -> Path:
    u = res["uplift_model"]
    c = pd.DataFrame(u["curve"])
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.plot(c.fraction * 100, c.qini, color=B_COLOR, lw=2, label="T-learner ranking")
    ax.plot(c.fraction * 100, c.random, color=MUTED, lw=1.2, label="Random targeting")
    ax.set_xlabel("Share of visitors targeted, highest predicted uplift first (%)")
    ax.set_ylabel("Incremental conversions")
    ax.legend(loc="upper left")
    verdict = "real targeting signal" if u["signal"] else "no reliable targeting signal"
    ax.set_title(f"Uplift model: {verdict}", pad=22)
    _subtitle(ax, f"Cross-fitted Qini; permutation p = {u['perm_p_value']:.2f} "
                  f"(n={res['n_sessions']:,}, features: {', '.join(u['features'])}).")
    return _save(fig, path)


def peeking(res: dict, path) -> Path:
    pk = res["peeking"]
    b = pd.DataFrame(pk["by_looks"])
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.grid(axis="x", visible=False)
    ax.bar(b.looks, b.fpr * 100, width=0.62, color=A_COLOR, lw=0)
    ax.axhline(pk["alpha"] * 100, color=CRITICAL, lw=1.2)
    ax.set_xlim(0.4, pk["looks"] + 1.6)
    ax.text(pk["looks"] + 0.5, pk["alpha"] * 100, " promised\n 5%", color=CRITICAL, fontsize=8,
            ha="left", va="center")
    last = b.iloc[-1]
    ax.text(last.looks, last.fpr * 100 + 0.5, f"{last.fpr:.0%}", ha="center", fontsize=9, color=INK)
    ax.set_xticks(b.looks)
    ax.set_xlabel("Number of times results were checked before stopping")
    ax.set_ylabel("False-positive rate (%)")
    ax.set_title("Peeking turns a 5% error rate into ~19%", pad=22)
    _subtitle(ax, f"{pk['sims']:,} simulated A/A tests. Fix: pre-set the sample size, or use a stricter "
                  f"per-look threshold (p < {pk['corrected_nominal_alpha']:.3f} for {pk['looks']} looks).")
    return _save(fig, path)


def power_curves(results: dict, path) -> Path:
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    for i, (k, r) in enumerate(results.items()):
        c = pd.DataFrame(r["power"]["power_curve"])
        ax.plot(c.mde_rel * 100, c.n_per_arm, color=SLOTS[i], lw=2,
                label=f"{NICE[k]} (baseline {r['power']['baseline']:.1%})")
    ax.axhline(600, color=MUTED, lw=1)
    ax.text(100, 640, "this test: ~600 per arm", ha="right", fontsize=8, color=INK_2)
    ax.set_yscale("log")
    ax.set_xlabel("Minimum detectable relative lift (%)")
    ax.set_ylabel("Sessions needed per arm (log)")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title("How big a test do we need?", pad=22)
    mdes = [r["power"]["mde_rel_at_n"] for r in results.values()]
    _subtitle(ax, f"Two-sided alpha = 0.05, power = 80%. ~600 per arm only reliably detects lifts of "
                  f"{min(mdes):.0%}-{max(mdes):.0%}.")
    return _save(fig, path)


def all_figures(results: dict, out_dir) -> list[Path]:
    out = Path(out_dir)
    paths = [uplift_forest(results, out / "overview_uplift.png"), power_curves(results, out / "power_curves.png")]
    first = next(iter(results.values()))
    paths.append(peeking(first, out / "peeking_simulation.png"))
    for k, r in results.items():
        paths += [
            funnel_bars(r, out / f"{k}_funnel.png"),
            posterior(r, out / f"{k}_posterior.png"),
            segment_forest(r, out / f"{k}_segments.png"),
            qini(r, out / f"{k}_qini.png"),
        ]
    return paths
