"""Copy a finished pipeline run into reports/platform/ and draw the README figures.

Usage:
    python -m abkit.workflow run --run-id release
    python scripts/publish_platform_report.py artifacts/runs/release
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "platform"
COLORS = {"pct10_discount": "#2a78d6", "free_shipping": "#1b9e77", "bundle": "#eb6834"}
GREY = "#898781"


def _axes(title: str, width: float = 8, height: float = 4.2):
    fig, ax = plt.subplots(figsize=(width, height), dpi=130)
    ax.set_title(title, loc="left", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color="#e1e0d9")
    return fig, ax


def overall_posteriors(draws: dict, rope: float, path: Path) -> None:
    fig, ax = _axes("Overall lift of each offer: the aggregate effect is close to zero")
    for arm, values in draws.items():
        ax.hist(values, bins=60, density=True, alpha=0.55, color=COLORS.get(arm), label=arm)
    ax.axvspan(-rope, rope, color=GREY, alpha=0.15, label="no practical difference")
    ax.axvline(0, color=GREY, lw=1)
    ax.set_xlabel("relative lift vs control (posterior)")
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_yticks([])
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def free_shipping_by_market(hierarchical: dict, path: Path) -> None:
    rows = [r for r in hierarchical["segments"]["market_device"]
            if r["arm"] == "free_shipping" and r["device"] == "mobile"]
    rows.sort(key=lambda r: r["rel_lift_mean"])
    fig, ax = _axes("Free shipping on mobile, by market: pooled estimate vs raw", height=5)
    y = np.arange(len(rows))
    pooled = np.array([r["rel_lift_mean"] for r in rows])
    low = np.array([r["ci_low"] for r in rows])
    high = np.array([r["ci_high"] for r in rows])
    raw = np.array([r["raw_rel_lift"] for r in rows], dtype=float)
    colors = [COLORS["free_shipping"] if r["verdict"] == "positive" else GREY for r in rows]
    ax.errorbar(pooled, y, xerr=[pooled - low, high - pooled], fmt="none", ecolor=colors, lw=2)
    ax.scatter(pooled, y, color=colors, zorder=3, label="hierarchical (95% interval)")
    ax.scatter(raw, y, marker="x", color="#444441", zorder=3, label="raw market estimate")
    ax.set_yticks(y, [f"{r['market']}  (n={r['users_arm']:,})" for r in rows], fontsize=8)
    ax.axvline(0, color=GREY, lw=1)
    ax.set_xlim(-0.8, 1.6)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("relative lift vs control")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def policy_lift(uplift: dict, path: Path) -> None:
    policies = uplift["policies"][1:]
    fig, ax = _axes("Conversion lift over control, measured on held-out users")
    names = [p["policy"].replace("all_", "everyone: ") for p in policies]
    values = np.array([p["rel_lift"] for p in policies])
    low = np.array([p["rel_lift_ci"][0] for p in policies])
    high = np.array([p["rel_lift_ci"][1] for p in policies])
    colors = [COLORS.get(p["policy"].replace("all_", ""), "#7b3fbf") for p in policies]
    y = np.arange(len(policies))
    ax.barh(y, values, color=colors, alpha=0.85)
    ax.errorbar(values, y, xerr=[values - low, high - values], fmt="none", ecolor="#444441")
    ax.set_yticks(y, names)
    ax.axvline(0, color=GREY, lw=1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def bundled_shrinkage(bundled: dict, path: Path) -> None:
    fig, ax = _axes("Four bundled A/B tests: headline lift vs hierarchical estimate", height=3.4)
    items = list(bundled.values())
    y = np.arange(len(items))
    pooled = np.array([i["overall"]["rel_lift_mean"] for i in items])
    low = np.array([i["overall"]["rel_lift_ci"][0] for i in items])
    high = np.array([i["overall"]["rel_lift_ci"][1] for i in items])
    ax.errorbar(pooled, y, xerr=[pooled - low, high - pooled], fmt="o", color="#2a78d6",
                label="hierarchical (95% interval)")
    ax.scatter([i["raw_rel_lift"] for i in items], y, marker="x", color="#eb6834", zorder=3,
               label="headline lift")
    ax.set_yticks(y, [i["title"].split(" - ")[0] for i in items])
    ax.axvline(0, color=GREY, lw=1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(__doc__)
        return 2
    run = Path(argv[0])
    bundle = json.loads((run / "dashboard.json").read_text(encoding="utf-8"))
    figures = OUT / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    for name in ("report.md", "dashboard.json", "manifest.json", "plan.json"):
        shutil.copyfile(run / name, OUT / name)
    draws = dict(np.load(run / "lift_draws.npz"))
    overall_posteriors(draws, bundle["hierarchical"]["overall"][0]["rope"],
                       figures / "overall_posteriors.png")
    free_shipping_by_market(bundle["hierarchical"], figures / "free_shipping_by_market.png")
    policy_lift(bundle["uplift"], figures / "policy_lift.png")
    bundled_shrinkage(bundle["bundled"], figures / "bundled_shrinkage.png")
    print(f"Wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
