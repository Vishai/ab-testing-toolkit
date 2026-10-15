"""Run the full analysis for every scenario and write all outputs.

Usage:
    python scripts/run_analysis.py            # all scenarios
    python scripts/run_analysis.py --monthly-sessions 50000

Outputs:
    reports/results.json   - every number, machine-readable
    reports/summary.md     - human-readable summary per scenario
    reports/figures/*.png  - charts used in the README and report
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from abkit import Settings, analyze_all  # noqa: E402
from abkit.pipeline import save_json  # noqa: E402
from abkit.plots import all_figures  # noqa: E402
from abkit.report import summary_markdown  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--target-mde", type=float, default=0.20, help="relative lift to plan for")
    ap.add_argument("--bounce-margin", type=float, default=0.02)
    ap.add_argument("--daily-sessions", type=int, default=1_000)
    ap.add_argument("--monthly-sessions", type=int, default=30_000)
    ap.add_argument("--out", type=Path, default=ROOT / "reports")
    args = ap.parse_args()

    settings = Settings(
        alpha=args.alpha,
        target_mde_rel=args.target_mde,
        bounce_margin=args.bounce_margin,
        daily_sessions=args.daily_sessions,
        monthly_sessions=args.monthly_sessions,
    )
    results = analyze_all(settings)

    args.out.mkdir(parents=True, exist_ok=True)
    save_json(results, args.out / "results.json")
    (args.out / "summary.md").write_text(summary_markdown(results))
    figs = all_figures(results, args.out / "figures")

    for k, r in results.items():
        p = r["primary"]
        print(f"{k:20s} {p['rate_a']:.2%} -> {p['rate_b']:.2%}  uplift {p['uplift']:+.0%}  "
              f"p={p['p_value']:.4f}  => {r['decision']['label']}")
    print(f"\nWrote results.json, summary.md and {len(figs)} figures to {args.out}")


if __name__ == "__main__":
    main()
