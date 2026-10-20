"""Posterior effects dashboard for product and engineering.

Reads finished pipeline runs, so nothing heavy runs in the app:
    python -m abkit.workflow run --run-id demo
    python -m streamlit run app/posterior_dashboard.py

ABKIT_RUNS_DIR points at the runs folder (default artifacts/runs).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

RUNS_DIR = Path(os.environ.get("ABKIT_RUNS_DIR", Path(__file__).resolve().parents[1] / "artifacts" / "runs"))
COLORS = {"pct10_discount": "#2a78d6", "free_shipping": "#1b9e77", "bundle": "#eb6834",
          "variation": "#2a78d6", "control": "#898781", "targeted": "#7b3fbf"}
VERDICT = {"positive": "Real lift", "negative": "Real drop", "no_effect": "No practical effect",
           "uncertain": "Not settled"}
GROUPINGS = {"Tenure": "tenure", "Device": "device", "Market": "market",
             "Device x tenure": "device_tenure", "Market x device": "market_device"}

st.set_page_config(page_title="Experiment posterior effects", layout="wide")


def style(fig: go.Figure, height: int = 380) -> go.Figure:
    fig.update_layout(height=height, margin={"l": 10, "r": 10, "t": 30, "b": 10},
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                      legend={"orientation": "h", "y": 1.1, "x": 0})
    fig.update_xaxes(gridcolor="#e1e0d9", zerolinecolor="#898781")
    fig.update_yaxes(gridcolor="#e1e0d9")
    return fig


def available_runs() -> list[Path]:
    if not RUNS_DIR.exists():
        return []
    runs = [p for p in RUNS_DIR.iterdir() if (p / "dashboard.json").exists()]
    return sorted(runs, key=lambda p: (p / "dashboard.json").stat().st_mtime, reverse=True)


@st.cache_data
def load_run(path: str) -> tuple[dict, dict[str, np.ndarray]]:
    folder = Path(path)
    bundle = json.loads((folder / "dashboard.json").read_text(encoding="utf-8"))
    draws_path = folder / "lift_draws.npz"
    draws = dict(np.load(draws_path)) if draws_path.exists() else {}
    return bundle, draws


def segment_label(row: dict) -> str:
    return " / ".join(str(row[k]) for k in ("market", "device", "tenure") if k in row)


def recommendations(hierarchical: dict) -> pd.DataFrame:
    """Per device x tenure segment: the offer with a settled positive lift, if any."""
    rows = pd.DataFrame(hierarchical["segments"]["device_tenure"])
    out = []
    for (device, tenure), group in rows.groupby(["device", "tenure"]):
        winners = group[group.verdict == "positive"].sort_values("rel_lift_mean", ascending=False)
        avoid = group[group.verdict == "negative"].arm.tolist()
        if len(winners):
            best = winners.iloc[0]
            action = (f"Show {best.arm}", (f"{best.rel_lift_mean:+.0%} "
                      f"({best.ci_low:+.0%} to {best.ci_high:+.0%})"))
        else:
            action = ("Keep control, or explore", "no offer settled yet")
        out.append({"Segment": f"{device} / {tenure}", "Action": action[0],
                    "Expected lift (95% interval)": action[1],
                    "Avoid": ", ".join(avoid) or "-"})
    return pd.DataFrame(out)


# -- sidebar ------------------------------------------------------------------

runs = available_runs()
st.sidebar.title("Posterior effects")
if not runs:
    st.title("No pipeline runs yet")
    st.info(f"Run the pipeline first, then reload:\n\n`python -m abkit.workflow run --run-id demo`\n\n"
            f"Looking in `{RUNS_DIR}`.")
    st.stop()

selected = st.sidebar.selectbox("Run", runs, format_func=lambda p: p.name)
bundle, lift_draws = load_run(str(selected))
manifest = bundle.get("manifest") or {}
st.sidebar.caption(f"Config `{manifest.get('config_fingerprint', '?')}` · "
                   f"commit `{(manifest.get('git_commit') or 'unknown')[:8]}`")

hier = bundle.get("hierarchical")
uplift = bundle.get("uplift")
integrity = bundle.get("integrity")
aggregate = bundle.get("aggregate")

st.title(f"Experiment readout · {bundle['run_id']}")
tabs = st.tabs(["Summary", "Posterior effects", "Who responds to what", "Trust and diagnostics",
                "Bundled A/B tests"])

# -- summary (product) ------------------------------------------------------------

with tabs[0]:
    cols = st.columns(3)
    if integrity:
        cols[0].metric("Trust checks", integrity["overall"].upper())
    if aggregate:
        winners = [o["offer"] for o in aggregate["offers"] if o["significant"]]
        cols[1].metric("Aggregate A/B winner", ", ".join(winners) or "none",
                       help="Each offer vs control, Dunnett-adjusted, as the launch plan specified.")
    if uplift:
        targeted = next(p for p in uplift["policies"] if p["policy"] == "targeted")
        cols[2].metric("Targeted policy lift", f"{targeted['rel_lift']:+.1%}",
                       help=f"95% interval {targeted['rel_lift_ci'][0]:+.1%} to "
                            f"{targeted['rel_lift_ci'][1]:+.1%}, on held-out users")
    st.subheader("What we learned")
    for sentence in bundle.get("findings", []):
        st.markdown(f"- {sentence}")
    if hier:
        st.subheader("Recommended action by segment")
        st.caption("From the hierarchical model: an offer is recommended only when 95% of its "
                   "posterior says it beats control by more than the practical threshold.")
        st.dataframe(recommendations(hier), hide_index=True, width="stretch")

# -- posterior effects ------------------------------------------------------------

with tabs[1]:
    if not hier:
        st.info("This run has no hierarchical model output.")
    else:
        rope = hier["overall"][0]["rope"]
        st.subheader("Overall lift, full posterior")
        fig = go.Figure()
        for arm, draws in lift_draws.items():
            fig.add_trace(go.Histogram(x=draws, name=arm, opacity=0.6, nbinsx=60,
                                       histnorm="probability density",
                                       marker_color=COLORS.get(arm)))
        fig.add_vrect(x0=-rope, x1=rope, fillcolor="#898781", opacity=0.15, line_width=0,
                      annotation_text="no practical difference", annotation_position="top left")
        fig.update_layout(barmode="overlay", xaxis_tickformat="+.0%",
                          xaxis_title="relative lift vs control")
        st.plotly_chart(style(fig), width="stretch")
        overall = pd.DataFrame(hier["overall"])
        overall["95% interval"] = overall.rel_lift_ci.map(lambda ci: f"{ci[0]:+.1%} to {ci[1]:+.1%}")
        overall["verdict"] = overall.verdict.map(VERDICT)
        st.dataframe(
            overall[["arm", "rel_lift_mean", "95% interval", "prob_positive", "prob_in_rope",
                     "verdict"]].rename(columns={"rel_lift_mean": "lift", "prob_positive": "P(lift > 0)",
                                                 "prob_in_rope": "P(no practical difference)"}),
            hide_index=True, width="stretch",
            column_config={"lift": st.column_config.NumberColumn(format="percent"),
                           "P(lift > 0)": st.column_config.NumberColumn(format="percent"),
                           "P(no practical difference)": st.column_config.NumberColumn(format="percent")},
        )

        st.subheader("By segment: pooled estimate vs raw")
        left, right = st.columns(2)
        grouping = GROUPINGS[left.selectbox("Segment by", list(GROUPINGS), index=3)]
        arm = right.selectbox("Offer", hier["arms"][1:])
        table = pd.DataFrame(hier["segments"][grouping])
        table = table[table.arm == arm].copy()
        table["segment"] = [segment_label(r) for r in table.to_dict("records")]
        table = table.sort_values("rel_lift_mean")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=table.rel_lift_mean, y=table.segment, mode="markers", name="pooled (hierarchical)",
            marker={"color": COLORS.get(arm, "#2a78d6"), "size": 9},
            error_x={"type": "data", "symmetric": False, "array": table.ci_high - table.rel_lift_mean,
                         "arrayminus": table.rel_lift_mean - table.ci_low}))
        fig.add_trace(go.Scatter(x=table.raw_rel_lift, y=table.segment, mode="markers",
                                 name="raw segment estimate",
                                 marker={"symbol": "x", "color": "#898781", "size": 8}))
        fig.add_vline(x=0, line_color="#898781")
        fig.update_layout(xaxis_tickformat="+.0%", xaxis_title="relative lift vs control")
        st.plotly_chart(style(fig, height=max(320, 26 * len(table))), width="stretch")
        st.caption("Raw estimates in small segments swing wildly. The pooled estimate borrows "
                   "strength from related segments and shrinks them toward what the data "
                   "supports.")
        shown = table[["segment", "users_arm", "raw_rel_lift", "rel_lift_mean", "ci_low", "ci_high",
                       "prob_positive", "verdict"]].copy()
        shown["verdict"] = shown.verdict.map(VERDICT)
        st.dataframe(shown.sort_values("rel_lift_mean", ascending=False), hide_index=True,
                     width="stretch")

# -- targeting -------------------------------------------------------------------

with tabs[2]:
    if not uplift:
        st.info("This run has no uplift model output.")
    else:
        policies = pd.DataFrame(uplift["policies"])
        fig = go.Figure(go.Bar(
            x=policies.policy, y=policies.rel_lift,
            marker_color=[COLORS.get(p.replace("all_", ""), "#7b3fbf") for p in policies.policy],
            error_y={"type": "data", "symmetric": False,
                         "array": [ci[1] - v for ci, v in zip(policies.rel_lift_ci, policies.rel_lift, strict=True)],
                         "arrayminus": [v - ci[0] for ci, v in zip(policies.rel_lift_ci, policies.rel_lift, strict=True)]}))
        fig.update_layout(yaxis_tickformat="+.0%", yaxis_title="conversion lift vs all control")
        st.subheader("Policies, scored on users the models never trained on")
        st.plotly_chart(style(fig), width="stretch")
        allocation = uplift["targeted_allocation"]
        st.caption("Targeted policy sends: " + ", ".join(
            f"{arm} {share:.0%}" for arm, share in sorted(allocation.items(), key=lambda kv: -kv[1])))

        st.subheader("Ranking signal per offer (Qini)")
        fig = go.Figure()
        for row in uplift["qini"]:
            curve = row["curve"]
            fig.add_trace(go.Scatter(x=curve["fraction"], y=curve["qini"], name=
                                     f"{row['offer']} (p={row['perm_p_value']:.3f})",
                                     line={"color": COLORS.get(row["offer"])}))
            fig.add_trace(go.Scatter(x=curve["fraction"], y=curve["random"], showlegend=False,
                                     line={"color": COLORS.get(row["offer"]), "dash": "dot"}))
        fig.update_layout(xaxis_title="share of users targeted, best first",
                          yaxis_title="incremental conversions", xaxis_tickformat=".0%")
        st.plotly_chart(style(fig), width="stretch")

        st.subheader("Predicted uplift by segment")
        responders = pd.DataFrame(uplift["responders"])
        st.dataframe(responders, hide_index=True, width="stretch")

# -- trust and diagnostics (engineering) -----------------------------------------------

with tabs[3]:
    if integrity:
        st.subheader("Trust checks")
        checks = pd.DataFrame(integrity["checks"])
        checks["status"] = checks.status.str.upper()
        st.dataframe(checks[["name", "status", "detail"]], hide_index=True,
                     width="stretch")
    if hier:
        st.subheader("Sampler and model checks")
        d, ppc, prior = hier["diagnostics"], hier["posterior_check"], hier["prior_check"]
        cols = st.columns(4)
        cols[0].metric("Max R-hat", f"{d['max_rhat']:.3f}", help="Should be below 1.01")
        cols[1].metric("Min bulk ESS", f"{d['min_ess_bulk']:,.0f}", help="Should be above 400")
        cols[2].metric("Divergences", d["divergences"])
        cols[3].metric("Posterior predictive", "passed" if ppc["passed"] else "failed",
                       help=f"{ppc['cell_coverage']:.0%} of cells inside the 90% interval, "
                            f"chi-square Bayesian p {ppc['chi2_bayes_p']:.2f}")
        st.caption(f"Prior predictive: 90% of prior control rates between "
                   f"{prior['control_rate_90'][0]:.1%} and {prior['control_rate_90'][1]:.1%}.")
        if "prior_sensitivity" in hier:
            st.subheader("Prior sensitivity")
            st.dataframe(pd.DataFrame(hier["prior_sensitivity"]), hide_index=True,
                         width="stretch")
        if "truth_check" in hier:
            st.subheader("Against the simulator's truth")
            st.dataframe(pd.DataFrame(hier["truth_check"]).T.rename_axis("offer").reset_index(),
                         hide_index=True, width="stretch")
    if manifest:
        st.subheader("Run manifest")
        tasks = pd.DataFrame(manifest["tasks"]).T.rename_axis("task").reset_index()
        st.dataframe(tasks, hide_index=True, width="stretch")
        st.json({k: manifest.get(k) for k in ("config_fingerprint", "git_commit", "python",
                                              "packages", "input_hashes")}, expanded=False)

# -- bundled datasets ---------------------------------------------------------------

with tabs[4]:
    bundled = bundle.get("bundled")
    if not bundled:
        st.info("This run has no bundled dataset output.")
    else:
        rows = []
        for result in bundled.values():
            overall = result["overall"]
            rows.append({"test": result["title"], "raw lift": result["raw_rel_lift"],
                         "pooled lift": overall["rel_lift_mean"],
                         "low": overall["rel_lift_ci"][0], "high": overall["rel_lift_ci"][1],
                         "P(lift > 0)": overall["prob_positive"],
                         "verdict": VERDICT[overall["verdict"]]})
        frame = pd.DataFrame(rows)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=frame["pooled lift"], y=frame.test, mode="markers", name="hierarchical",
            marker={"size": 10, "color": "#2a78d6"},
            error_x={"type": "data", "symmetric": False, "array": frame.high - frame["pooled lift"],
                         "arrayminus": frame["pooled lift"] - frame.low}))
        fig.add_trace(go.Scatter(x=frame["raw lift"], y=frame.test, mode="markers",
                                 name="raw headline", marker={"symbol": "x", "size": 10,
                                                                  "color": "#eb6834"}))
        fig.update_layout(xaxis_tickformat="+.0%", xaxis_title="relative lift")
        st.plotly_chart(style(fig, height=300), width="stretch")
        st.caption("The headline lifts from these small tests are inflated by noise (the "
                   "winner's curse). The hierarchical estimates are pulled back toward what "
                   "the device and user-type cells support together.")
        st.dataframe(frame, hide_index=True, width="stretch")
