"""Interactive A/B test dashboard.

Run from the repo root:
    python -m streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from abkit import SCENARIOS, Settings, analyze  # noqa: E402
from abkit import data as abdata  # noqa: E402
from abkit import power as abpower  # noqa: E402
from abkit.pipeline import to_jsonable  # noqa: E402
from abkit.scenarios import Scenario  # noqa: E402

A_COLOR, B_COLOR, MUTED, GRID = "#2a78d6", "#eb6834", "#898781", "#e1e0d9"
STATUS = {"pass": "✅", "warn": "⚠️", "fail": "⛔", "info": "ℹ️"}
VERDICT = {"ship": "success", "hold": "warning", "extend": "info", "reject": "error", "investigate": "error"}

st.set_page_config(page_title="A/B Test Analyzer", page_icon="🧪", layout="wide")


def style(fig: go.Figure, height: int = 360) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=30, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=1.08, x=0),
        font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif", size=13),
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


@st.cache_data(show_spinner="Running analysis...")
def run(key: str, alpha: float, target: float, margin: float, daily: int, monthly: int, upload: bytes | None):
    settings = Settings(alpha=alpha, target_mde_rel=target, bounce_margin=margin,
                        daily_sessions=daily, monthly_sessions=monthly, peeking_sims=10_000,
                        uplift_permutations=300)
    if upload is not None:
        import io
        df = pd.read_csv(io.BytesIO(upload))
        missing = [c for c in abdata.REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Uploaded file is missing columns: {missing}")
        df["group"] = df["AB_Group"].map(abdata.GROUP_MAP)
        df["treatment"] = (df["group"] == "B").astype(int)
        df["Revenue"] = df["Revenue"].astype(float)
        sc = Scenario(key="custom", title="Your uploaded test", file="upload.csv",
                      funnel=tuple(sorted(df.PageViewed.unique())), problem="(uploaded data)",
                      hypothesis="(uploaded data)", change="(uploaded data)",
                      revenue_label="Revenue" if (df.Revenue > 0).any() else None)
        return analyze(sc, df=df, settings=settings), df
    df = abdata.load(key)
    return analyze(key, df=df, settings=settings), df


# ---------------- Sidebar ----------------
st.sidebar.title("🧪 A/B Test Analyzer")
labels = {k: v.title for k, v in SCENARIOS.items()}
key = st.sidebar.selectbox("Scenario", list(labels), format_func=labels.get, index=1)
with st.sidebar.expander("Analyse your own test (CSV)"):
    st.caption("Same columns as the sample data: " + ", ".join(abdata.REQUIRED_COLUMNS))
    up = st.file_uploader("Upload CSV", type="csv")
st.sidebar.markdown("### Test settings")
alpha = st.sidebar.select_slider("Significance level (α)", [0.01, 0.05, 0.10], value=0.05)
target = st.sidebar.slider("Smallest lift worth detecting", 0.05, 0.50, 0.20, 0.05, format="%.2f")
margin = st.sidebar.slider("Bounce-rate guardrail margin (pp)", 0.5, 5.0, 2.0, 0.5) / 100
st.sidebar.markdown("### Business assumptions")
daily = st.sidebar.number_input("Daily sessions", 100, 1_000_000, 1_000, 100)
monthly = st.sidebar.number_input("Monthly sessions", 1_000, 10_000_000, 30_000, 1_000)

try:
    res, df = run(key, alpha, target, margin, int(daily), int(monthly), up.getvalue() if up else None)
except Exception as e:  # pragma: no cover - UI path
    st.error(str(e))
    st.stop()

sc, p, b, d = res["scenario"], res["primary"], res["bayes"], res["decision"]

# ---------------- Header ----------------
st.title(sc["title"])
st.caption(sc["problem"])
getattr(st, VERDICT[d["verdict"]])(f"**Decision: {d['label']}**  \n" + "  \n".join(d["reasons"]))

c1, c2, c3, c4 = st.columns(4)
c1.metric("Conversion A → B", f"{p['rate_b']:.2%}", f"{p['abs_diff'] * 100:+.2f} pp vs {p['rate_a']:.2%}")
c2.metric("Relative uplift", f"{p['uplift']:+.0%}", f"95% CI {p['uplift_ci'][0]:+.0%} to {p['uplift_ci'][1]:+.0%}",
          delta_color="off")
c3.metric("p-value", f"{p['p_value']:.4f}", "significant" if p["significant"] else "not significant",
          delta_color="normal" if p["significant"] else "off")
loss = b["expected_loss_b"] * 100
c4.metric("P(B beats A)", f"{b['prob_b_better']:.1%}",
          f"loss if wrong {'<0.001' if loss < 0.001 else f'{loss:.3f}'} pp",
          delta_color="off")

if d["caveats"]:
    with st.expander(f"⚠️ {len(d['caveats'])} caveat(s) before you ship", expanded=True):
        for c in d["caveats"]:
            st.markdown(f"- {c}")

tabs = st.tabs(["Trust checks", "Results", "Segments", "Uplift model", "Test planner", "Data"])

# ---------------- Trust ----------------
with tabs[0]:
    st.subheader("Can we trust this experiment?")
    st.caption("Run before reading any result. A significant uplift on a broken test is not a win.")
    chk = pd.DataFrame(res["quality"]["checks"])
    chk.insert(0, "", chk.status.map(STATUS))
    st.dataframe(chk[["", "name", "detail"]], hide_index=True, width="stretch")
    srm = res["quality"]["srm"]
    fig = go.Figure(go.Bar(x=["A - Control", "B - Variation"], y=[srm["n_a"], srm["n_b"]],
                           marker_color=[A_COLOR, B_COLOR], text=[srm["n_a"], srm["n_b"]], textposition="outside",
                           width=0.4))
    fig.add_hline(y=(srm["n_a"] + srm["n_b"]) / 2, line_color=MUTED, annotation_text="expected 50/50", annotation_position="top left")
    st.plotly_chart(style(fig, 300), width="stretch")
    bal = pd.DataFrame(res["quality"]["balance"])
    st.markdown("**Pre-treatment covariate balance** (|SMD| > 0.1 = notable)")
    st.dataframe(bal.style.format({"share_a": "{:.1%}", "share_b": "{:.1%}", "smd": "{:+.3f}", "chi2_p": "{:.3f}"}),
                 hide_index=True, width="stretch")

# ---------------- Results ----------------
with tabs[1]:
    left, right = st.columns(2)
    with left:
        st.subheader("Conversion by funnel page")
        f = pd.DataFrame(res["funnel"])
        fig = go.Figure()
        for g, col, name in (("A", A_COLOR, "A - Control"), ("B", B_COLOR, "B - Variation")):
            s = f[f.group == g]
            fig.add_bar(x=s.page.str.replace("_", " "), y=s.conversion * 100, name=name, marker_color=col,
                        hovertemplate="%{x}<br>%{y:.1f}%<extra>" + name + "</extra>")
        fig.update_yaxes(title="Conversion (%)")
        st.plotly_chart(style(fig), width="stretch")
    with right:
        st.subheader("Posterior conversion rates")
        dens = b["density"]
        fig = go.Figure()
        fig.add_scatter(x=np.array(dens["x"]) * 100, y=dens["a"], name="A - Control", fill="tozeroy",
                        line=dict(color=A_COLOR, width=2))
        fig.add_scatter(x=np.array(dens["x"]) * 100, y=dens["b"], name="B - Variation", fill="tozeroy",
                        line=dict(color=B_COLOR, width=2))
        fig.update_yaxes(showticklabels=False)
        fig.update_xaxes(title="Conversion rate (%)")
        st.plotly_chart(style(fig), width="stretch")

    st.subheader("All metrics")
    rows = [["Conversion rate (primary)", f"{p['rate_a']:.2%}", f"{p['rate_b']:.2%}", f"{p['uplift']:+.1%}",
             f"{p['p_value']:.4f}", "two-proportion z"]]
    t = res["secondary"]["time_on_page"]
    rows.append(["Time on page (s)", f"{t['mean_a']:.1f}", f"{t['mean_b']:.1f}", f"{t['uplift']:+.1%}",
                 f"{t['p_value']:.4f}", "Welch t"])
    if "revenue_per_visitor" in res["secondary"]:
        r = res["secondary"]["revenue_per_visitor"]
        rows.append(["Revenue per visitor", f"{r['mean_a']:,.2f}", f"{r['mean_b']:,.2f}", f"{r['uplift']:+.1%}",
                     f"{r['p_value']:.4f}", "bootstrap"])
        a = res["secondary"]["avg_order_value"]
        rows.append(["Avg value per conversion", f"{a['mean_a']:,.2f}", f"{a['mean_b']:,.2f}", f"{a['uplift']:+.1%}",
                     f"{a['p_value']:.4f}", "Welch t"])
    g = res["guardrails"]["bounce"]
    rows.append(["Bounce rate (guardrail)", f"{g['rate_a']:.1%}", f"{g['rate_b']:.1%}",
                 f"{g['abs_diff'] * 100:+.1f} pp", "PASS" if g["passed"] else "FAIL",
                 f"non-inferiority, +{g['margin'] * 100:.1f} pp margin"])
    st.dataframe(pd.DataFrame(rows, columns=["Metric", "A", "B", "Change", "p / result", "Test"]),
                 hide_index=True, width="stretch")

    adj = res["adjusted"]
    st.caption(f"Robustness: covariate-adjusted logistic regression gives OR {adj['odds_ratio']:.2f} "
               f"(95% CI {adj['or_ci'][0]:.2f}–{adj['or_ci'][1]:.2f}, p={adj['p_value']:.4f}).")

    imp = res["impact"]
    st.subheader("Business impact (illustrative)")
    i1, i2 = st.columns(2)
    i1.metric(f"Extra conversions / month @ {imp['monthly_sessions_assumed']:,} sessions",
              f"{imp['extra_conversions_per_month']:,.0f}",
              f"95% CI {imp['extra_conversions_ci'][0]:,.0f} – {imp['extra_conversions_ci'][1]:,.0f}", delta_color="off")
    if "extra_revenue_per_month" in imp:
        i2.metric("Extra revenue / month", f"{imp['extra_revenue_per_month']:,.0f}",
                  f"95% CI {imp['extra_revenue_ci'][0]:,.0f} – {imp['extra_revenue_ci'][1]:,.0f}", delta_color="off")

# ---------------- Segments ----------------
with tabs[2]:
    st.subheader("Segment effects (multiple-testing corrected)")
    st.caption("Filled markers survive Holm correction. Grey rows are page context, which B itself can change, "
               "so they are exploratory only.")
    s = pd.DataFrame(res["segments"]).iloc[::-1]
    s["label"] = s.variable + ": " + s.level.astype(str).str.replace("_", " ") + "  (n=" + (s.n_a + s.n_b).astype(str) + ")"
    fig = go.Figure()
    for _, r in s.iterrows():
        col = B_COLOR if r.kind == "pre-treatment" else MUTED
        fig.add_scatter(x=[r.abs_ci_lo * 100, r.abs_ci_hi * 100], y=[r.label] * 2, mode="lines",
                        line=dict(color=col, width=2), showlegend=False, hoverinfo="skip")
        fig.add_scatter(x=[r.abs_diff * 100], y=[r.label], mode="markers", showlegend=False,
                        marker=dict(size=11, color=col if r.sig_holm else "white", line=dict(color=col, width=2)),
                        hovertemplate=f"{r.label}<br>A {r.rate_a:.1%} → B {r.rate_b:.1%}<br>"
                                      f"diff %{{x:+.1f}} pp<br>raw p {r.p_raw:.3f} · Holm p {r.p_holm:.3f}<extra></extra>")
    fig.add_vline(x=0, line_color=MUTED)
    fig.update_xaxes(title="B − A (percentage points), 95% CI")
    st.plotly_chart(style(fig, 60 + 38 * len(s)), width="stretch")
    het = pd.DataFrame(res["heterogeneity"])
    st.markdown("**Interaction tests** (does B's effect differ across a segment?)")
    st.dataframe(het, hide_index=True, width="stretch")

# ---------------- Uplift ----------------
with tabs[3]:
    u = res["uplift_model"]
    st.subheader("Who is most persuaded by B?")
    st.caption(f"{u['learner']} on pre-treatment features ({', '.join(u['features'])}). "
               "The Qini curve compares targeting by predicted uplift against random targeting.")
    c = pd.DataFrame(u["curve"])
    fig = go.Figure()
    fig.add_scatter(x=c.fraction * 100, y=c.qini, name="Model ranking", line=dict(color=B_COLOR, width=2))
    fig.add_scatter(x=c.fraction * 100, y=c.random, name="Random", line=dict(color=MUTED, width=1.5))
    fig.update_xaxes(title="Share of visitors targeted (%)")
    fig.update_yaxes(title="Incremental conversions")
    st.plotly_chart(style(fig), width="stretch")
    if u["signal"]:
        st.success(f"Targeting signal detected (permutation p={u['perm_p_value']:.3f}).")
    else:
        st.info(f"No reliable targeting signal (permutation p={u['perm_p_value']:.3f}). "
                "Roll out to everyone rather than personalise.")
    seg = pd.DataFrame(u["segments"])
    st.dataframe(seg.style.format({c: "{:.1%}" for c in seg.columns if c.startswith(("p_", "obs_", "pred_"))}),
                 hide_index=True, width="stretch")

# ---------------- Planner ----------------
with tabs[4]:
    st.subheader("Plan the next test")
    pc1, pc2, pc3 = st.columns(3)
    base = pc1.number_input("Baseline conversion (%)", 0.1, 90.0, round(p["rate_a"] * 100, 2), 0.1) / 100
    mde = pc2.number_input("Minimum detectable lift (%)", 1.0, 200.0, target * 100, 1.0) / 100
    pw = pc3.select_slider("Power", [0.7, 0.8, 0.9, 0.95], value=0.8)
    try:
        n = abpower.sample_size_per_arm(base, mde, alpha, pw)
        days = abpower.runtime_days(n, daily)
        m1, m2, m3 = st.columns(3)
        m1.metric("Sessions per arm", f"{n:,}")
        m2.metric("Total sessions", f"{2 * n:,}")
        m3.metric("Runtime", f"{days} days", f"at {daily:,}/day, min. 2 weeks", delta_color="off")
    except ValueError as e:
        st.error(str(e))

    pw_info = res["power"]
    wc = pw_info["winners_curse"]
    st.markdown(
        f"**This test:** ~{pw_info['n_per_arm']:,} per arm could only reliably detect a "
        f"**{pw_info['mde_rel_at_n']:+.0%}** lift. If the true lift were {wc['assumed_true_lift']:.0%}, "
        f"power would be {wc['power']:.0%} and significant results would overstate it "
        f"~{wc['exaggeration_ratio']:.1f}x (winner's curse)."
    )

    st.subheader("Why you can't peek")
    looks = st.slider("How many times will you check the results?", 1, 20, 10)
    pk = abpower.peeking_simulation(base, n, looks=looks, alpha=alpha, sims=10_000)
    bl = pd.DataFrame(pk["by_looks"])
    fig = go.Figure(go.Bar(x=bl.looks, y=bl.fpr * 100, marker_color=A_COLOR,
                           hovertemplate="%{x} looks: %{y:.1f}% false positives<extra></extra>"))
    fig.add_hline(y=alpha * 100, line_color="#d03b3b", annotation_text=f"promised {alpha:.0%}", annotation_position="top left")
    fig.update_xaxes(title="Times results were checked", dtick=1)
    fig.update_yaxes(title="False-positive rate (%)")
    st.plotly_chart(style(fig, 320), width="stretch")
    st.caption(f"Stopping at the first p<{alpha} over {looks} looks gives a {pk['naive_fpr']:.0%} false-positive "
               f"rate. Using p<{pk['corrected_nominal_alpha']:.4f} at each look keeps it at "
               f"{pk['corrected_fpr']:.0%}.")

# ---------------- Data ----------------
with tabs[5]:
    st.subheader("Raw sessions")
    st.dataframe(df.drop(columns=["group", "treatment"], errors="ignore"), width="stretch", height=420)
    st.download_button("Download full results (JSON)",
                       data=json.dumps(to_jsonable(res), indent=2),
                       file_name=f"{sc['key']}_results.json", mime="application/json")

st.sidebar.markdown("---")
st.sidebar.caption("Built with `abkit` · data: MIS5308 simulated A/B datasets")
