"""Generate a Markdown summary from pipeline results."""

from __future__ import annotations

ICON = {"pass": "PASS", "warn": "WARN", "fail": "FAIL", "info": "INFO"}


def _pct(x, d=1):
    return "n/a" if x is None else f"{x * 100:.{d}f}%"


def _ci(ci, pct=True, d=0):
    lo, hi = ci
    if pct:
        return f"{lo * 100:+.{d}f}% to {hi * 100:+.{d}f}%"
    return f"{lo:+,.2f} to {hi:+,.2f}"


def scenario_markdown(r: dict) -> str:
    sc, p, b, pw = r["scenario"], r["primary"], r["bayes"], r["power"]
    d = r["decision"]
    lines = [
        f"## {sc['title']}",
        "",
        f"**Decision: {d['label']}**",
        "",
        *[f"- {x}" for x in d["reasons"]],
        "",
        f"**Problem.** {sc['problem']}",
        "",
        f"**Hypothesis.** {sc['hypothesis']}",
        "",
        "### Trust checks",
        "",
        "| Check | Status | Detail |",
        "|---|---|---|",
        *[f"| {c['name']} | {ICON[c['status']]} | {c['detail']} |" for c in r["quality"]["checks"]],
        "",
        "### Primary KPI - conversion rate",
        "",
        "| | Control (A) | Variation (B) |",
        "|---|---|---|",
        f"| Sessions | {p['n_a']:,} | {p['n_b']:,} |",
        f"| Conversions | {p['x_a']:,} | {p['x_b']:,} |",
        f"| Conversion rate | {_pct(p['rate_a'], 2)} | {_pct(p['rate_b'], 2)} |",
        "",
        f"- Absolute lift: **{p['abs_diff'] * 100:+.2f} pp** (95% CI {_ci(p['abs_ci'], d=2).replace('%', ' pp')})",
        f"- Relative uplift: **{p['uplift']:+.1%}** (95% CI {_ci(p['uplift_ci'])})",
        f"- Two-proportion z-test p = {p['p_value']:.4f}; Fisher exact p = {p['fisher_p']:.4f}; Cohen's h = {p['cohens_h']:.3f}",
        f"- Covariate-adjusted logistic regression: OR = {r['adjusted']['odds_ratio']:.2f} "
        f"(95% CI {r['adjusted']['or_ci'][0]:.2f}-{r['adjusted']['or_ci'][1]:.2f}), p = {r['adjusted']['p_value']:.4f}",
        f"- Bayesian: P(B > A) = **{b['prob_b_better']:.1%}**, expected loss of shipping B = {b['expected_loss_b'] * 100:.4f} pp",
        "",
        "### Secondary KPIs and guardrails",
        "",
    ]
    t = r["secondary"]["time_on_page"]
    lines.append(
        f"- Time on page: {t['mean_a']:.1f}s -> {t['mean_b']:.1f}s ({t['uplift']:+.1%}, Welch p = {t['p_value']:.4f})"
    )
    if "revenue_per_visitor" in r["secondary"]:
        rv = r["secondary"]["revenue_per_visitor"]
        lines.append(
            f"- Revenue per visitor: {rv['mean_a']:,.2f} -> {rv['mean_b']:,.2f} ({rv['uplift']:+.1%}; "
            f"bootstrap 95% CI for difference {_ci(rv['abs_ci'], pct=False)})"
        )
        aov = r["secondary"]["avg_order_value"]
        lines.append(
            f"- Average value per conversion: {aov['mean_a']:,.2f} -> {aov['mean_b']:,.2f} "
            f"({aov['uplift']:+.1%}, Welch p = {aov['p_value']:.3f})"
        )
    g = r["guardrails"]["bounce"]
    lines.append(
        f"- Guardrail - bounce rate: {_pct(g['rate_a'])} -> {_pct(g['rate_b'])}; non-inferiority margin "
        f"+{g['margin'] * 100:.0f} pp, one-sided upper bound {g['one_sided_bound'] * 100:+.2f} pp -> "
        f"**{'PASS' if g['passed'] else 'FAIL'}**"
    )
    wc = pw["winners_curse"]
    lines += [
        "",
        "### Power and design",
        "",
        f"- Baseline conversion {_pct(pw['baseline'], 2)}; ~{pw['n_per_arm']:,} sessions per arm.",
        f"- Minimum detectable effect at this size (80% power): **{pw['mde_rel_at_n']:+.0%}** relative.",
        f"- Sessions per arm needed to detect a {pw['target_mde_rel']:.0%} lift: **{pw['n_for_target']:,}** "
        f"(~{pw['runtime_days_for_target']} days at the assumed {r['settings']['daily_sessions']:,} sessions/day).",
        f"- Winner's curse: if the true lift were {wc['assumed_true_lift']:.0%}, this test would have "
        f"{wc['power']:.0%} power and significant results would overstate it ~{wc['exaggeration_ratio']:.1f}x.",
        "",
        "### Segments (multiple-testing corrected)",
        "",
        "| Segment | n | A | B | Diff (pp) | raw p | Holm p | BH q |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for s in r["segments"]:
        tag = "" if s["kind"] == "pre-treatment" else " *(exploratory)*"
        lines.append(
            f"| {s['variable']}: {s['level']}{tag} | {s['n_a'] + s['n_b']} | {_pct(s['rate_a'])} | {_pct(s['rate_b'])} | "
            f"{s['abs_diff'] * 100:+.1f} | {s['p_raw']:.3f} | {s['p_holm']:.3f} | {s['p_bh']:.3f} |"
        )
    lines += [
        "",
        "Interaction tests (does the effect differ by segment?): "
        + "; ".join(f"{h['covariate']} p = {h['p_value']:.3f}" for h in r["heterogeneity"]),
        "",
        "### Uplift model",
        "",
        f"- {r['uplift_model']['learner']} on {', '.join(r['uplift_model']['features'])}.",
        f"- Qini coefficient {r['uplift_model']['qini_coefficient']:.2f} vs permutation mean "
        f"{r['uplift_model']['perm_mean']:.2f}; permutation p = {r['uplift_model']['perm_p_value']:.3f} -> "
        + ("**targeting signal detected**." if r["uplift_model"]["signal"]
           else "**no reliable targeting signal** - roll out to everyone rather than personalise."),
        "",
    ]
    if d["caveats"]:
        lines += ["### Caveats", "", *[f"- {c}" for c in d["caveats"]], ""]
    return "\n".join(lines)


def summary_markdown(results: dict) -> str:
    head = [
        "# Results summary",
        "",
        "Auto-generated by `scripts/run_analysis.py`. Every number below is in `results.json`.",
        "",
        "| Scenario | A | B | Uplift (95% CI) | p | P(B>A) | SRM | Decision |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results.values():
        p = r["primary"]
        head.append(
            f"| {r['scenario']['title']} | {_pct(p['rate_a'])} | {_pct(p['rate_b'])} | "
            f"{p['uplift']:+.0%} ({_ci(p['uplift_ci'])}) | {p['p_value']:.4f} | {r['bayes']['prob_b_better']:.1%} | "
            f"{ICON[r['quality']['srm']['status']]} | {r['decision']['label']} |"
        )
    return "\n".join(head) + "\n\n" + "\n\n".join(scenario_markdown(r) for r in results.values())
