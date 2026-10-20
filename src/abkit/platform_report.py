"""Turn a pipeline run folder into a dashboard bundle and a readable report."""

from __future__ import annotations

import json
from pathlib import Path

VERDICT_TEXT = {
    "positive": "real lift",
    "negative": "real drop",
    "no_effect": "no practical effect",
    "uncertain": "not settled",
}


def _read(run_dir: Path, name: str):
    path = run_dir / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _manifest(folder: Path) -> dict | None:
    """Header plus every task's own record file, so nothing written in parallel is missed."""
    header = _read(folder, "manifest_header.json")
    if header is None:
        return _read(folder, "manifest.json")
    tasks = {path.stem: json.loads(path.read_text(encoding="utf-8"))
             for path in sorted((folder / "tasks").glob("*.json"))}
    return header | {"tasks": tasks}


def build_dashboard_data(run_dir: Path | str) -> dict:
    """Everything the dashboard shows, from one run, in one JSON-ready dict."""
    folder = Path(run_dir)
    bundle = {
        "run_id": folder.name,
        "manifest": _manifest(folder),
        "plan": _read(folder, "plan.json"),
        "extract": _read(folder, "extract.json"),
        "integrity": _read(folder, "integrity.json"),
        "aggregate": _read(folder, "aggregate.json"),
        "hierarchical": _read(folder, "hierarchical.json"),
        "uplift": _read(folder, "uplift.json"),
        "bundled": _read(folder, "bundled.json"),
    }
    bundle["findings"] = findings(bundle)
    return bundle


def credible_segments(hierarchical: dict, grouping: str) -> list[dict]:
    rows = hierarchical["segments"].get(grouping, [])
    return [r for r in rows if r["verdict"] in {"positive", "negative"}]


def findings(bundle: dict) -> list[str]:
    """A few plain sentences a product manager can read without the tables."""
    out: list[str] = []
    integrity = bundle.get("integrity")
    if integrity:
        problems = [c["name"] for c in integrity["checks"] if c["status"] != "pass"]
        out.append(
            "All trust checks passed."
            if not problems
            else f"Trust checks flagged: {', '.join(problems)}."
        )
    aggregate = bundle.get("aggregate")
    if aggregate:
        winners = [o["offer"] for o in aggregate["offers"]
                   if o["significant"] and o["rel_lift"] > 0]
        losers = [o["offer"] for o in aggregate["offers"]
                  if o["significant"] and o["rel_lift"] < 0]
        best = max(aggregate["offers"], key=lambda o: o["rel_lift"])
        if losers:
            out.append(
                f"The aggregate test (Dunnett-adjusted) says {', '.join(losers)} made things worse."
            )
        if winners:
            out.append(f"The aggregate test (Dunnett-adjusted) found a winner: {', '.join(winners)}.")
        else:
            out.append(
                "The aggregate A/B test found no winning offer after correcting for multiple "
                f"offers. The best was {best['offer']} at {best['rel_lift']:+.1%} "
                f"(adjusted p = {best['p_dunnett']:.2f})."
            )
    hierarchical = bundle.get("hierarchical")
    if hierarchical:
        coarse = credible_segments(hierarchical, "tenure") + credible_segments(hierarchical, "device")
        local = sorted(credible_segments(hierarchical, "market_device"),
                       key=lambda r: -abs(r["rel_lift_mean"]))[:3]
        for row in coarse + local:
            label = " ".join(str(row[k]) for k in ("market", "device", "tenure") if k in row)
            out.append(
                f"Hierarchical model: {row['arm']} for {label} users is a "
                f"{VERDICT_TEXT[row['verdict']]} ({row['rel_lift_mean']:+.0%}, 95% interval "
                f"{row['ci_low']:+.0%} to {row['ci_high']:+.0%})."
            )
    uplift = bundle.get("uplift")
    if uplift:
        targeted = next(p for p in uplift["policies"] if p["policy"] == "targeted")
        low, high = targeted["rel_lift_ci"]
        out.append(
            f"Sending each user their best predicted offer lifts conversion {targeted['rel_lift']:+.1%} "
            f"over control (95% interval {low:+.1%} to {high:+.1%}), measured on users the models "
            "never trained on."
        )
    return out


def _pct(value, digits: int = 1) -> str:
    return "n/a" if value is None else f"{value:+.{digits}%}"


def render_report(bundle: dict) -> str:
    lines = [f"# Experiment readout: `{bundle['run_id']}`", "", "## Summary", ""]
    lines += [f"- {sentence}" for sentence in bundle["findings"]]

    plan = bundle.get("plan")
    if plan:
        lines += [
            "", "## 1. Launch plan (before any data)", "",
            (f"- Baseline {plan['baseline']:.1%}, smallest lift worth detecting "
            f"{plan['mde_rel']:.0%}, power {plan['power']:.0%}, alpha {plan['alpha']} across "
            f"{len(plan['offers'])} offers ({plan['correction']}, critical z "
            f"{plan['critical_z']:.3f} vs Bonferroni {plan['bonferroni_z']:.3f})."),
            (f"- {plan['n_per_arm']:,} users per arm, {plan['planned_users']:,} planned with a "
            f"{plan['buffer']:.0%} buffer, about {plan['runtime_days']} days."),
            "- Simulated power at the target lift: "
            + ", ".join(f"{k} {v:.1%}" for k, v in plan["simulated_power_at_mde"].items())
            + f"; family-wise false positives {plan['simulated_familywise_error']:.1%}; "
            f"Bayesian assurance {plan['bayesian_assurance']:.0%}.",
            "", "| Segment | Share | Users per arm | Smallest detectable lift alone |",
            "|---|---:|---:|---:|",
            *(f"| {s['segment']} | {s['share']:.0%} | {s['users_per_arm']:,.0f} | "
              f"{s['detectable_lift']:+.0%} |" for s in plan["segments"]),
        ]

    integrity = bundle.get("integrity")
    if integrity:
        lines += ["", "## 2. Trust checks", "", "| Check | Status | Detail |", "|---|---|---|"]
        lines += [f"| {c['name']} | {c['status'].upper()} | {c['detail']} |"
                  for c in integrity["checks"]]

    aggregate = bundle.get("aggregate")
    if aggregate:
        lines += [
            "", "## 3. Aggregate A/B read-out", "",
            "| Offer | Control | Offer | Lift | p | p (Dunnett) | Significant |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
        lines += [
            f"| {o['offer']} | {o['control_rate']:.2%} | {o['offer_rate']:.2%} | "
            f"{o['rel_lift']:+.1%} | {o['p_value']:.3f} | {o['p_dunnett']:.3f} | "
            f"{'yes' if o['significant'] else 'no'} |"
            for o in aggregate["offers"]
        ]

    hierarchical = bundle.get("hierarchical")
    if hierarchical:
        d = hierarchical["diagnostics"]
        ppc = hierarchical["posterior_check"]
        prior = hierarchical["prior_check"]
        lines += [
            "", "## 4. Hierarchical Bayesian model", "",
            (f"Segments: {' x '.join(hierarchical['factors'])}, partially pooled. "
            f"Sampler: R-hat max {d['max_rhat']:.3f}, ESS min {d['min_ess_bulk']:,.0f}, "
            f"{d['divergences']} divergences ({'converged' if d['converged'] else 'NOT converged'})."),
            (f"Prior predictive: 90% of prior control rates in "
            f"{prior['control_rate_90'][0]:.1%} to {prior['control_rate_90'][1]:.1%}. "
            f"Posterior predictive: {ppc['cell_coverage']:.0%} of cells inside the 90% interval, "
            f"chi-square Bayesian p {ppc['chi2_bayes_p']:.2f} "
            f"({'passed' if ppc['passed'] else 'FAILED'})."),
            "", "### Overall", "",
            "| Offer | Lift | 95% credible interval | P(lift > 0) | Verdict |",
            "|---|---:|---|---:|---|",
            *(f"| {o['arm']} | {o['rel_lift_mean']:+.1%} | {_pct(o['rel_lift_ci'][0])} to "
              f"{_pct(o['rel_lift_ci'][1])} | {o['prob_positive']:.1%} | "
              f"{VERDICT_TEXT[o['verdict']]} |" for o in hierarchical["overall"]),
            "", "### Segments with a settled answer (device x tenure)", "",
            "| Segment | Offer | Raw lift | Pooled lift | 95% interval | Verdict |",
            "|---|---|---:|---:|---|---|",
        ]
        for row in credible_segments(hierarchical, "device_tenure"):
            lines.append(
                f"| {row['device']} {row['tenure']} | {row['arm']} | {_pct(row['raw_rel_lift'])} | "
                f"{row['rel_lift_mean']:+.1%} | {_pct(row['ci_low'])} to {_pct(row['ci_high'])} | "
                f"{VERDICT_TEXT[row['verdict']]} |"
            )
        if "prior_sensitivity" in hierarchical:
            lines += ["", "### Prior sensitivity", "",
                      "| Prior | Offer | Lift | 95% interval | Verdict |", "|---|---|---:|---|---|"]
            lines += [
                f"| {r['prior']} | {r['arm']} | {r['rel_lift_mean']:+.1%} | {_pct(r['ci_low'])} to "
                f"{_pct(r['ci_high'])} | {VERDICT_TEXT[r['verdict']]} |"
                for r in hierarchical["prior_sensitivity"]
            ]
        if "truth_check" in hierarchical:
            lines += ["", "### Against the simulator's truth", "",
                      "| Offer | Cell-level error, raw | Cell-level error, pooled |", "|---|---:|---:|"]
            lines += [f"| {arm} | {e['raw_rmse']:.3f} | {e['pooled_rmse']:.3f} |"
                      for arm, e in hierarchical["truth_check"].items()]

    uplift = bundle.get("uplift")
    if uplift:
        lines += [
            "", "## 5. Uplift modelling: who responds to which offer", "",
            ("Policy values are self-normalised inverse propensity estimates on users the models "
            "never trained on. The intervals are paired bootstrap intervals for each policy on its "
            "own; they are not adjusted for comparing several fixed offers, which is what the "
            "Dunnett-adjusted test in section 3 is for."),
            "",
            "| Policy | Conversion | Lift vs control | 95% interval |", "|---|---:|---:|---|",
            *(f"| {p['policy']} | {p['conversion_rate']:.2%} | {p['rel_lift']:+.1%} | "
              f"{_pct(p['rel_lift_ci'][0])} to {_pct(p['rel_lift_ci'][1])} |"
              for p in uplift["policies"]),
            "", "Targeted allocation: " + ", ".join(
                f"{arm} {share:.0%}" for arm, share in sorted(
                    uplift["targeted_allocation"].items(), key=lambda kv: -kv[1])),
            "", "| Offer | Qini coefficient | Permutation p |", "|---|---:|---:|",
            *(f"| {q['offer']} | {q['qini_coefficient']:.1f} | {q['perm_p_value']:.3f} |"
              for q in uplift["qini"]),
        ]
        if "truth" in uplift:
            truth = uplift["truth"]
            lines += ["", (f"Against the truth the targeted policy captures "
                      f"{truth['share_of_oracle_gain']:.0%} of what a perfect oracle would gain.")]

    bundled = bundle.get("bundled")
    if bundled:
        lines += [
            "", "## 6. The four bundled A/B tests, read hierarchically", "",
            "| Test | Raw lift | Pooled lift | 95% interval | P(lift > 0) | Verdict | Converged |",
            "|---|---:|---:|---|---:|---|---|",
        ]
        for result in bundled.values():
            o = result["overall"]
            lines.append(
                f"| {result['title']} | {result['raw_rel_lift']:+.0%} | {o['rel_lift_mean']:+.0%} | "
                f"{_pct(o['rel_lift_ci'][0], 0)} to {_pct(o['rel_lift_ci'][1], 0)} | "
                f"{o['prob_positive']:.1%} | {VERDICT_TEXT[o['verdict']]} | "
                f"{'yes' if result['diagnostics']['converged'] else 'no'} |"
            )

    manifest = bundle.get("manifest")
    if manifest:
        lines += [
            "", "## 7. Reproducibility", "",
            (f"- Config fingerprint `{manifest['config_fingerprint']}`, git commit "
            f"`{(manifest.get('git_commit') or 'unknown')[:10]}`, Python {manifest.get('python')}."),
            "- Packages: " + ", ".join(f"{k} {v}" for k, v in manifest.get("packages", {}).items()),
            "", "| Task | Status | Seconds |", "|---|---|---:|",
            *(f"| {name} | {t['status']} | {t['seconds']:.1f} |"
              for name, t in manifest["tasks"].items()),
        ]
    lines.append("")
    return "\n".join(lines)
