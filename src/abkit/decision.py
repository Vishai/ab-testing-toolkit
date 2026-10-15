"""Turn statistics into a ship / hold / investigate recommendation.

The rules are deliberately ordered: trust first, then guardrails, then the
primary metric. A significant result on a broken experiment is not a win.
"""

from __future__ import annotations


def recommend(result: dict) -> dict:
    quality = result["quality"]
    primary = result["primary"]
    guard = result["guardrails"]["bounce"]
    bayes = result["bayes"]
    power = result["power"]

    srm_status = quality["srm"]["status"]
    reasons: list[str] = []

    if srm_status == "fail" or any(c["status"] == "fail" for c in quality["checks"]):
        verdict, label = "investigate", "Do not trust - fix the experiment"
        reasons.append("A trust check failed hard; the result cannot be interpreted.")
    elif srm_status == "warn":
        verdict, label = "hold", "Hold - investigate traffic split first"
        reasons.append(
            f"Traffic split is off (SRM p={quality['srm']['p_value']:.3f}). "
            "Find the cause (redirect loss, bot filtering, assignment bug) before shipping, "
            "even though the primary metric looks positive."
        )
    elif primary["significant"] and primary["abs_diff"] > 0 and guard["passed"]:
        verdict, label = "ship", "Ship B - phased rollout"
        reasons.append(
            f"Primary KPI up {primary['uplift']:+.0%} (p={primary['p_value']:.3f}); "
            f"P(B > A) = {bayes['prob_b_better']:.1%}."
        )
        reasons.append("Bounce-rate guardrail passed the non-inferiority test.")
    elif primary["significant"] and primary["abs_diff"] > 0:
        verdict, label = "hold", "Hold - guardrail trade-off"
        reasons.append("Primary KPI improved but the bounce-rate guardrail did not pass.")
    elif primary["significant"]:
        verdict, label = "reject", "Keep A - B is worse"
        reasons.append("B significantly reduced the primary KPI.")
    elif primary["n_a"] < power["n_for_target"]:
        verdict, label = "extend", "Inconclusive - extend the test"
        reasons.append("Not significant and the test has not reached its planned sample size.")
    else:
        verdict, label = "reject", "No detectable effect - keep A"
        reasons.append("Fully powered and no significant difference.")

    # caveats that apply even to a 'ship'
    caveats = []
    bal = next((c for c in quality["checks"] if c["name"] == "Pre-treatment covariate balance"), None)
    if bal and bal["status"] == "warn":
        caveats.append(
            "Some pre-treatment imbalance detected; the covariate-adjusted model "
            f"gives OR {result['adjusted']['odds_ratio']:.2f} "
            f"(p={result['adjusted']['p_value']:.3f}), so the conclusion "
            + ("holds." if result["adjusted"]["p_value"] < 0.05 else "weakens - treat with care.")
        )
    sec = result.get("secondary", {})
    rpv = sec.get("revenue_per_visitor")
    if rpv and rpv["abs_ci"][0] <= 0 < primary["abs_diff"]:
        caveats.append(
            f"More conversions, but the revenue-per-visitor gain is not conclusive "
            f"(95% CI {rpv['abs_ci'][0]:+.2f} to {rpv['abs_ci'][1]:+.2f}). "
            "Don't promise a revenue uplift yet."
        )
    aov = sec.get("avg_order_value")
    if aov and aov["abs_diff"] < 0 and primary["abs_diff"] > 0:
        caveats.append(
            f"Average value per conversion fell {abs(aov['uplift']):.0%} "
            f"({aov['mean_a']:,.0f} -> {aov['mean_b']:,.0f}, p={aov['p_value']:.2f}). "
            "B may be pulling in lower-value conversions; monitor value mix after rollout."
        )
    for h in result.get("heterogeneity", []):
        if h["p_value"] < 0.05:
            segs = [
                s for s in result["segments"]
                if s["variable"] == h["covariate"] and s["abs_diff"] <= 0
            ]
            flat = ", ".join(str(s["level"]) for s in segs) or "some segments"
            caveats.append(
                f"The effect may differ by {h['covariate']} (interaction p={h['p_value']:.3f}); "
                f"{flat} showed no lift. This is one of several interaction tests, so confirm "
                "with a targeted follow-up before personalising."
            )
    wc = power.get("winners_curse", {})
    if primary["uplift"] > 0.5:
        msg = (
            "Lift above 50% is unusually large for a UI change (Twyman's law). "
            "Expect a smaller effect at full rollout; re-measure at 4-6 weeks."
        )
        if wc.get("exaggeration_ratio"):
            msg += (
                f" At this sample size, if the true lift were {wc['assumed_true_lift']:.0%}, "
                f"significant results would overstate it {wc['exaggeration_ratio']:.1f}x on average."
            )
        caveats.append(msg)
    if primary["n_a"] < power["n_for_target"]:
        caveats.append(
            f"The test is below the {power['n_for_target']:,}-per-arm sample planned for a "
            f"{power['target_mde_rel']:.0%} lift, so the uplift estimate is imprecise "
            f"(95% CI {primary['uplift_ci'][0]:+.0%} to {primary['uplift_ci'][1]:+.0%})."
        )

    return {"verdict": verdict, "label": label, "reasons": reasons, "caveats": caveats}
