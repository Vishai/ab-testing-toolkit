# Experiment readout: `release`

## Summary

- All trust checks passed.
- The aggregate A/B test found no winning offer after correcting for multiple offers. The best was free_shipping at +5.8% (adjusted p = 0.15).
- Hierarchical model: pct10_discount for new users is a real lift (+29%, 95% interval +17% to +42%).
- Hierarchical model: pct10_discount for returning users is a real drop (-13%, 95% interval -19% to -7%).
- Hierarchical model: bundle for desktop users is a real lift (+12%, 95% interval +3% to +23%).
- Hierarchical model: free_shipping for mobile users is a real lift (+14%, 95% interval +5% to +23%).
- Hierarchical model: free_shipping for NZ mobile users is a real lift (+42%, 95% interval +10% to +88%).
- Hierarchical model: free_shipping for AU mobile users is a real lift (+42%, 95% interval +17% to +72%).
- Hierarchical model: free_shipping for IE mobile users is a real lift (+38%, 95% interval +7% to +80%).
- Sending each user their best predicted offer lifts conversion +16.8% over control (95% interval +10.6% to +23.9%), measured on users the models never trained on.

## 1. Launch plan (before any data)

- Baseline 6.9%, smallest lift worth detecting 10%, power 80%, alpha 0.05 across 3 offers (Dunnett (many offers vs one control), critical z 2.349 vs Bonferroni 2.394).
- 28,723 users per arm, 117,000 planned with a 1% buffer, about 22 days.
- Simulated power at the target lift: bundle 79.7%, free_shipping 79.5%, pct10_discount 79.8%; family-wise false positives 5.1%; Bayesian assurance 80%.

| Segment | Share | Users per arm | Smallest detectable lift alone |
|---|---:|---:|---:|
| US | 30% | 8,775 | +16% |
| SE | 4% | 1,170 | +47% |
| ES | 4% | 1,170 | +47% |
| IT | 4% | 1,170 | +47% |
| IE | 2% | 585 | +68% |
| NZ | 2% | 585 | +68% |

## 2. Trust checks

| Check | Status | Detail |
|---|---|---|
| Sample ratio mismatch (assignment) | PASS | control 29,203, pct10_discount 29,284, free_shipping 29,435, bundle 29,078; chi-square p=0.5132 |
| Sample ratio mismatch (exposure) | PASS | control 29,203, pct10_discount 29,284, free_shipping 29,435, bundle 29,078; chi-square p=0.5132 |
| Assignment conflicts | PASS | 0 users assigned to more than one arm (0.00%) |
| Missing exposures | PASS | 0.00% of assigned users have no exposure (worst: bundle 0.00%); differs by arm p=1.0000 |
| Contamination | PASS | 0 users saw another arm (0.00%) |
| Delayed conversions | PASS | median lag 1.8 days, p99 11.5 vs a 14-day window |
| Pre-treatment covariate balance | PASS | largest SMD +0.019, min chi-square p=0.097 |
| Feature leakage | PASS | no leaky features |

## 3. Aggregate A/B read-out

| Offer | Control | Offer | Lift | p | p (Dunnett) | Significant |
|---|---:|---:|---:|---:|---:|---|
| pct10_discount | 6.82% | 6.90% | +1.3% | 0.677 | 0.954 | no |
| free_shipping | 6.82% | 7.22% | +5.8% | 0.059 | 0.147 | no |
| bundle | 6.82% | 6.84% | +0.4% | 0.901 | 0.999 | no |

## 4. Hierarchical Bayesian model

Segments: market x device x tenure, partially pooled. Sampler: R-hat max 1.005, ESS min 997, 0 divergences (converged).
Prior predictive: 90% of prior control rates in 1.2% to 30.9%. Posterior predictive: 99% of cells inside the 90% interval, chi-square Bayesian p 0.88 (passed).

### Overall

| Offer | Lift | 95% credible interval | P(lift > 0) | Verdict |
|---|---:|---|---:|---|
| pct10_discount | +1.3% | -4.4% to +7.5% | 65.3% | not settled |
| free_shipping | +5.8% | -0.3% to +12.2% | 96.9% | not settled |
| bundle | +0.7% | -4.9% to +6.6% | 58.9% | not settled |

### Segments with a settled answer (device x tenure)

| Segment | Offer | Raw lift | Pooled lift | 95% interval | Verdict |
|---|---|---:|---:|---|---|
| desktop new | pct10_discount | +18.3% | +27.7% | +12.9% to +43.6% | real lift |
| desktop returning | pct10_discount | -13.4% | -14.4% | -23.1% to -5.4% | real drop |
| desktop returning | bundle | +28.1% | +22.7% | +11.0% to +35.4% | real lift |
| mobile new | pct10_discount | +38.7% | +29.8% | +15.5% to +45.0% | real lift |
| mobile returning | pct10_discount | -14.7% | -12.9% | -20.5% to -4.9% | real drop |
| mobile returning | free_shipping | +12.6% | +14.5% | +5.0% to +25.1% | real lift |
| tablet new | pct10_discount | +42.1% | +33.4% | +10.5% to +60.8% | real lift |

### Prior sensitivity

| Prior | Offer | Lift | 95% interval | Verdict |
|---|---|---:|---|---|
| default | pct10_discount | +1.3% | -4.4% to +7.5% | not settled |
| default | free_shipping | +5.8% | -0.3% to +12.2% | not settled |
| default | bundle | +0.7% | -4.9% to +6.6% | not settled |
| skeptical | pct10_discount | +1.0% | -4.8% to +7.0% | not settled |
| skeptical | free_shipping | +5.5% | -0.4% to +11.7% | not settled |
| skeptical | bundle | +0.5% | -5.1% to +6.4% | not settled |
| wide | pct10_discount | +1.4% | -4.6% to +7.8% | not settled |
| wide | free_shipping | +5.9% | -0.3% to +12.5% | not settled |
| wide | bundle | +0.8% | -5.2% to +7.1% | not settled |

### Against the simulator's truth

| Offer | Cell-level error, raw | Cell-level error, pooled |
|---|---:|---:|
| bundle | 0.406 | 0.054 |
| free_shipping | 0.492 | 0.109 |
| pct10_discount | 0.549 | 0.053 |

## 5. Uplift modelling: who responds to which offer

| Policy | Conversion | Lift vs control | 95% interval |
|---|---:|---:|---|
| all_control | 6.81% | +0.0% | +0.0% to +0.0% |
| all_pct10_discount | 6.91% | +1.6% | -5.3% to +7.8% |
| all_free_shipping | 7.26% | +6.7% | +0.5% to +13.8% |
| all_bundle | 6.80% | -0.1% | -6.2% to +6.5% |
| targeted | 7.95% | +16.8% | +10.6% to +23.9% |

Targeted allocation: pct10_discount 33%, free_shipping 32%, bundle 24%, control 11%

| Offer | Qini coefficient | Permutation p |
|---|---:|---:|
| pct10_discount | 71.4 | 0.005 |
| free_shipping | 8.4 | 0.333 |
| bundle | 43.8 | 0.015 |

Against the truth the targeted policy captures 81% of what a perfect oracle would gain.

## 6. The four bundled A/B tests, read hierarchically

| Test | Raw lift | Pooled lift | 95% interval | P(lift > 0) | Verdict | Converged |
|---|---:|---:|---|---:|---|---|
| Charity - Online Donations | +91% | +61% | +7% to +133% | 99.0% | real lift | yes |
| SaaS - Free-Trial Sign-up | +57% | +45% | +8% to +91% | 99.3% | real lift | yes |
| Online Education - Course Enrolment | +61% | +46% | +4% to +99% | 98.6% | real lift | yes |
| Travel Booking - Hotel Reservations | +89% | +64% | +14% to +131% | 99.7% | real lift | yes |

## 7. Reproducibility

- Config fingerprint `5aaec13f585491ff`, git commit `32ed54b4c5`, Python 3.12.10.
- Packages: abkit 0.2.0, arviz 1.3.0, numpy 2.5.4, numpyro 0.22.0, pandas 3.0.6, pymc 6.3.2, scikit-learn 1.9.1, scipy 1.18.1

| Task | Status | Seconds |
|---|---|---:|
| aggregate | success | 0.9 |
| bundled | success | 43.5 |
| extract | success | 1.9 |
| hierarchical | success | 360.1 |
| plan | success | 1.2 |
| uplift | success | 9.1 |
| validate | success | 3.6 |
