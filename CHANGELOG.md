# Changelog

## 0.2.0 - 2026-10-20

The A/B test analyzer grows into an experimentation platform. Built on top of
Cyrus Namjoo's original toolkit (0.1.0), which is unchanged and still runs.

### Added

- Event-level simulator for a four-arm offer test across 12 markets, 3 devices and
  new vs returning customers, with known true effects and switchable defects.
- Hierarchical Bayesian treatment effects in PyMC: partially pooled market, device,
  tenure and pairwise interaction terms, weakly informative priors, prior and
  posterior predictive checks, R-hat, ESS and divergence diagnostics, credible
  intervals, a ROPE decision rule and prior sensitivity.
- Trust checks on raw logs: sample ratio mismatch at assignment and exposure,
  assignment conflicts, missing exposures, contamination, delayed conversions,
  covariate balance and feature leakage.
- Multi-offer uplift modelling with cross-fitted gradient-boosted outcome models,
  a targeted policy, inverse-propensity policy evaluation with bootstrap
  intervals and per-offer Qini tests.
- Pre-launch power analysis with Dunnett's correction, segment-level detectable
  lifts, Bayesian assurance, simulated power and family-wise error, and the
  matching aggregate read-out.
- A reproducible task pipeline with run manifests, an Airflow DAG generated from
  the same task list, Docker images, a compose stack, and Terraform for AWS.
- A posterior effects dashboard for product and engineering.

### Changed

- Requires Python 3.12 (PyMC 6).
