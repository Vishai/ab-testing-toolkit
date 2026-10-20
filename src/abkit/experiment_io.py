"""Read and write experiment logs as plain files.

A folder holds one experiment:

  experiment.json   design: arms, allocation, conversion window, when data was pulled
  users.csv         user_id plus pre-treatment attributes
  assignments.csv   user_id, arm, assigned_at
  exposures.csv     user_id, arm, exposed_at
  conversions.csv   user_id, converted_at, revenue
  truth.csv         optional, simulation only: true conversion probability per arm

This is the boundary where real logs would enter the platform instead of the
simulator.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from .simulate import Experiment, ExperimentConfig

TABLES = {
    "users": ["signup_date"],
    "assignments": ["assigned_at"],
    "exposures": ["exposed_at"],
    "conversions": ["converted_at"],
}


def save_experiment(experiment: Experiment, folder: Path | str) -> dict[str, str]:
    """Write the experiment and return a sha256 per file, for the run manifest."""
    target = Path(folder)
    target.mkdir(parents=True, exist_ok=True)
    cfg = experiment.config
    design = {
        "arms": list(cfg.arms),
        "allocation": list(cfg.allocation),
        "conversion_window_days": cfg.conversion_window_days,
        "start": cfg.start.isoformat(),
        "enrollment_days": cfg.enrollment_days,
        "analysis_at": pd.Timestamp(cfg.cutoff).isoformat(),
    }
    (target / "experiment.json").write_text(json.dumps(design, indent=2), encoding="utf-8")
    for name in TABLES:
        getattr(experiment, name).to_csv(target / f"{name}.csv", index=False)
    if experiment.truth is not None:
        experiment.truth.to_csv(target / "truth.csv")
    return file_hashes(target)


def file_hashes(folder: Path | str) -> dict[str, str]:
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(Path(folder).glob("*"))
        if path.is_file()
    }


def load_experiment(folder: Path | str) -> Experiment:
    source = Path(folder)
    design = json.loads((source / "experiment.json").read_text(encoding="utf-8"))
    config = ExperimentConfig(
        arms=tuple(design["arms"]),
        allocation=tuple(design["allocation"]),
        conversion_window_days=design["conversion_window_days"],
        start=datetime.fromisoformat(design["start"]),
        enrollment_days=design["enrollment_days"],
        analysis_at=datetime.fromisoformat(design["analysis_at"]),
    )
    tables = {
        name: pd.read_csv(source / f"{name}.csv", parse_dates=dates)
        for name, dates in TABLES.items()
    }
    truth_path = source / "truth.csv"
    truth = pd.read_csv(truth_path, index_col="user_id") if truth_path.exists() else None
    return Experiment(config=config, truth=truth, **tables)
