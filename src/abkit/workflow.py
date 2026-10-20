"""The experiment analysis pipeline as a graph of tasks.

Each task reads what earlier tasks wrote into the run folder and writes its own
outputs there, so tasks can run in separate processes or containers. The same
task list drives the local runner below and the Airflow DAG in
`airflow/dags/abkit_experiment.py`.

    plan ─► extract ─► validate ─┬─► aggregate ─────┐
                                 ├─► hierarchical ──┼─► report
                                 └─► uplift ────────┘
    bundled ────────────────────────────────────────┘

Every run writes `manifest.json`: the config, a hash of every input file, package
versions, the git commit, and each task's status and duration. With the same
config and seed, the analysis outputs come out the same.

Usage:
    python -m abkit.workflow run --run-id demo
    python -m abkit.workflow run --run-id demo --only hierarchical
    python -m abkit.workflow run --source path/to/experiment_folder
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, fields
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd

from . import data as session_data
from . import hierarchical as hb
from .experiment_io import file_hashes, load_experiment, save_experiment
from .integrity import run_integrity_checks
from .launch_plan import aggregate_readout, plan_launch
from .offer_uplift import run_offer_uplift
from .pipeline import to_jsonable
from .scenarios import SCENARIOS
from .simulate import (
    MARKETS,
    SEGMENT_COLUMNS,
    ExperimentConfig,
    analysis_frame,
    cell_counts,
    simulate,
)

SEVERITY = {"pass": 0, "info": 0, "warn": 1, "fail": 2}


class PipelineHalted(RuntimeError):
    """Raised when the trust checks reach the configured halt level."""


@dataclass
class PipelineConfig:
    run_id: str = "local"
    output_root: str = "artifacts/runs"
    source: str = "simulated"  # or a folder written by experiment_io.save_experiment
    experiment: dict = field(default_factory=dict)  # ExperimentConfig overrides
    plan_baseline: float = 0.069
    plan_mde: float = 0.10
    plan_daily_users: float = 5_500
    alpha: float = 0.05
    rope: float = hb.DEFAULT_ROPE
    draws: int = 1_000
    tune: int = 1_500
    chains: int = 4
    target_accept: float = 0.99
    seed: int = 11
    prior_sensitivity: bool = True
    uplift_boot: int = 500
    uplift_perm: int = 200
    bundled_scenarios: tuple[str, ...] = tuple(SCENARIOS)
    halt_on: str = "fail"

    @property
    def run_dir(self) -> Path:
        return Path(self.output_root) / self.run_id

    @property
    def sampler(self) -> hb.SamplerSettings:
        return hb.SamplerSettings(
            draws=self.draws, tune=self.tune, chains=self.chains,
            target_accept=self.target_accept, seed=self.seed,
        )

    def to_dict(self) -> dict:
        return to_jsonable(asdict(self))

    @classmethod
    def from_dict(cls, values: dict) -> PipelineConfig:
        known = {f.name for f in fields(cls)}
        clean = {k: v for k, v in values.items() if k in known}
        if "bundled_scenarios" in clean:
            clean["bundled_scenarios"] = tuple(clean["bundled_scenarios"])
        return cls(**clean)

    @property
    def fingerprint(self) -> str:
        payload = {k: v for k, v in self.to_dict().items() if k not in {"run_id", "output_root"}}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


# -- helpers ------------------------------------------------------------------


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True), encoding="utf-8")


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def experiment_config(overrides: dict) -> ExperimentConfig:
    """ExperimentConfig from JSON-friendly overrides (dates may arrive as ISO strings)."""
    values = dict(overrides)
    for key in ("start", "analysis_at"):
        if isinstance(values.get(key), str):
            values[key] = datetime.fromisoformat(values[key])
    for key in ("arms", "allocation"):
        if key in values:
            values[key] = tuple(values[key])
    return ExperimentConfig(**values)


def _experiment_dir(config: PipelineConfig) -> Path:
    return config.run_dir / "experiment"


def _frame(config: PipelineConfig):
    experiment = load_experiment(_experiment_dir(config))
    return experiment, analysis_frame(experiment)


# -- tasks --------------------------------------------------------------------


def task_plan(config: PipelineConfig) -> None:
    """Power analysis, written before any experiment data is read."""
    arms = experiment_config(config.experiment).arms
    shares = {market: share for market, share in MARKETS.items() if share <= 0.04}
    shares = {"US": MARKETS["US"], **shares}
    plan = plan_launch(
        config.plan_baseline, arms[1:], mde_rel=config.plan_mde, alpha=config.alpha,
        daily_users=config.plan_daily_users, segment_shares=shares,
    )
    _write(config.run_dir / "plan.json", plan)


def task_extract(config: PipelineConfig) -> None:
    """Pull the experiment logs into the run folder (simulated, or copied from a folder)."""
    target = _experiment_dir(config)
    if config.source == "simulated":
        plan = _read(config.run_dir / "plan.json")
        overrides = {"n_users": plan["planned_users"], **config.experiment}
        experiment = simulate(experiment_config(overrides))
    else:
        experiment = load_experiment(config.source)
    hashes = save_experiment(experiment, target)
    _write(config.run_dir / "extract.json", {"source": config.source, "files": hashes,
                                             "users": len(experiment.users)})


def task_validate(config: PipelineConfig) -> None:
    """Trust checks. Stops the pipeline when a check reaches `halt_on`."""
    experiment = load_experiment(_experiment_dir(config))
    report = run_integrity_checks(experiment)
    _write(config.run_dir / "integrity.json", report)
    if SEVERITY[report["overall"]] >= SEVERITY[config.halt_on]:
        bad = [c["name"] for c in report["checks"] if SEVERITY[c["status"]] >= SEVERITY[config.halt_on]]
        raise PipelineHalted(f"trust checks failed: {', '.join(bad)}")


def task_aggregate(config: PipelineConfig) -> None:
    experiment, frame = _frame(config)
    readout = aggregate_readout(frame, experiment.config.arms, config.alpha)
    _write(config.run_dir / "aggregate.json", {"users": len(frame), "offers": readout})


def task_hierarchical(config: PipelineConfig) -> None:
    experiment, frame = _frame(config)
    arms = experiment.config.arms
    cells = cell_counts(frame)
    fit = hb.fit(cells, SEGMENT_COLUMNS, arms, sampler=config.sampler)
    groupings = {"tenure": ["tenure"], "device": ["device"], "market": ["market"],
                 "device_tenure": ["device", "tenure"], "market_device": ["market", "device"]}
    result = {
        "factors": list(SEGMENT_COLUMNS),
        "arms": list(arms),
        "overall": [hb.summarize_lift(fit, arm, rope=config.rope) for arm in arms[1:]],
        "segments": {
            name: hb.segment_table(fit, by=columns, rope=config.rope).to_dict(orient="records")
            for name, columns in groupings.items()
        },
        "diagnostics": hb.diagnostics(fit),
        "prior_check": hb.prior_predictive_check(fit),
        "posterior_check": hb.posterior_predictive_check(fit),
    }
    if experiment.truth is not None:
        truth = experiment.truth.loc[frame.user_id].reset_index(drop=True)
        joined = frame[list(SEGMENT_COLUMNS)].join(truth).groupby(list(SEGMENT_COLUMNS)).mean()
        merged = fit.cells.merge(joined.reset_index(), on=list(SEGMENT_COLUMNS))
        errors = {}
        for arm in arms[1:]:
            true_lift = (merged[arm] / merged[arms[0]] - 1).to_numpy()
            raw = hb.raw_cell_lifts(fit, arm)
            ok = np.isfinite(raw)
            errors[arm] = {
                "raw_rmse": float(np.sqrt(np.mean((raw[ok] - true_lift[ok]) ** 2))),
                "pooled_rmse": float(
                    np.sqrt(np.mean((hb.posterior_cell_lifts(fit, arm) - true_lift) ** 2))
                ),
            }
        result["truth_check"] = errors
    if config.prior_sensitivity:
        sensitivity = hb.prior_sensitivity(cells, SEGMENT_COLUMNS, arms, sampler=config.sampler,
                                           rope=config.rope)
        result["prior_sensitivity"] = sensitivity.to_dict(orient="records")
    _write(config.run_dir / "hierarchical.json", result)
    draws = {arm: hb.lift_draws(fit, arm)[2][:: max(1, fit.rate_draws.shape[0] // 1000)]
             for arm in arms[1:]}
    np.savez_compressed(config.run_dir / "lift_draws.npz", **draws)


def task_uplift(config: PipelineConfig) -> None:
    experiment, frame = _frame(config)
    integrity = _read(config.run_dir / "integrity.json")
    features = [f for f in integrity["details"]["leakage"]["safe_features"]
                if f in SEGMENT_COLUMNS] or list(SEGMENT_COLUMNS)
    cfg = experiment.config
    result = run_offer_uplift(
        frame, features, cfg.arms, dict(zip(cfg.arms, cfg.allocation, strict=True)),
        truth=experiment.truth, n_boot=config.uplift_boot, n_perm=config.uplift_perm,
        seed=config.seed,
    )
    _write(config.run_dir / "uplift.json", result)


def task_bundled(config: PipelineConfig) -> None:
    """Hierarchical read-out of the four bundled A/B session datasets."""
    results = {}
    for key in config.bundled_scenarios:
        sessions = session_data.load(key)
        cells = hb.cells_from_sessions(sessions, ["Device", "UserType"])
        fit = hb.fit(cells, ["Device", "UserType"], ["control", "variation"],
                     sampler=config.sampler)
        raw = sessions.groupby("group").Conversion.mean()
        results[key] = {
            "title": SCENARIOS[key].title,
            "raw_rel_lift": float(raw["B"] / raw["A"] - 1),
            "overall": hb.summarize_lift(fit, "variation", rope=config.rope),
            "device": hb.segment_table(fit, by=["Device"], rope=config.rope).to_dict(
                orient="records"),
            "diagnostics": hb.diagnostics(fit),
            "posterior_check": hb.posterior_predictive_check(fit),
        }
    _write(config.run_dir / "bundled.json", results)


def task_report(config: PipelineConfig) -> None:
    from .platform_report import build_dashboard_data, render_report

    bundle = build_dashboard_data(config.run_dir)
    _write(config.run_dir / "dashboard.json", bundle)
    (config.run_dir / "report.md").write_text(render_report(bundle), encoding="utf-8")


@dataclass(frozen=True)
class TaskSpec:
    name: str
    function: Callable[[PipelineConfig], None]
    upstream: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()


TASKS: tuple[TaskSpec, ...] = (
    TaskSpec("plan", task_plan, (), ("plan.json",)),
    TaskSpec("extract", task_extract, ("plan",), ("extract.json",)),
    TaskSpec("validate", task_validate, ("extract",), ("integrity.json",)),
    TaskSpec("aggregate", task_aggregate, ("validate",), ("aggregate.json",)),
    TaskSpec("hierarchical", task_hierarchical, ("validate",), ("hierarchical.json",)),
    TaskSpec("uplift", task_uplift, ("validate",), ("uplift.json",)),
    TaskSpec("bundled", task_bundled, (), ("bundled.json",)),
    TaskSpec("report", task_report, ("aggregate", "hierarchical", "uplift", "bundled"),
             ("report.md", "dashboard.json")),
)
TASKS_BY_NAME = {task.name: task for task in TASKS}


def topological_order(tasks: tuple[TaskSpec, ...] = TASKS) -> list[str]:
    order, done = [], set()
    pending = list(tasks)
    while pending:
        ready = [t for t in pending if set(t.upstream) <= done]
        if not ready:
            raise ValueError("the task graph has a cycle or a missing dependency")
        for task in ready:
            order.append(task.name)
            done.add(task.name)
            pending.remove(task)
    return order


# -- running ------------------------------------------------------------------


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True, cwd=Path(__file__).parent).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _versions() -> dict[str, str]:
    packages = ("abkit", "numpy", "pandas", "scipy", "scikit-learn", "pymc", "arviz", "numpyro")
    found = {}
    for package in packages:
        try:
            found[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            continue
    return found


def _manifest_path(config: PipelineConfig) -> Path:
    return config.run_dir / "manifest.json"


def load_manifest(config: PipelineConfig) -> dict:
    path = _manifest_path(config)
    if path.exists():
        return _read(path)
    return {"run_id": config.run_id, "config": config.to_dict(),
            "config_fingerprint": config.fingerprint, "tasks": {}}


def run_task(name: str, config: PipelineConfig) -> dict:
    """Run one task and record it in the manifest. This is what Airflow calls."""
    task = TASKS_BY_NAME[name]
    config.run_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(config)
    if manifest["config_fingerprint"] != config.fingerprint:
        raise ValueError(
            f"run {config.run_id} was started with a different config; use a new run id"
        )
    started = time.perf_counter()
    record = {"started_at": datetime.now(UTC).isoformat()}
    try:
        task.function(config)
        record["status"] = "success"
    except PipelineHalted as error:
        record["status"] = "halted"
        record["error"] = str(error)
        raise
    except Exception as error:
        record["status"] = "failed"
        record["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        record["seconds"] = round(time.perf_counter() - started, 2)
        manifest = load_manifest(config)
        manifest["tasks"][name] = record
        manifest |= {
            "git_commit": _git_commit(),
            "python": platform.python_version(),
            "packages": _versions(),
        }
        if (config.run_dir / "experiment").exists():
            manifest["input_hashes"] = file_hashes(config.run_dir / "experiment")
        _write(_manifest_path(config), manifest)
    return record


def run_pipeline(config: PipelineConfig, only: list[str] | None = None,
                 force: bool = False) -> dict:
    """Run every task in dependency order, skipping ones that already succeeded."""
    _write(config.run_dir / "config.json", config.to_dict())
    manifest = load_manifest(config)
    for name in topological_order():
        if only and name not in only:
            continue
        previous = manifest["tasks"].get(name, {})
        outputs_exist = all((config.run_dir / o).exists() for o in TASKS_BY_NAME[name].outputs)
        if previous.get("status") == "success" and outputs_exist and not force and not only:
            continue
        run_task(name, config)
        manifest = load_manifest(config)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the experiment analysis pipeline")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--run-id", default="local")
    run.add_argument("--output-root", default="artifacts/runs")
    run.add_argument("--source", default="simulated")
    run.add_argument("--config", type=Path, help="JSON file of PipelineConfig fields")
    run.add_argument("--only", nargs="*", choices=[t.name for t in TASKS])
    run.add_argument("--force", action="store_true", help="Rerun tasks that already succeeded")
    run.add_argument("--quick", action="store_true", help="Fewer draws, for smoke tests")
    args = parser.parse_args(argv)

    values = _read(args.config) if args.config else {}
    values |= {"run_id": args.run_id, "output_root": args.output_root, "source": args.source}
    if args.quick:
        values |= {"draws": 300, "tune": 500, "chains": 2, "prior_sensitivity": False,
                   "uplift_boot": 100, "uplift_perm": 30}
    config = PipelineConfig.from_dict(values)
    try:
        manifest = run_pipeline(config, only=args.only, force=args.force)
    except PipelineHalted as error:
        print(f"Pipeline halted: {error}", file=sys.stderr)
        return 3
    for name, record in manifest["tasks"].items():
        print(f"{name:13s} {record['status']:8s} {record['seconds']:7.1f}s")
    print(f"Outputs in {config.run_dir}")
    return 0


if __name__ == "__main__":
    pd.set_option("display.width", 160)
    sys.exit(main())
