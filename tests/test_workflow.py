import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

from abkit import workflow
from abkit.experiment_io import save_experiment
from abkit.publish import parse_s3_uri, publish_run
from abkit.simulate import ExperimentConfig, simulate
from abkit.workflow import (
    PipelineConfig,
    PipelineHalted,
    run_pipeline,
    topological_order,
)

ROOT = Path(__file__).resolve().parents[1]
QUICK = {
    "draws": 200, "tune": 300, "chains": 2, "prior_sensitivity": False,
    "uplift_boot": 50, "uplift_perm": 10, "bundled_scenarios": ("online_education",),
    "experiment": {"n_users": 30_000},
}
ANALYSIS_OUTPUTS = ("plan.json", "integrity.json", "aggregate.json", "hierarchical.json",
                    "uplift.json", "bundled.json")


def quick_config(tmp_path: Path, run_id: str, **overrides) -> PipelineConfig:
    return PipelineConfig.from_dict(
        QUICK | {"run_id": run_id, "output_root": str(tmp_path)} | overrides
    )


@pytest.fixture(scope="module")
def finished(tmp_path_factory):
    root = tmp_path_factory.mktemp("runs")
    first = quick_config(root, "first")
    run_pipeline(first)
    second = quick_config(root, "second")
    run_pipeline(second)
    return first, second


def test_task_graph_is_valid():
    order = topological_order()
    assert set(order) == set(workflow.TASKS_BY_NAME)
    for task in workflow.TASKS:
        assert all(order.index(up) < order.index(task.name) for up in task.upstream)
        assert task.outputs
    cyclic = (workflow.TaskSpec("a", print, ("b",)), workflow.TaskSpec("b", print, ("a",)))
    with pytest.raises(ValueError, match="cycle"):
        topological_order(cyclic)


def test_quick_run_produces_every_output(finished):
    config, _ = finished
    for name in (*ANALYSIS_OUTPUTS, "report.md", "dashboard.json", "manifest.json",
                 "lift_draws.npz"):
        assert (config.run_dir / name).exists(), name
    manifest = json.loads((config.run_dir / "manifest.json").read_text())
    assert {t["status"] for t in manifest["tasks"].values()} == {"success"}
    assert set(manifest["tasks"]) == set(workflow.TASKS_BY_NAME)
    assert "users.csv" in manifest["input_hashes"]
    assert manifest["packages"]["pymc"]
    report = (config.run_dir / "report.md").read_text()
    for heading in ("Launch plan", "Trust checks", "Aggregate A/B", "Hierarchical Bayesian",
                    "Uplift modelling", "bundled A/B tests", "Reproducibility"):
        assert heading in report


def test_same_config_and_seed_give_identical_results(finished):
    first, second = finished
    for name in ANALYSIS_OUTPUTS:
        assert (first.run_dir / name).read_bytes() == (second.run_dir / name).read_bytes(), name
    hashes = [json.loads((c.run_dir / "manifest.json").read_text())["input_hashes"]
              for c in finished]
    assert hashes[0] == hashes[1]


def test_completed_tasks_are_skipped_on_rerun(finished):
    config, _ = finished
    before = json.loads((config.run_dir / "manifest.json").read_text())
    run_pipeline(config)
    after = json.loads((config.run_dir / "manifest.json").read_text())
    assert before["tasks"] == after["tasks"]


def test_a_run_refuses_a_different_config(finished):
    config, _ = finished
    changed = PipelineConfig.from_dict(config.to_dict() | {"seed": 999})
    with pytest.raises(ValueError, match="different config"):
        workflow.run_task("aggregate", changed)


def test_failed_trust_checks_halt_the_pipeline(tmp_path):
    config = quick_config(tmp_path, "broken", experiment={"n_users": 20_000,
                                                          "contamination_share": 0.05})
    with pytest.raises(PipelineHalted, match="Contamination"):
        run_pipeline(config)
    manifest = json.loads((config.run_dir / "manifest.json").read_text())
    assert manifest["tasks"]["validate"]["status"] == "halted"
    assert "hierarchical" not in manifest["tasks"]
    assert not (config.run_dir / "uplift.json").exists()


def test_cli_reports_a_halt(tmp_path):
    settings = tmp_path / "config.json"
    settings.write_text(json.dumps(QUICK | {"experiment": {"n_users": 20_000,
                                                           "exposure_loss": {"bundle": 0.1}}}))
    code = workflow.main(["run", "--run-id", "cli", "--output-root", str(tmp_path),
                          "--config", str(settings), "--only", "plan", "extract", "validate"])
    assert code == 3


def test_pipeline_reads_a_folder_of_real_format_logs(tmp_path):
    folder = tmp_path / "logs"
    experiment = simulate(ExperimentConfig(n_users=12_000, seed=4))
    save_experiment(experiment.__class__(**{**experiment.__dict__, "truth": None}), folder)
    config = quick_config(tmp_path, "from-folder", source=str(folder))
    run_pipeline(config, only=["plan", "extract", "validate", "aggregate"])
    extract = json.loads((config.run_dir / "extract.json").read_text())
    assert extract["users"] == 12_000
    assert not (config.run_dir / "experiment" / "truth.csv").exists()
    assert json.loads((config.run_dir / "aggregate.json").read_text())["users"] > 0


def test_exported_graph_matches_the_task_list():
    exported = json.loads((ROOT / "airflow" / "dags" / "abkit_tasks.json").read_text())
    assert exported == workflow.task_graph()


def test_airflow_dag_mirrors_the_task_graph(monkeypatch):
    created, edges = {}, []

    class FakeOperator:
        def __init__(self, task_id, bash_command, retries=None, **_):
            self.task_id, self.command, self.retries = task_id, bash_command, retries
            created[task_id] = self

        def __rshift__(self, other):
            edges.append((self.task_id, other.task_id))
            return other

    class FakeDAG:
        def __init__(self, dag_id, **kwargs):
            self.dag_id, self.kwargs = dag_id, kwargs

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    airflow = types.ModuleType("airflow")
    airflow.DAG = FakeDAG
    operators = types.ModuleType("airflow.operators")
    bash = types.ModuleType("airflow.operators.bash")
    bash.BashOperator = FakeOperator
    monkeypatch.setitem(sys.modules, "airflow", airflow)
    monkeypatch.setitem(sys.modules, "airflow.operators", operators)
    monkeypatch.setitem(sys.modules, "airflow.operators.bash", bash)
    monkeypatch.setenv("ABKIT_S3_URI", "s3://bucket/readouts")
    monkeypatch.setenv("ABKIT_PYTHON", "/venv/bin/python")

    spec = importlib.util.spec_from_file_location(
        "abkit_dag", ROOT / "airflow" / "dags" / "abkit_experiment.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.dag.dag_id == "abkit_experiment_readout"
    assert set(created) == set(workflow.TASKS_BY_NAME) | {"publish"}
    expected = {(up, t.name) for t in workflow.TASKS for up in t.upstream} | {("report", "publish")}
    assert set(edges) == expected
    assert created["validate"].retries == 0
    command = created["hierarchical"].command
    assert command.startswith("/venv/bin/python -m abkit.workflow run --only hierarchical")
    assert "--run-id {{ ds_nodash }}" in command
    assert "dag_run.conf" in command
    assert created["publish"].command.endswith("s3://bucket/readouts")


def test_cli_overrides_match_what_airflow_passes(tmp_path):
    code = workflow.main(["run", "--run-id", "conf", "--output-root", str(tmp_path),
                          "--only", "plan", "--overrides", '{"quick": true, "seed": 3}'])
    assert code == 0
    saved = json.loads((tmp_path / "conf" / "config.json").read_text())
    assert saved["draws"] == 300 and saved["seed"] == 3


def test_publish_uploads_results_but_not_raw_logs(finished):
    config, _ = finished
    uploaded = []

    class FakeS3:
        def upload_file(self, filename, bucket, key):
            uploaded.append((bucket, key))

    keys = publish_run(config.run_dir, "s3://readouts/abkit/", client=FakeS3())
    assert ("readouts", "abkit/first/report.md") in uploaded
    assert all("experiment/" not in key for key in keys)
    assert parse_s3_uri("s3://bucket/a/b") == ("bucket", "a/b")
    with pytest.raises(ValueError):
        parse_s3_uri("https://example.com/x")


def test_a_significantly_worse_offer_is_never_called_a_winner():
    from abkit.platform_report import findings

    bundle = {"aggregate": {"offers": [
        {"offer": "a", "rel_lift": -0.15, "significant": True, "p_dunnett": 0.001},
        {"offer": "b", "rel_lift": 0.02, "significant": False, "p_dunnett": 0.6},
    ]}}
    text = " ".join(findings(bundle))
    assert "winner: a" not in text
    assert "a made things worse" in text
