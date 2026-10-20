"""Daily experiment readout.

Runs the abkit pipeline: launch plan, log extract, trust checks, aggregate
read-out, hierarchical Bayesian model, uplift model, the bundled A/B tests and the
report. The task list and its dependencies come from `abkit.workflow.TASKS`, so
this DAG and `python -m abkit.workflow run` always run the same graph.

Settings (environment):
  ABKIT_OUTPUT_ROOT   where run folders go (default /opt/airflow/artifacts/runs)
  ABKIT_SOURCE        "simulated" or a folder of experiment logs
  ABKIT_S3_URI        optional s3://bucket/prefix to publish each finished run to

Trigger with config, for example {"quick": true}, to run with fewer draws.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

from abkit.workflow import TASKS, PipelineConfig, run_task

OUTPUT_ROOT = os.environ.get("ABKIT_OUTPUT_ROOT", "/opt/airflow/artifacts/runs")
SOURCE = os.environ.get("ABKIT_SOURCE", "simulated")
S3_URI = os.environ.get("ABKIT_S3_URI", "")
QUICK = {"draws": 300, "tune": 500, "chains": 2, "prior_sensitivity": False,
         "uplift_boot": 100, "uplift_perm": 30}


def build_config(run_id: str, conf: dict | None = None) -> PipelineConfig:
    overrides = dict(conf or {})
    quick = overrides.pop("quick", False)
    values = {"run_id": run_id, "output_root": OUTPUT_ROOT, "source": SOURCE}
    values |= QUICK if quick else {}
    values |= overrides
    return PipelineConfig.from_dict(values)


def run_pipeline_task(task_name: str, run_id: str, dag_run=None, **_) -> None:
    conf = getattr(dag_run, "conf", None) or {}
    run_task(task_name, build_config(run_id, conf))


def publish(run_id: str, dag_run=None, **_) -> None:
    from abkit.publish import publish_run

    config = build_config(run_id, getattr(dag_run, "conf", None) or {})
    publish_run(config.run_dir, S3_URI)


with DAG(
    dag_id="abkit_experiment_readout",
    description="Trust checks, hierarchical Bayes and uplift for the running offer test",
    schedule="@daily",
    start_date=datetime(2025, 9, 1),
    catchup=False,
    max_active_runs=1,
    default_args={"retries": 1, "retry_delay": timedelta(minutes=5)},
    tags=["experimentation", "bayesian", "uplift"],
    doc_md=__doc__,
) as dag:
    operators = {
        spec.name: PythonOperator(
            task_id=spec.name,
            python_callable=run_pipeline_task,
            op_kwargs={"task_name": spec.name, "run_id": "{{ ds_nodash }}"},
            # A failed trust check is a finding, not a flaky error, so don't retry it.
            retries=0 if spec.name == "validate" else 1,
        )
        for spec in TASKS
    }
    for spec in TASKS:
        for upstream in spec.upstream:
            operators[upstream] >> operators[spec.name]

    if S3_URI:
        operators["report"] >> PythonOperator(
            task_id="publish",
            python_callable=publish,
            op_kwargs={"run_id": "{{ ds_nodash }}"},
        )
