"""Daily experiment readout.

Runs the abkit pipeline: launch plan, log extract, trust checks, aggregate
read-out, hierarchical Bayesian model, uplift model, the bundled A/B tests and the
report.

abkit lives in its own virtualenv inside the Airflow image, so PyMC, JAX and pandas
never have to agree with Airflow's pinned dependencies. Each task runs
`python -m abkit.workflow run --only <task>` in that virtualenv. The task list
comes from `abkit_tasks.json`, which `python -m abkit.workflow graph` writes and a
test keeps identical to `abkit.workflow.TASKS`. That way this DAG and the local
runner always run the same graph, and the scheduler never imports the science
stack to parse the DAG.

Settings (environment):
  ABKIT_PYTHON        interpreter with abkit installed (default /opt/abkit-venv/bin/python)
  ABKIT_OUTPUT_ROOT   where run folders go (default /opt/airflow/artifacts/runs)
  ABKIT_SOURCE        "simulated" or a folder of experiment logs
  ABKIT_S3_URI        optional s3://bucket/prefix to publish each finished run to

Trigger with config, for example {"quick": true} or {"seed": 3}, to override
pipeline settings for that run.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.bash import BashOperator

PYTHON = os.environ.get("ABKIT_PYTHON", "/opt/abkit-venv/bin/python")
OUTPUT_ROOT = os.environ.get("ABKIT_OUTPUT_ROOT", "/opt/airflow/artifacts/runs")
SOURCE = os.environ.get("ABKIT_SOURCE", "simulated")
S3_URI = os.environ.get("ABKIT_S3_URI", "")
GRAPH = json.loads((Path(__file__).parent / "abkit_tasks.json").read_text(encoding="utf-8"))

RUN_ID = "{{ ds_nodash }}"
OVERRIDES = "{{ (dag_run.conf or {}) | tojson }}"


def task_command(task: str) -> str:
    return (
        f"{PYTHON} -m abkit.workflow run --only {task} --run-id {RUN_ID} "
        f"--output-root {OUTPUT_ROOT} --source {SOURCE} --overrides '{OVERRIDES}'"
    )


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
        task: BashOperator(
            task_id=task,
            bash_command=task_command(task),
            # A failed trust check is a finding, not a flaky error, so don't retry it.
            retries=0 if task == "validate" else 1,
        )
        for task in GRAPH
    }
    for task, upstream in GRAPH.items():
        for parent in upstream:
            operators[parent] >> operators[task]

    if S3_URI:
        operators["report"] >> BashOperator(
            task_id="publish",
            bash_command=f"{PYTHON} -m abkit.publish {OUTPUT_ROOT}/{RUN_ID} {S3_URI}",
        )
