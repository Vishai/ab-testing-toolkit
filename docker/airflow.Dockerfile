# Airflow with abkit installed in a separate virtualenv.
#
# Airflow pins its own dependency versions. PyMC, JAX and pandas 3 go into
# /opt/abkit-venv instead, and the DAG runs every task with that interpreter, so
# the two never have to agree.
FROM apache/airflow:2.10.5-python3.12

USER root
COPY --chown=airflow:root pyproject.toml README.md LICENSE /opt/abkit/
COPY --chown=airflow:root src /opt/abkit/src
COPY --chown=airflow:root data /opt/abkit/data
RUN python -m venv /opt/abkit-venv \
    && /opt/abkit-venv/bin/pip install --no-cache-dir --upgrade pip \
    && /opt/abkit-venv/bin/pip install --no-cache-dir "/opt/abkit[fast,aws]" \
    && chown -R airflow:root /opt/abkit-venv

USER airflow
ENV ABKIT_PYTHON=/opt/abkit-venv/bin/python \
    ABKIT_DATA_DIR=/opt/abkit/data/raw \
    ABKIT_OUTPUT_ROOT=/opt/airflow/artifacts/runs
