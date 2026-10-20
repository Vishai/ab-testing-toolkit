# Analysis image: runs the pipeline end to end, or serves the dashboard.
#   docker build -t abkit .
#   docker run --rm -v "$PWD/artifacts:/app/artifacts" abkit                    # full pipeline
#   docker run --rm -v "$PWD/artifacts:/app/artifacts" abkit \
#       python -m abkit.workflow run --run-id smoke --output-root /app/artifacts/runs --quick
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    ABKIT_DATA_DIR=/app/data/raw \
    ABKIT_RUNS_DIR=/app/artifacts/runs

WORKDIR /app
RUN useradd --create-home --uid 1000 abkit

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN python -m pip install --upgrade pip && python -m pip install ".[app,fast,aws]"

COPY data ./data
COPY app ./app
RUN mkdir -p /app/artifacts/runs && chown -R abkit /app/artifacts
USER abkit

CMD ["python", "-m", "abkit.workflow", "run", "--run-id", "docker", "--output-root", "/app/artifacts/runs"]
