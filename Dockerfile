# Serving image: only the API and its runtime deps, not the training stack.
# Build after `dvc repro` so models/model.joblib exists.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_PATH=/app/models/model.joblib \
    PREDICTION_LOG=/app/logs/predictions.jsonl

WORKDIR /app

# Install dependencies first so this layer is cached between code changes.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[serve]"

COPY models/model.joblib /app/models/model.joblib

# Don't run as root.
RUN useradd --create-home --uid 1000 app && mkdir -p /app/logs && chown app /app/logs
USER app

ARG MODEL_VERSION=local
ENV MODEL_VERSION=${MODEL_VERSION}

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "churn.serve.app:app", "--host", "0.0.0.0", "--port", "8000"]
