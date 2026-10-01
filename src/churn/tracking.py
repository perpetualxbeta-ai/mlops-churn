"""MLflow setup shared by the training and evaluation stages."""

from __future__ import annotations

import os

import mlflow

from churn.config import ROOT


def setup_mlflow(experiment: str) -> None:
    # Default: a local SQLite store (needed for the model registry).
    # Point MLFLOW_TRACKING_URI at a server (e.g. http://localhost:5000 from
    # docker compose) to share runs instead.
    uri = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'mlflow.db'}")
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment)
