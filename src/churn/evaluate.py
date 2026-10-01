"""Evaluate on the held-out test set and apply the quality gate (DVC stage: evaluate).

If the model clears `evaluate.min_roc_auc`, its registered version gets the
"champion" alias in the MLflow registry. If not, the script exits non-zero,
which fails `dvc repro` locally and the GitHub Actions pipeline in CI.
"""

from __future__ import annotations

import json
import sys

import joblib
import mlflow
import pandas as pd
from mlflow import MlflowClient
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from churn.config import (
    FEATURES,
    METRICS_PATH,
    MODEL_PATH,
    RUN_INFO_PATH,
    TARGET,
    TEST_PATH,
    load_params,
)
from churn.tracking import setup_mlflow


def compute_metrics(y_true, y_proba, threshold: float = 0.5) -> dict[str, float]:
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "roc_auc": round(roc_auc_score(y_true, y_proba), 4),
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred), 4),
        "f1": round(f1_score(y_true, y_pred), 4),
    }


def main() -> None:
    params = load_params()
    setup_mlflow(params["mlflow"]["experiment"])

    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(TEST_PATH)
    metrics = compute_metrics(df[TARGET], model.predict_proba(df[FEATURES])[:, 1])

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))

    run_id = json.loads(RUN_INFO_PATH.read_text())["run_id"]
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics({f"test_{k}": v for k, v in metrics.items()})

    min_auc = params["evaluate"]["min_roc_auc"]
    if metrics["roc_auc"] < min_auc:
        print(f"QUALITY GATE FAILED: roc_auc {metrics['roc_auc']} < {min_auc}", file=sys.stderr)
        sys.exit(1)

    # Promote: point the "champion" alias at the version this run registered.
    client = MlflowClient()
    name = params["mlflow"]["registered_model"]
    versions = client.search_model_versions(f"name='{name}' and run_id='{run_id}'")
    if versions:
        client.set_registered_model_alias(name, "champion", versions[0].version)
        print(f"Quality gate passed. {name} v{versions[0].version} is now 'champion'.")


if __name__ == "__main__":
    main()
