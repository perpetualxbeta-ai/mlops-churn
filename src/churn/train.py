"""Train the model and track the run in MLflow (DVC stage: train).

Each run records its params, cross-validation metrics, git commit and the model
itself, then registers a new model version. `evaluate.py` decides whether that
version is good enough to become the "champion".
"""

from __future__ import annotations

import json
import subprocess

import joblib
import mlflow
import pandas as pd
from mlflow.models import infer_signature
from sklearn.model_selection import cross_val_score

from churn.config import (
    FEATURES,
    MODEL_PATH,
    RUN_INFO_PATH,
    TARGET,
    TRAIN_METRICS_PATH,
    TRAIN_PATH,
    load_params,
)
from churn.model import build_model
from churn.tracking import setup_mlflow


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def main() -> None:
    params = load_params()
    train_params = params["train"]
    setup_mlflow(params["mlflow"]["experiment"])

    df = pd.read_csv(TRAIN_PATH)
    X, y = df[FEATURES], df[TARGET]
    model = build_model(train_params)

    with mlflow.start_run() as run:
        mlflow.set_tag("git_commit", git_commit())
        mlflow.log_params({"model": train_params["model"], **train_params[train_params["model"]]})
        mlflow.log_params({"n_train_rows": len(df), "data_seed": params["data"]["seed"]})

        cv_auc = cross_val_score(model, X, y, cv=5, scoring="roc_auc")
        mlflow.log_metrics({"cv_roc_auc_mean": cv_auc.mean(), "cv_roc_auc_std": cv_auc.std()})

        model.fit(X, y)

        mlflow.sklearn.log_model(
            model,
            name="model",
            signature=infer_signature(X, model.predict_proba(X)[:, 1]),
            input_example=X.head(3),
            registered_model_name=params["mlflow"]["registered_model"],
            # MLflow saves sklearn models with skops (safer than pickle) and asks
            # us to explicitly trust tree internals that we produced ourselves.
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )

        # Also save a plain file: DVC versions it and the Docker image ships it.
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, MODEL_PATH)

        RUN_INFO_PATH.write_text(json.dumps({"run_id": run.info.run_id}))
        TRAIN_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
        TRAIN_METRICS_PATH.write_text(
            json.dumps({"cv_roc_auc_mean": round(cv_auc.mean(), 4)}, indent=2)
        )
        print(f"run_id={run.info.run_id}  cv_roc_auc={cv_auc.mean():.4f} ± {cv_auc.std():.4f}")


if __name__ == "__main__":
    main()
