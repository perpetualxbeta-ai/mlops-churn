"""Model definition: preprocessing + estimator in a single scikit-learn Pipeline.

Bundling preprocessing with the model is what prevents training/serving skew:
the API feeds raw features in and gets the exact same transformations used in training.
"""

from __future__ import annotations

from typing import Any

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def build_model(train_params: dict[str, Any]) -> Pipeline:
    name = train_params["model"]
    seed = train_params["random_state"]

    if name == "logistic_regression":
        estimator = LogisticRegression(random_state=seed, **train_params[name])
    elif name == "random_forest":
        estimator = RandomForestClassifier(random_state=seed, n_jobs=-1, **train_params[name])
    else:
        raise ValueError(f"Unknown model '{name}'")

    preprocess = ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline([("preprocess", preprocess), ("model", estimator)])
