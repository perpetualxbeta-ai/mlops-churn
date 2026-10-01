"""Shared paths, feature definitions and params loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
PARAMS_PATH = ROOT / "params.yaml"

DATA_DIR = ROOT / "data"
RAW_PATH = DATA_DIR / "raw" / "customers.csv"
TRAIN_PATH = DATA_DIR / "processed" / "train.csv"
TEST_PATH = DATA_DIR / "processed" / "test.csv"

MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "model.joblib"
RUN_INFO_PATH = MODEL_DIR / "run.json"

REPORTS_DIR = ROOT / "reports"
METRICS_PATH = REPORTS_DIR / "metrics.json"
TRAIN_METRICS_PATH = REPORTS_DIR / "train_metrics.json"

TARGET = "churn"
NUMERIC_FEATURES = ["tenure_months", "monthly_charges", "support_calls", "senior_citizen"]
CATEGORICAL_FEATURES = ["contract", "internet_service", "payment_method"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

CATEGORIES: dict[str, list[str]] = {
    "contract": ["month_to_month", "one_year", "two_year"],
    "internet_service": ["dsl", "fiber", "none"],
    "payment_method": ["electronic_check", "mailed_check", "bank_transfer", "credit_card"],
}


def load_params(path: Path = PARAMS_PATH) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)
