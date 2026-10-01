"""Validate raw data and split it into train/test sets (DVC stage: prepare)."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from churn.config import RAW_PATH, TARGET, TEST_PATH, TRAIN_PATH, load_params
from churn.data.validate import validate


def main() -> None:
    params = load_params()["data"]
    df = validate(pd.read_csv(RAW_PATH))

    train, test = train_test_split(
        df, test_size=params["test_size"], random_state=params["seed"], stratify=df[TARGET]
    )
    TRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)
    train.to_csv(TRAIN_PATH, index=False)
    test.to_csv(TEST_PATH, index=False)
    print(f"train={len(train):,} rows, test={len(test):,} rows")


if __name__ == "__main__":
    main()
