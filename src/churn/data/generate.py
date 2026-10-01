"""Generate a synthetic, telco-style customer churn dataset.

Synthetic data keeps the project self-contained: no downloads, no Kaggle account, and
CI can rebuild everything from scratch. Swap in a real dataset later (see README).

Usage:
    python -m churn.data.generate                          # reference data for training
    python -m churn.data.generate --drift --n-rows 1000 \
        --out data/simulated/drifted.csv                   # "production" data that has drifted
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from churn.config import CATEGORIES, RAW_PATH, load_params


def generate(n_rows: int, seed: int, drift: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # With drift=True we simulate a market shift: newer customers, higher prices,
    # more fiber and more month-to-month contracts.
    contract_p = [0.75, 0.15, 0.10] if drift else [0.55, 0.25, 0.20]
    internet_p = [0.20, 0.70, 0.10] if drift else [0.35, 0.45, 0.20]
    tenure_scale = 10 if drift else 24
    price_shift = 15 if drift else 0

    contract = rng.choice(CATEGORIES["contract"], size=n_rows, p=contract_p)
    internet = rng.choice(CATEGORIES["internet_service"], size=n_rows, p=internet_p)
    payment = rng.choice(CATEGORIES["payment_method"], size=n_rows, p=[0.35, 0.2, 0.22, 0.23])
    tenure = np.clip(rng.exponential(tenure_scale, n_rows), 0, 72).round().astype(int)
    senior = rng.binomial(1, 0.16, n_rows)
    support_calls = rng.poisson(1.5, n_rows)

    base_price = np.select([internet == "fiber", internet == "dsl"], [80.0, 55.0], default=22.0)
    monthly = np.clip(base_price + rng.normal(price_shift, 12, n_rows), 18, 130).round(2)

    # The "true" churn mechanism the model has to learn.
    logit = (
        -1.2
        + 1.4 * (contract == "month_to_month")
        - 1.0 * (contract == "two_year")
        + 0.6 * (internet == "fiber")
        + 0.5 * (payment == "electronic_check")
        + 0.35 * support_calls
        + 0.4 * senior
        + 0.02 * (monthly - 65)
        - 0.045 * tenure
    )
    churn = rng.binomial(1, 1 / (1 + np.exp(-logit)))

    return pd.DataFrame(
        {
            "customer_id": [f"C{seed:03d}-{i:06d}" for i in range(n_rows)],
            "tenure_months": tenure,
            "monthly_charges": monthly,
            "support_calls": support_calls,
            "senior_citizen": senior,
            "contract": contract,
            "internet_service": internet,
            "payment_method": payment,
            "churn": churn,
        }
    )


def main() -> None:
    params = load_params()["data"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-rows", type=int, default=params["n_rows"])
    parser.add_argument("--seed", type=int, default=params["seed"])
    parser.add_argument("--drift", action="store_true", help="simulate a shifted population")
    parser.add_argument("--out", type=Path, default=RAW_PATH)
    args = parser.parse_args()

    df = generate(args.n_rows, args.seed, args.drift)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df):,} rows to {args.out} (churn rate {df['churn'].mean():.1%})")


if __name__ == "__main__":
    main()
