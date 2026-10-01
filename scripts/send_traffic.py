"""Simulate production traffic: send rows from a CSV to the running API.

    python scripts/send_traffic.py                                   # normal customers
    python scripts/send_traffic.py --csv data/simulated/drifted.csv  # drifted customers

Then run `make drift` to see whether the monitor notices.
"""

from __future__ import annotations

import argparse
import json
import urllib.request

import pandas as pd

FEATURES = [
    "tenure_months",
    "monthly_charges",
    "support_calls",
    "senior_citizen",
    "contract",
    "internet_service",
    "payment_method",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/processed/test.csv")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=50)
    args = parser.parse_args()

    rows = pd.read_csv(args.csv)[FEATURES].head(args.n).to_dict(orient="records")
    churners = 0
    for i in range(0, len(rows), args.batch_size):
        body = json.dumps({"customers": rows[i : i + args.batch_size]}).encode()
        req = urllib.request.Request(
            f"{args.url}/predict/batch", data=body, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req) as resp:
            churners += sum(p["will_churn"] for p in json.load(resp)["predictions"])

    rate = churners / len(rows)
    print(f"Sent {len(rows)} customers; predicted churners: {churners} ({rate:.0%})")


if __name__ == "__main__":
    main()
