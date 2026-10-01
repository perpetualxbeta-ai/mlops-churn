"""Data drift check with Evidently: is production data still like the training data?

Compares a "current" dataset (the API's prediction log, or any CSV) against the
"reference" training set, writes an HTML report plus a JSON summary, and exits
non-zero if too many columns drifted. Schedule this and you have a retraining trigger.

Usage:
    python -m churn.monitoring.drift                                   # uses logs/predictions.jsonl
    python -m churn.monitoring.drift --current data/simulated/drifted.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset

from churn.config import (
    CATEGORICAL_FEATURES,
    FEATURES,
    NUMERIC_FEATURES,
    REPORTS_DIR,
    ROOT,
    TRAIN_PATH,
    load_params,
)

DEFAULT_CURRENT = ROOT / "logs" / "predictions.jsonl"


def load_current(path: Path) -> pd.DataFrame:
    if path.suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    return pd.read_csv(path)


def run_drift(reference: pd.DataFrame, current: pd.DataFrame, max_drift_share: float) -> dict:
    definition = DataDefinition(
        numerical_columns=NUMERIC_FEATURES, categorical_columns=CATEGORICAL_FEATURES
    )
    snapshot = Report([DataDriftPreset(drift_share=max_drift_share)]).run(
        Dataset.from_pandas(current[FEATURES], data_definition=definition),
        Dataset.from_pandas(reference[FEATURES], data_definition=definition),
    )

    summary: dict = {"columns": {}}
    for metric in snapshot.dict()["metrics"]:
        cfg, value = metric["config"], metric["value"]
        if cfg["type"].endswith("DriftedColumnsCount"):
            summary["drifted_columns"] = int(value["count"])
            summary["drift_share"] = round(float(value["share"]), 3)
        elif cfg["type"].endswith("ValueDrift"):
            # For p-value methods, drift = p < threshold; for distance methods, score >= threshold.
            is_pvalue = "p_value" in cfg["method"]
            drifted = value < cfg["threshold"] if is_pvalue else value >= cfg["threshold"]
            summary["columns"][cfg["column"]] = {
                "method": cfg["method"],
                "score": round(float(value), 6),
                "drifted": bool(drifted),
            }

    summary["dataset_drift"] = summary["drift_share"] > max_drift_share
    summary["n_reference"], summary["n_current"] = len(reference), len(current)
    summary["_snapshot"] = snapshot
    return summary


def main() -> None:
    params = load_params()["monitoring"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current", type=Path, default=DEFAULT_CURRENT)
    parser.add_argument("--reference", type=Path, default=TRAIN_PATH)
    parser.add_argument("--max-drift-share", type=float, default=params["max_drift_share"])
    args = parser.parse_args()

    if not args.current.exists():
        sys.exit(f"No current data at {args.current}. Send some requests to the API first.")

    summary = run_drift(
        pd.read_csv(args.reference), load_current(args.current), args.max_drift_share
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    summary.pop("_snapshot").save_html(str(REPORTS_DIR / "drift.html"))
    (REPORTS_DIR / "drift.json").write_text(json.dumps(summary, indent=2))

    for col, info in summary["columns"].items():
        print(f"  {'DRIFT' if info['drifted'] else 'ok   '}  {col:<18} {info['method']}")
    print(
        f"Drifted columns: {summary['drifted_columns']}/{len(FEATURES)} "
        f"(share {summary['drift_share']}, limit {args.max_drift_share})"
    )
    print(f"Report: {REPORTS_DIR / 'drift.html'}")

    if summary["dataset_drift"]:
        print("DATASET DRIFT DETECTED: consider retraining.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
