from churn.data.generate import generate
from churn.monitoring.drift import run_drift


def test_no_drift_on_same_distribution():
    summary = run_drift(generate(2000, seed=1), generate(1000, seed=2), max_drift_share=0.3)
    assert not summary["dataset_drift"]


def test_detects_simulated_drift():
    summary = run_drift(
        generate(2000, seed=1), generate(1000, seed=2, drift=True), max_drift_share=0.3
    )
    assert summary["dataset_drift"]
    assert summary["columns"]["tenure_months"]["drifted"]
