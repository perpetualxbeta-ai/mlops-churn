import pandas as pd
import pandera.errors
import pytest

from churn.data.generate import generate
from churn.data.validate import validate


def test_generation_is_reproducible():
    pd.testing.assert_frame_equal(generate(100, seed=1), generate(100, seed=1))


def test_generated_data_passes_contract(sample_df):
    validate(sample_df)


def test_churn_rate_is_realistic(sample_df):
    assert 0.1 < sample_df["churn"].mean() < 0.5


def test_contract_rejects_bad_values(sample_df):
    bad = sample_df.copy()
    bad.loc[0, "contract"] = "lifetime"
    bad.loc[1, "tenure_months"] = -3
    with pytest.raises(pandera.errors.SchemaErrors):
        validate(bad)


def test_contract_rejects_unexpected_columns(sample_df):
    with pytest.raises(pandera.errors.SchemaErrors):
        validate(sample_df.assign(surprise=1))


def test_drift_simulation_shifts_distribution():
    normal, drifted = generate(2000, seed=1), generate(2000, seed=1, drift=True)
    assert drifted["tenure_months"].mean() < normal["tenure_months"].mean()
    assert drifted["monthly_charges"].mean() > normal["monthly_charges"].mean()
