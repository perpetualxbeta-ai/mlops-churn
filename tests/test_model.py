import pytest
from sklearn.model_selection import train_test_split

from churn.config import FEATURES, TARGET
from churn.data.generate import generate
from churn.evaluate import compute_metrics
from churn.model import build_model


@pytest.mark.parametrize("name", ["logistic_regression", "random_forest"])
def test_model_beats_quality_gate(params, name):
    """Training smoke test: each supported model clears the AUC gate on fresh data."""
    df = generate(3000, seed=11)
    train, test = train_test_split(df, test_size=0.25, random_state=0, stratify=df[TARGET])
    model = build_model({**params["train"], "model": name})
    model.fit(train[FEATURES], train[TARGET])
    metrics = compute_metrics(test[TARGET], model.predict_proba(test[FEATURES])[:, 1])
    assert metrics["roc_auc"] >= params["evaluate"]["min_roc_auc"]


def test_predictions_are_probabilities(trained_model, sample_df):
    probas = trained_model.predict_proba(sample_df[FEATURES])[:, 1]
    assert ((probas >= 0) & (probas <= 1)).all()


def test_unknown_category_does_not_crash(trained_model, sample_df):
    row = sample_df[FEATURES].head(1).copy()
    row["payment_method"] = "crypto"
    trained_model.predict_proba(row)


def test_directional_expectation(trained_model, sample_df):
    """Behavioural test: a long two-year contract should churn less than a new monthly one."""
    base = sample_df[FEATURES].head(1).copy()
    risky = base.assign(contract="month_to_month", tenure_months=1, support_calls=6)
    safe = base.assign(contract="two_year", tenure_months=60, support_calls=0)
    p_risky = trained_model.predict_proba(risky)[0, 1]
    p_safe = trained_model.predict_proba(safe)[0, 1]
    assert p_risky > p_safe


def test_unknown_model_name_raises(params):
    with pytest.raises(ValueError):
        build_model({**params["train"], "model": "xgboost"})
