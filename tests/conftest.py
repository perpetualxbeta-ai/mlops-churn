import pytest

from churn.config import load_params
from churn.data.generate import generate


@pytest.fixture(scope="session")
def params():
    return load_params()


@pytest.fixture(scope="session")
def sample_df():
    return generate(n_rows=800, seed=7)


@pytest.fixture(scope="session")
def trained_model(sample_df, params):
    from churn.config import FEATURES, TARGET
    from churn.model import build_model

    model = build_model(params["train"])
    model.fit(sample_df[FEATURES], sample_df[TARGET])
    return model
