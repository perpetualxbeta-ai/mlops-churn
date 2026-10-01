import json

import joblib
import pytest
from fastapi.testclient import TestClient

CUSTOMER = {
    "tenure_months": 2,
    "monthly_charges": 95.0,
    "support_calls": 4,
    "senior_citizen": 0,
    "contract": "month_to_month",
    "internet_service": "fiber",
    "payment_method": "electronic_check",
}


@pytest.fixture()
def client(tmp_path, monkeypatch, trained_model):
    model_path = tmp_path / "model.joblib"
    joblib.dump(trained_model, model_path)
    log_path = tmp_path / "predictions.jsonl"

    import churn.serve.app as app_module

    monkeypatch.setattr(app_module, "MODEL_FILE", model_path)
    monkeypatch.setattr(app_module, "PREDICTION_LOG", log_path)
    with TestClient(app_module.app) as c:
        c.log_path = log_path
        yield c


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["model_loaded"] is True


def test_predict(client):
    resp = client.post("/predict", json=CUSTOMER)
    assert resp.status_code == 200
    body = resp.json()
    assert 0 <= body["churn_probability"] <= 1
    assert isinstance(body["will_churn"], bool)


def test_predict_batch(client):
    resp = client.post("/predict/batch", json={"customers": [CUSTOMER, CUSTOMER]})
    assert resp.status_code == 200
    assert len(resp.json()["predictions"]) == 2


def test_invalid_input_is_rejected(client):
    assert client.post("/predict", json={**CUSTOMER, "contract": "forever"}).status_code == 422
    assert client.post("/predict", json={**CUSTOMER, "tenure_months": -1}).status_code == 422


def test_predictions_are_logged_for_monitoring(client):
    client.post("/predict", json=CUSTOMER)
    record = json.loads(client.log_path.read_text().splitlines()[-1])
    assert record["contract"] == "month_to_month" and "prediction" in record
