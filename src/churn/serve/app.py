"""FastAPI service that serves the trained churn model.

Run locally:   uvicorn churn.serve.app:app --reload
Docs:          http://localhost:8000/docs

Every prediction is appended to a JSONL log. That log is the "current" data the
drift monitor compares against the training data (see churn.monitoring.drift).
"""

from __future__ import annotations

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request

from churn.config import FEATURES, MODEL_PATH, ROOT
from churn.serve.schemas import BatchRequest, BatchResponse, Customer, Prediction

logger = logging.getLogger("churn.api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

MODEL_FILE = Path(os.getenv("MODEL_PATH", MODEL_PATH))
PREDICTION_LOG = Path(os.getenv("PREDICTION_LOG", ROOT / "logs" / "predictions.jsonl"))
MODEL_VERSION = os.getenv("MODEL_VERSION", "local")
THRESHOLD = float(os.getenv("CHURN_THRESHOLD", "0.5"))

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if MODEL_FILE.exists():
        state["model"] = joblib.load(MODEL_FILE)
        logger.info("Loaded model from %s (version=%s)", MODEL_FILE, MODEL_VERSION)
    else:
        logger.warning("No model at %s. Run `dvc repro` first; /predict will 503.", MODEL_FILE)
    yield
    state.clear()


app = FastAPI(title="Churn Prediction API", version="0.1.0", lifespan=lifespan)


@app.middleware("http")
async def log_latency(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    ms = (time.perf_counter() - start) * 1000
    logger.info("%s %s -> %s in %.1fms", request.method, request.url.path, response.status_code, ms)
    return response


def _predict(customers: list[Customer]) -> list[Prediction]:
    model = state.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    df = pd.DataFrame([c.model_dump() for c in customers])[FEATURES]
    probas = model.predict_proba(df)[:, 1]
    _log_predictions(df, probas)
    return [
        Prediction(
            churn_probability=round(float(p), 4),
            will_churn=bool(p >= THRESHOLD),
            model_version=MODEL_VERSION,
        )
        for p in probas
    ]


def _log_predictions(df: pd.DataFrame, probas) -> None:
    PREDICTION_LOG.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).isoformat()
    with PREDICTION_LOG.open("a") as f:
        for row, p in zip(df.to_dict(orient="records"), probas, strict=True):
            f.write(json.dumps({"timestamp": ts, **row, "prediction": float(p)}) + "\n")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": "model" in state, "model_version": MODEL_VERSION}


@app.post("/predict", response_model=Prediction)
def predict(customer: Customer) -> Prediction:
    return _predict([customer])[0]


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(request: BatchRequest) -> BatchResponse:
    return BatchResponse(predictions=_predict(request.customers))
