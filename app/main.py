import os
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from mp171.config import load_params
from mp171.logging_config import get_json_logger

logger = get_json_logger("mp171.api")


class BikeFeatures(BaseModel):
    age_years: float = Field(..., ge=0)
    km_driven: float = Field(..., ge=0, le=200000)
    engine_cc: float = Field(..., gt=0)
    owners: int = Field(..., ge=0)
    service_records: int = Field(..., ge=0)
    accidents: int = Field(..., ge=0)
    brand_tier: str
    city_zone: str


class PredictionResponse(BaseModel):
    prediction: float


def load_champion_model():
    env_path = os.getenv("MODEL_PATH")
    if env_path:
        path = Path(env_path)
    elif (ROOT / "params.yaml").exists():
        path = ROOT / load_params(ROOT / "params.yaml")["paths"]["model_path"]
    else:
        path = ROOT / "models/model.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Model file not found at {path}")
    return joblib.load(path)


app = FastAPI(title="MP171 Used Two-Wheeler Price API", version="1.0")


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "service": "MP171 Used Two-Wheeler Price API",
        "endpoints": {
            "health": "GET /health",
            "predict": "POST /predict",
            "docs": "GET /docs",
        },
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: BikeFeatures) -> PredictionResponse:
    start = time.perf_counter()
    if getattr(app.state, "model", None) is None:
        try:
            app.state.model = load_champion_model()
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    row = pd.DataFrame([features.model_dump()])
    pred = float(app.state.model.predict(row)[0])
    latency_ms = round((time.perf_counter() - start) * 1000.0, 2)

    logger.info(
        "prediction_served",
        extra={
            "extra_fields": {
                "event": "predict",
                "city_zone": features.city_zone,
                "brand_tier": features.brand_tier,
                "prediction": round(pred, 4),
                "latency_ms": latency_ms,
            }
        },
    )
    return PredictionResponse(prediction=round(pred, 4))
