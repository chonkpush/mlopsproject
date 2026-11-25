import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from app.main import app
from mp171.audit import run_audit
from mp171.clean import clean_training_data
from mp171.config import load_params
from mp171.modeling import build_pipeline
from mp171.monitor import compute_psi


def _synthetic_train_df() -> pd.DataFrame:
    rows = [
        {
            "record_id": f"R{i:05d}",
            "date": f"2026-01-{(i % 20) + 1:02d}",
            "age_years": 60.0 if i % 3 == 0 else 5.0,
            "km_driven": 20000 + i * 1000,
            "engine_cc": 150 + (i % 5) * 25,
            "owners": -999 if i == 1 else (i % 3) + 1,
            "service_records": 5 + (i % 4),
            "accidents": i % 2,
            "brand_tier": ["budget", "mid", "premium"][i % 3],
            "city_zone": "North" if i % 3 == 0 else ("South" if i % 3 == 1 else "Central"),
            "resale_thousand_rs": 950.0 if i == 2 else 75.0 - (i % 4) * 3.0,
        }
        for i in range(18)
    ]
    df = pd.DataFrame(rows)
    return pd.concat([df, df.iloc[[0]]], ignore_index=True)


def _fit_test_model() -> None:
    params = load_params(ROOT / "params.yaml")
    df = clean_training_data(_synthetic_train_df(), params)
    num_cols = params["model"]["numeric_features"]
    cat_cols = params["model"]["categorical_features"]
    pipe = build_pipeline(params["model"]["candidates"][0], num_cols, cat_cols)
    pipe.fit(df[num_cols + cat_cols], df[params["model"]["target"]])
    app.state.model = pipe


def test_health_endpoint() -> None:
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_predict_valid_and_json_log(caplog) -> None:
    _fit_test_model()
    client = TestClient(app)
    payload = {
        "age_years": 4.5,
        "km_driven": 35000,
        "engine_cc": 150,
        "owners": 1,
        "service_records": 6,
        "accidents": 0,
        "brand_tier": "mid",
        "city_zone": "South",
    }
    api_logger = logging.getLogger("mp171.api")
    api_logger.propagate = True
    try:
        with caplog.at_level(logging.INFO, logger="mp171.api"):
            resp = client.post("/predict", json=payload)
    finally:
        api_logger.propagate = False

    assert resp.status_code == 200
    assert isinstance(resp.json()["prediction"], float)
    formatted = api_logger.handlers[0].format(caplog.records[-1])
    assert json.loads(formatted)["event"] == "predict"


def test_predict_rejects_client_rules() -> None:
    _fit_test_model()
    client = TestClient(app)
    base = {
        "age_years": 4.5,
        "km_driven": 35000,
        "engine_cc": 150,
        "owners": 1,
        "service_records": 6,
        "accidents": 0,
        "brand_tier": "mid",
        "city_zone": "South",
    }
    assert client.post("/predict", json={**base, "km_driven": 250000}).status_code == 422
    assert client.post("/predict", json={**base, "service_records": -1}).status_code == 422


def test_audit_and_cleaning() -> None:
    params = load_params(ROOT / "params.yaml")
    raw = _synthetic_train_df()
    holdout = raw.drop(columns=["resale_thousand_rs"]).copy()
    later = raw.drop_duplicates().copy()
    later.loc[later["city_zone"] == "North", "age_years"] /= 12.0
    later.loc[later["owners"] == -999, "owners"] = 2
    later.loc[later["resale_thousand_rs"] > 300, "resale_thousand_rs"] /= 10.0

    issues = run_audit(
        train_df=raw,
        later_df=later,
        holdout_df=holdout,
        numeric_cols=params["model"]["numeric_features"],
        categorical_cols=params["model"]["categorical_features"],
        target_col=params["model"]["target"],
    )
    assert issues["duplicate_rows"] == {"present": True, "count": 1}
    assert issues["unit_mixup"]["group_value"] == "North"
    assert issues["unit_mixup"]["factor"] == 12
    assert issues["leaking_column"]["present"] is False

    cleaned = clean_training_data(raw, params)
    assert len(cleaned) == 18
    assert (cleaned["owners"] == -999).sum() == 0
    assert cleaned["resale_thousand_rs"].max() < 300.0


def test_psi_detects_shift() -> None:
    rng = np.random.default_rng(42)
    base = rng.normal(loc=40000, scale=10000, size=200)
    same = rng.normal(loc=40000, scale=10000, size=200)
    shifted = rng.normal(loc=90000, scale=10000, size=200)
    assert compute_psi(base, shifted, bins=10) > compute_psi(base, same, bins=10)
