import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

from src.mlproject.train_v1 import (
    build_preprocessor,
    FEATURES,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
)


def sample_training_data():
    return pd.DataFrame({
        "age_years": [2, 3, 4, 5, 6, 7, 2, 3, 4, 5],
        "km_driven": [10000, 20000, 30000, 40000, 50000,
                      60000, 15000, 25000, 35000, 45000],
        "engine_cc": [100, 125, 150, 100, 125, 150, 100, 125, 150, 100],
        "owners": [1, 1, 2, 1, 2, 3, 1, 2, 1, 2],
        "service_records": [1, 2, 3, 1, 2, 3, 1, 2, 3, 1],
        "accidents": [0, 0, 1, 0, 1, 2, 0, 0, 1, 0],
        "brand_tier": [
            "Budget", "Mid", "Premium", "Budget", "Mid",
            "Premium", "Budget", "Mid", "Premium", "Budget"
        ],
        "city_zone": [
            "North", "South", "East", "West", "North",
            "South", "East", "West", "North", "South"
        ],
        "resale_thousand_rs": [30, 40, 50, 35, 45, 55, 32, 42, 52, 37],
    })


def test_preprocessor_returns_valid_transformer():
    preprocessor = build_preprocessor()

    assert preprocessor is not None
    assert hasattr(preprocessor, "fit")
    assert hasattr(preprocessor, "transform")


def test_preprocessor_handles_missing_numeric_values():
    df = sample_training_data()

    df.loc[0, "engine_cc"] = np.nan
    df.loc[1, "owners"] = np.nan

    X = df[FEATURES]

    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(X)

    assert np.isfinite(transformed).all()


def test_preprocessor_encodes_categorical_features():
    df = sample_training_data()

    X = df[FEATURES]

    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(X)

    # Output must contain encoded categorical columns
    assert transformed.shape[1] > len(NUMERIC_FEATURES)


def test_training_pipeline_can_fit_and_predict():
    df = sample_training_data()

    X = df[FEATURES]
    y = df["resale_thousand_rs"]

    pipeline = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("model", LinearRegression()),
    ])

    pipeline.fit(X, y)

    predictions = pipeline.predict(X)

    assert len(predictions) == len(y)
    assert np.isfinite(predictions).all()


def test_predictions_are_numeric():
    df = sample_training_data()

    X = df[FEATURES]
    y = df["resale_thousand_rs"]

    pipeline = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("model", LinearRegression()),
    ])

    pipeline.fit(X, y)

    predictions = pipeline.predict(X)

    assert np.issubdtype(predictions.dtype, np.number)