from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler


class NonNegativeRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, estimator: BaseEstimator) -> None:
        self.estimator = estimator

    def fit(self, X: Any, y: Any) -> "NonNegativeRegressor":
        self.estimator_ = clone(self.estimator).fit(X, y)
        return self

    def predict(self, X: Any) -> np.ndarray:
        return np.clip(self.estimator_.predict(X), 0.0, None)


def build_pipeline(
    candidate: dict[str, Any],
    numeric_features: list[str],
    categorical_features: list[str],
    random_state: int = 42,
) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    est = candidate["estimator"]
    if est == "ridge":
        steps = [("pre", preprocessor), ("reg", NonNegativeRegressor(Ridge(alpha=float(candidate.get("alpha", 1.0)))))]
    elif est == "poly_ridge":
        steps = [
            ("pre", preprocessor),
            ("poly", PolynomialFeatures(degree=int(candidate.get("degree", 2)), include_bias=False)),
            ("reg", NonNegativeRegressor(Ridge(alpha=float(candidate.get("alpha", 10.0))))),
        ]
    elif est == "huber":
        steps = [
            ("pre", preprocessor),
            ("reg", NonNegativeRegressor(HuberRegressor(alpha=float(candidate.get("alpha", 0.001)), max_iter=1000))),
        ]
    elif est == "random_forest":
        steps = [
            ("pre", preprocessor),
            (
                "reg",
                NonNegativeRegressor(
                    RandomForestRegressor(
                        n_estimators=int(candidate.get("n_estimators", 200)),
                        max_depth=candidate.get("max_depth", 12),
                        random_state=random_state,
                    )
                ),
            ),
        ]
    elif est == "hist_gbr":
        steps = [
            ("pre", preprocessor),
            (
                "reg",
                NonNegativeRegressor(
                    HistGradientBoostingRegressor(
                        max_iter=int(candidate.get("max_iter", 200)),
                        max_depth=candidate.get("max_depth", 6),
                        random_state=random_state,
                    )
                ),
            ),
        ]
    else:
        raise ValueError(f"Unsupported estimator: {est}")

    return Pipeline(steps=steps)
