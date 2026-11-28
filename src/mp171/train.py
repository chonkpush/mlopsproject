import sys
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

SRC_DIR = str(Path(__file__).resolve().parents[1])
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from mp171.clean import clean_training_data
from mp171.config import compute_sha256, load_params
from mp171.modeling import build_pipeline


def log_model_artifact(model: Any) -> None:
    mlflow.sklearn.log_model(
        sk_model=model,
        name="model",
        serialization_format="cloudpickle",
    )


def train_v1(params: dict[str, Any]) -> dict[str, Any]:
    raw_train_path = Path(params["paths"]["raw_train"])
    cleaned_train_path = Path(params["paths"]["cleaned_train"])
    model_path = Path(params["paths"]["model_path"])
    cleaned_train_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    data_sha256 = compute_sha256(raw_train_path)
    clean_df = clean_training_data(pd.read_csv(raw_train_path), params)
    clean_df.to_csv(cleaned_train_path, index=False)

    num_cols = params["model"]["numeric_features"]
    cat_cols = params["model"]["categorical_features"]
    feature_cols = num_cols + cat_cols
    target_col = params["model"]["target"]
    random_state = int(params["model"]["random_state"])

    X_train, X_val, y_train, y_val = train_test_split(
        clean_df[feature_cols],
        clean_df[target_col],
        test_size=float(params["model"]["val_size"]),
        random_state=random_state,
    )

    mlflow_cfg = params["mlflow"]
    mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
    mlflow.set_experiment(mlflow_cfg["experiment_name"])

    best_candidate: dict[str, Any] | None = None
    best_mae = float("inf")
    best_run_id = ""

    for candidate in params["model"]["candidates"]:
        pipe = build_pipeline(candidate, num_cols, cat_cols, random_state=random_state)
        pipe.fit(X_train, y_train)
        mae_val = float(mean_absolute_error(y_val, pipe.predict(X_val)))

        with mlflow.start_run(run_name=f"v1_{candidate['name']}") as run:
            mlflow.log_param("stage", "v1_candidate")
            mlflow.log_param("data_sha256", data_sha256)
            mlflow.log_param("model_name", candidate["name"])
            mlflow.log_param("estimator", candidate["estimator"])
            for k, v in candidate.items():
                if k not in {"name", "estimator"}:
                    mlflow.log_param(k, v)
            mlflow.log_metric("mae_val", mae_val)
            log_model_artifact(pipe)
            run_id = run.info.run_id

        if mae_val < best_mae:
            best_mae = mae_val
            best_candidate = candidate
            best_run_id = run_id

    assert best_candidate is not None

    v1_model = build_pipeline(best_candidate, num_cols, cat_cols, random_state=random_state)
    v1_model.fit(clean_df[feature_cols], clean_df[target_col])
    train_mae = float(mean_absolute_error(clean_df[target_col], v1_model.predict(clean_df[feature_cols])))

    with mlflow.start_run(run_name=f"v1_champion_{best_candidate['name']}") as run:
        mlflow.log_param("stage", "v1_final")
        mlflow.log_param("data_sha256", data_sha256)
        mlflow.log_param("model_name", best_candidate["name"])
        mlflow.log_param("estimator", best_candidate["estimator"])
        mlflow.log_param("selected_from_run_id", best_run_id)
        mlflow.log_metric("mae_val", best_mae)
        mlflow.log_metric("mae_train_full", train_mae)
        log_model_artifact(v1_model)
        v1_run_id = run.info.run_id

    reg_name = mlflow_cfg["registered_model_name"]
    mv = mlflow.register_model(model_uri=f"runs:/{v1_run_id}/model", name=reg_name)
    MlflowClient().set_registered_model_alias(reg_name, "champion", mv.version)
    joblib.dump(v1_model, model_path)

    return {
        "data_sha256": data_sha256,
        "best_candidate": best_candidate,
        "best_mae_val": best_mae,
        "best_run_id": v1_run_id,
        "registered_model_name": reg_name,
        "v1_version": str(mv.version),
    }


if __name__ == "__main__":
    print(train_v1(load_params()))
