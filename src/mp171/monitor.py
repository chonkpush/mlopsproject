import sys
from pathlib import Path
from typing import Any

import joblib
import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.metrics import mean_absolute_error

SRC_DIR = str(Path(__file__).resolve().parents[1])
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from mp171.config import compute_sha256, load_params
from mp171.modeling import build_pipeline
from mp171.train import log_model_artifact


def compute_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    expected_arr = np.asarray(expected, dtype=float)
    actual_arr = np.asarray(actual, dtype=float)

    edges = np.unique(np.quantile(expected_arr, np.linspace(0.0, 1.0, bins + 1)))
    if len(edges) < 2:
        return 0.0

    edges[0] = -np.inf
    edges[-1] = np.inf

    exp_counts, _ = np.histogram(expected_arr, bins=edges)
    act_counts, _ = np.histogram(actual_arr, bins=edges)

    exp_pct = np.clip(exp_counts / len(expected_arr), 1e-6, None)
    act_pct = np.clip(act_counts / len(actual_arr), 1e-6, None)

    return round(float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))), 4)


def monitor_and_retrain(
    params: dict[str, Any],
    best_candidate: dict[str, Any],
    v1_version: str,
) -> dict[str, Any]:
    data_sha256 = compute_sha256(params["paths"]["raw_train"])
    clean_train = pd.read_csv(params["paths"]["cleaned_train"])
    later_df = pd.read_csv(params["paths"]["raw_later"])

    num_cols = params["model"]["numeric_features"]
    cat_cols = params["model"]["categorical_features"]
    feature_cols = num_cols + cat_cols
    target_col = params["model"]["target"]

    psi_scores = {
        col: compute_psi(
            clean_train[col].to_numpy(),
            later_df[col].to_numpy(),
            bins=int(params["monitoring"]["psi_bins"]),
        )
        for col in num_cols
    }
    most_drifted = max(psi_scores, key=psi_scores.get)

    mlflow_cfg = params["mlflow"]
    mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
    mlflow.set_experiment(mlflow_cfg["experiment_name"])
    reg_name = mlflow_cfg["registered_model_name"]

    v1_model = mlflow.sklearn.load_model(f"models:/{reg_name}/{v1_version}")
    later_preds_v1 = v1_model.predict(later_df[feature_cols])

    mae_by_group = {
        zone: round(
            float(mean_absolute_error(later_df.loc[later_df["city_zone"] == zone, target_col], later_preds_v1[later_df["city_zone"] == zone])),
            4,
        )
        for zone in ("North", "South", "Central")
    }
    worst_served = max(mae_by_group, key=mae_by_group.get)

    sort_cols = ["date", "record_id"] if "record_id" in later_df.columns else ["date"]
    later_sorted = later_df.sort_values(sort_cols).reset_index(drop=True)
    split_idx = int(len(later_sorted) * float(params["monitoring"]["retrain_later_fraction"]))
    later_first_70 = later_sorted.iloc[:split_idx]
    later_last_30 = later_sorted.iloc[split_idx:]

    retrain_df = pd.concat([clean_train, later_first_70], ignore_index=True)
    v2_model = build_pipeline(
        best_candidate,
        num_cols,
        cat_cols,
        random_state=int(params["model"]["random_state"]),
    )
    v2_model.fit(retrain_df[feature_cols], retrain_df[target_col])

    mae_v1_check = round(
        float(mean_absolute_error(later_last_30[target_col], v1_model.predict(later_last_30[feature_cols]))),
        4,
    )
    mae_v2_check = round(
        float(mean_absolute_error(later_last_30[target_col], v2_model.predict(later_last_30[feature_cols]))),
        4,
    )

    improvement_ratio = (mae_v1_check - mae_v2_check) / mae_v1_check if mae_v1_check > 0 else 0.0
    decision = "promote" if improvement_ratio >= float(params["monitoring"]["promotion_min_improvement"]) else "keep"

    with mlflow.start_run(run_name=f"v2_retrain_{best_candidate['name']}") as run:
        mlflow.log_param("stage", "v2_retrain")
        mlflow.log_param("data_sha256", data_sha256)
        mlflow.log_param("model_name", best_candidate["name"])
        mlflow.log_param("decision", decision)
        mlflow.log_metric("mae_v1_on_check", mae_v1_check)
        mlflow.log_metric("mae_v2_on_check", mae_v2_check)
        mlflow.log_metric("mae_later_v1_all", float(mean_absolute_error(later_df[target_col], later_preds_v1)))
        for col, score in psi_scores.items():
            mlflow.log_metric(f"psi_{col}", score)
        log_model_artifact(v2_model)
        v2_run_id = run.info.run_id

    mv_v2 = mlflow.register_model(model_uri=f"runs:/{v2_run_id}/model", name=reg_name)
    if decision == "promote":
        champion_version = str(mv_v2.version)
        MlflowClient().set_registered_model_alias(reg_name, "champion", champion_version)
        joblib.dump(v2_model, Path(params["paths"]["model_path"]))
    else:
        champion_version = str(v1_version)

    return {
        "psi": psi_scores,
        "most_drifted_feature": most_drifted,
        "mae_by_group": mae_by_group,
        "worst_served_group": worst_served,
        "mae_v1_on_check": mae_v1_check,
        "mae_v2_on_check": mae_v2_check,
        "decision": decision,
        "champion_version": int(champion_version),
    }


if __name__ == "__main__":
    p = load_params()
    print(monitor_and_retrain(p, p["model"]["candidates"][1], "1"))
