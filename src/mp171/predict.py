import sys
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd

SRC_DIR = str(Path(__file__).resolve().parents[1])
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from mp171.config import load_params


def predict_holdout(params: dict[str, Any]) -> Path:
    out_path = Path(params["paths"]["predictions_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    mlflow_cfg = params["mlflow"]
    mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
    champion_model = mlflow.sklearn.load_model(f"models:/{mlflow_cfg['registered_model_name']}@champion")

    holdout_df = pd.read_csv(params["paths"]["holdout_features"])
    feature_cols = params["model"]["numeric_features"] + params["model"]["categorical_features"]
    preds = champion_model.predict(holdout_df[feature_cols])

    pd.DataFrame({"record_id": holdout_df["record_id"], "prediction": preds.round(4)}).to_csv(out_path, index=False)
    return out_path


if __name__ == "__main__":
    print(predict_holdout(load_params()))
