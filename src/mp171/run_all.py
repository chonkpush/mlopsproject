import json
import sys
from pathlib import Path

import pandas as pd

SRC_DIR = str(Path(__file__).resolve().parents[1])
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from mp171.audit import run_audit
from mp171.config import load_params
from mp171.monitor import monitor_and_retrain
from mp171.predict import predict_holdout
from mp171.train import train_v1


def main() -> None:
    params = load_params()
    data_issues = run_audit(
        train_df=pd.read_csv(params["paths"]["raw_train"]),
        later_df=pd.read_csv(params["paths"]["raw_later"]),
        holdout_df=pd.read_csv(params["paths"]["holdout_features"]),
        numeric_cols=params["model"]["numeric_features"],
        categorical_cols=params["model"]["categorical_features"],
        target_col=params["model"]["target"],
    )
    train_info = train_v1(params)
    monitor_info = monitor_and_retrain(
        params=params,
        best_candidate=train_info["best_candidate"],
        v1_version=train_info["v1_version"],
    )
    predict_holdout(params)
    print(
        json.dumps(
            {
                "data_issues": data_issues,
                "best_run_id": train_info["best_run_id"],
                "champion_version": monitor_info["champion_version"],
                "monitoring": monitor_info,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
