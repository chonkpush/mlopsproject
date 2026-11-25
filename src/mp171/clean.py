from typing import Any

import numpy as np
import pandas as pd


def clean_training_data(df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    clean = df.drop_duplicates().copy()
    cfg = params["cleaning"]
    target_col = params["model"]["target"]

    unit_col = cfg["unit_mixup_column"]
    zone_mask = clean["city_zone"] == cfg["unit_mixup_zone"]
    clean.loc[zone_mask, unit_col] = clean.loc[zone_mask, unit_col] / float(cfg["unit_mixup_factor"])

    sentinel_col = cfg["sentinel_column"]
    clean.loc[clean[sentinel_col] == cfg["sentinel_value"], sentinel_col] = np.nan
    clean[sentinel_col] = clean[sentinel_col].fillna(clean[sentinel_col].median())

    stuck_col = cfg["stuck_sensor_column"]
    stuck_mask = (clean["date"] >= cfg["stuck_sensor_start"]) & (clean["date"] <= cfg["stuck_sensor_end"])
    clean.loc[stuck_mask, stuck_col] = np.nan
    clean[stuck_col] = clean[stuck_col].fillna(clean[stuck_col].median())

    if target_col in clean.columns:
        typo_mask = clean[target_col] > float(cfg["target_typo_threshold"])
        clean.loc[typo_mask, target_col] = clean.loc[typo_mask, target_col] / float(cfg["target_typo_divisor"])

    return clean.reset_index(drop=True)
