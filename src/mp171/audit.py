from typing import Any

import numpy as np
import pandas as pd

KNOWN_UNIT_FACTORS = (10, 12, 60, 100, 1000)
COMMON_SENTINELS = (-9999, -999, -1, 999, 9999)


def detect_leaking_column(
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    target_col: str,
    corr_threshold: float = 0.95,
) -> dict[str, Any]:
    extra_cols = sorted(set(train_df.columns) - set(holdout_df.columns) - {target_col})
    for col in extra_cols:
        if pd.api.types.is_numeric_dtype(train_df[col]):
            if abs(float(train_df[col].corr(train_df[target_col]))) >= corr_threshold:
                return {"present": True, "column": col}
        return {"present": True, "column": col}
    return {"present": False, "column": None}


def detect_unit_mixup(
    train_df: pd.DataFrame,
    group_col: str,
    numeric_cols: list[str],
) -> dict[str, Any]:
    best_match: dict[str, Any] | None = None
    best_score = float("inf")

    for col in numeric_cols:
        groups = train_df[group_col].dropna().unique()
        if len(groups) < 2:
            continue
        for grp in groups:
            grp_vals = train_df.loc[train_df[group_col] == grp, col].dropna()
            other_vals = train_df.loc[train_df[group_col] != grp, col].dropna()
            if len(grp_vals) == 0 or len(other_vals) == 0:
                continue
            ratios = []
            for fn in (np.min, np.median, np.mean, np.max):
                other_stat = float(fn(other_vals))
                if other_stat > 0:
                    ratios.append(float(fn(grp_vals)) / other_stat)
            if len(ratios) < 3:
                continue
            for factor in KNOWN_UNIT_FACTORS:
                mean_err = float(np.mean([abs(r - factor) / factor for r in ratios]))
                if mean_err < 0.15 and mean_err < best_score:
                    best_score = mean_err
                    best_match = {
                        "present": True,
                        "column": col,
                        "group_value": str(grp),
                        "factor": int(factor),
                    }

    if best_match is not None:
        return best_match
    return {"present": False, "column": None, "group_value": None, "factor": None}


def detect_sentinel_value(train_df: pd.DataFrame, numeric_cols: list[str]) -> dict[str, Any]:
    for col in numeric_cols:
        vc = train_df[col].value_counts()
        for sentinel in COMMON_SENTINELS:
            if sentinel in vc.index and int(vc[sentinel]) >= 5:
                return {"present": True, "column": col, "value": int(sentinel), "count": int(vc[sentinel])}
    return {"present": False, "column": None, "value": None, "count": None}


def detect_duplicate_rows(train_df: pd.DataFrame) -> dict[str, Any]:
    dup_count = int(train_df.duplicated().sum())
    return {"present": dup_count > 0, "count": dup_count if dup_count > 0 else None}


def detect_stuck_sensor(
    train_df: pd.DataFrame,
    date_col: str,
    numeric_cols: list[str],
    min_run_length: int = 20,
) -> dict[str, Any]:
    sort_cols = [date_col, "record_id"] if "record_id" in train_df.columns else [date_col]
    sorted_df = train_df.sort_values(sort_cols).reset_index(drop=True)

    best = {"present": False, "column": None, "start_date": None, "end_date": None}
    best_len = 0

    for col in numeric_cols:
        runs = (sorted_df[col] != sorted_df[col].shift()).cumsum()
        grouped = sorted_df.groupby(runs).agg(
            length=(col, "size"),
            start_date=(date_col, "first"),
            end_date=(date_col, "last"),
        )
        top = grouped.sort_values("length", ascending=False).iloc[0]
        if int(top["length"]) >= min_run_length and int(top["length"]) > best_len:
            best_len = int(top["length"])
            best = {
                "present": True,
                "column": col,
                "start_date": str(top["start_date"]),
                "end_date": str(top["end_date"]),
            }

    return best


def detect_target_typos(
    train_df: pd.DataFrame,
    target_col: str,
    gap_multiplier: float = 3.0,
) -> dict[str, Any]:
    vals = np.sort(train_df[target_col].dropna().to_numpy())
    if len(vals) < 10:
        return {"present": False, "count": None}
    idx = int(np.argmax(np.diff(vals)))
    if vals[idx] > 0 and vals[idx + 1] >= vals[idx] * gap_multiplier:
        count = int(len(vals) - (idx + 1))
        if 0 < count < len(vals) * 0.1:
            return {"present": True, "count": count}
    return {"present": False, "count": None}


def detect_spelling_variants(
    train_df: pd.DataFrame,
    later_df: pd.DataFrame,
    text_cols: list[str],
) -> dict[str, Any]:
    for col in text_cols:
        train_levels = set(train_df[col].dropna().astype(str).unique())
        later_levels = set(later_df[col].dropna().astype(str).unique())
        if train_levels != later_levels and len(train_levels) > len(later_levels):
            return {"present": True, "column": col, "true_levels": int(len(later_levels))}
    return {"present": False, "column": None, "true_levels": None}


def run_audit(
    train_df: pd.DataFrame,
    later_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    target_col: str,
    group_col: str = "city_zone",
    date_col: str = "date",
) -> dict[str, Any]:
    return {
        "leaking_column": detect_leaking_column(train_df, holdout_df, target_col),
        "unit_mixup": detect_unit_mixup(train_df, group_col, numeric_cols),
        "sentinel_value": detect_sentinel_value(train_df, numeric_cols),
        "duplicate_rows": detect_duplicate_rows(train_df),
        "stuck_sensor": detect_stuck_sensor(train_df, date_col, numeric_cols),
        "target_typos": detect_target_typos(train_df, target_col),
        "spelling_variants": detect_spelling_variants(train_df, later_df, categorical_cols),
    }
