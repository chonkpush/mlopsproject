import pandas as pd
import pytest

from src.mlproject.clean_data import clean_data


@pytest.fixture
def sample_data():
    return pd.DataFrame({
        "record_id": [1, 2, 3, 4, 5, 6, 6],
        "date": [
            "2026-01-10",
            "2026-01-10",
            "2026-01-10",
            "2026-02-15",
            "2026-03-01",
            "2026-01-10",
            "2026-01-10",
        ],
        "age_years": [36, 36, 25, 25, 30, 25, 25],
        "km_driven": [10000, 20000, 15000, 25000, 30000, 5000, 5000],
        "engine_cc": [125, 150, 100, 162, 162, 110, 110],
        "owners": [1, -999, 2, 1, 2, 1, 1],
        "service_records": [2, 3, 1, 4, 2, 1, 1],
        "accidents": [0, 1, 0, 0, 1, 0, 0],
        "brand_tier": ["Budget", "Premium", "Budget", "Mid", "Premium", "Budget", "Budget"],
        "city_zone": ["North", "North", "South", "South", "North", "South", "South"],
        "resale_thousand_rs": [500, 100, 50, 200, 350, 60, 60],
    })


def test_removes_duplicate_rows(sample_data):
    cleaned = clean_data(sample_data)

    assert cleaned.duplicated().sum() == 0
    assert len(cleaned) == 6


def test_corrects_north_age_unit_mixup(sample_data):
    cleaned = clean_data(sample_data)

    assert cleaned.loc[cleaned["record_id"] == 1, "age_years"].iloc[0] == 3


def test_does_not_change_normal_age(sample_data):
    cleaned = clean_data(sample_data)

    assert cleaned.loc[cleaned["record_id"] == 3, "age_years"].iloc[0] == 25


def test_replaces_sentinel_owner_value(sample_data):
    cleaned = clean_data(sample_data)

    assert pd.isna(cleaned.loc[cleaned["record_id"] == 2, "owners"].iloc[0])


def test_corrects_target_values(sample_data):
    cleaned = clean_data(sample_data)

    assert cleaned.loc[cleaned["record_id"] == 1, "resale_thousand_rs"].iloc[0] == 50


def test_corrects_stuck_engine_sensor(sample_data):
    cleaned = clean_data(sample_data)

    row = cleaned.loc[cleaned["record_id"] == 4, "engine_cc"].iloc[0]

    assert pd.isna(row)


def test_does_not_modify_engine_outside_stuck_period(sample_data):
    cleaned = clean_data(sample_data)

    row = cleaned.loc[cleaned["record_id"] == 5, "engine_cc"].iloc[0]

    assert row == 162


def test_original_dataframe_is_not_modified(sample_data):
    original = sample_data.copy(deep=True)

    clean_data(sample_data)

    pd.testing.assert_frame_equal(sample_data, original)