import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = ROOT / "data" / "raw" / "train.csv"
LATER_PATH = ROOT / "data" / "raw" / "later.csv"
HOLDOUT_PATH = ROOT / "data" / "raw" / "holdout_features.csv"


def load_data():
    train = pd.read_csv(TRAIN_PATH)
    later = pd.read_csv(LATER_PATH)
    holdout = pd.read_csv(HOLDOUT_PATH)
    return train, later, holdout


def main():
    train, later, holdout = load_data()

    print("=" * 60)
    print("DATA INVESTIGATION - MP171")
    print("=" * 60)

    print("\nDataset Shapes:")
    print("Train:", train.shape)
    print("Later:", later.shape)
    print("Holdout:", holdout.shape)

    print("\nTrain Columns:")
    print(train.columns.tolist())

    print("\nHoldout Columns:")
    print(holdout.columns.tolist())

    print("\nTrain-only Columns:")
    print(set(train.columns) - set(holdout.columns))

    print("\nHoldout-only Columns:")
    print(set(holdout.columns) - set(train.columns))

    print("\nMissing Values:")
    print(train.isnull().sum())

    print("\nDuplicate Rows:")
    print(train.duplicated().sum())

    print("\nSentinel Value Check (-999):")
    for col in train.columns:
        count = (train[col] == -999).sum()
        if count > 0:
            print(f"{col}: {count}")

    print("\nAge Anomaly Check (age_years > 30):")
    print(train[train["age_years"] > 30][
        ["record_id", "date", "age_years", "city_zone"]
    ].head(20))

    print("\nTarget Anomaly Check (> 300):")
    print(train[train["resale_thousand_rs"] > 300][
        ["record_id", "date", "resale_thousand_rs"]
    ])

    print("\nEngine CC = 162 Check:")
    stuck = train[train["engine_cc"] == 162]
    print("Count:", len(stuck))
    print("Dates:")
    print(stuck["date"].value_counts().sort_index())

    print("\nCategory Values:")
    for col in ["brand_tier", "city_zone"]:
        print(f"\n{col}:")
        print("Train:", sorted(train[col].dropna().unique()))
        print("Later:", sorted(later[col].dropna().unique()))


if __name__ == "__main__":
    main()