
import pandas as pd
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "raw" / "train.csv"
OUTPUT_PATH = ROOT / "data" / "processed" / "train_clean.csv"


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the MP171 training dataset."""

    df = df.copy()

    # 1. Remove exact duplicate rows
    df = df.drop_duplicates()

    # 2. Correct the age unit mix-up in the North zone
    # Affected values are recorded in months instead of years.
    mask = (df["city_zone"] == "North") & (df["age_years"] > 30)
    df.loc[mask, "age_years"] = df.loc[mask, "age_years"] / 12

    # 3. Replace sentinel values in owners with missing values
    df["owners"] = df["owners"].replace(-999, float("nan"))

    # 4. Correct target values that are 10 times larger than expected
    mask = df["resale_thousand_rs"] > 300
    df.loc[mask, "resale_thousand_rs"] = (
        df.loc[mask, "resale_thousand_rs"] / 10
    )

    # 5. Handle the stuck engine sensor during February 9-25
    mask = (
        (df["engine_cc"] == 162)
        & (df["date"] >= "2026-02-09")
        & (df["date"] <= "2026-02-25")
    )
    df.loc[mask, "engine_cc"] = float("nan")

    return df


def main():
    # Load original training data
    df = pd.read_csv(DATA_PATH)

    print("=" * 60)
    print("MP171 DATA CLEANING")
    print("=" * 60)

    print("\nOriginal shape:", df.shape)

    # Apply cleaning
    cleaned = clean_data(df)

    # Display cleaning results
    print("Cleaned shape:", cleaned.shape)

    print("\nMissing values after cleaning:")
    print(cleaned.isnull().sum())

    print("\nRemaining duplicate rows:", cleaned.duplicated().sum())

    # Save cleaned dataset
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(OUTPUT_PATH, index=False)

    print("\nCleaned dataset saved successfully!")
    print("Output path:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
