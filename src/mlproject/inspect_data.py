
import pandas as pd

# =========================================================
# 1. LOAD DATASETS
# =========================================================

train = pd.read_csv("data/raw/train.csv")
later = pd.read_csv("data/raw/later.csv")
holdout = pd.read_csv("data/raw/holdout_features.csv")


# =========================================================
# 2. DATASET INSPECTION FUNCTION
# =========================================================

def inspect_dataset(name, df):
    print("\n" + "=" * 50)
    print(f"DATASET: {name}")
    print("=" * 50)

    print("\nShape (rows, columns):")
    print(df.shape)

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nFirst 5 rows:")
    print(df.head().to_string())

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nStatistical summary:")
    print(df.describe(include="all").T)


# =========================================================
# 3. INSPECT TRAIN DATASET
# =========================================================

inspect_dataset("TRAIN", train)


# =========================================================
# 4. BASIC DATA QUALITY INVESTIGATION
# =========================================================

print("\n" + "=" * 50)
print("DATA QUALITY CHECKS - TRAIN")
print("=" * 50)


# ---------------------------------------------------------
# Check 1: Exact duplicate rows
# ---------------------------------------------------------

print("\n1. Exact duplicate rows:")
print(train.duplicated().sum())


# ---------------------------------------------------------
# Check 2: Duplicate record IDs
# ---------------------------------------------------------

print("\n2. Duplicate record IDs:")
print(train["record_id"].duplicated().sum())

print("\nRows with duplicate record IDs:")
duplicate_ids = train[
    train["record_id"].duplicated(keep=False)
]

print(
    duplicate_ids[
        ["record_id", "date", "age_years", "km_driven",
         "resale_thousand_rs"]
    ].sort_values("record_id").to_string(index=False)
)


# ---------------------------------------------------------
# Check 3: Suspicious vehicle ages
# ---------------------------------------------------------

print("\n3. Vehicles with age above 30 years:")
suspicious_age = train[train["age_years"] > 30]

print(
    suspicious_age[
        ["record_id", "date", "age_years",
         "resale_thousand_rs"]
    ].to_string(index=False)
)


# ---------------------------------------------------------
# Check 4: Invalid owner values
# ---------------------------------------------------------

print("\n4. Records with owners less than 1:")
invalid_owners = train[train["owners"] < 1]

print(
    invalid_owners[
        ["record_id", "date", "owners", "age_years",
         "resale_thousand_rs"]
    ].to_string(index=False)
)


# ---------------------------------------------------------
# Check 5: Unusually high resale prices
# ---------------------------------------------------------

print("\n5. Records with resale price above 300 thousand:")
high_prices = train[train["resale_thousand_rs"] > 300]

print(
    high_prices[
        ["record_id", "date", "resale_thousand_rs",
         "age_years", "engine_cc", "brand_tier", "city_zone"]
    ].to_string(index=False)
)


# ---------------------------------------------------------
# Check 6: Numerical column ranges
# ---------------------------------------------------------

print("\n6. Numerical column minimum and maximum values:")

numeric_columns = train.select_dtypes(include="number").columns

print(
    train[numeric_columns]
    .agg(["min", "max"])
    .T.to_string()
)


# ---------------------------------------------------------
# Check 7: Category values
# ---------------------------------------------------------

print("\n7. Unique category values:")

categorical_columns = ["brand_tier", "city_zone"]

for column in categorical_columns:
    print(f"\n{column}:")
    print(train[column].value_counts(dropna=False))


# =========================================================
# 5. INVESTIGATION COMPLETE
# =========================================================

print("\nInitial data investigation completed.")
