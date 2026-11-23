
import hashlib
from pathlib import Path

import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import mean_absolute_error


# --------------------------------------------------
# PATHS AND CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "data" / "processed" / "train_clean.csv"
RAW_DATA_PATH = ROOT / "data" / "raw" / "train.csv"

MLFLOW_DB = ROOT / "mlflow.db"

EXPERIMENT_NAME = "G257_MP171_V1"
REGISTERED_MODEL_NAME = "G257_MP171_V1"

TARGET = "resale_thousand_rs"

NUMERIC_FEATURES = [
    "age_years",
    "km_driven",
    "engine_cc",
    "owners",
    "service_records",
    "accidents",
]

CATEGORICAL_FEATURES = [
    "brand_tier",
    "city_zone",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# --------------------------------------------------
# DATA HASH
# --------------------------------------------------

def calculate_sha256(file_path):
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:
        for chunk in iter(lambda: file.read(4096), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


# --------------------------------------------------
# PREPROCESSING
# --------------------------------------------------

def build_preprocessor():

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ])

    return preprocessor


# --------------------------------------------------
# MAIN TRAINING
# --------------------------------------------------

def main():

    # Load cleaned data
    df = pd.read_csv(DATA_PATH)

    print("Dataset shape:", df.shape)

    X = df[FEATURES]
    y = df[TARGET]

    # Train-validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
    )

    print("Training rows:", X_train.shape[0])
    print("Validation rows:", X_val.shape[0])

    # Original dataset SHA256
    data_sha256 = calculate_sha256(RAW_DATA_PATH)

    print("Dataset SHA256:", data_sha256)

    # MLflow configuration
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB}")
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Models to compare
    models = {
        "LinearRegression": LinearRegression(),

        "RandomForest": RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            random_state=42,
            n_jobs=-1,
        ),

        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=42,
        ),
    }

    results = []

    # Run experiments
    for model_name, model in models.items():

        print("\n" + "=" * 50)
        print("Training:", model_name)
        print("=" * 50)

        pipeline = Pipeline([
            ("preprocessor", build_preprocessor()),
            ("model", model),
        ])

        with mlflow.start_run(run_name=model_name) as run:

            # Train
            pipeline.fit(X_train, y_train)

            # Predict
            predictions = pipeline.predict(X_val)

            # Evaluate
            mae = mean_absolute_error(y_val, predictions)

            print("Validation MAE:", mae)

            # Log parameters
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("train_rows", len(X_train))
            mlflow.log_param("validation_rows", len(X_val))
            mlflow.log_param("random_state", 42)

            # Dataset tracking
            mlflow.log_param("data_sha256", data_sha256)

            # Log metrics
            mlflow.log_metric("mae", mae)

            # Log trained model
            mlflow.sklearn.log_model(
                sk_model=pipeline,
                name="model",
                serialization_format="cloudpickle",
            )

            results.append({
                "model_name": model_name,
                "mae": mae,
                "run_id": run.info.run_id,
            })

    # --------------------------------------------------
    # SELECT BEST MODEL
    # --------------------------------------------------

    results_df = pd.DataFrame(results).sort_values("mae")

    print("\n" + "=" * 50)
    print("MODEL COMPARISON")
    print("=" * 50)

    print(results_df.to_string(index=False))

    best_model_name = results_df.iloc[0]["model_name"]
    best_run_id = results_df.iloc[0]["run_id"]
    best_mae = results_df.iloc[0]["mae"]

    print("\nBest model:", best_model_name)
    print("Best MAE:", best_mae)
    print("Best Run ID:", best_run_id)

    # --------------------------------------------------
    # REGISTER BEST MODEL
    # --------------------------------------------------

    best_model_uri = f"runs:/{best_run_id}/model"

    registered_model = mlflow.register_model(
        model_uri=best_model_uri,
        name=REGISTERED_MODEL_NAME,
    )

    # Set champion alias
    client = mlflow.MlflowClient()

    client.set_registered_model_alias(
        name=REGISTERED_MODEL_NAME,
        alias="champion",
        version=registered_model.version,
    )

    print("\nModel registered successfully!")
    print("Registered model:", REGISTERED_MODEL_NAME)
    print("Model version:", registered_model.version)
    print("Champion alias: champion")

    # Save comparison results
    results_path = ROOT / "reports" / "v1_model_comparison.csv"
    results_path.parent.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(results_path, index=False)

    print("\nComparison results saved to:", results_path)


if __name__ == "__main__":
    main()
