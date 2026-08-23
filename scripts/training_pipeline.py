"""
Daily Training Pipeline
Retrains the Ridge Regression model on the latest
data and updates the Hopsworks Model Registry.
"""

import os
import sys
import json
import shutil
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "lahore" / "lahore_features_hourly.csv"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Dedicated folder only for ridge artifacts to avoid uploading unrelated models (like XGBoost)
RIDGE_EXPORT_DIR = MODEL_DIR / "ridge_export"

FEATURE_COLUMNS = [
    "temperature_2m", "relative_humidity_2m", "surface_pressure",
    "precipitation", "cloud_cover", "wind_speed_10m", "wind_direction_10m",
    "pm2_5", "pm10", "carbon_monoxide", "nitrogen_dioxide",
    "sulphur_dioxide", "ozone", "us_aqi", "hour", "day", "month",
    "day_of_week", "is_weekend", "hour_sin", "hour_cos",
    "month_sin", "month_cos", "aqi_change_rate", "aqi_lag_1h",
    "aqi_lag_24h", "aqi_rolling_mean_6h", "aqi_rolling_std_6h",
    "pm25_rolling_mean_6h"
]


def load_and_prepare_data():
    """Load CSV, create target, time-based split."""
    print("📂 Loading dataset...")
    df = pd.read_csv(DATA_PATH, parse_dates=["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    print(f"Total rows: {len(df)}")
    print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")

    # Create target: AQI 1 hour ahead
    df["target_aqi_1h"] = df["us_aqi"].shift(-1)

    initial_count = len(df)
    df = df.dropna().reset_index(drop=True)
    removed = initial_count - len(df)
    if removed > 0:
        print(f"Dropped {removed} rows with NaN values (recent hourly data without full lag history)")

    # Time-based split (70% train, 15% val, 15% test)
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    X = df[FEATURE_COLUMNS]
    y = df["target_aqi_1h"]

    splits = {
        "X_train": X.iloc[:train_end],
        "y_train": y.iloc[:train_end],
        "X_val":   X.iloc[train_end:val_end],
        "y_val":   y.iloc[train_end:val_end],
        "X_test":  X.iloc[val_end:],
        "y_test":  y.iloc[val_end:],
    }
    print(f"Train: {len(splits['X_train'])} | Val: {len(splits['X_val'])} | Test: {len(splits['X_test'])}")
    return splits


def train_and_evaluate(splits):
    """Train Ridge Regression, evaluate on validation and test sets."""
    print("\n🧠 Training Ridge Regression...")

    model = Ridge(alpha=1.0, random_state=42)
    model.fit(splits["X_train"], splits["y_train"])

    # Validation metrics
    val_pred = model.predict(splits["X_val"])
    val_mae = mean_absolute_error(splits["y_val"], val_pred)
    val_rmse = np.sqrt(mean_squared_error(splits["y_val"], val_pred))
    val_r2 = r2_score(splits["y_val"], val_pred)

    # Test metrics
    test_pred = model.predict(splits["X_test"])
    test_mae = mean_absolute_error(splits["y_test"], test_pred)
    test_rmse = np.sqrt(mean_squared_error(splits["y_test"], test_pred))
    test_r2 = r2_score(splits["y_test"], test_pred)

    metrics = {
        "val_mae":   round(float(val_mae), 4),
        "val_rmse":  round(float(val_rmse), 4),
        "val_r2":    round(float(val_r2), 4),
        "test_mae":  round(float(test_mae), 4),
        "test_rmse": round(float(test_rmse), 4),
        "test_r2":   round(float(test_r2), 4),
    }

    print(f"Validation: MAE={val_mae:.4f} | RMSE={val_rmse:.4f} | R²={val_r2:.4f}")
    print(f"Test      : MAE={test_mae:.4f} | RMSE={test_rmse:.4f} | R²={test_r2:.4f}")
    return model, metrics


def save_model_locally(model, metrics):
    """Save model, metrics, and feature list locally."""
    print("\n💾 Saving model locally...")

    # 1. Save standard local artifacts in models/
    joblib.dump(model, MODEL_DIR / "ridge_aqi_1h.pkl")
    with open(MODEL_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(MODEL_DIR / "feature_columns.json", "w") as f:
        json.dump(FEATURE_COLUMNS, f, indent=2)

    # 2. Prepare clean, isolated directory for Hopsworks upload
    if RIDGE_EXPORT_DIR.exists():
        shutil.rmtree(RIDGE_EXPORT_DIR)
    RIDGE_EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, RIDGE_EXPORT_DIR / "ridge_aqi_1h.pkl")
    with open(RIDGE_EXPORT_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(RIDGE_EXPORT_DIR / "feature_columns.json", "w") as f:
        json.dump(FEATURE_COLUMNS, f, indent=2)

    print(f"✅ Saved clean artifacts to: {RIDGE_EXPORT_DIR}")


def upload_to_hopsworks(model, metrics):
    """Upload only the Ridge model directory to Hopsworks Model Registry."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.getenv("HOPSWORKS_API_KEY")

        if not api_key:
            print("ℹ️ No HOPSWORKS_API_KEY, skipping cloud upload")
            return

        print("\n📤 Uploading to Hopsworks Model Registry...")
        import hopsworks
        project = hopsworks.login(api_key_value=api_key)
        mr = project.get_model_registry()

        input_example = pd.DataFrame([{
            "temperature_2m": 25.0, "relative_humidity_2m": 60.0,
            "surface_pressure": 1013.0, "precipitation": 0.0,
            "cloud_cover": 30.0, "wind_speed_10m": 5.0,
            "wind_direction_10m": 180.0, "pm2_5": 80.0,
            "pm10": 120.0, "carbon_monoxide": 700.0,
            "nitrogen_dioxide": 40.0, "sulphur_dioxide": 10.0,
            "ozone": 55.0, "us_aqi": 160.0,
            "hour": 10, "day": 15, "month": 6,
            "day_of_week": 2, "is_weekend": 0,
            "hour_sin": 0.5, "hour_cos": 0.866,
            "month_sin": 0.5, "month_cos": 0.866,
            "aqi_change_rate": -0.5, "aqi_lag_1h": 161.0,
            "aqi_lag_24h": 155.0, "aqi_rolling_mean_6h": 158.0,
            "aqi_rolling_std_6h": 3.0, "pm25_rolling_mean_6h": 78.0
        }])

        # Let Hopsworks auto-increment the version
        model_entry = mr.sklearn.create_model(
            name="lahore_aqi_ridge_1h",
            metrics=metrics,
            description=(
                f"Daily retrained Ridge Regression. "
                f"Test RMSE: {metrics['test_rmse']} | "
                f"Test R²: {metrics['test_r2']}"
            ),
            input_example=input_example
        )

        # Upload ONLY the isolated folder with 3 small files (< 50 KB)
        model_entry.save(str(RIDGE_EXPORT_DIR))
        print("✅ Model successfully uploaded to Hopsworks!")

    except Exception as e:
        print(f"⚠️ Hopsworks upload failed: {e}")


def run_daily_pipeline():
    """Main daily training execution."""
    print("=" * 60)
    print("📅 DAILY TRAINING PIPELINE STARTED")
    print(f"Run time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    if not DATA_PATH.exists():
        print(f"❌ Data not found at {DATA_PATH}")
        return False

    splits = load_and_prepare_data()
    model, metrics = train_and_evaluate(splits)
    save_model_locally(model, metrics)
    upload_to_hopsworks(model, metrics)

    print("=" * 60)
    print("✅ DAILY TRAINING COMPLETE")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = run_daily_pipeline()
    sys.exit(0 if success else 1)
