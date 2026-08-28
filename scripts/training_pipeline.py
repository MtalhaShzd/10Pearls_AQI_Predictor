"""
Daily Training Pipeline — Lahore AQI (Ridge, 1h ahead)
======================================================
Reads features from the Hopsworks Feature Store (v2, streaming), trains a
scaled Ridge regressor, evaluates against a persistence baseline, and
registers the model in the Hopsworks Model Registry.

Usage:
    python src/training_pipeline.py
    python src/training_pipeline.py --only-if-better   # recommended for daily cron
    python src/training_pipeline.py --source csv       # train from local CSV
    python src/training_pipeline.py --no-upload
"""

import argparse
import json
import os
import shutil
import sys
import traceback
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "lahore" / "lahore_features_hourly.csv"
MODEL_DIR = PROJECT_ROOT / "models"
RIDGE_EXPORT_DIR = MODEL_DIR / "ridge_export"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

FG_NAME = os.getenv("HOPSWORKS_FEATURE_GROUP", "lahore_air_quality_features")
FG_VERSION = int(os.getenv("HOPSWORKS_FEATURE_GROUP_VERSION", "2"))   # v2 = streaming
MODEL_NAME = "lahore_aqi_ridge_1h"

MIN_ROWS = 500
RIDGE_ALPHA = 1.0

FEATURE_COLUMNS = [
    "temperature_2m", "relative_humidity_2m", "surface_pressure",
    "precipitation", "cloud_cover", "wind_speed_10m", "wind_direction_10m",
    "pm2_5", "pm10", "carbon_monoxide", "nitrogen_dioxide",
    "sulphur_dioxide", "ozone", "us_aqi", "hour", "day", "month",
    "day_of_week", "is_weekend", "hour_sin", "hour_cos",
    "month_sin", "month_cos", "aqi_change_rate", "aqi_lag_1h",
    "aqi_lag_24h", "aqi_rolling_mean_6h", "aqi_rolling_std_6h",
    "pm25_rolling_mean_6h",
]


def hopsworks_login():
    """Login with explicit host/project — the bare login() defaults elsewhere."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    api_key = os.getenv("HOPSWORKS_API_KEY")
    if not api_key:
        return None

    import hopsworks
    return hopsworks.login(
        host=os.getenv("HOPSWORKS_HOST", "eu-west.cloud.hopsworks.ai"),
        project=os.getenv("HOPSWORKS_PROJECT", "internship10P"),
        api_key_value=api_key,
    )


# --------------------------------------------------------------------------
# DATA
# --------------------------------------------------------------------------
def _prep(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"]).dt.tz_localize(None)
    return (df.drop_duplicates(subset="datetime", keep="last")
              .sort_values("datetime").reset_index(drop=True))


def load_dataframe(source: str) -> pd.DataFrame:
    """Load features. For --source auto, reads BOTH Hopsworks and the local
    CSV and keeps whichever is actually more current — a successful Hopsworks
    read doesn't guarantee fresh data if offline materialization is stuck
    (e.g. compute quota exhausted). Hopsworks wins exact ties.

    --source hopsworks / --source csv bypass the comparison and force that
    exact source, raising if it's unavailable.
    """
    df_hw = None
    if source in ("auto", "hopsworks"):
        project = hopsworks_login()
        if project is None:
            if source == "hopsworks":
                raise RuntimeError("--source hopsworks but HOPSWORKS_API_KEY is not set")
            print("ℹ️ No HOPSWORKS_API_KEY — falling back to local CSV")
        else:
            try:
                print(f"📥 Reading {FG_NAME} v{FG_VERSION} from feature store...")
                fg = project.get_feature_store().get_feature_group(FG_NAME, version=FG_VERSION)
                df_hw = _prep(fg.read())
                print(f"   {len(df_hw)} rows retrieved (latest {df_hw['datetime'].max()})")
            except Exception as e:
                if source == "hopsworks":
                    raise
                print(f"⚠️ Feature Store read failed ({e}) — falling back to local CSV")

    if source == "hopsworks":
        df = df_hw
    elif source == "csv":
        if not DATA_PATH.exists():
            raise FileNotFoundError(f"No CSV at {DATA_PATH}")
        print(f"📂 Reading local CSV: {DATA_PATH}")
        df = _prep(pd.read_csv(DATA_PATH, parse_dates=["datetime"]))
    else:  # auto: compare and pick the fresher one
        df_csv = None
        if DATA_PATH.exists():
            print(f"📂 Reading local CSV: {DATA_PATH}")
            df_csv = _prep(pd.read_csv(DATA_PATH, parse_dates=["datetime"]))
            print(f"   {len(df_csv)} rows (latest {df_csv['datetime'].max()})")

        if df_hw is None and df_csv is None:
            raise FileNotFoundError(f"No feature store access and no CSV at {DATA_PATH}")
        elif df_hw is None:
            df = df_csv
            print("ℹ️ Using local CSV — Hopsworks unavailable")
        elif df_csv is None:
            df = df_hw
        elif df_csv["datetime"].max() > df_hw["datetime"].max():
            df = df_csv
            print(f"ℹ️ Local CSV is newer (latest {df_csv['datetime'].max()}) than "
                  f"Hopsworks (latest {df_hw['datetime'].max()}) — training on local CSV")
        else:
            df = df_hw
            
    missing = set(FEATURE_COLUMNS + ["datetime"]) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")

    span = int((df["datetime"].max() - df["datetime"].min()).total_seconds() // 3600) + 1
    print(f"   Range   : {df['datetime'].min()} → {df['datetime'].max()}")
    print(f"   Coverage: {len(df)}/{span} hours ({100 * len(df) / span:.1f}%)")
    if len(df) < span * 0.95:
        print(f"   ⚠️ {span - len(df)} hourly gaps — consider running the backfill pipeline")

    return df


def build_target(df: pd.DataFrame) -> pd.DataFrame:
    """Target = us_aqi one hour LATER, joined on the clock.

    FIX: the old shift(-1) silently jumped across time gaps, pairing (say)
    Jul-31 23:00 with Aug-25 00:00 and teaching the model nonsense.
    """
    nxt = df[["datetime", "us_aqi"]].rename(columns={"us_aqi": "target_aqi_1h"})
    nxt["datetime"] = nxt["datetime"] - pd.Timedelta(hours=1)
    out = df.merge(nxt, on="datetime", how="left")

    before = len(out)
    out = out.dropna(subset=FEATURE_COLUMNS + ["target_aqi_1h"]).reset_index(drop=True)
    print(f"   Dropped {before - len(out)} rows (gap-adjacent or NaN)")
    return out


def time_split(df: pd.DataFrame):
    n = len(df)
    if n < MIN_ROWS:
        raise ValueError(f"Only {n} usable rows — need at least {MIN_ROWS}")

    train_end, val_end = int(n * 0.70), int(n * 0.85)
    X, y = df[FEATURE_COLUMNS], df["target_aqi_1h"]

    s = {
        "X_train": X.iloc[:train_end],       "y_train": y.iloc[:train_end],
        "X_val":   X.iloc[train_end:val_end], "y_val":   y.iloc[train_end:val_end],
        "X_test":  X.iloc[val_end:],          "y_test":  y.iloc[val_end:],
    }
    print(f"   Split → train {len(s['X_train'])} | val {len(s['X_val'])} | test {len(s['X_test'])}")
    return s


# --------------------------------------------------------------------------
# TRAIN
# --------------------------------------------------------------------------
def evaluate(y_true, y_pred, prefix):
    return {
        f"{prefix}_mae":  round(float(mean_absolute_error(y_true, y_pred)), 4),
        f"{prefix}_rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        f"{prefix}_r2":   round(float(r2_score(y_true, y_pred)), 4),
    }


def train_and_evaluate(s):
    """Train scaled Ridge.

    FIX: raw Ridge on unscaled features penalised surface_pressure (~1000) and
    hour_sin (~1) with the same alpha, effectively ignoring the small-scale
    features. StandardScaler inside a Pipeline fixes this and keeps
    joblib.load(...).predict(X) working exactly as before.
    """
    print("\n🧠 Training Ridge (StandardScaler + Ridge)...")
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=RIDGE_ALPHA)),
    ])
    model.fit(s["X_train"], s["y_train"])

    metrics = {}
    metrics.update(evaluate(s["y_val"],  model.predict(s["X_val"]),  "val"))
    metrics.update(evaluate(s["y_test"], model.predict(s["X_test"]), "test"))

    # Persistence baseline: "next hour = this hour". If we can't beat it, stop.
    base = evaluate(s["y_test"], s["X_test"]["us_aqi"].values, "baseline")
    metrics.update(base)

    print(f"   Val      : MAE={metrics['val_mae']:<8} RMSE={metrics['val_rmse']:<8} R²={metrics['val_r2']}")
    print(f"   Test     : MAE={metrics['test_mae']:<8} RMSE={metrics['test_rmse']:<8} R²={metrics['test_r2']}")
    print(f"   Baseline : MAE={base['baseline_mae']:<8} RMSE={base['baseline_rmse']:<8} R²={base['baseline_r2']}")

    if metrics["test_rmse"] < base["baseline_rmse"]:
        gain = 100 * (1 - metrics["test_rmse"] / base["baseline_rmse"])
        print(f"   ✅ Beats persistence baseline by {gain:.1f}%")
    else:
        print("   ⚠️ Model does NOT beat the naive persistence baseline")

    return model, metrics


def save_model_locally(model, metrics):
    print("\n💾 Saving artifacts...")
    joblib.dump(model, MODEL_DIR / "ridge_aqi_1h.pkl")
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (MODEL_DIR / "feature_columns.json").write_text(json.dumps(FEATURE_COLUMNS, indent=2))

    if RIDGE_EXPORT_DIR.exists():
        shutil.rmtree(RIDGE_EXPORT_DIR)
    RIDGE_EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, RIDGE_EXPORT_DIR / "ridge_aqi_1h.pkl")
    (RIDGE_EXPORT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (RIDGE_EXPORT_DIR / "feature_columns.json").write_text(json.dumps(FEATURE_COLUMNS, indent=2))
    print(f"   ✅ {RIDGE_EXPORT_DIR}")


# --------------------------------------------------------------------------
# REGISTRY
# --------------------------------------------------------------------------
def build_input_example(X_train: pd.DataFrame) -> pd.DataFrame:
    """Derive the example from real data instead of hardcoding 29 numbers."""
    return X_train.tail(1).reset_index(drop=True)


def upload_to_hopsworks(metrics, input_example, only_if_better=False) -> bool:
    """Register the model. Raises on failure — never silently skips."""
    project = hopsworks_login()
    if project is None:
        print("ℹ️ No HOPSWORKS_API_KEY — skipping model registry upload")
        return True

    print("\n📤 Uploading to Hopsworks Model Registry...")
    mr = project.get_model_registry()

    if only_if_better:
        best = None
        try:
            for m in mr.get_models(MODEL_NAME):
                r = (m.training_metrics or {}).get("test_rmse")
                if r is not None:
                    r = float(r)
                    if best is None or r < best:
                        best = r
        except Exception as e:
            print(f"   ⚠️ Could not read existing models ({e}) — registering anyway")

        if best is not None:
            print(f"   Best registered test_rmse: {best} | new: {metrics['test_rmse']}")
            if metrics["test_rmse"] >= best:
                print("   ⏭️ Not an improvement — skipping registration (no version bump)")
                return True

    entry = mr.sklearn.create_model(
        name=MODEL_NAME,
        metrics=metrics,
        description=(
            f"Daily retrained Ridge (scaled). "
            f"Test RMSE {metrics['test_rmse']} | R² {metrics['test_r2']} | "
            f"baseline RMSE {metrics['baseline_rmse']}"
        ),
        input_example=input_example,
    )
    entry.save(str(RIDGE_EXPORT_DIR))
    print(f"   ✅ Registered {MODEL_NAME} v{entry.version}")
    return True


# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------
def run_daily_pipeline(source="auto", upload=True, only_if_better=False) -> bool:
    print("=" * 60)
    print("📅 DAILY TRAINING PIPELINE STARTED")
    print(f"Run time: {datetime.utcnow():%Y-%m-%d %H:%M:%S} UTC")
    print("=" * 60)

    df = load_dataframe(source)
    df = build_target(df)
    splits = time_split(df)
    model, metrics = train_and_evaluate(splits)
    save_model_locally(model, metrics)

    if upload:
        upload_to_hopsworks(metrics, build_input_example(splits["X_train"]), only_if_better)
    else:
        print("\nℹ️ --no-upload set, skipping registry")

    print("=" * 60)
    print("✅ DAILY TRAINING COMPLETE")
    print("=" * 60)
    return True


def main():
    p = argparse.ArgumentParser(description="Lahore AQI daily training pipeline")
    p.add_argument("--source", choices=["auto", "hopsworks", "csv"], default="auto")
    p.add_argument("--no-upload", action="store_true")
    p.add_argument("--only-if-better", action="store_true",
                   help="Only register if test_rmse improves on the best existing version")
    a = p.parse_args()

    try:
        ok = run_daily_pipeline(a.source, not a.no_upload, a.only_if_better)
    except Exception:
        print("\n❌ TRAINING PIPELINE FAILED")
        traceback.print_exc()
        sys.exit(1)          
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
