"""
Hourly Feature Pipeline — Lahore AQI
====================================
Fetches recent weather + pollutant data, engineers features, appends to
lahore_features_hourly.csv, and upserts into the Hopsworks Feature Store
(lahore_air_quality_features v2, streaming).

Usage:
    python src/feature_pipeline.py                  # normal hourly run
    python src/feature_pipeline.py --past-days 30   # close a data gap
    python src/feature_pipeline.py --no-upload      # local CSV only
"""

import argparse
import json
import os
import sys 
import time
import traceback
import urllib.request
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "lahore" / "lahore_features_hourly.csv"
DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

LAT, LON = 31.5204, 74.3587
CITY_TZ = "Asia/Karachi"          # Open-Meteo timezone=auto returns LOCAL naive times

FG_NAME = os.getenv("HOPSWORKS_FEATURE_GROUP", "lahore_air_quality_features")
FG_VERSION = int(os.getenv("HOPSWORKS_FEATURE_GROUP_VERSION", "2"))                   # v2 = streaming-enabled
FG_PRIMARY_KEY = ["datetime"]
FG_EVENT_TIME = "datetime"

DEFAULT_PAST_DAYS = 3              # forecast API supports up to 92
OVERLAP_HOURS = 24                 # re-upload recent rows so lags self-correct

INT_COLS = ["hour", "day", "month", "day_of_week", "is_weekend"]

FEATURE_COLUMNS = [
    "datetime", "temperature_2m", "relative_humidity_2m", "surface_pressure",
    "precipitation", "cloud_cover", "wind_speed_10m", "wind_direction_10m",
    "pm2_5", "pm10", "carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide",
    "ozone", "us_aqi", "hour", "day", "month", "day_of_week", "is_weekend",
    "hour_sin", "hour_cos", "month_sin", "month_cos", "aqi_change_rate",
    "aqi_lag_1h", "aqi_lag_24h", "aqi_rolling_mean_6h", "aqi_rolling_std_6h",
    "pm25_rolling_mean_6h",
]


# --------------------------------------------------------------------------
# HELPERS
# --------------------------------------------------------------------------
def now_local() -> pd.Timestamp:
    """Current Lahore wall-clock time as a NAIVE timestamp.

    FIX: previously used pd.Timestamp.now() which is UTC on CI runners.
    Lahore is UTC+5, so the newest 5 hours were silently discarded.
    """
    return pd.Timestamp.now(tz=CITY_TZ).tz_localize(None)


def http_get_json(url: str, retries: int = 3, timeout: int = 30) -> dict:
    """GET JSON with retries and exponential backoff."""
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                return json.loads(resp.read().decode())
        except Exception as e:
            last_err = e
            print(f"   ⚠️ attempt {attempt}/{retries} failed: {e}")
            if attempt < retries:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"Request failed after {retries} attempts:\n{url}\n{last_err}")


def fg_column_names(fg) -> list:
    """Read FG schema, tolerating the .features -> .columns deprecation."""
    cols = getattr(fg, "columns", None)
    if cols:
        return [c if isinstance(c, str) else c.name for c in cols]
    return [f.name for f in fg.features]


# --------------------------------------------------------------------------
# DATA FETCHING
# --------------------------------------------------------------------------
WEATHER_VARS = (
    "temperature_2m,relative_humidity_2m,surface_pressure,precipitation,"
    "cloud_cover,wind_speed_10m,wind_direction_10m"
)
POLLUTANT_VARS = (
    "pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone,us_aqi"
)


def fetch_weather(past_days=DEFAULT_PAST_DAYS, start_date=None, end_date=None) -> pd.DataFrame:
    """Fetch hourly weather.

    FIX: the old code used archive-api (ERA5), which lags ~5 days behind and
    returns nulls for recent hours -> dropna() wiped everything -> "no new data".
    We now use the forecast endpoint with past_days for recent data, and only
    fall back to the archive for explicit historical date ranges.
    """
    if start_date and end_date:
        url = (
            f"https://archive-api.open-meteo.com/v1/archive?"
            f"latitude={LAT}&longitude={LON}"
            f"&start_date={start_date}&end_date={end_date}"
            f"&hourly={WEATHER_VARS}&timezone=auto"
        )
    else:
        url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={LAT}&longitude={LON}"
            f"&hourly={WEATHER_VARS}"
            f"&past_days={past_days}&forecast_days=1&timezone=auto"
        )

    h = http_get_json(url)["hourly"]
    df = pd.DataFrame({
        "datetime": pd.to_datetime(h["time"]),
        "temperature_2m": h["temperature_2m"],
        "relative_humidity_2m": h["relative_humidity_2m"],
        "surface_pressure": h["surface_pressure"],
        "precipitation": h["precipitation"],
        "cloud_cover": h["cloud_cover"],
        "wind_speed_10m": h["wind_speed_10m"],
        "wind_direction_10m": h["wind_direction_10m"],
    })
    df = df[df["datetime"] <= now_local()].dropna()
    return df.reset_index(drop=True)


def fetch_pollutants(past_days=DEFAULT_PAST_DAYS, start_date=None, end_date=None) -> pd.DataFrame:
    """Fetch hourly pollutants.

    FIX: removed the old fallback that duplicated the last CSV row — that
    fabricated fake data. Now it raises so the failure is visible.
    """
    base = f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={LAT}&longitude={LON}&hourly={POLLUTANT_VARS}&timezone=auto"
    url = (
        f"{base}&start_date={start_date}&end_date={end_date}"
        if start_date and end_date else f"{base}&past_days={past_days}"
    )

    h = http_get_json(url)["hourly"]
    df = pd.DataFrame({
        "datetime": pd.to_datetime(h["time"]),
        "pm2_5": h["pm2_5"],
        "pm10": h["pm10"],
        "carbon_monoxide": h["carbon_monoxide"],
        "nitrogen_dioxide": h["nitrogen_dioxide"],
        "sulphur_dioxide": h["sulphur_dioxide"],
        "ozone": h["ozone"],
        "us_aqi": h["us_aqi"],
    })
    df = df[df["datetime"] <= now_local()].dropna()
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------
# FEATURE ENGINEERING
# --------------------------------------------------------------------------
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df["hour"] = df["datetime"].dt.hour
    df["day"] = df["datetime"].dt.day
    df["month"] = df["datetime"].dt.month
    df["day_of_week"] = df["datetime"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12.0)
    return df


def add_lag_and_rolling_features(df: pd.DataFrame, recompute_tail: int = 48) -> pd.DataFrame:
    """Vectorised lag/rolling features.

    FIX: the old row-by-row loop with df.at[] was O(n) Python and error-prone.
    Same semantics, ~1000x faster. Only the last `recompute_tail` rows are
    overwritten so previously-computed history stays byte-identical.
    """
    df = df.sort_values("datetime").reset_index(drop=True)
    aqi, pm25 = df["us_aqi"], df["pm2_5"]

    lag_1, lag_2, lag_24 = aqi.shift(1), aqi.shift(2), aqi.shift(24)

    computed = pd.DataFrame({
        "aqi_change_rate": ((lag_1 - lag_2) / (lag_2 + 1e-5)).replace([np.inf, -np.inf], 0.0).fillna(0.0),
        "aqi_lag_1h": lag_1.fillna(aqi),
        "aqi_lag_24h": lag_24.fillna(aqi),
        "aqi_rolling_mean_6h": aqi.rolling(6, min_periods=1).mean(),
        "aqi_rolling_std_6h": aqi.rolling(6, min_periods=1).std().fillna(0.0),
        "pm25_rolling_mean_6h": pm25.rolling(6, min_periods=1).mean(),
    }, index=df.index)

    for col in computed.columns:
        if col not in df.columns:
            df[col] = np.nan

    if recompute_tail and len(df) > recompute_tail:
        tail = df.index[-recompute_tail:]
        df.loc[tail, computed.columns] = computed.loc[tail]
    else:
        df[computed.columns] = computed

    return df


# --------------------------------------------------------------------------
# HOPSWORKS UPLOAD
# --------------------------------------------------------------------------
def upload_to_hopsworks(combined_df: pd.DataFrame) -> bool:
    """Upsert recent rows into the streaming feature group."""

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    api_key = os.getenv("HOPSWORKS_API_KEY")
    if not api_key:
        print("ℹ️ No HOPSWORKS_API_KEY set — skipping cloud upload (local CSV only)")
        return True

    print("\n📤 Uploading to Hopsworks Feature Store...")
    import hopsworks

    project = hopsworks.login(
        host=os.getenv("HOPSWORKS_HOST", "eu-west.cloud.hopsworks.ai"),
        project=os.getenv("HOPSWORKS_PROJECT", "internship10P"),
        api_key_value=api_key,
    )
    fs = project.get_feature_store()

    # stream=True routes data via Kafka instead of direct HDFS RPC,
    # which is blocked for external clients on Hopsworks Serverless.
    fg = fs.get_or_create_feature_group(
        name=FG_NAME,
        version=FG_VERSION,
        primary_key=FG_PRIMARY_KEY,
        event_time=FG_EVENT_TIME,
        description="Lahore AQI hourly features — streaming ingestion",
        online_enabled=True,
        stream=True,
    )
    # get_or_create silently IGNORES stream=True if the FG already exists.
    # Fail loudly rather than crashing later with 'RPC listener disconnected'.
    
    if getattr(fg, "stream", False) is not True:
        raise RuntimeError(
            f"{FG_NAME} v{FG_VERSION} has stream=False. External writes will fail. "
            f"Set HOPSWORKS_FEATURE_GROUP_VERSION to a streaming-enabled version."
        )

    
    try:
        existing = fg.read()
        remote_hours = pd.DatetimeIndex(
            pd.to_datetime(existing["datetime"]).dt.floor("h").unique()
        )
        print(f"   Feature store holds {len(existing)} rows"
              + (f", latest {remote_hours.max()}" if len(remote_hours) else " (empty)"))
    except Exception as e:
        print(f"   ⚠️ Could not read feature group ({e}) — will upload everything")
        remote_hours = pd.DatetimeIndex([])

    local_hours = combined_df["datetime"].dt.floor("h")

    missing_mask = ~local_hours.isin(remote_hours)          # anywhere in the series
    if len(remote_hours):
        overlap_cutoff = combined_df["datetime"].max() - pd.Timedelta(hours=OVERLAP_HOURS)
        overlap_mask = combined_df["datetime"] > overlap_cutoff   # refresh recent lags
    else:
        overlap_mask = pd.Series(False, index=combined_df.index)

    n_missing = int(missing_mask.sum())
    upload_df = combined_df[missing_mask | overlap_mask].copy()

    if upload_df.empty:
        print("ℹ️ Feature store already in sync — nothing to upload")
        return True
        
    # --- dtype hygiene ---
    upload_df["datetime"] = (
        pd.to_datetime(upload_df["datetime"])
          .dt.tz_localize(None)
          .astype("datetime64[us]")          # avoids ns->us precision warning
    )
    for col in INT_COLS:
        upload_df[col] = upload_df[col].fillna(0).astype("int64")   # schema says bigint

    # --- align to live FG schema ---
    schema = fg_column_names(fg)
    missing = set(schema) - set(upload_df.columns)
    if missing:
        raise ValueError(f"DataFrame missing columns required by feature group: {missing}")
    upload_df = upload_df[schema]

    print(f"   Sending {len(upload_df)} rows ({n_missing} missing, "
          f"{len(upload_df) - n_missing} re-sent to refresh lag features)")

    # datetime is the primary key, so overlapping rows upsert rather than duplicate.
    fg.insert(upload_df, write_options={"wait_for_job": False})

    print(f"✅ Upload complete — latest timestamp sent: {upload_df['datetime'].max()}")
    return True


# --------------------------------------------------------------------------
# MAIN PIPELINE
# --------------------------------------------------------------------------
def run_hourly_pipeline(past_days=DEFAULT_PAST_DAYS, upload=True) -> bool:
    print("=" * 60)
    print("⏰ HOURLY FEATURE PIPELINE STARTED")
    print(f"Run time (UTC)   : {datetime.utcnow():%Y-%m-%d %H:%M:%S}")
    print(f"Run time (Lahore): {now_local():%Y-%m-%d %H:%M:%S}")
    print(f"Fetch window     : last {past_days} day(s)")
    print("=" * 60)

    # 1. Load existing CSV
    if DATA_PATH.exists():
        existing_df = pd.read_csv(DATA_PATH, parse_dates=["datetime"])
        print(f"Existing CSV rows: {len(existing_df)}")
        before = len(existing_df)
        existing_df = existing_df[existing_df["datetime"] <= now_local()]
        if before - len(existing_df):
            print(f" Removed {before - len(existing_df)} future-dated rows")
    else:
        existing_df = pd.DataFrame()
        print("No existing CSV — starting fresh")

    # 2 & 3. Fetch
    print("\n📡 Fetching weather...")
    weather_df = fetch_weather(past_days=past_days)
    print(f"   {len(weather_df)} weather rows"
          + (f" ({weather_df['datetime'].min()} → {weather_df['datetime'].max()})" if len(weather_df) else ""))

    print("📡 Fetching pollutants...")
    pollutants_df = fetch_pollutants(past_days=past_days)
    print(f"   {len(pollutants_df)} pollutant rows"
          + (f" ({pollutants_df['datetime'].min()} → {pollutants_df['datetime'].max()})" if len(pollutants_df) else ""))

    if weather_df.empty or pollutants_df.empty:
        print("❌ One or both APIs returned no usable rows. Aborting.")
        return False

    # 4. Merge
    merged_df = pd.merge(weather_df, pollutants_df, on="datetime", how="inner").dropna()
    print(f"🔗 Merged: {len(merged_df)} rows")
    if merged_df.empty:
        print("❌ Merge produced 0 rows (no overlapping timestamps). Aborting.")
        return False

    # 5. Combine with history
    combined_df = (
        pd.concat([existing_df, merged_df], ignore_index=True)
          .drop_duplicates(subset="datetime", keep="last")
          .sort_values("datetime")
          .reset_index(drop=True)
    )

    # 6. Feature engineering
    print(" Engineering features...")
    combined_df = add_time_features(combined_df)
    combined_df = add_lag_and_rolling_features(
        combined_df, recompute_tail=max(48, len(merged_df) + 24)
    )

    missing = set(FEATURE_COLUMNS) - set(combined_df.columns)
    if missing:
        print(f"❌ Missing engineered columns: {missing}")
        return False
    combined_df = combined_df[FEATURE_COLUMNS]

    # 7. Save CSV
    combined_df.to_csv(DATA_PATH, index=False)
    print(f"💾 Saved {len(combined_df)} rows → {DATA_PATH}")
    print(f"   Range: {combined_df['datetime'].min()} → {combined_df['datetime'].max()}")

    # 8. Upload
    if upload:
        if not upload_to_hopsworks(combined_df):
            return False
    else:
        print("\nℹ️ --no-upload flag set, skipping Hopsworks")

    print("=" * 60)
    print("✅ HOURLY PIPELINE COMPLETE")
    print("=" * 60)
    return True


def main():
    parser = argparse.ArgumentParser(description="Lahore AQI hourly feature pipeline")
    parser.add_argument("--past-days", type=int, default=DEFAULT_PAST_DAYS,
                        help="Days of history to fetch (max 92). Use ~30 to close a gap.")
    parser.add_argument("--no-upload", action="store_true",
                        help="Write CSV only, skip Hopsworks")
    args = parser.parse_args()

    if not 1 <= args.past_days <= 92:
        print("❌ --past-days must be between 1 and 92")
        sys.exit(1)

    try:
        ok = run_hourly_pipeline(past_days=args.past_days, upload=not args.no_upload)
    except Exception:
        print("\n❌ PIPELINE CRASHED")
        traceback.print_exc()
        sys.exit(1)         

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
