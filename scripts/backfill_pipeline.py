"""
One-Time Backfill Pipeline

Fetches missing hourly weather and pollutant data and fills gaps in:
data/processed/lahore/lahore_features_hourly.csv

This pipeline:
1. Detects missing hourly timestamps.
2. Fetches historical weather and air-quality data from Open-Meteo.
3. Recomputes time, lag, and rolling features for the full dataset.
4. Saves the corrected CSV.
5. Uploads the corrected data to Hopsworks Feature Store if API key exists.
"""

import os
import sys
import time
import json
import warnings
import urllib.request
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "lahore" / "lahore_features_hourly.csv"
DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

LAT = 31.5204
LON = 74.3587

FEATURE_GROUP_NAME = os.getenv("HOPSWORKS_FEATURE_GROUP", "lahore_air_quality_features")
FEATURE_GROUP_VERSION = int(os.getenv("HOPSWORKS_FEATURE_GROUP_VERSION", "1"))

FINAL_COLUMNS = [
    "datetime",
    "temperature_2m",
    "relative_humidity_2m",
    "surface_pressure",
    "precipitation",
    "cloud_cover",
    "wind_speed_10m",
    "wind_direction_10m",
    "pm2_5",
    "pm10",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
    "us_aqi",
    "hour",
    "day",
    "month",
    "day_of_week",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    "aqi_change_rate",
    "aqi_lag_1h",
    "aqi_lag_24h",
    "aqi_rolling_mean_6h",
    "aqi_rolling_std_6h",
    "pm25_rolling_mean_6h",
]


def read_json_from_url(url, timeout=60):
    """Read JSON from URL with small retry handling."""
    last_error = None

    for attempt in range(1, 4):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as e:
            last_error = e
            print(f"⚠️ API request failed attempt {attempt}/3: {e}")
            if attempt < 3:
                time.sleep(5 * attempt)

    raise last_error


def fetch_historical_weather(start_date, end_date):
    """Fetch historical weather between two dates."""
    url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={LAT}&longitude={LON}"
        f"&start_date={start_date}&end_date={end_date}"
        f"&hourly=temperature_2m,relative_humidity_2m,surface_pressure,"
        f"precipitation,cloud_cover,wind_speed_10m,wind_direction_10m"
        f"&timezone=auto"
    )

    print(f"📡 Fetching weather from {start_date} to {end_date}...")
    data = read_json_from_url(url)

    if "hourly" not in data:
        raise ValueError(f"Weather API returned no hourly data: {data}")

    hourly = data["hourly"]

    df = pd.DataFrame({
        "datetime": pd.to_datetime(hourly["time"]),
        "temperature_2m": hourly["temperature_2m"],
        "relative_humidity_2m": hourly["relative_humidity_2m"],
        "surface_pressure": hourly["surface_pressure"],
        "precipitation": hourly["precipitation"],
        "cloud_cover": hourly["cloud_cover"],
        "wind_speed_10m": hourly["wind_speed_10m"],
        "wind_direction_10m": hourly["wind_direction_10m"],
    })

    df["datetime"] = pd.to_datetime(df["datetime"]).dt.tz_localize(None)
    return df


def fetch_historical_pollutants(start_date, end_date):
    """Fetch historical pollutant data."""
    url = (
        f"https://air-quality-api.open-meteo.com/v1/air-quality?"
        f"latitude={LAT}&longitude={LON}"
        f"&hourly=pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,"
        f"sulphur_dioxide,ozone,us_aqi"
        f"&start_date={start_date}&end_date={end_date}"
        f"&timezone=auto"
    )

    print(f"📡 Fetching pollutants from {start_date} to {end_date}...")
    data = read_json_from_url(url)

    if "hourly" not in data:
        raise ValueError(f"Air-quality API returned no hourly data: {data}")

    hourly = data["hourly"]

    df = pd.DataFrame({
        "datetime": pd.to_datetime(hourly["time"]),
        "pm2_5": hourly["pm2_5"],
        "pm10": hourly["pm10"],
        "carbon_monoxide": hourly["carbon_monoxide"],
        "nitrogen_dioxide": hourly["nitrogen_dioxide"],
        "sulphur_dioxide": hourly["sulphur_dioxide"],
        "ozone": hourly["ozone"],
        "us_aqi": hourly["us_aqi"],
    })

    df["datetime"] = pd.to_datetime(df["datetime"]).dt.tz_localize(None)
    return df


def add_time_features(df):
    """Add cyclical and calendar features."""
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"]).dt.tz_localize(None)

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


def add_lag_and_rolling_features(df):
    """
    Recompute lag and rolling features for the full dataset.

    This is important for backfill because gaps may exist in the middle,
    not only at the latest rows.
    """
    df = df.copy()
    df = df.sort_values("datetime").reset_index(drop=True)

    df["aqi_lag_1h"] = df["us_aqi"].shift(1)
    df["aqi_lag_24h"] = df["us_aqi"].shift(24)

    df["aqi_change_rate"] = (
        (df["us_aqi"].shift(1) - df["us_aqi"].shift(2)) /
        (df["us_aqi"].shift(2) + 1e-5)
    )

    df["aqi_rolling_mean_6h"] = (
        df["us_aqi"]
        .rolling(window=6, min_periods=1)
        .mean()
    )

    df["aqi_rolling_std_6h"] = (
        df["us_aqi"]
        .rolling(window=6, min_periods=1)
        .std()
        .fillna(0.0)
    )

    df["pm25_rolling_mean_6h"] = (
        df["pm2_5"]
        .rolling(window=6, min_periods=1)
        .mean()
    )

    df["aqi_lag_1h"] = df["aqi_lag_1h"].fillna(df["us_aqi"])
    df["aqi_lag_24h"] = df["aqi_lag_24h"].fillna(df["us_aqi"])
    df["aqi_change_rate"] = df["aqi_change_rate"].replace([np.inf, -np.inf], 0.0).fillna(0.0)

    return df


def enforce_schema(df):
    """Clean column order and enforce stable dtypes."""
    df = df.copy()

    df["datetime"] = pd.to_datetime(df["datetime"]).dt.tz_localize(None)

    int_columns = [
        "hour",
        "day",
        "month",
        "day_of_week",
        "is_weekend",
    ]

    numeric_columns = [col for col in FINAL_COLUMNS if col not in ["datetime"]]

    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in int_columns:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype("int64")

    float_columns = [col for col in numeric_columns if col not in int_columns]
    for col in float_columns:
        if col in df.columns:
            df[col] = df[col].astype("float64")

    df = df[FINAL_COLUMNS]
    return df


def find_missing_hourly_ranges(existing_df):
    """Find missing hourly timestamps and convert them into date ranges for API calls."""
    existing_df = existing_df.copy()
    existing_df["datetime"] = pd.to_datetime(existing_df["datetime"]).dt.tz_localize(None)
    existing_df = existing_df.sort_values("datetime").reset_index(drop=True)

    first_ts = existing_df["datetime"].min().floor("h")
    last_ts = existing_df["datetime"].max().floor("h")

    expected_hours = pd.date_range(start=first_ts, end=last_ts, freq="h")
    existing_hours = pd.DatetimeIndex(existing_df["datetime"].dt.floor("h").unique())

    missing_hours = expected_hours.difference(existing_hours)

    if len(missing_hours) == 0:
        return missing_hours, []

    missing_dates = sorted(pd.to_datetime(pd.Series(missing_hours.date).unique()))

    ranges = []
    start = pd.Timestamp(missing_dates[0])
    prev = pd.Timestamp(missing_dates[0])

    for d in missing_dates[1:]:
        d = pd.Timestamp(d)
        if (d - prev).days > 1:
            ranges.append((start, prev))
            start = d
        prev = d

    ranges.append((start, prev))
    return missing_hours, ranges


def upload_to_hopsworks(df):
    """Upload corrected data to Hopsworks Feature Store."""
    try:
        from dotenv import load_dotenv
        load_dotenv()

        api_key = os.getenv("HOPSWORKS_API_KEY")

        if not api_key:
            print("ℹ️ No HOPSWORKS_API_KEY found. Skipping Hopsworks upload.")
            return

        print("\n📤 Uploading corrected data to Hopsworks Feature Store...")
        print(f"Feature group: {FEATURE_GROUP_NAME}, version: {FEATURE_GROUP_VERSION}")
        print(f"Rows to upload: {len(df)}")

        import hopsworks

        project = hopsworks.login(api_key_value=api_key)
        fs = project.get_feature_store()
        fg = fs.get_feature_group(FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)

        last_error = None

        for attempt in range(1, 4):
            try:
                print(f"Upload attempt {attempt}/3...")

                try:
                    fg.insert(
                        df,
                        write_options={"wait_for_job": True}
                    )
                except TypeError:
                    # Compatibility fallback for older HSFS versions
                    fg.insert(df, wait=True)

                print("✅ Hopsworks upload complete")
                return

            except Exception as e:
                last_error = e
                print(f"⚠️ Hopsworks upload attempt {attempt}/3 failed: {e}")
                if attempt < 3:
                    time.sleep(15 * attempt)

        print(f"⚠️ Hopsworks upload skipped after retries: {last_error}")

    except Exception as e:
        print(f"⚠️ Hopsworks upload skipped: {e}")


def run_backfill():
    """Main backfill execution function."""
    print("=" * 60)
    print("🔄 BACKFILL PIPELINE STARTED")
    print(f"Run time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    if not DATA_PATH.exists():
        print(f"❌ No existing data file found at: {DATA_PATH}")
        print("Please run the hourly feature pipeline first.")
        return False

    existing_df = pd.read_csv(DATA_PATH, parse_dates=["datetime"])
    existing_df["datetime"] = pd.to_datetime(existing_df["datetime"]).dt.tz_localize(None)
    existing_df = existing_df.sort_values("datetime").reset_index(drop=True)

    print(f"Existing rows: {len(existing_df)}")
    print(f"Existing date range: {existing_df['datetime'].min()} to {existing_df['datetime'].max()}")

    missing_hours, ranges = find_missing_hourly_ranges(existing_df)

    if len(missing_hours) == 0:
        print("✅ No hourly gaps found in existing data.")
        print("=" * 60)
        print("✅ BACKFILL COMPLETE")
        print("=" * 60)
        return True

    print(f"📅 Found {len(missing_hours)} missing hourly timestamps.")
    print(f"📅 Found {len(ranges)} date ranges to fetch:")

    for start_date, end_date in ranges:
        print(f"   {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

    all_backfill_dfs = []

    for start_date, end_date in ranges:
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")

        try:
            weather_df = fetch_historical_weather(start_str, end_str)
            pollutants_df = fetch_historical_pollutants(start_str, end_str)

            merged_df = pd.merge(weather_df, pollutants_df, on="datetime", how="inner")
            merged_df["datetime"] = pd.to_datetime(merged_df["datetime"]).dt.tz_localize(None)

            # Keep only truly missing hours
            merged_df = merged_df[
                merged_df["datetime"].dt.floor("h").isin(missing_hours)
            ]

            merged_df = merged_df.dropna().sort_values("datetime").reset_index(drop=True)

            print(f"   ✅ Fetched {len(merged_df)} missing rows for {start_str} to {end_str}")

            if len(merged_df) > 0:
                all_backfill_dfs.append(merged_df)

        except Exception as e:
            print(f"   ⚠️ Failed to fetch {start_str} to {end_str}: {e}")

    if not all_backfill_dfs:
        print("❌ No backfill data was fetched.")
        return False

    backfill_df = pd.concat(all_backfill_dfs, ignore_index=True)
    backfill_df = backfill_df.drop_duplicates(subset="datetime", keep="last")
    backfill_df = backfill_df.sort_values("datetime").reset_index(drop=True)

    print(f"\n🔧 Backfill rows fetched: {len(backfill_df)}")
    print(f"Backfill date range: {backfill_df['datetime'].min()} to {backfill_df['datetime'].max()}")

    # Combine old + new rows
    combined_df = pd.concat([existing_df, backfill_df], ignore_index=True)
    combined_df["datetime"] = pd.to_datetime(combined_df["datetime"]).dt.tz_localize(None)

    combined_df = combined_df.drop_duplicates(subset="datetime", keep="last")
    combined_df = combined_df.sort_values("datetime").reset_index(drop=True)

    print("\n🔧 Recomputing features for full corrected dataset...")
    combined_df = add_time_features(combined_df)
    combined_df = add_lag_and_rolling_features(combined_df)
    combined_df = enforce_schema(combined_df)

    # Save corrected CSV
    combined_df.to_csv(DATA_PATH, index=False)

    print(f"\n✅ Saved corrected dataset to: {DATA_PATH}")
    print(f"Total rows now: {len(combined_df)}")
    print(f"Date range now: {combined_df['datetime'].min()} to {combined_df['datetime'].max()}")

    # Verify remaining gaps
    remaining_missing_hours, _ = find_missing_hourly_ranges(combined_df)

    if len(remaining_missing_hours) == 0:
        print("✅ Verification passed: no remaining hourly gaps.")
    else:
        print(f"⚠️ Verification warning: {len(remaining_missing_hours)} hourly gaps still remain.")

    # Upload full corrected dataset to Hopsworks
    upload_to_hopsworks(combined_df)

    print("=" * 60)
    print("✅ BACKFILL COMPLETE")
    print("=" * 60)

    return True


if __name__ == "__main__":
    success = run_backfill()
    sys.exit(0 if success else 1)
