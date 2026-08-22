"""
One-Time Backfill Pipeline
Fetches a long historical range of weather and pollutant
data to fill gaps in lahore_features_hourly.csv.
"""

import os
import sys
import warnings
import urllib.request
from datetime import datetime
from pathlib import Path
import json

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "lahore" / "lahore_features_hourly.csv"
DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

LAT = 31.5204
LON = 74.3587


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
    with urllib.request.urlopen(url) as response:
        data = json.loads(response.read().decode())

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
    with urllib.request.urlopen(url) as response:
        data = json.loads(response.read().decode())

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
    return df


def add_time_features(df):
    """Add cyclical and time-based features."""
    df["hour"] = df["datetime"].dt.hour
    df["day"] = df["datetime"].dt.day
    df["month"] = df["datetime"].dt.month
    df["day_of_week"] = df["datetime"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].apply(lambda x: 1 if x >= 5 else 0)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12.0)
    return df


def add_lag_and_rolling_features(df):
    """Add lag and rolling features to the most recent rows."""
    df = df.sort_values("datetime").reset_index(drop=True)

    for i in range(max(0, len(df) - 24), len(df)):
        idx = i

        if idx >= 2:
            lag_1h = df.iloc[idx - 1]["us_aqi"]
            lag_2h = df.iloc[idx - 2]["us_aqi"]
            df.at[idx, "aqi_change_rate"] = (lag_1h - lag_2h) / (lag_2h + 1e-5)
        else:
            df.at[idx, "aqi_change_rate"] = 0.0

        if idx >= 1:
            df.at[idx, "aqi_lag_1h"] = df.iloc[idx - 1]["us_aqi"]
        else:
            df.at[idx, "aqi_lag_1h"] = df.iloc[idx]["us_aqi"]

        if idx >= 24:
            df.at[idx, "aqi_lag_24h"] = df.iloc[idx - 24]["us_aqi"]
        else:
            df.at[idx, "aqi_lag_24h"] = df.iloc[idx]["us_aqi"]

        if idx >= 5:
            window = df.iloc[idx - 5:idx + 1]["us_aqi"]
            df.at[idx, "aqi_rolling_mean_6h"] = window.mean()
            df.at[idx, "aqi_rolling_std_6h"] = window.std() if len(window) > 1 else 0.0
            df.at[idx, "pm25_rolling_mean_6h"] = df.iloc[idx - 5:idx + 1]["pm2_5"].mean()
        else:
            df.at[idx, "aqi_rolling_mean_6h"] = df.iloc[idx]["us_aqi"]
            df.at[idx, "aqi_rolling_std_6h"] = 0.0
            df.at[idx, "pm25_rolling_mean_6h"] = df.iloc[idx]["pm2_5"]

    return df


def run_backfill():
    """Main backfill execution function."""
    print("=" * 60)
    print("🔄 BACKFILL PIPELINE STARTED")
    print(f"Run time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Load existing data
    if DATA_PATH.exists():
        existing_df = pd.read_csv(DATA_PATH, parse_dates=["datetime"])
        existing_df = existing_df.sort_values("datetime").reset_index(drop=True)
        print(f"Existing data: {len(existing_df)} rows")
        print(f"Date range: {existing_df['datetime'].min()} to {existing_df['datetime'].max()}")

        # Determine the gap to backfill
        last_existing_date = existing_df["datetime"].max().strftime("%Y-%m-%d")
        first_existing_date = existing_df["datetime"].min().strftime("%Y-%m-%d")

        # Find missing date ranges
        all_dates = pd.date_range(start=first_existing_date, end=last_existing_date, freq="D")
        existing_dates = pd.to_datetime(existing_df["datetime"].dt.date).unique()
        missing_dates = [d for d in all_dates if d.date() not in [ed.date() for ed in existing_dates]]

        if not missing_dates:
            print("✅ No gaps found in existing data")
            return True

        # Group consecutive missing dates into ranges
        missing_dates = sorted(missing_dates)
        ranges = []
        start = missing_dates[0]
        prev = missing_dates[0]
        for d in missing_dates[1:]:
            if (d - prev).days > 1:
                ranges.append((start, prev))
                start = d
            prev = d
        ranges.append((start, prev))

        print(f"📅 Found {len(ranges)} date ranges to backfill:")
        for s, e in ranges:
            print(f"   {s.strftime('%Y-%m-%d')} to {e.strftime('%Y-%m-%d')}")
    else:
        print("❌ No existing data file found. Run feature_pipeline.py first.")
        return False

    # Fetch backfill data for each range
    all_backfill_dfs = []
    for start_date, end_date in ranges:
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")

        try:
            weather_df = fetch_historical_weather(start_str, end_str)
            pollutants_df = fetch_historical_pollutants(start_str, end_str)

            merged_df = pd.merge(weather_df, pollutants_df, on="datetime", how="inner")
            merged_df = merged_df.dropna()
            print(f"   Merged: {len(merged_df)} rows for {start_str} to {end_str}")
            all_backfill_dfs.append(merged_df)
        except Exception as e:
            print(f"   ⚠️ Failed to fetch {start_str} to {end_str}: {e}")

    if not all_backfill_dfs:
        print("❌ No backfill data fetched")
        return False

    # Combine backfill with existing
    backfill_df = pd.concat(all_backfill_dfs, ignore_index=True)
    backfill_df = backfill_df.drop_duplicates(subset="datetime", keep="last")
    backfill_df = backfill_df.sort_values("datetime").reset_index(drop=True)

    # Remove any future timestamps
    now = pd.Timestamp.now(tz=backfill_df["datetime"].dt.tz)
    backfill_df = backfill_df[backfill_df["datetime"] <= now]

    print(f"\n🔧 Engineering features for {len(backfill_df)} backfill rows...")
    backfill_df = add_time_features(backfill_df)

    # Combine with existing
    combined_df = pd.concat([existing_df, backfill_df], ignore_index=True)
    combined_df = combined_df.drop_duplicates(subset="datetime", keep="last")
    combined_df = combined_df.sort_values("datetime").reset_index(drop=True)

    # Recompute lag/rolling for the new rows
    combined_df = add_lag_and_rolling_features(combined_df)

    # Save
    combined_df.to_csv(DATA_PATH, index=False)
    print(f"✅ Saved to: {DATA_PATH}")
    print(f"Total rows: {len(combined_df)}")
    print(f"Date range: {combined_df['datetime'].min()} to {combined_df['datetime'].max()}")

    # Upload to Hopsworks
    try:
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.getenv("HOPSWORKS_API_KEY")
        if api_key:
            print("\n📤 Uploading to Hopsworks Feature Store...")
            import hopsworks
            project = hopsworks.login(api_key_value=api_key)
            fs = project.get_feature_store()
            fg = fs.get_feature_group("lahore_air_quality_features", version=1)
            fg.insert(combined_df.tail(len(backfill_df) + 24), wait=True)
            print("✅ Hopsworks upload complete")
    except Exception as e:
        print(f"⚠️ Hopsworks upload skipped: {e}")

    print("=" * 60)
    print("✅ BACKFILL COMPLETE")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = run_backfill()
    sys.exit(0 if success else 1)