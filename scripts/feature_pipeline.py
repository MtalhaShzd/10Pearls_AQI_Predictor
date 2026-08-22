"""
Hourly Feature Pipeline
Fetches the current hour's data and appends
engineered features to lahore_features_hourly.csv
"""

import os
import sys
import json
import warnings
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Make sure Python can find project modules
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "lahore" / "lahore_features_hourly.csv"
DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

# Lahore coordinates
LAT = 31.5204
LON = 74.3587


def fetch_current_weather():
    """Fetch current and past 48h weather for feature engineering."""
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d")

    url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={LAT}&longitude={LON}"
        f"&start_date={start_date}&end_date={end_date}"
        f"&hourly=temperature_2m,relative_humidity_2m,surface_pressure,"
        f"precipitation,cloud_cover,wind_speed_10m,wind_direction_10m"
        f"&timezone=auto"
    )

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

    #  Filter out any future timestamps
    now = pd.Timestamp.now(tz=df["datetime"].dt.tz)
    df = df[df["datetime"] <= now]
    df = df.dropna()
    return df

def fetch_current_pollutants():
    """
    Fetch current pollutant levels.
    Uses Open-Meteo Air Quality API (free, no key required).
    """
    url = (
        f"https://air-quality-api.open-meteo.com/v1/air-quality?"
        f"latitude={LAT}&longitude={LON}"
        f"&hourly=pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,"
        f"sulphur_dioxide,ozone,us_aqi"
        f"&timezone=auto&past_days=2"
    )

    try:
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

        # Filter out any future timestamps
        now = pd.Timestamp.now(tz=df["datetime"].dt.tz)
        df = df[df["datetime"] <= now]
        df = df.dropna()
        return df

    except Exception as e:
        print(f"⚠️ Using fallback pollutant estimation: {e}")
        if DATA_PATH.exists():
            last = pd.read_csv(DATA_PATH, parse_dates=["datetime"]).iloc[-1]
            return pd.DataFrame([{
                "datetime": last["datetime"],
                "pm2_5": last["pm2_5"],
                "pm10": last["pm10"],
                "carbon_monoxide": last["carbon_monoxide"],
                "nitrogen_dioxide": last["nitrogen_dioxide"],
                "sulphur_dioxide": last["sulphur_dioxide"],
                "ozone": last["ozone"],
                "us_aqi": last["us_aqi"],
            }])
        return None


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

    # Only compute for the last 24 rows (most recent data)
    for i in range(max(0, len(df) - 24), len(df)):
        idx = i

        # AQI change rate
        if idx >= 2:
            lag_1h = df.iloc[idx - 1]["us_aqi"]
            lag_2h = df.iloc[idx - 2]["us_aqi"]
            df.at[idx, "aqi_change_rate"] = (lag_1h - lag_2h) / (lag_2h + 1e-5)
        else:
            df.at[idx, "aqi_change_rate"] = 0.0

        # AQI lag 1h
        if idx >= 1:
            df.at[idx, "aqi_lag_1h"] = df.iloc[idx - 1]["us_aqi"]
        else:
            df.at[idx, "aqi_lag_1h"] = df.iloc[idx]["us_aqi"]

        # AQI lag 24h
        if idx >= 24:
            df.at[idx, "aqi_lag_24h"] = df.iloc[idx - 24]["us_aqi"]
        else:
            df.at[idx, "aqi_lag_24h"] = df.iloc[idx]["us_aqi"]

        # Rolling 6h
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


def run_hourly_pipeline():
    """Main hourly execution function."""
    print("=" * 60)
    print("⏰ HOURLY FEATURE PIPELINE STARTED")
    print(f"Run time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # 1. Load existing data
    if DATA_PATH.exists():
        existing_df = pd.read_csv(DATA_PATH, parse_dates=["datetime"])
        print(f"Existing data: {len(existing_df)} rows")
        
        # CLEANUP: Remove any future timestamps that may have been added
        # by previous versions of the pipeline
        original_count = len(existing_df)
        existing_df = existing_df[
            existing_df["datetime"] <= pd.Timestamp.now(tz=existing_df["datetime"].dt.tz)
        ]
        removed = original_count - len(existing_df)
        if removed > 0:
            print(f"🧹 Cleaned {removed} future rows from previous runs")
            existing_df.to_csv(DATA_PATH, index=False)
    else:
        existing_df = pd.DataFrame()
        print("No existing data found, starting fresh")
        
    # 2. Fetch latest weather
    print("\n📡 Fetching current weather data...")
    weather_df = fetch_current_weather()
    print(f"Fetched {len(weather_df)} weather rows")

    # 3. Fetch latest pollutants
    print("📡 Fetching current pollutant data...")
    pollutants_df = fetch_current_pollutants()
    if pollutants_df is None:
        print("❌ Failed to fetch pollutants. Aborting.")
        return False
    print(f"Fetched {len(pollutants_df)} pollutant rows")

    # 4. Merge weather + pollutants
    merged_df = pd.merge(weather_df, pollutants_df, on="datetime", how="inner")
    merged_df = merged_df.dropna()
    print(f"Merged dataset: {len(merged_df)} rows")

    # 5. Combine with historical
    combined_df = pd.concat([existing_df, merged_df], ignore_index=True)
    combined_df = combined_df.drop_duplicates(subset="datetime", keep="last")
    combined_df = combined_df.sort_values("datetime").reset_index(drop=True)

    # 6. Apply feature engineering
    print("🔧 Engineering features...")
    combined_df = add_time_features(combined_df)
    combined_df = add_lag_and_rolling_features(combined_df)

    # 7. Get only the truly new rows
    last_existing = pd.to_datetime(existing_df["datetime"]).max() if not existing_df.empty else pd.Timestamp.min
    new_rows = combined_df[combined_df["datetime"] > last_existing]

    if len(new_rows) == 0:
        print("ℹ️ No new data to add")
        return True

    print(f"New rows to add: {len(new_rows)}")
    print(f"Latest timestamp: {new_rows['datetime'].max()}")

    # 8. Save back to CSV
    combined_df.to_csv(DATA_PATH, index=False)
    print(f"✅ Saved to: {DATA_PATH}")
    print(f"Total rows now: {len(combined_df)}")

    # 9. Upload to Hopsworks (if API key is available)
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

            new_data = combined_df.tail(len(new_rows) + 24)
            fg.insert(new_data, wait=True)
            print("✅ Hopsworks upload complete")
        else:
            print("ℹ️ No HOPSWORKS_API_KEY, skipping cloud upload (local only)")
    except Exception as e:
        print(f"⚠️ Hopsworks upload skipped: {e}")

    print("=" * 60)
    print("✅ HOURLY PIPELINE COMPLETE")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = run_hourly_pipeline()
    sys.exit(0 if success else 1)