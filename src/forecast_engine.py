# src/forecast_engine.py

import os
import json
import joblib
import warnings
import urllib.request
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "lahore"

FEATURE_COLUMNS = [
    "temperature_2m", "relative_humidity_2m", "surface_pressure", 
    "precipitation", "cloud_cover", "wind_speed_10m", "wind_direction_10m",
    "pm2_5", "pm10", "carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide", "ozone",
    "us_aqi", "hour", "day", "month", "day_of_week", "is_weekend",
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    "aqi_change_rate", "aqi_lag_1h", "aqi_lag_24h",
    "aqi_rolling_mean_6h", "aqi_rolling_std_6h", "pm25_rolling_mean_6h"
]

# Held-out test performance metrics from Notebook 06
MODEL_METRICS = {
    "test_mae": 2.89,
    "test_rmse": 4.69,
    "test_r2": 0.987,
    "val_rmse": 1.52,
    "version": "v1.0"
}

def fetch_weather_forecast(lat=31.5204, lon=74.3587):
    """Fetch hourly weather forecast for the next 3 days from Open-Meteo."""
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}&hourly="
        f"temperature_2m,relative_humidity_2m,surface_pressure,"
        f"precipitation,cloud_cover,wind_speed_10m,wind_direction_10m"
        f"&timezone=auto&forecast_days=3"
    )
    with urllib.request.urlopen(url) as response:
        data = json.loads(response.read().decode())
    hourly_data = data["hourly"]
    df = pd.DataFrame({
        "datetime": pd.to_datetime(hourly_data["time"]),
        "temperature_2m": hourly_data["temperature_2m"],
        "relative_humidity_2m": hourly_data["relative_humidity_2m"],
        "surface_pressure": hourly_data["surface_pressure"],
        "precipitation": hourly_data["precipitation"],
        "cloud_cover": hourly_data["cloud_cover"],
        "wind_speed_10m": hourly_data["wind_speed_10m"],
        "wind_direction_10m": hourly_data["wind_direction_10m"]
    })
    return df

def add_time_features(df):
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

def get_current_conditions():
    """Fetch the latest available historical row for real-time display."""
    historical_path = DATA_DIR / "lahore_features_hourly.csv"
    df = pd.read_csv(historical_path, parse_dates=["datetime"])
    latest = df.sort_values("datetime").iloc[-1]
    
    # Get 24-hour AQI trend (last 24 recorded hours)
    trend_24h = df.tail(24)[["datetime", "us_aqi"]].to_dict("records")
    
    return {
        "current_aqi": round(float(latest["us_aqi"]), 1),
        "timestamp": latest["datetime"].strftime("%Y-%m-%d %H:%M:%S"),
        "pollutants": {
            "pm2_5": round(float(latest["pm2_5"]), 1),
            "pm10": round(float(latest["pm10"]), 1),
            "ozone": round(float(latest["ozone"]), 1),
            "nitrogen_dioxide": round(float(latest["nitrogen_dioxide"]), 1),
            "sulphur_dioxide": round(float(latest["sulphur_dioxide"]), 1),
            "carbon_monoxide": round(float(latest["carbon_monoxide"]), 1)
        },
        "weather": {
            "temperature": round(float(latest["temperature_2m"]), 1),
            "humidity": round(float(latest["relative_humidity_2m"]), 1),
            "pressure": round(float(latest["surface_pressure"]), 1),
            "wind_speed": round(float(latest["wind_speed_10m"]), 1),
            "wind_direction": round(float(latest["wind_direction_10m"]), 1),
            "cloud_cover": round(float(latest["cloud_cover"]), 1)
        },
        "trend_24h": [
            {
                "datetime": t["datetime"].strftime("%Y-%m-%d %H:%M:%S"),
                "aqi": round(float(t["us_aqi"]), 1)
            } for t in trend_24h
        ]
    }

def generate_72h_forecast():
    """Run the recursive 72-hour forecasting pipeline."""
    model = joblib.load(MODEL_DIR / "ridge_aqi_1h.pkl")
    
    weather_df = fetch_weather_forecast()
    forecast_df = add_time_features(weather_df)
    
    historical_df = pd.read_csv(DATA_DIR / "lahore_features_hourly.csv", parse_dates=["datetime"])
    historical_df = historical_df.sort_values("datetime").reset_index(drop=True)
    context_history = historical_df.tail(24).copy()
    last_historical_time = context_history["datetime"].max()
    
    forecast_df = forecast_df[forecast_df["datetime"] > last_historical_time].reset_index(drop=True)
    if len(forecast_df) < 72:
        forecast_df = add_time_features(weather_df).head(72)
    else:
        forecast_df = forecast_df.head(72)
    
    history_queue = context_history.copy()
    predicted_aqis, predicted_times = [], []
    
    for _, row in forecast_df.iterrows():
        current_time = row["datetime"]
        lag_1h = history_queue.iloc[-1]["us_aqi"]
        lag_24h = history_queue.iloc[-24]["us_aqi"] if len(history_queue) >= 24 else history_queue.iloc[0]["us_aqi"]
        lag_2h = history_queue.iloc[-2]["us_aqi"] if len(history_queue) >= 2 else lag_1h
        
        rolling_6h_aqi = history_queue.tail(6)["us_aqi"]
        rolling_mean_6h = rolling_6h_aqi.mean()
        rolling_std_6h = rolling_6h_aqi.std() if not np.isnan(rolling_6h_aqi.std()) else 0.0
        rolling_mean_6h_pm25 = history_queue.tail(6)["pm2_5"].mean()
        aqi_change_rate = (lag_1h - lag_2h) / (lag_2h + 1e-5)
        
        pm2_5_pred = history_queue.iloc[-1]["pm2_5"] * 0.98 + (rolling_mean_6h_pm25 * 0.02)
        pm10_pred = history_queue.iloc[-1]["pm10"] * 0.98
        co_pred = history_queue.iloc[-1]["carbon_monoxide"]
        no2_pred = history_queue.iloc[-1]["nitrogen_dioxide"]
        so2_pred = history_queue.iloc[-1]["sulphur_dioxide"]
        o3_pred = history_queue.iloc[-1]["ozone"]
        
        feature_dict = {
            "temperature_2m": row["temperature_2m"], "relative_humidity_2m": row["relative_humidity_2m"],
            "surface_pressure": row["surface_pressure"], "precipitation": row["precipitation"],
            "cloud_cover": row["cloud_cover"], "wind_speed_10m": row["wind_speed_10m"],
            "wind_direction_10m": row["wind_direction_10m"], "pm2_5": pm2_5_pred, "pm10": pm10_pred,
            "carbon_monoxide": co_pred, "nitrogen_dioxide": no2_pred, "sulphur_dioxide": so2_pred, "ozone": o3_pred,
            "us_aqi": lag_1h, "hour": row["hour"], "day": row["day"], "month": row["month"],
            "day_of_week": row["day_of_week"], "is_weekend": row["is_weekend"],
            "hour_sin": row["hour_sin"], "hour_cos": row["hour_cos"],
            "month_sin": row["month_sin"], "month_cos": row["month_cos"],
            "aqi_change_rate": aqi_change_rate, "aqi_lag_1h": lag_1h, "aqi_lag_24h": lag_24h,
            "aqi_rolling_mean_6h": rolling_mean_6h, "aqi_rolling_std_6h": rolling_std_6h,
            "pm25_rolling_mean_6h": rolling_mean_6h_pm25
        }
        X_step = pd.DataFrame([feature_dict])[FEATURE_COLUMNS]
        predicted_value = max(0.0, float(model.predict(X_step)[0]))
        predicted_aqis.append(predicted_value)
        predicted_times.append(current_time)
        
        new_obs = feature_dict.copy()
        new_obs["datetime"] = current_time
        new_obs["us_aqi"] = predicted_value
        history_queue = pd.concat([history_queue, pd.DataFrame([new_obs])], ignore_index=True)
        history_queue = history_queue.tail(24).reset_index(drop=True)
    
    return pd.DataFrame({"datetime": predicted_times, "predicted_aqi": predicted_aqis})

def get_shap_explanation(horizon="24h"):
    """Compute SHAP feature contributions for a specific forecast horizon."""
    import shap
    model = joblib.load(MODEL_DIR / "ridge_aqi_1h.pkl")
    historical_df = pd.read_csv(DATA_DIR / "lahore_features_hourly.csv", parse_dates=["datetime"])
    historical_df = historical_df.sort_values("datetime").reset_index(drop=True)
    
    # Compute SHAP on the latest sample
    X_train = historical_df[FEATURE_COLUMNS].tail(5000)
    latest_sample = historical_df[FEATURE_COLUMNS].tail(1)
    
    explainer = shap.LinearExplainer(model, X_train)
    shap_values = explainer.shap_values(latest_sample)[0]
    
    # Feature name mapping to human-readable format
    label_map = {
        "temperature_2m": "Temperature", "relative_humidity_2m": "Humidity",
        "surface_pressure": "Pressure", "precipitation": "Precipitation",
        "cloud_cover": "Cloud Cover", "wind_speed_10m": "Wind Speed",
        "wind_direction_10m": "Wind Direction", "pm2_5": "PM2.5", "pm10": "PM10",
        "carbon_monoxide": "Carbon Monoxide (CO)", "nitrogen_dioxide": "Nitrogen Dioxide (NO₂)",
        "sulphur_dioxide": "Sulfur Dioxide (SO₂)", "ozone": "Ozone (O₃)",
        "us_aqi": "Current AQI", "hour": "Hour of Day", "day": "Day",
        "month": "Month", "day_of_week": "Day of Week", "is_weekend": "Weekend Flag",
        "hour_sin": "Hour Cycle (sin)", "hour_cos": "Hour Cycle (cos)",
        "month_sin": "Season Cycle (sin)", "month_cos": "Season Cycle (cos)",
        "aqi_change_rate": "AQI Change Rate", "aqi_lag_1h": "AQI 1h Ago",
        "aqi_lag_24h": "AQI 24h Ago", "aqi_rolling_mean_6h": "AQI 6h Avg",
        "aqi_rolling_std_6h": "AQI 6h Volatility", "pm25_rolling_mean_6h": "PM2.5 6h Avg"
    }
    
    contributions = [
        {"feature": label_map.get(f, f), "value": round(float(v), 2)}
        for f, v in zip(FEATURE_COLUMNS, shap_values)
    ]
    contributions_sorted = sorted(contributions, key=lambda x: abs(x["value"]), reverse=True)
    return contributions_sorted