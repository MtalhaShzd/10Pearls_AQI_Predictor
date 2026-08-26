# src/forecast_engine.py

import os
import json
import warnings
import urllib.request
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from src.hopsworks_store import (
    get_model,
    get_feature_columns,
    get_metrics,
    get_features_df,
    get_source_info,
)

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Fallback list used only if feature_columns.json is unavailable
DEFAULT_FEATURE_COLUMNS = [
    "temperature_2m", "relative_humidity_2m", "surface_pressure",
    "precipitation", "cloud_cover", "wind_speed_10m", "wind_direction_10m",
    "pm2_5", "pm10", "carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide", "ozone",
    "us_aqi", "hour", "day", "month", "day_of_week", "is_weekend",
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    "aqi_change_rate", "aqi_lag_1h", "aqi_lag_24h",
    "aqi_rolling_mean_6h", "aqi_rolling_std_6h", "pm25_rolling_mean_6h"
]

FEATURE_COLUMNS = DEFAULT_FEATURE_COLUMNS

# Default metrics; replaced at runtime by Model Registry metrics
MODEL_METRICS = {
    "test_mae": 2.89,
    "test_rmse": 4.69,
    "test_r2": 0.987,
    "val_rmse": 1.52,
    "version": "v1.0",
    "source": "default"
}


def get_feature_list():
    """Feature columns from Model Registry, fallback to default list."""
    try:
        cols = get_feature_columns()
        return cols if cols else DEFAULT_FEATURE_COLUMNS
    except Exception as e:
        print(f"⚠️ Could not load feature columns: {e}")
        return DEFAULT_FEATURE_COLUMNS


def get_model_metrics():
    """Metrics from Model Registry, fallback to defaults."""
    global MODEL_METRICS
    try:
        m = get_metrics()
        if m:
            MODEL_METRICS = m
    except Exception as e:
        print(f"⚠️ Could not load metrics: {e}")
    return MODEL_METRICS


def get_data_source_info():
    """Expose which store the data came from."""
    return get_source_info()


def fetch_weather_forecast(lat=31.5204, lon=74.3587):
    """Fetch hourly weather forecast for the next 3 days from Open-Meteo."""
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}&hourly="
        f"temperature_2m,relative_humidity_2m,surface_pressure,"
        f"precipitation,cloud_cover,wind_speed_10m,wind_direction_10m"
        f"&timezone=auto&forecast_days=3"
    )
    with urllib.request.urlopen(url, timeout=45) as response:
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
    df["datetime"] = pd.to_datetime(df["datetime"]).dt.tz_localize(None)
    return df


def add_time_features(df):
    df = df.copy()
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
    """Latest observation, read from Hopsworks Feature Store."""
    df = get_features_df()
    df = df.sort_values("datetime").reset_index(drop=True)
    latest = df.iloc[-1]

    trend_24h = df.tail(24)[["datetime", "us_aqi"]].to_dict("records")

    return {
        "current_aqi": round(float(latest["us_aqi"]), 1),
        "timestamp": pd.to_datetime(latest["datetime"]).strftime("%Y-%m-%d %H:%M:%S"),
        "source": get_source_info(),
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
                "datetime": pd.to_datetime(t["datetime"]).strftime("%Y-%m-%d %H:%M:%S"),
                "aqi": round(float(t["us_aqi"]), 1)
            } for t in trend_24h
        ]
    }


def generate_72h_forecast(return_features: bool = False):
    """Recursive 72-hour forecast using Model Registry model + Feature Store data."""
    model = get_model()
    feature_cols = get_feature_list()

    weather_df = fetch_weather_forecast()
    forecast_df = add_time_features(weather_df)

    historical_df = get_features_df()
    historical_df = historical_df.sort_values("datetime").reset_index(drop=True)

    context_history = historical_df.tail(24).copy()
    last_historical_time = context_history["datetime"].max()

    future = forecast_df[forecast_df["datetime"] > last_historical_time].reset_index(drop=True)
    if len(future) < 72:
        future = add_time_features(weather_df).head(72)
    else:
        future = future.head(72)

    history_queue = context_history.copy()
    predicted_aqis, predicted_times = [], []
    feature_rows = []  # one X_step per forecast hour

    for _, row in future.iterrows():
        current_time = row["datetime"]

        lag_1h = float(history_queue.iloc[-1]["us_aqi"])
        lag_24h = float(
            history_queue.iloc[-24]["us_aqi"]
            if len(history_queue) >= 24
            else history_queue.iloc[0]["us_aqi"]
        )
        lag_2h = float(
            history_queue.iloc[-2]["us_aqi"] if len(history_queue) >= 2 else lag_1h
        )

        rolling_6h_aqi = history_queue.tail(6)["us_aqi"]
        rolling_mean_6h = float(rolling_6h_aqi.mean())
        std_val = rolling_6h_aqi.std()
        rolling_std_6h = float(std_val) if not np.isnan(std_val) else 0.0
        rolling_mean_6h_pm25 = float(history_queue.tail(6)["pm2_5"].mean())
        aqi_change_rate = float((lag_1h - lag_2h) / (lag_2h + 1e-5))

        pm2_5_pred = float(history_queue.iloc[-1]["pm2_5"]) * 0.98 + (rolling_mean_6h_pm25 * 0.02)
        pm10_pred = float(history_queue.iloc[-1]["pm10"]) * 0.98
        co_pred = float(history_queue.iloc[-1]["carbon_monoxide"])
        no2_pred = float(history_queue.iloc[-1]["nitrogen_dioxide"])
        so2_pred = float(history_queue.iloc[-1]["sulphur_dioxide"])
        o3_pred = float(history_queue.iloc[-1]["ozone"])

        feature_dict = {
            "temperature_2m": float(row["temperature_2m"]),
            "relative_humidity_2m": float(row["relative_humidity_2m"]),
            "surface_pressure": float(row["surface_pressure"]),
            "precipitation": float(row["precipitation"]),
            "cloud_cover": float(row["cloud_cover"]),
            "wind_speed_10m": float(row["wind_speed_10m"]),
            "wind_direction_10m": float(row["wind_direction_10m"]),
            "pm2_5": pm2_5_pred, "pm10": pm10_pred,
            "carbon_monoxide": co_pred, "nitrogen_dioxide": no2_pred,
            "sulphur_dioxide": so2_pred, "ozone": o3_pred,
            "us_aqi": lag_1h,
            "hour": int(row["hour"]), "day": int(row["day"]), "month": int(row["month"]),
            "day_of_week": int(row["day_of_week"]), "is_weekend": int(row["is_weekend"]),
            "hour_sin": float(row["hour_sin"]), "hour_cos": float(row["hour_cos"]),
            "month_sin": float(row["month_sin"]), "month_cos": float(row["month_cos"]),
            "aqi_change_rate": aqi_change_rate,
            "aqi_lag_1h": lag_1h, "aqi_lag_24h": lag_24h,
            "aqi_rolling_mean_6h": rolling_mean_6h,
            "aqi_rolling_std_6h": rolling_std_6h,
            "pm25_rolling_mean_6h": rolling_mean_6h_pm25
        }

        X_step = pd.DataFrame([feature_dict])
        for col in feature_cols:
            if col not in X_step.columns:
                X_step[col] = 0.0
        X_step = X_step[feature_cols]

        predicted_value = max(0.0, float(model.predict(X_step)[0]))
        predicted_aqis.append(predicted_value)
        predicted_times.append(current_time)
        feature_rows.append(X_step.iloc[0])

        new_obs = feature_dict.copy()
        new_obs["datetime"] = current_time
        new_obs["us_aqi"] = predicted_value
        history_queue = pd.concat([history_queue, pd.DataFrame([new_obs])], ignore_index=True)
        history_queue = history_queue.tail(24).reset_index(drop=True)

    forecast_df_out = pd.DataFrame({"datetime": predicted_times, "predicted_aqi": predicted_aqis})

    if return_features:
        features_df = pd.DataFrame(feature_rows).reset_index(drop=True)
        return forecast_df_out, features_df
    return forecast_df_out


HORIZON_TO_STEP = {"24h": 24, "48h": 48, "72h": 72}

def get_shap_explanation(horizon="24h"):
    """SHAP feature contributions for a specific forecast horizon."""
    import shap

    model = get_model()
    feature_cols = get_feature_list()

    step = HORIZON_TO_STEP.get(horizon, 24)

    # Recompute the recursive forecast to get the exact feature vector
    # the model saw at this horizon (not just the latest historical row).
    _, features_df = generate_72h_forecast(return_features=True)
    idx = min(step, len(features_df)) - 1
    target_row = features_df.iloc[[idx]][feature_cols]

    historical_df = get_features_df().sort_values("datetime").reset_index(drop=True)
    for col in feature_cols:
        if col not in historical_df.columns:
            historical_df[col] = 0.0
    X_train_raw = historical_df[feature_cols].tail(5000)

    scaler = model.named_steps["scaler"]
    ridge = model.named_steps["ridge"]

    X_train_scaled = scaler.transform(X_train_raw)
    target_scaled = scaler.transform(target_row)

    explainer = shap.LinearExplainer(ridge, X_train_scaled)
    shap_values = explainer.shap_values(target_scaled)[0]

    label_map = { ... }  # unchanged

    contributions = [
        {"feature": label_map.get(f, f), "value": round(float(v), 2)}
        for f, v in zip(feature_cols, shap_values)
    ]
    return sorted(contributions, key=lambda x: abs(x["value"]), reverse=True)
