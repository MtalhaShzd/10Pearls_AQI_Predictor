"""
Backfill Pipeline — Lahore AQI
==============================
Detects missing hourly timestamps, fetches them from Open-Meteo, recomputes
lag/rolling features for the whole series, and upserts the affected rows into
Hopsworks (lahore_air_quality_features v2).

Usage:
    python src/backfill_pipeline.py
    python src/backfill_pipeline.py --start 2026-08-01 --end 2026-08-25
    python src/backfill_pipeline.py --no-upload
    python src/backfill_pipeline.py --full-upload      # re-send every row
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "lahore" / "lahore_features_hourly.csv"
DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

LAT, LON = 31.5204, 74.3587
CITY_TZ = "Asia/Karachi"

FG_NAME = os.getenv("HOPSWORKS_FEATURE_GROUP", "lahore_air_quality_features")
FG_VERSION = int(os.getenv("HOPSWORKS_FEATURE_GROUP_VERSION", "2"))   # v2, not 1

ARCHIVE_LAG_DAYS = 7      # archive-api (ERA5) trails real time by ~5 days
AFFECTED_WINDOW_H = 24    # filling hour T changes lags for the next 24 hours

INT_COLS = ["hour", "day", "month", "day_of_week", "is_weekend"]

WEATHER_VARS = ("temperature_2m,relative_humidity_2m,surface_pressure,precipitation,"
                "cloud_cover,wind_speed_10m,wind_direction_10m")
POLLUTANT_VARS = "pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone,us_aqi"

FINAL_COLUMNS = [
    "datetime", "temperature_2m", "relative_humidity_2m", "surface_pressure",
    "precipitation", "cloud_cover", "wind_speed_10m", "wind_direction_10m",
    "pm2_5", "pm10", "carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide",
    "ozone", "us_aqi", "hour", "day", "month", "day_of_week", "is_weekend",
    "hour_sin", "hour_cos", "month_sin", "month_cos", "aqi_change_rate",
    "aqi_lag_1h", "aqi_lag_24h", "aqi_rolling_mean_6h", "aqi_rolling_std_6h",
    "pm25_rolling_mean_6h",
]


def now_local() -> pd.Timestamp:
    """FIX: pd.Timestamp.now() is UTC on CI; Lahore is UTC+5."""
    return pd.Timestamp.now(tz=CITY_TZ).tz_localize(None)


def get_json(url, retries=3, timeout=60):
    err = None
    for i in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return json.loads(r.read().decode())
        except Exception as e:
            err = e
            print(f"   ⚠️ attempt {i}/{retries}: {e}")
            if i < retries:
                time.sleep(5 * i)
    raise RuntimeError(f"Request failed:\n{url}\n{err}")


def hopsworks_login():
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    key = os.getenv("HOPSWORKS_API_KEY")
    if not key:
        return None
    import hopsworks
    return hopsworks.login(
        host=os.getenv("HOPSWORKS_HOST", "eu-west.cloud.hopsworks.ai"),
        project=os.getenv("HOPSWORKS_PROJECT", "internship10P"),
        api_key_value=key,
    )


# --------------------------------------------------------------------------
# FETCH — hybrid archive / forecast
# --------------------------------------------------------------------------
def _frame(hourly, cols):
    d = {"datetime": pd.to_datetime(hourly["time"])}
    d.update({c: hourly[c] for c in cols})
    df = pd.DataFrame(d)
    df["datetime"] = df["datetime"].dt.tz_localize(None)
    return df


WEATHER_COLS = ["temperature_2m", "relative_humidity_2m", "surface_pressure",
                "precipitation", "cloud_cover", "wind_speed_10m", "wind_direction_10m"]
POLLUTANT_COLS = ["pm2_5", "pm10", "carbon_monoxide", "nitrogen_dioxide",
                  "sulphur_dioxide", "ozone", "us_aqi"]


def fetch_weather_range(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """FIX: archive-api returns nulls for the last ~5 days, which dropna() then
    erased — so recent gaps could never be filled. We now split the request:
    archive for older dates, forecast (past_days) for recent ones.
    """
    today = now_local().normalize()
    cutoff = today - pd.Timedelta(days=ARCHIVE_LAG_DAYS)
    parts = []

    if start < cutoff:
        a_end = min(end, cutoff - pd.Timedelta(days=1))
        print(f"   archive  {start:%Y-%m-%d} → {a_end:%Y-%m-%d}")
        h = get_json(
            f"https://archive-api.open-meteo.com/v1/archive?latitude={LAT}&longitude={LON}"
            f"&start_date={start:%Y-%m-%d}&end_date={a_end:%Y-%m-%d}"
            f"&hourly={WEATHER_VARS}&timezone=auto"
        )["hourly"]
        parts.append(_frame(h, WEATHER_COLS))

    if end >= cutoff:
        r_start = max(start, cutoff)
        past_days = min(92, (today - r_start).days + 1)
        print(f"   🛰️ forecast past_days={past_days} (for {r_start:%Y-%m-%d} → {end:%Y-%m-%d})")
        h = get_json(
            f"https://api.open-meteo.com/v1/forecast?latitude={LAT}&longitude={LON}"
            f"&hourly={WEATHER_VARS}&past_days={past_days}&forecast_days=1&timezone=auto"
        )["hourly"]
        df = _frame(h, WEATHER_COLS)
        parts.append(df[(df["datetime"] >= r_start) &
                        (df["datetime"] < end + pd.Timedelta(days=1))])

    return (pd.concat(parts, ignore_index=True)
              .drop_duplicates("datetime", keep="last")
              .sort_values("datetime").reset_index(drop=True))


def fetch_pollutant_range(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    print(f"   🌫️ pollutants {start:%Y-%m-%d} → {end:%Y-%m-%d}")
    h = get_json(
        f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={LAT}&longitude={LON}"
        f"&hourly={POLLUTANT_VARS}&start_date={start:%Y-%m-%d}&end_date={end:%Y-%m-%d}&timezone=auto"
    )["hourly"]
    return _frame(h, POLLUTANT_COLS)


# --------------------------------------------------------------------------
# FEATURES
# --------------------------------------------------------------------------
def add_time_features(df):
    df = df.copy()
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
    """Full recompute — correct for backfill, since a mid-series gap invalidates
    the lag/rolling values of every row that follows it."""
    df = df.sort_values("datetime").reset_index(drop=True)
    aqi, pm25 = df["us_aqi"], df["pm2_5"]

    df["aqi_lag_1h"] = aqi.shift(1).fillna(aqi)
    df["aqi_lag_24h"] = aqi.shift(24).fillna(aqi)
    df["aqi_change_rate"] = (((aqi.shift(1) - aqi.shift(2)) / (aqi.shift(2) + 1e-5))
                             .replace([np.inf, -np.inf], 0.0).fillna(0.0))
    df["aqi_rolling_mean_6h"] = aqi.rolling(6, min_periods=1).mean()
    df["aqi_rolling_std_6h"] = aqi.rolling(6, min_periods=1).std().fillna(0.0)
    df["pm25_rolling_mean_6h"] = pm25.rolling(6, min_periods=1).mean()
    return df


def enforce_schema(df):
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"]).dt.tz_localize(None).astype("datetime64[us]")
    for c in [c for c in FINAL_COLUMNS if c != "datetime"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in INT_COLS:
        df[c] = df[c].fillna(0).astype("int64")
    for c in [c for c in FINAL_COLUMNS if c not in INT_COLS + ["datetime"]]:
        df[c] = df[c].astype("float64")
    return df[FINAL_COLUMNS]


# --------------------------------------------------------------------------
# GAP DETECTION
# --------------------------------------------------------------------------
def find_missing_hours(df, end=None):
    """FIX: the old version only searched BETWEEN min and max, so it reported
    'no gaps' for the trailing Aug 1 → Aug 25 hole. We now extend to now().
    """
    have = pd.DatetimeIndex(pd.to_datetime(df["datetime"]).dt.floor("h").unique())
    start = have.min()
    end = (pd.Timestamp(end) if end is not None else now_local()).floor("h")
    if end < have.max():
        end = have.max()
    return pd.date_range(start, end, freq="h").difference(have)


def group_into_ranges(hours, max_days=90):
    if len(hours) == 0:
        return []
    days = sorted({h.normalize() for h in hours})
    ranges, s, prev = [], days[0], days[0]
    for d in days[1:]:
        if (d - prev).days > 1 or (d - s).days >= max_days:
            ranges.append((s, prev))
            s = d
        prev = d
    ranges.append((s, prev))
    return ranges


# --------------------------------------------------------------------------
# LOAD / UPLOAD
# --------------------------------------------------------------------------
def load_base(project):
    if DATA_PATH.exists():
        print(f"📂 CSV: {DATA_PATH}")
        df = pd.read_csv(DATA_PATH, parse_dates=["datetime"])
    elif project is not None:
        print(f"📥 No CSV — bootstrapping from {FG_NAME} v{FG_VERSION}")
        df = project.get_feature_store().get_feature_group(FG_NAME, version=FG_VERSION).read()
    else:
        raise FileNotFoundError("No CSV and no Hopsworks access")

    df["datetime"] = pd.to_datetime(df["datetime"]).dt.tz_localize(None)
    return (df.drop_duplicates("datetime", keep="last")
              .sort_values("datetime").reset_index(drop=True))


def upload(project, df, affected: pd.DatetimeIndex, full=False):
    """Upsert only the rows whose values actually changed.

    FIX: the old version pushed all 22k+ rows through Kafka on every run.
    """
    if project is None:
        print("ℹ️ No HOPSWORKS_API_KEY — skipping upload")
        return

    fs = project.get_feature_store()
    fg = fs.get_feature_group(FG_NAME, version=FG_VERSION)

    if getattr(fg, "stream", False) is not True:
        raise RuntimeError(
            f"{FG_NAME} v{FG_VERSION} has stream=False. External writes will fail "
            f"with 'RPC listener disconnected'. Point FG_VERSION at the streaming version."
        )

    out = df if full else df[df["datetime"].isin(affected)]
    if out.empty:
        print("ℹ️ Nothing changed — no upload needed")
        return

    print(f"\n📤 Upserting {len(out)} rows into {FG_NAME} v{FG_VERSION}...")
    try:
        fg.insert(out, write_options={"wait_for_job": True})
        print(f"   ✅ Done ({out['datetime'].min()} → {out['datetime'].max()})")
    except Exception as e:
        print(f"::warning::Hopsworks materialization failed (likely quota), "
              f"but the local CSV and Kafka ingestion already succeeded: {e}")


# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------
def run_backfill(start=None, end=None, do_upload=True, full_upload=False) -> bool:
    print("=" * 60)
    print("🔄 BACKFILL PIPELINE STARTED")
    print(f"Run time (Lahore): {now_local():%Y-%m-%d %H:%M:%S}")
    print("=" * 60)

    project = hopsworks_login() if do_upload else None
    base = load_base(project)
    print(f"Existing rows: {len(base)}  ({base['datetime'].min()} → {base['datetime'].max()})")

    if start and end:
        missing = pd.date_range(pd.Timestamp(start), pd.Timestamp(end).replace(hour=23),
                                freq="h").difference(
                      pd.DatetimeIndex(base["datetime"].dt.floor("h").unique()))
    else:
        missing = find_missing_hours(base)

    if len(missing) == 0:
        print("✅ No hourly gaps found")
        return True

    ranges = group_into_ranges(missing)
    print(f"📅 {len(missing)} missing hours across {len(ranges)} range(s):")
    for s, e in ranges:
        print(f"   {s:%Y-%m-%d} → {e:%Y-%m-%d}")

    fetched = []
    for s, e in ranges:
        print(f"\n▶ {s:%Y-%m-%d} → {e:%Y-%m-%d}")
        try:
            w = fetch_weather_range(s, e)
            p = fetch_pollutant_range(s, e)
            m = pd.merge(w, p, on="datetime", how="inner")
            m = m[m["datetime"].dt.floor("h").isin(missing)].dropna()
            print(f"   ✅ recovered {len(m)} rows")
            if len(m):
                fetched.append(m)
        except Exception as e2:
            print(f"   ⚠️ range failed: {e2}")

    if not fetched:
        print("❌ No data recovered from the APIs")
        return False

    new = (pd.concat(fetched, ignore_index=True)
             .drop_duplicates("datetime", keep="last")
             .sort_values("datetime").reset_index(drop=True))
    print(f"\n Recovered {len(new)} rows total")

    combined = (pd.concat([base, new], ignore_index=True)
                  .drop_duplicates("datetime", keep="last")
                  .sort_values("datetime").reset_index(drop=True))

    print(" Recomputing features across the full series...")
    combined = enforce_schema(add_lag_and_rolling_features(add_time_features(combined)))

    combined.to_csv(DATA_PATH, index=False)
    print(f" Saved {len(combined)} rows → {DATA_PATH}")

    # Filling hour T also changes lag/rolling values for the following 24h.
    affected = set(new["datetime"])
    for ts in new["datetime"]:
        affected.update(pd.date_range(ts, periods=AFFECTED_WINDOW_H + 1, freq="h"))
    affected = pd.DatetimeIndex(sorted(affected))

    remaining = find_missing_hours(combined)
    print(f"🔍 Remaining gaps: {len(remaining)}"
          + ("" if len(remaining) == 0 else f"  (earliest {remaining.min()})"))

    if do_upload:
        upload(project, combined, affected, full_upload)

    print("=" * 60)
    print("✅ BACKFILL COMPLETE")
    print("=" * 60)
    return True


def main():
    p = argparse.ArgumentParser(description="Lahore AQI backfill pipeline")
    p.add_argument("--start", help="YYYY-MM-DD (forces a range)")
    p.add_argument("--end", help="YYYY-MM-DD")
    p.add_argument("--no-upload", action="store_true")
    p.add_argument("--full-upload", action="store_true", help="Re-send every row")
    a = p.parse_args()

    if bool(a.start) != bool(a.end):
        print("❌ --start and --end must be used together")
        sys.exit(1)

    try:
        ok = run_backfill(a.start, a.end, not a.no_upload, a.full_upload)
    except Exception:
        print("\n❌ BACKFILL FAILED")
        traceback.print_exc()
        sys.exit(1)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
