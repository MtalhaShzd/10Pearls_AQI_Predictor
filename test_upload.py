"""Try several insert strategies to find one that works."""

import os
import time
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

df = pd.read_csv("data/processed/lahore/lahore_features_hourly.csv", parse_dates=["datetime"])
df["datetime"] = pd.to_datetime(df["datetime"]).dt.tz_localize(None)
df = df.sort_values("datetime").tail(24).reset_index(drop=True)

print(f"Test payload: {len(df)} rows, {df['datetime'].min()} -> {df['datetime'].max()}\n")

import hopsworks

project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
fs = project.get_feature_store()
fg = fs.get_feature_group("lahore_air_quality_features", version=1)

strategies = [
    ("A: default",                {}),
    ("B: no offline materialization", {"start_offline_materialization": False}),
    ("C: wait for job",           {"wait_for_job": True}),
    ("D: internal_kafka False",   {"internal_kafka": False}),
]

for name, opts in strategies:
    print("=" * 60)
    print(name, opts)
    print("=" * 60)
    try:
        fg.insert(df, write_options=opts)
        print(f">>> SUCCESS with {name}\n")
        break
    except Exception as e:
        print(f">>> FAILED: {type(e).__name__}: {e}\n")
        time.sleep(5)