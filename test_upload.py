import os
from datetime import datetime, timedelta, timezone

import hopsworks
import numpy as np
import pandas as pd

FG_NAME = "github_actions_debug_fg"
FG_VERSION = 1

# 1. Connect
project = hopsworks.login(
    host="eu-west.cloud.hopsworks.ai",
    project="internship10P",
    api_key_value=os.environ["HOPSWORKS_API_KEY"],
)
fs = project.get_feature_store()
print("Connected OK")

# 2. Build 24 rows of fake data
now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
timestamps = [now - timedelta(hours=i) for i in range(24)]

df = pd.DataFrame({
    "city": ["Lahore"] * 24,
    "datetime": pd.to_datetime(timestamps).tz_localize(None).astype("datetime64[us]"),
    "aqi_test": np.random.uniform(50.0, 150.0, size=24).astype(np.float64),
})
print(df.dtypes)

# 3. Get the streaming feature group (already created in your notebook)
fg = fs.get_or_create_feature_group(
    name=FG_NAME,
    version=FG_VERSION,
    primary_key=["city"],
    event_time="datetime",
    description="CI/CD debug FG (streaming)",
    online_enabled=True,
    stream=True,
)

# 4. Insert
print("Inserting...")
fg.insert(df, write_options={"wait_for_job": True})

# 5. Verify
online_df = fg.read(online=True)
offline_df = fg.read()
print(f"Online rows : {len(online_df)}  (1 is expected - one row per city)")
print(f"Offline rows: {len(offline_df)}")

assert len(online_df) > 0, "FAILED: nothing in online store"
print(">>> SUCCESS from GitHub Actions")
