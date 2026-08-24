import hopsworks
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def test_isolated_upload():
    project = hopsworks.login()
    fs = project.get_feature_store()

    now = datetime.utcnow()
    df = pd.DataFrame({
        "city": ["Lahore"] * 24,
        "datetime": pd.to_datetime([now - timedelta(hours=i) for i in range(24)]).astype("datetime64[us]"),
        "aqi_test": np.random.uniform(50.0, 150.0, size=24).astype(np.float64),
    })

    # Use a new name/version so schema conflicts cannot occur
    test_fg = fs.get_or_create_feature_group(
        name="github_actions_debug_fg",
        version=1,
        primary_key=["city"],
        event_time="datetime",
        description="Temporary test FG for CI/CD pipeline",
        online_enabled=True
    )

    print("Inserting into github_actions_debug_fg...")
    test_fg.insert(df, write_options={"wait_for_job": False})
    print(">>> Test write succeeded!")

if __name__ == "__main__":
    test_isolated_upload()
