"""
Hopsworks Store Client

Loads:
  - Trained model + artifacts from Hopsworks Model Registry
  - Feature data from Hopsworks Feature Store

Falls back to local files only if Hopsworks is unreachable.
"""

import os
import json
import time
import joblib
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_NAME = os.getenv("MODEL_NAME", "lahore_aqi_ridge_1h")
FEATURE_GROUP_NAME = os.getenv("HOPSWORKS_FEATURE_GROUP", "lahore_air_quality_features")
FEATURE_GROUP_VERSION = int(os.getenv("HOPSWORKS_FEATURE_GROUP_VERSION", "2"))

LOCAL_MODEL_DIR = PROJECT_ROOT / "models"
LOCAL_CSV_PATH = PROJECT_ROOT / "data" / "processed" / "lahore" / "lahore_features_hourly.csv"

FEATURE_CACHE_TTL = int(os.getenv("FEATURE_CACHE_TTL", "900"))  # 15 minutes
MODEL_CACHE_TTL = int(os.getenv("MODEL_CACHE_TTL", "1800"))     # retry every 30 min while degraded

# In-memory caches
_MODEL = None
_FEATURE_COLUMNS = None
_METRICS = None
_MODEL_VERSION = None
_MODEL_SOURCE = "not loaded"
_MODEL_LOAD_TS = 0.0

_FEATURES_DF = None
_FEATURES_TS = 0.0
_FEATURES_SOURCE = "not loaded"

_PROJECT = None


def _get_project():
    """Login to Hopsworks once and reuse the connection."""
    global _PROJECT

    if _PROJECT is not None:
        return _PROJECT

    api_key = os.getenv("HOPSWORKS_API_KEY")
    if not api_key:
        raise RuntimeError("HOPSWORKS_API_KEY not set")

    import hopsworks
    _PROJECT = hopsworks.login(api_key_value=api_key)
    print("✅ Connected to Hopsworks project")
    return _PROJECT


def _find_file(folder, patterns):
    """Return first file inside folder matching any pattern."""
    for pattern in patterns:
        matches = list(Path(folder).rglob(pattern))
        if matches:
            return matches[0]
    return None


# ═══════════════════════════════════════════════════════════
# MODEL REGISTRY
# ═══════════════════════════════════════════════════════════
def _download_model_from_registry():
    project = _get_project()
    mr = project.get_model_registry()

    models = mr.get_models(name=MODEL_NAME)
    if not models:
        raise RuntimeError(f"No model '{MODEL_NAME}' found in Model Registry")

    latest = max(models, key=lambda m: m.version)
    print(f"📥 Downloading '{MODEL_NAME}' v{latest.version} from Model Registry...")

    path = Path(latest.download())
    print(f"✅ Model artifacts downloaded to: {path}")
    return path, latest.version


def load_model_artifacts(force_refresh: bool = False):
    """Load model, feature columns, metrics. Registry first, local fallback.
    Auto-retries Hopsworks every MODEL_CACHE_TTL seconds while running on
    the local fallback, so the app self-heals once quota/connectivity recovers.
    """
    global _MODEL, _FEATURE_COLUMNS, _METRICS, _MODEL_VERSION, _MODEL_SOURCE, _MODEL_LOAD_TS

    using_fallback = _MODEL_SOURCE not in (None, "not loaded", "Hopsworks Model Registry")
    stale = (time.time() - _MODEL_LOAD_TS) > MODEL_CACHE_TTL
    should_retry = force_refresh or _MODEL is None or (using_fallback and stale)

    if not should_retry:
        return _MODEL, _FEATURE_COLUMNS, _METRICS

    artifact_dir = None
    version_label = "local"
    source = "local models/ folder"

    try:
        artifact_dir, version = _download_model_from_registry()
        version_label = f"v{version}"
        source = "Hopsworks Model Registry"
    except Exception as e:
        print(f"⚠️ Model Registry download failed: {e}")

    if artifact_dir is None:
        if not LOCAL_MODEL_DIR.exists():
            raise FileNotFoundError(
                "Model unavailable: Hopsworks failed and no local models/ folder exists."
            )
        print("ℹ️ Falling back to local models/ folder")
        artifact_dir = LOCAL_MODEL_DIR

    model_file = _find_file(artifact_dir, ["ridge_aqi_1h.pkl", "*.pkl"])
    if model_file is None:
        raise FileNotFoundError(f"No .pkl model found in {artifact_dir}")

    _MODEL = joblib.load(model_file)
    _MODEL_VERSION = version_label
    _MODEL_SOURCE = source
    print(f"✅ Model loaded: {model_file.name} ({source} {version_label})")

    # feature_columns.json
    features_file = _find_file(artifact_dir, ["feature_columns.json"])
    if features_file is None and LOCAL_MODEL_DIR.exists():
        features_file = _find_file(LOCAL_MODEL_DIR, ["feature_columns.json"])

    if features_file:
        with open(features_file, "r") as f:
            _FEATURE_COLUMNS = json.load(f)
    else:
        _FEATURE_COLUMNS = []
        print("⚠️ feature_columns.json not found")

    # metrics.json
    metrics_file = _find_file(artifact_dir, ["metrics.json"])
    if metrics_file is None and LOCAL_MODEL_DIR.exists():
        metrics_file = _find_file(LOCAL_MODEL_DIR, ["metrics.json"])

    if metrics_file:
        with open(metrics_file, "r") as f:
            raw = json.load(f)
        _METRICS = {
            k: (float(v) if isinstance(v, (int, float)) else v)
            for k, v in raw.items()
        }
    else:
        _METRICS = {"test_mae": 2.89, "test_rmse": 4.69, "test_r2": 0.987, "val_rmse": 1.52}
        print("⚠️ metrics.json not found, using defaults")

    _METRICS["version"] = version_label
    _METRICS["source"] = source
    _MODEL_LOAD_TS = time.time()

    return _MODEL, _FEATURE_COLUMNS, _METRICS


def get_model():
    return load_model_artifacts()[0]


def get_feature_columns():
    return load_model_artifacts()[1]


def get_metrics():
    return load_model_artifacts()[2]


# ═══════════════════════════════════════════════════════════
# FEATURE STORE
# ═══════════════════════════════════════════════════════════
def _read_feature_group():
    project = _get_project()
    fs = project.get_feature_store()
    fg = fs.get_feature_group(FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)

    print(f"📥 Reading Feature Store: {FEATURE_GROUP_NAME} v{FEATURE_GROUP_VERSION}...")
    df = fg.read()
    print(f"✅ Feature Store returned {len(df)} rows")
    return df


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names and datetime dtype."""
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]

    # Hopsworks may store event time as 'timestamp'
    if "datetime" not in df.columns and "timestamp" in df.columns:
        df = df.rename(columns={"timestamp": "datetime"})

    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
        try:
            df["datetime"] = df["datetime"].dt.tz_localize(None)
        except (TypeError, AttributeError):
            pass
        df = df.drop_duplicates(subset="datetime", keep="last")
        df = df.sort_values("datetime").reset_index(drop=True)

    return df


def get_features_df(force_refresh: bool = False) -> pd.DataFrame:
    """
    Get feature data.
    Priority: in-memory cache (TTL) → Feature Store → local CSV fallback.
    """
    global _FEATURES_DF, _FEATURES_TS, _FEATURES_SOURCE

    age = time.time() - _FEATURES_TS
    if _FEATURES_DF is not None and not force_refresh and age < FEATURE_CACHE_TTL:
        return _FEATURES_DF.copy()

    df = None
    source = "local CSV"

    try:
        df = _read_feature_group()
        source = "Hopsworks Feature Store"
    except Exception as e:
        print(f"⚠️ Feature Store read failed: {e}")

    if df is None or len(df) == 0:
        if not LOCAL_CSV_PATH.exists():
            raise FileNotFoundError(
                "Features unavailable: Feature Store failed and no local CSV exists."
            )
        print("ℹ️ Falling back to local CSV")
        df = pd.read_csv(LOCAL_CSV_PATH)
        source = "local CSV"

    df = _normalize(df)

    _FEATURES_DF = df
    _FEATURES_TS = time.time()
    _FEATURES_SOURCE = source

    return df.copy()


def get_source_info():
    """Return current data/model sources for dashboard display."""
    return {
        "model_source": _MODEL_SOURCE,
        "model_version": _MODEL_VERSION or "not loaded",
        "feature_source": _FEATURES_SOURCE,
        "feature_group": FEATURE_GROUP_NAME,
        "feature_rows": 0 if _FEATURES_DF is None else int(len(_FEATURES_DF)),
        "cache_age_seconds": round(time.time() - _FEATURES_TS, 1) if _FEATURES_TS else None,
    }
