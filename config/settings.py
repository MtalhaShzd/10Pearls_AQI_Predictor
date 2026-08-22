"""Project settings and environment configuration."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FINAL_DATA_DIR = DATA_DIR / "final"
MODELS_DIR = BASE_DIR / "models"

API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

CITY_LIST_PATH = DATA_DIR / "cities.csv"
if not CITY_LIST_PATH.exists():
    CITY_LIST_PATH = RAW_DATA_DIR / "cities.csv"
if not CITY_LIST_PATH.exists():
    CITY_LIST_PATH = BASE_DIR / "config" / "cities.csv"
