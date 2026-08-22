# 🌬️ 10Pearls AQI Predictor

> AI-powered air quality intelligence for Lahore, Pakistan. End-to-end machine learning pipeline predicting the Air Quality Index (AQI) for the next 72 hours with automated hourly data ingestion, daily model retraining, and a production-ready interactive dashboard.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![ML](https://img.shields.io/badge/ML-Scikit--Learn%20%7C%20XGBoost%20%7C%20TensorFlow-orange)
![Cloud](https://img.shields.io/badge/Cloud-Hopsworks-red)
![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-green)

---

## 📋 Table of Contents

1. [Project Overview](#-project-overview)
2. [Key Features](#-key-features)
3. [Architecture](#-architecture)
4. [Tech Stack](#-tech-stack)
5. [Project Structure](#-project-structure)
6. [Installation & Setup](#-installation--setup)
7. [Usage](#-usage)
8. [ML Pipeline Stages](#-ml-pipeline-stages)
9. [Model Performance](#-model-performance)
10. [Dashboard Screenshots](#-dashboard-screenshots)
11. [CI/CD Automation](#-cicd-automation)
12. [API Endpoints](#-api-endpoints)
13. [Contributing](#-contributing)
14. [License](#-license)

---

## 🎯 Project Overview

The **10Pearls AQI Predictor** is a complete end-to-end serverless machine learning system that:

- 📡 Fetches real-time weather and pollutant data for Lahore, Pakistan
- 🧮 Engineers 29 time-series, cyclical, lag, and rolling features
- 🤖 Trains and compares 5 ML models (Baseline, Ridge, Random Forest, XGBoost, LSTM)
- 🔮 Predicts the next 72 hours (3 days) of AQI values via recursive autoregressive forecasting
- 📊 Visualizes predictions through an interactive Streamlit dashboard
- ☁️ Stores features in Hopsworks Feature Store and models in Hopsworks Model Registry
- 🔄 Runs entirely serverless via GitHub Actions with **zero infrastructure management**

The dashboard presents real-time pollutant concentrations (PM2.5, PM10, O₃, NO₂, SO₂, CO), current weather conditions, 24-hour AQI history, 24h/48h/72h forecast cards, predicted trend curves, model performance metrics, and SHAP-based feature importance explanations.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| **Hourly Feature Pipeline** | GitHub Action fetches live weather and pollutant data every hour, appends to feature store |
| **Daily Training Pipeline** | Retrains Ridge Regression on latest data, uploads new version to Hopsworks Model Registry |
| **72-Hour Forecast** | Recursive autoregressive forecasting predicts AQI for the next 3 days |
| **Interactive Dashboard** | Streamlit UI with pollutant cards, forecast cards, Plotly charts, SHAP explanations |
| **REST API** | Flask backend serving JSON endpoints for current conditions, forecast, and SHAP |
| **Model Comparison** | Trained and evaluated Baseline, Ridge, Random Forest, XGBoost, LSTM |
| **SHAP Explainability** | Feature contribution analysis for the 24h, 48h, and 72h horizons |
| **Dark Mode Dashboard** | Premium dark theme with pollutant severity color coding |
| **CSV Download** | Export the full 72-hour forecast directly from the dashboard |
| **100% Serverless** | No servers to manage — runs entirely on GitHub Actions and free-tier cloud APIs |

---

## 🏗️ Architecture
┌──────────────────┐
│ External APIs │ (Open-Meteo Weather, Open-Meteo Air Quality)
└────────┬─────────┘
↓
┌──────────────────────────────────────────────┐
│ GitHub Actions (Hourly Cron) │
│ ┌────────────────────────────────────────┐ │
│ │ scripts/feature_pipeline.py │ │
│ │ - Fetch weather + pollutants │ │
│ │ - Engineer features │ │
│ │ - Append to CSV │ │
│ │ - Upload to Hopsworks Feature Store │ │
│ └────────────────────────────────────────┘ │
└────────┬─────────────────────────────────────┘
↓
┌──────────────────┐ ┌──────────────────┐
│ Hopsworks FS │ │ Local CSV │
└────────┬─────────┘ └────────┬─────────┘
↓ ↓
┌──────────────────────────────────────────────┐
│ GitHub Actions (Daily Cron @ 02:00 UTC) │
│ ┌────────────────────────────────────────┐ │
│ │ scripts/training_pipeline.py │ │
│ │ - Load latest data │ │
│ │ - Train Ridge Regression │ │
│ │ - Evaluate MAE, RMSE, R² │ │
│ │ - Save local model + metrics │ │
│ │ - Upload to Hopsworks Model Registry │ │
│ └────────────────────────────────────────┘ │
└────────┬─────────────────────────────────────┘
↓
┌──────────────────────────────────────────────┐
│ Production Layer │
│ ┌─────────────┐ ┌──────────────────────┐ │
│ │ Flask API │ ←→ │ Streamlit Dashboard │ │
│ │ (port 5000) │ │ (port 8501) │ │
│ └─────────────┘ └──────────────────────┘ │
└──────────────────────────────────────────────

---

## 🛠️ Tech Stack

### Languages & Frameworks
- **Python 3.11** — Core language
- **Flask** — REST API backend
- **Streamlit** — Interactive dashboard
- **Plotly** — Charts and visualizations

### Machine Learning
- **Scikit-learn** — Ridge Regression, Random Forest, baseline
- **XGBoost** — Gradient boosting
- **TensorFlow / Keras** — LSTM deep learning
- **SHAP** — Model explainability

### Data & Storage
- **Pandas / NumPy** — Data manipulation
- **Hopsworks** — Feature Store + Model Registry
- **PyArrow** — High-performance data transfer
- **Joblib** — Model serialization

### APIs
- **Open-Meteo Weather API** — Real-time weather data (free, no key)
- **Open-Meteo Air Quality API** — Real-time pollutant data (free, no key)

### DevOps & CI/CD
- **GitHub Actions** — Hourly + daily workflow automation
- **APScheduler** — Local scheduling
- **python-dotenv** — Secret management

---

## 📁 Project Structure
Pearls_AQI_Predictor/
│
├── .github/
│ └── workflows/
│ ├── hourly.yml # Runs feature pipeline every hour
│ ├── daily.yml # Retrains model every day at 02:00 UTC
│ └── backfill.yml # Manual backfill workflow
│
├── app/
│ └── dashboard.py # Streamlit dashboard
│
├── api/
│ └── app.py # Flask REST API
│
├── config/
│ ├── init.py
│ └── settings.py
│
├── data/
│ └── processed/
│ └── lahore/
│ └── lahore_features_hourly.csv
│
├── models/
│ ├── ridge_aqi_1h.pkl
│ ├── random_forest_aqi_1h.pkl
│ ├── xgboost_aqi_1h.json
│ ├── lstm_aqi_1h.keras
│ ├── scaler_X.pkl
│ ├── scaler_y.pkl
│ ├── metrics.json
│ └── feature_columns.json
│
├── notebooks/
│ ├── 01_data_collection.ipynb
│ ├── 02_data_preprocessing.ipynb
│ ├── 03_EDA.ipynb
│ ├── 04_feature_engineering.ipynb
│ ├── 05_hopsworks_feature_store.ipynb
│ ├── 06_model_training.ipynb
│ ├── 07_model_registry.ipynb
│ ├── 08_shap_explainability.ipynb
│ └── 09_forecasting_pipeline.ipynb
│
├── scripts/
│ ├── feature_pipeline.py # Hourly data ingestion
│ ├── training_pipeline.py # Daily model retraining
│ └── backfill_pipeline.py # One-time gap filling
│
├── src/
│ └── forecast_engine.py # 72h recursive forecasting engine
│
├── tests/
│
├── .env # Local secrets (gitignored)
├── .gitignore
├── README.md
└── requirements.txt

---

## ⚙️ Installation & Setup

### 1. Clone the Repository

https://github.com/MtalhaShzd/10Pearls_AQI_Predictor
cd Pearls_AQI_Predictor

2. Create Virtual Environment

python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

3. Install Dependencies

pip install -r requirements.txt

4. Configure Environment Variables
Create .env in the project root:

HOPSWORKS_API_KEY=your_hopsworks_api_key_here

5. Run the Pipelines

# Fetch latest data
python scripts/feature_pipeline.py

# Retrain the model
python scripts/training_pipeline.py

# (Optional) Backfill historical gaps
python scripts/backfill_pipeline.py
6. Start the Backend API

python api/app.py
API runs on http://127.0.0.1:5000

7. Start the Dashboard

streamlit run app/dashboard.py
Dashboard runs on http://localhost:8501

🚀 Usage
Local Development

# Terminal 1: Backend API
python api/app.py

# Terminal 2: Streamlit Dashboard
streamlit run app/dashboard.py
Open your browser to http://localhost:8501 to see the dashboard.

Production (GitHub Actions)
Once pushed to GitHub, the pipelines run automatically:

Workflow	Schedule	Purpose
Hourly Feature Pipeline	Every hour at minute 0	Fetches latest data
Daily Training Pipeline	Every day at 02:00 UTC	Retrains the model
Backfill Pipeline	Manual trigger	Fills historical gaps
🔄 ML Pipeline Stages
Stage 1: Data Collection
Historical backfill from 2024-01-02 to present
Hourly weather variables: temperature, humidity, pressure, wind, precipitation
Hourly pollutants: PM2.5, PM10, CO, NO₂, SO₂, O₃, US AQI
Stage 2: Data Preprocessing
Removed duplicates by timestamp
Verified no missing hourly intervals
Validated all pollutant readings
Stage 3: Exploratory Data Analysis
Correlation analysis
Seasonal pattern detection
Pollutant trend visualization
AQI distribution by hour, day, month
Stage 4: Feature Engineering
Temporal features: hour, day, month, day_of_week, is_weekend
Cyclical encoding: sin/cos for hour and month
Lag features: aqi_lag_1h, aqi_lag_24h
Rolling features: aqi_rolling_mean_6h, aqi_rolling_std_6h, pm25_rolling_mean_6h
Change rate: aqi_change_rate
Total: 29 engineered features

Stage 5: Hopsworks Feature Store
Created feature group lahore_air_quality_features (v1)
Primary key: datetime
Event time: datetime
Online + offline enabled
Stage 6: Model Training
Time-based split: 70% train / 15% val / 15% test
Models compared:
Baseline (Persistence)
Ridge Regression
Random Forest
XGBoost
LSTM (TensorFlow)
Stage 7: Model Registry
Best model registered in Hopsworks Model Registry as lahore_aqi_ridge_1h
Stage 8: SHAP Explainability
LinearExplainer for Ridge model
Summary plot, waterfall plot, dependence plots
Stage 9: 72-Hour Forecasting
Recursive autoregressive prediction
72-step loop using the trained Ridge model
Live weather from Open-Meteo Forecast API
📊 Model Performance
Validation Set Results
Model	MAE	RMSE	R²
Ridge Regression ⭐	1.0044	1.5177	0.9988
Baseline (Persistence)	1.1930	1.8827	0.9981
XGBoost	1.0790	2.4960	0.9966
Random Forest	1.6369	2.6725	0.9961
LSTM	4.3263	5.7691	0.9819
Final Test Set (Ridge Regression)
Metric	Value
MAE	2.8946
RMSE	4.6925
R²	0.9868
90% of predictions within	±7.06 AQI units
95% of predictions within	±10.26 AQI units
Why Ridge Won
Hour-to-hour AQI changes are almost perfectly linear. The current AQI + recent history predicts the next hour AQI very well, making the linear Ridge model ideal. XGBoost and Random Forest slightly overfit the validation set.

🖥️ Dashboard Screenshots
The dashboard features:

6 pollutant metric cards (PM2.5, PM10, O₃, NO₂, SO₂, CO)
24-hour AQI trend with CURRENT / AVG / MIN / MAX statistics
Live weather conditions (temperature, humidity, pressure)
3 forecast horizon cards (24h / 48h / 72h) with health category badges
Predicted AQI trend line chart
Model performance info panel
SHAP-based "Why This Prediction" feature importance chart
Expandable 72-hour forecast table with CSV download
🔁 CI/CD Automation
GitHub Actions Workflows
Located in .github/workflows/:

hourly.yml
Trigger: Cron  (every hour)
Steps:
Checkout code
Set up Python 3.11
Install dependencies
Run scripts/feature_pipeline.py
Commit updated CSV back to repo
Push to main branch
daily.yml
Trigger: Cron (daily)
Steps:
Checkout code
Set up Python 3.11
Install dependencies
Run scripts/training_pipeline.py
Commit updated model back to repo
Push to main branch
backfill.yml
Trigger: Manual (workflow_dispatch)
Purpose: One-time gap filling for historical data
Required GitHub Secret
Secret	Description
HOPSWORKS_API_KEY	Your Hopsworks API key for cloud uploads
Set in: Settings → Secrets and variables → Actions

🌐 API Endpoints
The Flask backend (api/app.py) exposes the following endpoints:

Method	Endpoint	Description
GET	/	Health check
GET	/api/current	Current AQI, pollutants, weather, 24h trend
GET	/api/forecast	72-hour forecast with hourly predictions
GET	/api/shap/<horizon>	SHAP contributions for 24h, 48h, or 72h
Example: GET /api/current
JSON

{
  "current_aqi": 143.0,
  "timestamp": "2026-08-22 09:00:00",
  "pollutants": {
    "pm2_5": 53.3, "pm10": 74.3, "ozone": 66.0,
    "nitrogen_dioxide": 42.2, "sulphur_dioxide": 11.1,
    "carbon_monoxide": 814.0
  },
  "weather": {
    "temperature": 28.9, "humidity": 87.0,
    "pressure": 976.7, "wind_speed": 5.2
  }
}
📝 Notebooks (Research Documentation)
The notebooks/ folder contains the full research and development history:

#	Notebook	Purpose
01	01_data_collection.ipynb	Initial data fetching from APIs
02	02_data_preprocessing.ipynb	Cleaning, validation, deduplication
03	03_EDA.ipynb	Exploratory data analysis
04	04_feature_engineering.ipynb	Time, cyclical, lag, rolling features
05	05_hopsworks_feature_store.ipynb	Hopsworks Feature Store registration
06	06_model_training.ipynb	Training 5 models, evaluation
07	07_model_registry.ipynb	Best model registration in Hopsworks
08	08_shap_explainability.ipynb	SHAP analysis on Ridge model
09	09_forecasting_pipeline.ipynb	72-hour recursive forecast pipeline
🤝 Contributing
This is an internship project. For questions or suggestions, please open an issue on GitHub.

Acknowledgments
Open-Meteo for free, no-API-key weather and air quality data
Hopsworks for the Feature Store and Model Registry platform
10Pearls for the internship opportunity and project guidance


Project by: TALHA SHAHZAD
Repository: https://github.com/MtalhaShzd/10Pearls_AQI_Predictor
