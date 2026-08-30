# 🌍 Pearls AQI Predictor

### AI-Powered Air Quality Forecasting System for Lahore, Pakistan

An end-to-end **Machine Learning and MLOps system** that collects real-time weather and air-quality data, engineers predictive features, stores them in **Hopsworks Feature Store**, trains and evaluates multiple ML models, provides **SHAP-based explainability**, and generates **72-hour recursive AQI forecasts** through a Flask REST API and Streamlit dashboard.

> **Developed as part of the 10Pearls Shine Program — Data Science Learning Path**

---

## 📊 Project Overview

Air Quality Index (AQI) can change significantly throughout the day due to weather conditions and pollutant concentrations.

The **Pearls AQI Predictor** addresses this problem by building an automated forecasting pipeline that:

* Collects hourly weather and pollutant data
* Processes and validates historical and real-time data
* Engineers temporal, lag, rolling, and cyclical features
* Stores features in Hopsworks Feature Store
* Trains and compares multiple machine-learning models
* Registers the best model in Hopsworks Model Registry
* Explains predictions using SHAP
* Generates recursive **24-hour, 48-hour, and 72-hour AQI forecasts**
* Provides predictions through a Flask REST API
* Displays results through an interactive Streamlit dashboard
* Automates hourly data ingestion and daily model retraining using GitHub Actions

---

## ✨ Key Features

| Feature                | Description                                                                |
| ---------------------- | -------------------------------------------------------------------------- |
| 🌦️ Weather Data       | Temperature, humidity, pressure, precipitation, wind and cloud information |
| 🏭 Air Quality Data    | PM2.5, PM10, CO, NO₂, SO₂, O₃ and US AQI                                   |
| 🧹 Data Validation     | Duplicate, missing-value and hourly-continuity checks                      |
| ⚙️ Feature Engineering | Temporal, cyclical, lag, rolling and change-rate features                  |
| 🗄️ Feature Store      | Hopsworks Feature Store                                                    |
| 🤖 Model Training      | Ridge, Random Forest, XGBoost and LSTM                                     |
| 🏆 Model Selection     | Time-based validation and test evaluation                                  |
| 📦 Model Registry      | Hopsworks Model Registry                                                   |
| 🔎 Explainability      | SHAP feature contributions                                                 |
| 🔮 Forecasting         | Recursive 72-hour AQI forecasting                                          |
| 🚀 REST API            | Flask backend                                                              |
| 📊 Dashboard           | Streamlit interactive dashboard                                            |
| 🔄 Automation          | GitHub Actions hourly and daily workflows                                  |
| 📥 Export              | 72-hour forecast CSV download                                              |

---

#  System Architecture

```text
                         ┌─────────────────────────────┐
                         │       External APIs         │
                         │                             │
                         │  Open-Meteo Weather API     │
                         │  Open-Meteo Air Quality API │
                         └──────────────┬──────────────┘
                                        │
                                        │ Hourly data
                                        ▼
                    ┌────────────────────────────────────┐
                    │       GitHub Actions                │
                    │       Hourly Workflow               │
                    └────────────────┬───────────────────┘
                                     │
                                     ▼
                    ┌────────────────────────────────────┐
                    │     Feature Pipeline                │
                    │ scripts/feature_pipeline.py         │
                    │                                    │
                    │ • Fetch data                        │
                    │ • Validate data                     │
                    │ • Engineer features                 │
                    │ • Update local dataset              │
                    │ • Upload to Hopsworks               │
                    └───────────────┬────────────────────┘
                                    │
                    ┌───────────────┴────────────────┐
                    │                                │
                    ▼                                ▼
        ┌──────────────────────┐         ┌──────────────────────┐
        │ Hopsworks Feature    │         │ Local Processed CSV  │
        │ Store                │         │                      │
        └──────────┬───────────┘         └──────────┬───────────┘
                   │                                │
                   └──────────────┬─────────────────┘
                                  │
                                  ▼
                    ┌────────────────────────────────────┐
                    │      GitHub Actions                 │
                    │      Daily Training Workflow        │
                    └────────────────┬───────────────────┘
                                     │
                                     ▼
                    ┌────────────────────────────────────┐
                    │       Training Pipeline             │
                    │ scripts/training_pipeline.py        │
                    │                                    │
                    │ • Load latest data                  │
                    │ • Train models                      │
                    │ • Evaluate models                   │
                    │ • Select best model                 │
                    │ • Save metrics                      │
                    │ • Register model                    │
                    └────────────────┬───────────────────┘
                                     │
                                     ▼
                    ┌────────────────────────────────────┐
                    │     Hopsworks Model Registry        │
                    └────────────────┬───────────────────┘
                                     │
                                     ▼
                    ┌────────────────────────────────────┐
                    │        Production Layer             │
                    │                                    │
                    │       ┌──────────────────┐         │
                    │       │    Flask API     │         │
                    │       │    Port 5000     │         │
                    │       └────────┬─────────┘         │
                    │                │                   │
                    │                ▼                   │
                    │       ┌──────────────────┐         │
                    │       │ Streamlit        │         │
                    │       │ Dashboard        │         │
                    │       │ Port 8501        │         │
                    │       └──────────────────┘         │
                    └────────────────────────────────────┘
```

---

# 🧠 Machine Learning Pipeline

The project is organized into nine major stages.

## Stage 1 — Data Collection

Historical data was collected from **January 2024 onward** using Open-Meteo weather and air-quality APIs.

### Weather Features

* Temperature
* Relative humidity
* Surface pressure
* Precipitation
* Cloud cover
* Wind speed
* Wind direction

### Pollutant Features

* PM2.5
* PM10
* Carbon monoxide
* Nitrogen dioxide
* Sulphur dioxide
* Ozone
* US AQI

---

## Stage 2 — Data Preprocessing

The dataset was validated and cleaned through:

* Timestamp validation
* Duplicate removal
* Missing-value checks
* Hourly interval validation
* Data type validation
* Chronological ordering

---

## Stage 3 — Exploratory Data Analysis

The EDA stage investigates:

* AQI distribution
* Pollutant relationships
* Weather/AQI correlations
* Hourly AQI patterns
* Daily patterns
* Monthly and seasonal patterns
* Pollutant trends
* AQI variability

---

## Stage 4 — Feature Engineering

The forecasting pipeline uses temporal, cyclical, lagged and rolling features.

### Temporal Features

* `hour`
* `day`
* `month`
* `day_of_week`
* `is_weekend`

### Cyclical Features

* `hour_sin`
* `hour_cos`
* `month_sin`
* `month_cos`

### Lag Features

* `aqi_lag_1h`
* `aqi_lag_24h`

### Rolling Features

* `aqi_rolling_mean_6h`
* `aqi_rolling_std_6h`
* `pm25_rolling_mean_6h`

### Change Feature

* `aqi_change_rate`

These features allow the model to capture both **short-term AQI momentum** and **daily/seasonal patterns**.

---

# 🗄️ Stage 5 — Hopsworks Feature Store

The project uses Hopsworks Feature Store for centralized feature management.

### Feature Group

```text
lahore_air_quality_features
```

### Configuration

```text
Version:       2
Primary Key:   datetime
Event Time:    datetime
Online Store:  Enabled
Offline Store: Enabled
```

The feature store provides a consistent source of engineered features for model training and future production workflows.

---

# 🤖 Stage 6 — Model Training

Multiple approaches were evaluated using a chronological, time-based split.

### Models Compared

1. Persistence Baseline
2. Ridge Regression
3. Random Forest
4. XGBoost
5. LSTM

The data was divided into:

```text
70% Training
15% Validation
15% Testing
```

A chronological split was used instead of random shuffling to better represent real-world forecasting.

---

# 🏆 Model Performance

## Validation Results

| Model                |        MAE |       RMSE |         R² |
| -------------------- | ---------: | ---------: | ---------: |
| 🥇 Ridge Regression  | **1.0044** | **1.5177** | **0.9988** |
| Persistence Baseline |     1.1930 |     1.8827 |     0.9981 |
| XGBoost              |     1.0790 |     2.4960 |     0.9966 |
| Random Forest        |     1.6369 |     2.6725 |     0.9961 |
| LSTM                 |     4.3263 |     5.7691 |     0.9819 |

## Final Test Results — Ridge Regression

| Metric                        |     Result |
| ----------------------------- | ---------: |
| MAE                           | **2.8946** |
| RMSE                          | **4.6925** |
| R²                            | **0.9868** |
| Predictions within ±7.06 AQI  |    **90%** |
| Predictions within ±10.26 AQI |    **95%** |

### Why Ridge Regression?

Ridge Regression achieved the strongest overall performance on the validation set while remaining computationally lightweight.

AQI has strong short-term temporal dependence, and the engineered lag and rolling features provide substantial predictive information.

The Ridge model therefore provides a good balance between:

* Accuracy
* Training speed
* Interpretability
* Deployment simplicity
* Explainability
Model metrics evolve with each daily retraining cycle; the table above reflects results at initial model selection — live metrics are available via /api/forecast

---

# 📦 Stage 7 — Model Registry

The selected production model is registered through the **Hopsworks Model Registry**.

The registry provides a centralized location for:

* Model versions
* Model artifacts
* Model metadata
* Model metrics
* Production model management

The project uses the registered Ridge model for forecasting.

---

# 🔎 Stage 8 — SHAP Explainability

The project uses **SHAP** to explain individual AQI predictions.

The explainability layer answers:

> **"Why did the model predict this AQI?"**

The dashboard can display feature contributions for the selected forecast horizon.

Examples of explainable features include:

* Previous AQI
* PM2.5
* PM10
* Temperature
* Humidity
* Wind speed
* Lag features
* Rolling AQI statistics
* Temporal features

The project uses SHAP analysis to improve model transparency and make predictions easier to interpret.

---

# 🔮 Stage 9 — 72-Hour Forecasting

The forecasting engine generates a recursive **72-step hourly forecast**.

```text
Current AQI
     │
     ▼
Predict t+1
     │
     ▼
Update lag features
     │
     ▼
Predict t+2
     │
     ▼
Update features
     │
     ▼
     ...
     │
     ▼
Predict t+72
```

The system supports:

* 24-hour forecast
* 48-hour forecast
* 72-hour forecast

Future weather information is obtained from the Open-Meteo Forecast API.

The resulting predictions are presented as hourly AQI forecasts in the dashboard.

---

# 📊 Streamlit Dashboard

The dashboard provides an interactive interface for monitoring current conditions and future AQI.

### Dashboard Components

* Current AQI
* PM2.5
* PM10
* O₃
* NO₂
* SO₂
* CO
* 24-hour AQI trend
* Current / Average / Minimum / Maximum AQI
* Current weather conditions
* 24-hour forecast
* 48-hour forecast
* 72-hour forecast
* AQI health categories
* Predicted AQI trend
* Model performance information
* SHAP-based feature importance
* Expandable 72-hour forecast table
* CSV forecast download

---

# 🌐 REST API

The backend is implemented using **Flask**.

## Base URL

```text
local host
http://localhost:5000
Render Api
https://one0pearls-aqi-predictor.onrender.com
```

## Endpoints

### Health Check

```http
GET /
```

Returns the API health/status information.

---

### Current AQI

```http
GET /api/current
```

Returns current AQI, pollutants, weather and recent AQI information.

Example response:

```json
{
  "current_aqi": 143.0,
  "timestamp": "2026-08-22 09:00:00",
  "pollutants": {
    "pm2_5": 53.3,
    "pm10": 74.3,
    "ozone": 66.0,
    "nitrogen_dioxide": 42.2,
    "sulphur_dioxide": 11.1,
    "carbon_monoxide": 814.0
  },
  "weather": {
    "temperature": 28.9,
    "humidity": 87.0,
    "pressure": 976.7,
    "wind_speed": 5.2
  }
}
```

---

### AQI Forecast

```http
GET /api/forecast
```

Returns the hourly 72-hour AQI forecast.

---

### SHAP Explanation

```http
GET /api/shap/<horizon>
```

Supported horizons:

```text
24h
48h
72h
```

Example:

```http
GET /api/shap/24h
```

Returns SHAP contributions used by the dashboard to explain the selected prediction.

### Health Check

```http
GET /health
```

Returns a simple service-level health status, separate from the root `/` check.

---

### List Available Routes

```http
GET /routes
```

Returns a list of all registered API endpoints — useful for verifying deployment.

---

### Data Source Status

```http
GET /api/source
```

Returns which data source (Hopsworks or local CSV fallback) and which
model source (Hopsworks Model Registry or local artifact) are
currently being served, along with row counts and cache age. Used by
the dashboard to detect and display fallback/degraded-mode status.

Example response:

```json
{
  "success": true,
  "source": {
    "feature_source": "Hopsworks Feature Store",
    "model_source": "Hopsworks Model Registry",
    "model_version": "v33",
    "feature_group": "lahore_air_quality_features",
    "feature_rows": 23283,
    "cache_age_seconds": 142.6
  }
}
```

---

### Force Reload

```http
GET /api/reload
POST /api/reload
```

Forces a fresh read of both the model and feature data, bypassing the
in-memory cache. Useful for manually confirming Hopsworks connectivity
has recovered without waiting for the normal cache TTL to expire.
---

# 🔄 CI/CD & Automation

GitHub Actions automates the ML pipeline.

## Hourly Feature Pipeline

**Workflow:**

```text
.github/workflows/hourly.yml
```

### Schedule

```text
Every hour
```

### Tasks

1. Checkout repository
2. Set up Python
3. Install dependencies
4. Fetch latest weather and AQI data
5. Engineer features
6. Update processed dataset
7. Upload features to Hopsworks
8. Commit updated data when required

---

## Daily Training Pipeline

**Workflow:**

```text
.github/workflows/daily.yml
```

### Schedule

```text
Every day at 02:00 UTC & PKT 07:00
```

### Tasks

1. Checkout repository
2. Set up Python
3. Install dependencies
4. Load latest feature data
5. Train the production model
6. Evaluate model performance
7. Save metrics
8. Register the model in Hopsworks
9. Update production artifacts when required

---

## Backfill Pipeline

**Workflow:**

```text
.github/workflows/backfill.yml
```

The workflow runs after every 6 hours for backfilling the gaps or when historical data contains gaps or requires backfilling.

---

# 🔐 Secrets & Configuration

Sensitive credentials are **not committed to GitHub**.

Local development uses environment variables.

Example:

```text
HOPSWORKS_API_KEY=your_api_key
```

For GitHub Actions, configure the required credentials under:

```text
Repository
→ Settings
→ Secrets and variables
→ Actions
```

Required secret:

```text
HOPSWORKS_API_KEY
```

The following files/directories are intentionally excluded from Git:

```text
.env
.venv/
hopsworks-certs/
.streamlit/secrets.toml
models/*.pkl
models/*.keras
models/shap_analysis/
data/raw/
```

---

# 📁 Project Structure

```text
Pearls_AQI_Predictor/
│
├── .github/
│   └── workflows/
│       ├── hourly.yml
│       ├── daily.yml
│       └── backfill.yml
│
├── app/
│   └── dashboard.py
│
├── api/
│   └── app.py
│
├── config/
│   ├── __init__.py
│   └── settings.py
│
├── data/
│   └── processed/
│       └── lahore/
│           └── lahore_features_hourly.csv
│
├── models/
│   ├── metrics.json
│   └── feature_columns.json
│
├── notebooks/
│   ├── 01_data_collection.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_EDA.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_hopsworks_feature_store.ipynb
│   ├── 06_model_training.ipynb
│   ├── 07_model_registry.ipynb
│   ├── 08_shap_explainability.ipynb
│   └── 09_forecasting_pipeline.ipynb
│
├── scripts/
│   ├── feature_pipeline.py
│   ├── training_pipeline.py
│   └── backfill_pipeline.py
│
├── src/
│   └── forecast_engine.py
|   └── hopsworks_store.py
│
├── tests/
│
├── .env
├── .gitignore
├── README.md
└── requirements.txt
```

> Large model binaries and credentials are intentionally excluded from the public repository.

---

# 🚀 Installation & Setup

## 1. Clone the Repository

```bash
git clone https://github.com/MtalhaShzd/10Pearls_AQI_Predictor.git
cd Pearls_AQI_Predictor
```

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Configure Environment Variables

Create a `.env` file:

```text
HOPSWORKS_API_KEY=your_hopsworks_api_key
```

Do not commit this file.

---

# ▶️ Running the Application Locally

The project contains two application components:

```text
Flask API        → Port 5000
Streamlit        → Port 8501
```

### Terminal 1 — Start Flask API

Run the Flask application according to the entry point configured in:

```text
api/app.py
```

The API should be available at:

```text
http://localhost:5000
```

---

### Terminal 2 — Start Streamlit Dashboard

```bash
streamlit run app/dashboard.py
```

Then open:

```text
http://localhost:8501
```

> The Streamlit dashboard communicates with the Flask API for current AQI, forecasts and SHAP explanations.
> Deployed at Streamlit Cloud and Render
> Live URL: https://10pearls-lahore-aqi-predictor.streamlit.app

---
---

# ☁️ Deployment

The project is deployed across three separate platforms, each handling
a different layer of the stack:

| Component              | Platform                | Notes                                      |
| ----------------------- | ------------------------ | ------------------------------------------- |
| Flask REST API          | Render (Web Service)     | Runs via Gunicorn; auto-deploys on push to `main` |
| Streamlit Dashboard     | Streamlit Community Cloud | Auto-deploys on push to `main`             |
| Feature Store & Model Registry | Hopsworks Serverless | Managed platform; not self-hosted          |
| CI/CD Automation        | GitHub Actions           | Scheduled + manually-triggerable workflows |

### Live URLs

```text
API (Render):
https://one0pearls-aqi-predictor.onrender.com

Dashboard (Streamlit Cloud):
https://10pearls-lahore-aqi-predictor.streamlit.app
```

### Render Configuration

The API service runs via Gunicorn rather than Flask's built-in
development server:

```text
gunicorn api.app:app --timeout 300 --workers 1 --threads 4 --graceful-timeout 120
```

Render injects the listening port via the `PORT` environment variable
at runtime rather than a fixed port. Required environment variables
(configured in Render's dashboard, not committed to the repo):

```text
HOPSWORKS_API_KEY
HOPSWORKS_HOST
HOPSWORKS_PROJECT
HOPSWORKS_FEATURE_GROUP
HOPSWORKS_FEATURE_GROUP_VERSION
```

### Streamlit Cloud Configuration

The dashboard reads the API's base URL from an environment variable,
defaulting to the Render deployment above if unset:

```text
API_URL=https://one0pearls-aqi-predictor.onrender.com
```

### Free-Tier Constraints

Both Render and Streamlit Cloud's free tiers spin down idle services
after a period of inactivity, resulting in a cold-start delay
(typically 30-50 seconds) on the first request after idle time. The
dashboard surfaces this explicitly to the user rather than presenting
it as an error.

# 🧪 Research Notebooks

The notebooks document the complete research and development process.

| Notebook                           | Purpose                                 |
| ---------------------------------- | --------------------------------------- |
| `01_data_collection.ipynb`         | Initial weather and AQI data collection |
| `02_data_preprocessing.ipynb`      | Cleaning and validation                 |
| `03_EDA.ipynb`                     | Exploratory data analysis               |
| `04_feature_engineering.ipynb`     | Feature creation                        |
| `05_hopsworks_feature_store.ipynb` | Feature Store integration               |
| `06_model_training.ipynb`          | Model comparison and evaluation         |
| `07_model_registry.ipynb`          | Model registration                      |
| `08_shap_explainability.ipynb`     | SHAP explainability                     |
| `09_forecasting_pipeline.ipynb`    | 72-hour forecasting                     |

---

# 🛠️ Technology Stack

### Programming

* Python
* SQL / data-processing workflows

### Machine Learning

* Scikit-learn
* XGBoost
* TensorFlow / Keras
* SHAP

### MLOps

* Hopsworks Feature Store
* Hopsworks Model Registry
* GitHub Actions

### Backend

* Flask
* REST API

### Frontend

* Streamlit

### Data Sources

* Open-Meteo Weather API
* Open-Meteo Air Quality API

### Development

* VS Code
* Jupyter Notebook
* Git
* GitHub

---

# 📈 Current Production Flow

```text
External Data
     ↓
Hourly Feature Pipeline
     ↓
Feature Engineering
     ↓
Hopsworks Feature Store
     ↓
Daily Model Training
     ↓
Model Evaluation
     ↓
Hopsworks Model Registry
     ↓
Forecast Engine
     ↓
Flask REST API
     ↓
Streamlit Dashboard
     ↓
24h / 48h / 72h AQI Forecast
```

---

# 🎯 Project Goals

The project demonstrates an end-to-end practical implementation of:

* Machine Learning
* Time-series forecasting
* Feature engineering
* Feature stores
* Model registries
* Model explainability
* REST API development
* Data pipelines
* CI/CD automation
* Production-oriented ML workflows

The primary goal is to transform raw environmental data into an automated and explainable AQI forecasting system.

---

# 🔮 Future Improvements

Potential improvements include:

* Automated model drift detection
* Data quality monitoring
* Model performance monitoring
* Automated rollback to the previous model version
* More advanced multi-step forecasting strategies
* Additional cities and geographical regions
* Weather forecast uncertainty integration
---

# 🤝 Contributing

This project was developed as part of the **10Pearls Shine Internship Program**.

Suggestions, and improvements are welcome.

---

#  Acknowledgments

Special thanks to:

* **10Pearls** — for the internship/program opportunity and project guidance.

---

# 👨‍💻 Author

**Talha Shahzad**

Data Science / AI & MLOps Project

**Project:** Pearls AQI Predictor

---
---

## 📄 Project Documentation

The full technical report — covering system architecture, methodology, model evaluation, resilience engineering, and a detailed log of production challenges and their resolutions — is available here:

- 📥 **Download (GitHub):** [10Pearls_AQI_Predictor_Technical_Report.pdf](docs/10Pearls_AQI_Predictor_Technical_Report.pdf)
- 📖 **View Online (Google Drive):** [10Pearls AQI Predictor — Technical Report](https://drive.google.com/file/d/1cpcCwbP05ACp0devY_1jKa2W8P1a-mSd/view?usp=sharing)

> If GitHub's inline preview have some issues, download the file and open it — it will populate automatically.

---

## ⭐ Project Summary

> **Pearls AQI Predictor is an end-to-end AI and MLOps system that automatically collects environmental data, engineers forecasting features, stores them using Hopsworks, trains and evaluates machine-learning models, explains predictions using SHAP, and provides automated 72-hour AQI forecasts through a Flask API and Streamlit dashboard.**
