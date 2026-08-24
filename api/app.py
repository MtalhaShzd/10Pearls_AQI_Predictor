# api/app.py

import sys
from pathlib import Path
from flask import Flask, jsonify
from flask_cors import CORS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.forecast_engine import (
    generate_72h_forecast,
    get_current_conditions,
    get_shap_explanation,
    get_model_metrics,
    get_data_source_info,
)
from src.hopsworks_store import load_model_artifacts, get_features_df

app = Flask(__name__)
CORS(app)


@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"status": "online", "city": "Lahore"}), 200


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "Lahore AQI API"}), 200


@app.route("/routes", methods=["GET"])
def list_routes():
    return jsonify({
        "routes": sorted([str(r.rule) for r in app.url_map.iter_rules()])
    }), 200


@app.route("/api/source", methods=["GET"])
def api_source():
    try:
        return jsonify({"success": True, "source": get_data_source_info()}), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/reload", methods=["GET", "POST"])
def api_reload():
    try:
        _, _, metrics = load_model_artifacts(force_refresh=True)
        df = get_features_df(force_refresh=True)
        return jsonify({
            "success": True,
            "metrics": metrics,
            "feature_rows": int(len(df))
        }), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/current", methods=["GET"])
def api_current_conditions():
    try:
        return jsonify({"success": True, "data": get_current_conditions()}), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/forecast", methods=["GET"])
def api_forecast():
    try:
        forecast_df = generate_72h_forecast()
        forecast_df["datetime"] = forecast_df["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")

        predictions = []
        for _, row in forecast_df.iterrows():
            aqi = float(row["predicted_aqi"])
            if aqi <= 50:
                label = "Good"
            elif aqi <= 100:
                label = "Moderate"
            elif aqi <= 150:
                label = "Unhealthy for Sensitive Groups"
            elif aqi <= 200:
                label = "Unhealthy"
            elif aqi <= 300:
                label = "Very Unhealthy"
            else:
                label = "Hazardous"

            predictions.append({
                "datetime": str(row["datetime"]),
                "predicted_aqi": round(aqi, 2),
                "label": label
            })

        return jsonify({
            "success": True,
            "city": "Lahore",
            "predictions": predictions,
            "model_metrics": get_model_metrics(),
            "source": get_data_source_info()
        }), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/shap/<horizon>", methods=["GET"])
def api_shap(horizon):
    try:
        contributions = get_shap_explanation(horizon)
        return jsonify({
            "success": True,
            "horizon": horizon,
            "contributions": contributions
        }), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
