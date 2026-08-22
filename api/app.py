# api/app.py

import sys
from pathlib import Path
from flask import Flask, jsonify

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.forecast_engine import (
    generate_72h_forecast, get_current_conditions, 
    get_shap_explanation, MODEL_METRICS
)

app = Flask(__name__)

@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"status": "online", "city": "Lahore"}), 200

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
        forecast_df["datetime"] = forecast_df["datetime"].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        predictions = []
        for _, row in forecast_df.iterrows():
            aqi = float(row["predicted_aqi"])
            if aqi <= 50: label = "Good"
            elif aqi <= 100: label = "Moderate"
            elif aqi <= 150: label = "Unhealthy for Sensitive Groups"
            elif aqi <= 200: label = "Unhealthy"
            elif aqi <= 300: label = "Very Unhealthy"
            else: label = "Hazardous"
            predictions.append({
                "datetime": row["datetime"],
                "predicted_aqi": round(aqi, 2),
                "label": label
            })
        return jsonify({
            "success": True, 
            "city": "Lahore",
            "predictions": predictions,
            "model_metrics": MODEL_METRICS
        }), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/shap/<horizon>", methods=["GET"])
def api_shap(horizon):
    try:
        contributions = get_shap_explanation(horizon)
        return jsonify({"success": True, "horizon": horizon, "contributions": contributions}), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)