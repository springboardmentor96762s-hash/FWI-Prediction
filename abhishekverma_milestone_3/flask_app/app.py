from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle
import os

app = Flask(__name__)

# Load your trained Ridge model and scaler
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ridge_path = os.path.join(BASE_DIR, 'ridge.pkl')
scaler_path = os.path.join(BASE_DIR, 'scaler.pkl')

with open(ridge_path, 'rb') as f:
    model = pickle.load(f)

with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

# FWI risk thresholds
def get_fwi_risk(fwi):
    if fwi < 5:
        return "Very Low Risk"
    elif fwi < 12:
        return "Low Risk"
    elif fwi < 25:
        return "Moderate Risk"
    elif fwi < 50:
        return "High Risk"
    elif fwi < 75:
        return "Very High Risk"
    else:
        return "Extreme Risk"

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    try:
        # ✅ Correct order: day, month, year first (as in training), then features, then region
        features = [
            float(data.get("day", 1)),
            float(data.get("month", 1)),
            float(data.get("year", 2024)),
            float(data.get("temperature", 0)),
            float(data.get("rh", 0)),
            float(data.get("ws", 0)),
            float(data.get("rain", 0)),
            float(data.get("ffmc", 0)),
            float(data.get("dmc", 0)),
            float(data.get("dc", 0)),
            float(data.get("isi", 0)),
            float(data.get("bui", 0)),

            0,  # <-- FIX: Classes dummy value because your model expects 14 features

            float(data.get("region", 0)),
        ]

        # Scale features
        features_scaled = scaler.transform([features])

        # Predict FWI
        fwi = model.predict(features_scaled)[0]
        risk = get_fwi_risk(fwi)

        response = {
            "fwi": round(fwi, 2),
            "risk": risk
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)