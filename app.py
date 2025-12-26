from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load ML model and scaler
model = joblib.load("ridge_model.pkl")
scaler = joblib.load("scaler.pkl")


def get_risk(fwi):
    if fwi < 50:
        return "LOW RISK", "low"
    elif fwi < 100:
        return "MODERATE RISK", "moderate"
    elif fwi < 150:
        return "HIGH RISK", "high"
    else:
        return "EXTREME RISK", "extreme"


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        values = [
            float(request.form["Temperature"]),
            float(request.form["RH"]),
            float(request.form["Ws"]),
            float(request.form["Rain"]),
            float(request.form["FFMC"]),
            float(request.form["DMC"]),
            float(request.form["DC"]),
            float(request.form["ISI"]),
            float(request.form["BUI"]),
        ]

        features = np.array([values])
        scaled = scaler.transform(features)
        prediction = round(model.predict(scaled)[0], 2)

        risk_label, risk_class = get_risk(prediction)

        return render_template(
            "index.html",
            result=prediction,
            risk=risk_label,
            risk_class=risk_class
        )

    except Exception:
        return render_template(
            "index.html",
            error="Please enter valid numeric values"
        )


if __name__ == "__main__":
    app.run(debug=True)
