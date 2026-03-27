from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load model and scaler
model = joblib.load("ridge.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Collect input values from form
        features = [
            float(request.form["Temperature"]),
            float(request.form["RH"]),
            float(request.form["Ws"]),
            float(request.form["Rain"]),
            float(request.form["FFMC"]),
            float(request.form["DMC"]),
            float(request.form["DC"]),
            float(request.form["ISI"]),
            float(request.form["BUI"])
        ]

        # Convert to 2D numpy array
        arr = np.array(features).reshape(1, -1)

        # Scale input
        arr_scaled = scaler.transform(arr)

        # Predict FWI value
        fwi = model.predict(arr_scaled)[0]
        fwi = round(fwi, 2)

        # Decide fire or not
        if fwi > 20:
            fire_status = "🔥 FIRE RISK HIGH"
        else:
            fire_status = "🟢 NO FIRE RISK"

        return render_template("home.html", fwi=fwi, fire_status=fire_status)

    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    app.run(debug=True)
