from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load model + scaler
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ridge_model = pickle.load(open(os.path.join(BASE_DIR, "ridge.pkl"), "rb"))
scaler = pickle.load(open(os.path.join(BASE_DIR, "scaler.pkl"), "rb"))

FEATURES = ['FFMC', 'DMC', 'DC', 'ISI', 'BUI', 'TEMP', 'RH', 'RAIN']

@app.route('/')
def index():
    return render_template('index.html', features=FEATURES)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        values = [float(request.form[f]) for f in FEATURES]
        scaled = scaler.transform([values])
        prediction = ridge_model.predict(scaled)[0]

        # Assign Risk Levels
        if prediction < 5:
            risk = "No Risk"
        elif prediction < 15:
            risk = "Moderate Risk"
        elif prediction < 30:
            risk = "High Risk"
        else:
            risk = "Extreme Risk"

        return render_template(
            "home.html",
            prediction=round(prediction, 2),
            risk=risk
        )
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
