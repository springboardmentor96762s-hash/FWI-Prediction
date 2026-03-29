from flask import Flask, render_template, request, flash
import numpy as np
import pickle

app = Flask(__name__)
app.secret_key = "some_secret_key"

with open('ridge.pkl', 'rb') as f:
    ridge_model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)


def fwi_to_risk(fwi_value: float) -> str:
    """Convert numeric FWI to a simple risk label."""
    if fwi_value < 5:
        return "Low"
    elif fwi_value < 15:
        return "Moderate"
    else:
        return "High"


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [
            float(request.form['Temperature']),
            float(request.form['RH']),
            float(request.form['Ws']),
            float(request.form['Rain']),
            float(request.form['DMC']),
            float(request.form['DC']),
            float(request.form['ISI']),
            float(request.form['BUI']),
            float(request.form['FFMC'])
        ]
    except (ValueError, KeyError):
        flash("Please fill all fields with valid numeric values.")
        return render_template('index.html')

    X = np.array([features])
    X_scaled = scaler.transform(X)

    fwi = ridge_model.predict(X_scaled)[0]
    fwi = round(float(fwi), 3)

    risk_label = fwi_to_risk(fwi)

    return render_template(
        'home.html',
        prediction=fwi,
        risk=risk_label
    )


if __name__ == '__main__':
    app.run(debug=True)
