from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model and scaler
with open('ridge.pkl', 'rb') as f:
    model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

feature_names = ['Region','Temperature','RH','Ws','Rain','FFMC','DMC','DC','ISI','BUI']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = [float(request.form[f]) for f in feature_names]
        df_input = pd.DataFrame([data], columns=feature_names)
        scaled_input = scaler.transform(df_input)
        pred = model.predict(scaled_input)[0]
        pred = round(float(pred), 3)

        if pred < 2:   risk = "Very Low"
        elif pred < 5: risk = "Low–Moderate"
        elif pred < 12: risk = "High"
        elif pred < 25: risk = "Very High"
        else: risk = "Extreme"

        return render_template('home.html', prediction=pred, risk=risk)
    except Exception as e:
        return f"<h2>Error: {e}</h2>"

if __name__ == '__main__':
    app.run(debug=True)
