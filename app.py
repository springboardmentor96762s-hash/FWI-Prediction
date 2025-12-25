from flask import Flask, render_template, request
import numpy as np
import pickle
import os

app = Flask(__name__)

try:
    
    ridge = pickle.load(open("ridge.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
except FileNotFoundError:
    print("Error: ridge.pkl or scaler.pkl not found. The application will not be able to make predictions.")
  
    ridge = None
    scaler = None


def get_fwi_status(fwi_value):
    """Determines the fire danger status based on the FWI value."""
    if fwi_value < 5.2:
        return "Low", "The Fire Weather Index is **Low**. Fire danger is minimal, but caution is always advised."
    elif 5.2 <= fwi_value < 11.2:
        return "Moderate", "The Fire Weather Index is **Moderate**. Fires can start easily and may be difficult to control."
    elif 11.2 <= fwi_value < 21.3:
        return "High", "The Fire Weather Index is **High**. Fires will start easily and spread rapidly. Extreme caution is necessary."
    elif 21.3 <= fwi_value < 38.0:
        return "Very High", "The Fire Weather Index is **Very High**. Fires will start very easily, spread quickly, and be challenging to suppress."
    else: 
        return "Extreme", "The Fire Weather Index is **Extreme**. Fires will start and spread aggressively. All fire activity should be prohibited."

@app.route("/")
def home():

    regions = ["Bejaia Region", "Sidi Bel-Abbès Region"]
    return render_template("index.html", regions=regions)

@app.route("/predict", methods=["POST"])
def predict():
    if ridge is None or scaler is None:
        return "Model files not loaded. Cannot make prediction.", 500

    try:
        region = request.form["Region"]
        data = [
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
    except ValueError:
        return "Invalid input. Please ensure all fields are filled with valid numbers.", 400
    except KeyError as e:
        return f"Missing form field: {e}", 400

    arr = np.array([data])

    scaled = scaler.transform(arr)

    prediction = ridge.predict(scaled)[0]
    fwi_value = round(prediction, 2)

    status, message = get_fwi_status(fwi_value)

    return render_template("home.html", 
                           fwi_value=fwi_value, 
                           status=status, 
                           message=message,
                           region=region)

if __name__ == "__main__":
    app.run(debug=True)
