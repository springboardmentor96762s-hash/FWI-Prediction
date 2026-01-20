from flask import Flask, render_template, request
import numpy as np
import pickle
import os

app = Flask(__name__)

# uploading ridge.pkl and scaler.pkl 
MODEL_PATH = os.path.join(os.path.dirname(__file__), "ridge.pkl")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "scaler.pkl")

# Load model and scaler (will raise a clear exception if missing)
model = pickle.load(open(MODEL_PATH, "rb"))
scaler = pickle.load(open(SCALER_PATH, "rb"))

# selected features
FEATURES = ["temperature", "humidity", "rain", "FFMC", "DMC", "DC", "ISI", "BUI"]

def classify_danger(fwi):
   
    if fwi < 5:
        return ("Low Danger", "low", "low.gif", "low.mp3",
                f"FWI {fwi:.1f}. Low danger. Conditions are safe.")
    elif fwi < 15:
        return ("Moderate Danger", "moderate", "moderate.gif", "moderate.mp3",
                f"FWI {fwi:.1f}. Moderate danger. Stay alert.")
    elif fwi < 30:
        return ("High Danger", "high", "high.gif", "high.mp3",
                f"FWI {fwi:.1f}. High danger. Fire can spread easily. Be cautious.")
    elif fwi < 45:
        return ("Very High Danger", "veryhigh", "veryhigh.gif", "alert.mp3",
                f"FWI {fwi:.1f}. Very high danger. Avoid ignition sources.")
    else:
        return ("Extreme Danger", "extreme", "extreme.gif", "alert.mp3",
                f"FWI {fwi:.1f}. Extreme danger. Immediate action required.")

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        values = []
        for feat in FEATURES:
            v = request.form.get(feat)
            if v is None or str(v).strip() == "":
                
                return render_template("home.html",
                                       error=True,
                                       message="Missing input: please fill all fields.",
                                       css_class=None)
            
            values.append(float(v))

        arr = np.array(values).reshape(1, -1)
        arr_scaled = scaler.transform(arr)
        fwi = float(model.predict(arr_scaled)[0])
        fwi_rounded = round(fwi, 2)

        level, css_class, gif_file, sound_file, voice_text = classify_danger(fwi_rounded)

        return render_template("home.html",
                               error=False,
                               fwi=fwi_rounded,
                               level=level,
                               css_class=css_class,
                               gif_file=gif_file,
                               sound_file=sound_file,
                               voice_text=voice_text)

    except ValueError:
        return render_template("home.html", error=True, message="Enter valid numeric values.", css_class=None)
    except Exception as e:
       
        return render_template("home.html", error=True, message=str(e), css_class=None)

if __name__ == "__main__":
   
    app.run(debug=True)
