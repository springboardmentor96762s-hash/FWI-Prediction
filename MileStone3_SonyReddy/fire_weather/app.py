from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)


model = pickle.load(open("ridge.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    
    month = int(request.form['month'])
    day = int(request.form['day'])
    ffmc = float(request.form['ffmc'])
    dmc = float(request.form['dmc'])
    dc = float(request.form['dc'])
    isi = float(request.form['isi'])
    temp = float(request.form['temp'])
    wind = float(request.form['wind'])
    bui = float(request.form['bui'])

    input_data = np.array([[month, day, ffmc, dmc, dc, isi, temp, wind, bui]])

    fwi_value = model.predict(input_data)[0]

    if fwi_value < 5:
        level = "Low Risk"
    elif fwi_value < 20:
        level = "Moderate Risk"
    else:
        level = "High Risk"

    result_text = f"FWI Value: {fwi_value:.2f} → {level}"

    return render_template("index.html", prediction_text=result_text)

if __name__ == "__main__":
    app.run(debug=True)
