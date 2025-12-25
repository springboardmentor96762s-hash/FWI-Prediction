from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

with open("ridge.pkl", "rb") as f:
    model = pickle.load(f)

try:
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
except FileNotFoundError:
    scaler = None


FEATURES = [
    "Region", "Temperature", "RH", "Ws",
    "Rain", "FFMC", "DMC", "ISI"
]


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    error = None

    try:
        region = int(request.form["Region"])           
        temperature = float(request.form["Temperature"])
        rh = float(request.form["RH"])
        ws = float(request.form["Ws"])
        rain = float(request.form["Rain"])
        ffmc = float(request.form["FFMC"])
        dmc = float(request.form["DMC"])
        isi = float(request.form["ISI"])

        input_vector = np.array([[region, temperature, rh, ws,
                                  rain, ffmc, dmc, isi]])

        if scaler is not None:
            input_vector = scaler.transform(input_vector)

        predicted_fwi = float(model.predict(input_vector)[0])

    except ValueError:
        error = "Please enter valid numeric values for all fields."
        return render_template("index.html", error=error, form=request.form)
    except Exception as e:
        error = f"Something went wrong while making prediction: {e}"
        return render_template("index.html", error=error, form=request.form)

    input_data = {
        "Region": "Bejaia" if region == 0 else "Sidi Bel-Abbes",
        "Temperature": temperature,
        "RH": rh,
        "Ws": ws,
        "Rain": rain,
        "FFMC": ffmc,
        "DMC": dmc,
        "ISI": isi,
    }

    return render_template(
        "home.html",
        prediction=round(predicted_fwi, 3),
        input_data=input_data
    )


if __name__ == "__main__":
    app.run(debug=True)
