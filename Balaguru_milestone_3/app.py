from flask import Flask, render_template, request, redirect, url_for, session
import numpy as np
import joblib

app = Flask(__name__)
app.config['SECRET_KEY'] = 'super_secret_key_123'   


USERS = {
    "admin": "securepassword123",
    "user": "testpass"
}


model = joblib.load("ridge.pkl")
scaler = joblib.load("scaler.pkl")


@app.route('/login', methods=['GET', 'POST'])
def login():

    if request.method == 'POST':

        username = request.form.get('username')
        password = request.form.get('password')

        if username in USERS and USERS[username] == password:
            session['username'] = username
            return redirect(url_for('fwi_predict'))
        else:
            return render_template("login.html", error="Invalid Credentials")

    return render_template("login.html", error=None)



@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))




@app.route('/predict', methods=['GET', 'POST'])
def fwi_predict():

    if 'username' not in session:
        return redirect(url_for('login'))

    output = None

    if request.method == 'POST':
        try:
            inputs = [
                float(request.form["temperature"]),
                float(request.form["rh"]),
                float(request.form["ws"]),
                float(request.form["rain"]),
                float(request.form["ffmc"]),
                float(request.form["dmc"]),
                float(request.form["dc"]),
                float(request.form["isi"]),
                float(request.form["bui"])
            ]

            final_data = np.array(inputs).reshape(1, -1)

            
            scaled_data = scaler.transform(final_data)

            
            prediction = model.predict(scaled_data)[0]
            prediction = round(prediction, 2)

            output = f"FWI Index: {prediction}"

        except Exception as e:
            output = f"Error: {str(e)}"

    return render_template("fwi_predict.html", output=output)



@app.route('/')
def home():
    return redirect(url_for('login'))



if __name__ == "__main__":
    app.run(debug=True)
