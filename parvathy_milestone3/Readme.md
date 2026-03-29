This project is a small Flask web app that predicts the Fire Weather Index (FWI) and shows the fire risk level.
A Ridge Regression model is trained on the Bejaia Region dataset using weather features (Temperature, RH, Ws, Rain, DMC, DC, ISI, BUI, FFMC) to predict FWI, then saved as ridge.pkl along with a scaler.pkl for preprocessing.​
app.py loads the saved model and scaler, shows an input form (index.html), and, after the user submits values, returns a prediction page (home.html) with the FWI value and a risk label: Low, Moderate, or High.​
To run: keep app.py, ridge.pkl, scaler.pkl, and the templates folder together, install Flask and scikit‑learn, run python app.py, and open http://127.0.0.1:5000 in a browser to use the app.


<img width="1916" height="1008" alt="image" src="https://github.com/user-attachments/assets/9dd780da-6f90-4bb9-87fd-de021fe0adc6" />

<img width="1853" height="969" alt="image" src="https://github.com/user-attachments/assets/f87a2be6-55fe-4bad-b21a-c34249521c73" />
