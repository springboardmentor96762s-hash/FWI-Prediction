## Forest Fire FWI Prediction (Machine Learning + Flask Web App)

A simple, clean, and modern web application that predicts the Fire Weather Index (FWI) using a trained Machine Learning model (Ridge Regression) integrated with a Flask backend.


## Overview

This project takes several environmental inputs such as temperature, humidity, wind speed, etc., and predicts the Fire Weather Index, indicating the likelihood of forest fire risk.



## Machine Learning Model

- Model: Ridge Regression
- Dataset: Algerian Forest Fires Dataset (cleaned)
  

## Steps to Run the Project

```bash
# Navigate to project directory
cd forest_fwi_app

# Create virtual environment
python -m venv venv

# Activate environment (Windows)
venv\Scripts\activate
# OR (Mac / Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the Flask app
python app.py

# Open in browser
http://127.0.0.1:5000/

