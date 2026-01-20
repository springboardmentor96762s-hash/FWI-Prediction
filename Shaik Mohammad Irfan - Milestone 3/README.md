
# Tempest FWI Predictor – Milestone 3: Model Deployment (Module 6)

**A Machine Learning Web Application to Predict Fire Weather Index (FWI)**

## Overview

This repository contains the complete deployment of the trained Ridge Regression model as a user-friendly web application using **Flask**. The application allows real-time prediction of the Fire Weather Index (FWI) based on 10 meteorological and fuel moisture inputs.

The deployment transforms the machine learning model into an interactive tool suitable for forest fire management teams and stakeholders.

## Project Structure


FWI Predictor/
├── app.py                  # Main Flask application
├── ridge.pkl               # Trained Ridge Regression model
├── scaler.pkl              # Fitted StandardScaler for preprocessing
├── templates/
│   ├── index.html          # Input form with all 10 features
│   └── home.html           # Result page showing predicted FWI and danger level
└── static/                 # (Optional for future CSS/JS files)


## Features

- **Input Form** (`index.html`):  
  Clean, responsive form with all required features:  
  - Region (dropdown: Bejaia = 0, Sidi-Bel Abbes = 1)  
  - Temperature, RH, Ws, Rain, FFMC, DMC, DC, ISI, BUI  
  - Realistic default values pre-filled

- **Prediction Logic** (`app.py`):  
  - Loads `ridge.pkl` and `scaler.pkl` at startup  
  - Collects user input → converts to DataFrame → applies scaling → predicts FWI  
  - Interprets result into fire danger category

- **Result Page** (`home.html`):  
  - Displays predicted FWI prominently  
  - Shows fire danger level:  
    - FWI < 1 → Very Low  
    - 1 ≤ FWI < 5 → Low–Moderate  
    - 5 ≤ FWI < 12 → High  
    - 12 ≤ FWI < 25 → Very High  
    - FWI ≥ 25 → Extreme  
  - Link to make new prediction

## How to Run Locally

1. Install dependencies:
   ```bash
   pip install flask pandas scikit-learn
   ```

2. Place `ridge.pkl` and `scaler.pkl` in the root folder.

3. Run the app:
   ```bash
   python app.py
   ```

4. Open browser: `http://127.0.0.1:5000`

## How to Run in Google Colab (with Public URL)

1. Upload/Mount your Google Drive folder containing the files.
2. Install ngrok:
   ```python
   !pip install pyngrok flask -q
   ```
3. Start tunnel and run:
   ```python
   from pyngrok import ngrok
   public_url = ngrok.connect(5000)
   print("Live URL:", public_url)
   !python app.py
   ```
4. Click the ngrok URL → live demo accessible to anyone.

## Sample Test Cases (All Danger Levels)

| Danger Level       | Region       | Temp | RH | Ws | Rain | FFMC | DMC | DC  | ISI | BUI | Expected FWI |
|--------------------|--------------|------|----|----|------|------|-----|-----|-----|-----|--------------|
| Very Low           | Bejaia (0)   | 22   | 85 | 10 | 2.5  | 65   | 5   | 20  | 2.0 | 8   | ~0.6         |
| Low–Moderate       | Sidi-Bel (1) | 28   | 70 | 14 | 0.0  | 80   | 10  | 60  | 4.5 | 15  | ~3.8         |
| High               | Bejaia (0)   | 33   | 45 | 18 | 0.0  | 90   | 30  | 200 | 10  | 40  | ~10.2        |
| Very High          | Sidi-Bel (1) | 36   | 30 | 20 | 0.0  | 92   | 50  | 400 | 14  | 70  | ~21.5        |
| Extreme            | Bejaia (0)   | 39   | 20 | 25 | 0.0  | 94   | 80  | 600 | 18  | 100 | ~34+         |

## Key Achievements (Module 6)

- Full Flask application with proper routing and templates
- Correct loading and use of saved model and scaler
- Consistent preprocessing during inference
- Professional, intuitive user interface
- Public deployment via ngrok for live demonstration
- End-to-end ML pipeline completed: Data → Model → Web App


**Author:** Shaik Mohammad Irfan  
**Date:** December 14, 2025

---
**Tempest FWI Predictor is now fully deployed and ready for real-world use.**
