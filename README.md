
# Tempest Fire Weather Index (FWI) Predictor

## Overview
This project, completed as part of the Infosys Springboard Virtual Internship by Shaik Mohammad Irfan in December 2025, develops a predictive model for the Fire Weather Index (FWI). The FWI is a critical metric used to assess forest fire danger based on meteorological data and fire behavior indices. The dataset focuses on observations from Algeria's Bejaia and Sidi-Bel Abbes regions during the fire-prone months of June to September 2012.

Key components include:
- Data cleaning and preprocessing.
- Exploratory Data Analysis (EDA) with statistics, visualizations, and correlations.
- A Ridge Regression model to predict FWI, tuned to handle multicollinearity.
- Deployment via a Flask web application for interactive predictions.
- A demo video demonstrating the app's functionality.

The model uses Ridge Regression with L2 regularization, achieving a best Mean Squared Error (MSE) of approximately 0.401 and Mean Absolute Error (MAE) of 0.483 on the test set.

## Table of Contents
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Model: Ridge Regression](#model-ridge-regression)
- [Deployment: Flask Web App](#deployment-flask-web-app)
- [Requirements](#requirements)
- [Installation and Usage](#installation-and-usage)
- [Demo Video](#demo-video)
- [Files in Repository](#files-in-repository)
- [Author](#author)

## Dataset
The dataset (`FWI_UPDATE.csv`) contains 244 entries with 15 columns from forest fire observations in Algeria (Bejaia encoded as 0, Sidi-Bel Abbes as 1) from June to September 2012. It includes meteorological variables and components of the Canadian Fire Weather Index system.

Features:
- **Region**: Binary (0: Bejaia, 1: Sidi-Bel Abbes).
- **day, month, year**: Date components (all year=2012, months 6-9).
- **Temperature**: Air temperature in °C (range: 22-42).
- **RH**: Relative humidity in % (range: 21-90).
- **Ws**: Wind speed in km/h (range: 6-29).
- **Rain**: Rainfall in mm (range: 0-16.8).
- **FFMC**: Fine Fuel Moisture Code (range: 28.6-96.0).
- **DMC**: Duff Moisture Code (range: 0.7-65.9).
- **DC**: Drought Code (range: 6.9-177.3).
- **ISI**: Initial Spread Index (range: 0-19.0).
- **BUI**: Buildup Index (range: 1.1-68.0).
- **FWI**: Fire Weather Index (target variable, range: 0-31.1).
- **Classes**: Binary (1: fire, 0: no fire).

The cleaned version is saved as `Cleaned_FWI_dataset.csv`.

## Preprocessing
The raw dataset was prepared for analysis and modeling through the following steps:
- Stripped whitespace from column names.
- Converted features to numeric types, coercing errors to NaN.
- Dropped rows with missing 'day' values.
- Interpolated remaining missing numeric values using linear method with both forward and backward limits.
- Encoded 'Region' (Bejaia: 0, Sidi-Bel: 1) and 'Classes' (stripped and mapped 'fire' to 1, others to 0).
- Cast appropriate columns to integers (day, month, year, Temperature, RH, Ws) and floats for others.
- Standardized features using scikit-learn's `StandardScaler` (mean=0, std=1) to aid Ridge Regression in handling scale differences and multicollinearity.

The processed dataset was saved as `Cleaned_FWI_dataset.csv` for downstream tasks.

## Exploratory Data Analysis (EDA)
EDA provided insights into the dataset's structure and relationships:
- **Descriptive Statistics**: From pandas `.describe()`:
  - Means: Temperature ≈32.17°C, RH 62%, Ws 15.5 km/h, Rain 0.76 mm. Fire indices: FFMC 77.89, DMC 14.68, DC 49.43, ISI 4.74, BUI 16.66, FWI 7.05.
  - Variability: High std in DC (47.67) and BUI (14.20); low in Temperature (3.63).
  - Extremes: Max Temperature 42°C, Rain 16.8 mm, FWI 31.1.
  - Class balance: 56% fire (137 instances) vs. 44% no fire.
  - Skewness: Positive skew in Rain (>4), DC (1.5), DMC/BUI/ISI/FWI (1.2-1.8); Temperature near-symmetric (-0.3); RH slightly left-skewed.
- **Histograms**: Temperature bimodal (~30-35°C), RH right-skewed (50-80%), Ws moderate (10-20 km/h), Rain heavily skewed to 0, FFMC left-skewed (high values common), others right-skewed. Saved as `histograms-features.png`.
- **Correlation Analysis**: Heatmap shows strong positive correlations with FWI: ISI (0.919), BUI (0.857), DMC (0.875), DC (0.738), FFMC (0.691), Temperature (0.566). Negative: RH (-0.580), Rain (-0.325). Multicollinearity evident (e.g., DMC-BUI: 0.98). Saved as `correlation-heatmap.png`.

These findings justified standardization and informed model selection.

## Model: Ridge Regression
Ridge Regression was selected to predict FWI while addressing multicollinearity via L2 regularization. The objective function minimized is:

\[ J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_{\theta}(x^{(i)}) - y^{(i)})^2 + \alpha \sum_{j=1}^{n} \theta_j^2 \]

- **Features**: Region, Temperature, RH, Ws, Rain, FFMC, DMC, DC, ISI, BUI.
- **Target**: FWI.
- **Data Split**: 80% train, 20% test; features standardized.
- **Hyperparameter Tuning**: Alpha values from logspace(10^{-3} to 10^{3}, 100 points). Best α ≈0.132 via 5-fold CV.
- **Performance**: MSE ≈0.401, MAE ≈0.483. MSE decreases initially with α, reaches minimum, then increases; MAE plateaus. Plots: Validation curve and predicted vs. actual (saved from notebook).

Model and scaler saved as `ridge.pkl` and `scaler.pkl`.

## Deployment: Flask Web App
The trained model was deployed as a Flask web application for FWI predictions:
- Loads model and scaler from pickles.
- Routes: `/` renders input form (`index.html`); `/predict` (POST) processes inputs, scales them, predicts FWI (rounded to 3 decimals), and renders results (`home.html`).
- Inputs: Align with model features.
- Risk Categorization: Very Low (<2), Low-Moderate (<5), High (<12), Very High (<25), Extreme (≥25).
- UI: Simple HTML forms with card-style output displaying FWI and risk level.
- Runs locally in debug mode; can be publicized via ngrok.

## Requirements
- Python 3.x
- Libraries: pandas, numpy, scikit-learn, flask, matplotlib, seaborn, pyngrok (for public exposure)

Install with:
```
pip install pandas numpy scikit-learn flask matplotlib seaborn pyngrok
```

## Installation and Usage
1. Clone the repository:
   ```
   git clone <repository-url>
   cd <repository-directory>
   ```
2. Run the Jupyter Notebook (`SHAIK MOHAMMAD IRFAN - FWI Predictor Infosys Springboard.ipynb`) to preprocess data, perform EDA, train the model, and generate pickles/plots.
3. Start the Flask app:
   ```
   python app.py
   ```
   - Access locally at `http://127.0.0.1:5000`.
   - For public access, configure ngrok authtoken and run the app (see notebook for example logs).
4. Enter feature values in the web form and submit to get FWI prediction and risk level.

## Demo Video
A screen-recorded demo of the Flask app in action is available [here](<link-to-demo-video>). It demonstrates input submission, prediction generation, and risk categorization for sample cases.

## Files in Repository
- `SHAIK MOHAMMAD IRFAN FWI-Predictor Report.pdf`: Detailed project report.
- `SHAIK MOHAMMAD IRFAN - FWI Predictor Infosys Springboard.ipynb`: Jupyter Notebook with full code for preprocessing, EDA, modeling, and deployment setup.
- `Cleaned_FWI_dataset.csv`: Processed dataset.
- `app.py`: Flask application script.
- `templates/index.html` and `templates/home.html`: HTML templates for the web app (assume standard folder structure).
- `ridge.pkl` and `scaler.pkl`: Serialized model and scaler (generated via notebook).
- `histograms-features.png` and `correlation-heatmap.png`: EDA visualizations (generated via notebook).
- `README.md`: This documentation file.

## Author
- **Shaik Mohammad Irfan**
- Infosys Springboard Virtual Internship
- Completed: December 2025
