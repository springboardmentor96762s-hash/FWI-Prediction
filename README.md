<<<<<<< HEAD
# Forest Fire Weather Index (FWI) Prediction using Ridge Regression

This project builds a Ridge Regression model to predict the Forest Fire Weather Index (FWI) using meteorological and fire-related features. It includes data loading, preprocessing, feature scaling, model training, hyperparameter tuning, evaluation, and visualizations.

## Project Structure

```
project/
│── forestfire.csv
│── script.py
│── scaler.pkl
│── ridge.pkl
└── README.md
```

---

## Features Used

The following input features are used for prediction:

* Temperature
* RH (Relative Humidity)
* Ws (Wind Speed)
* Rain
* FFMC
* DMC
* DC
* ISI
* BUI

Target variable: **FWI**

---

## Workflow Summary

### 1. Load Dataset

Reads data from `forestfire.csv`.

### 2. Feature Selection

Selects the most important meteorological and fire-related predictors.

### 3. Train–Test Split

Splits data into 80% training and 20% testing.

### 4. Feature Scaling

Uses StandardScaler for normalization.
Saves the scaler as:


scaler.pkl


### 5. Ridge Regression Model

Trains Ridge Regression with multiple alpha values:

0.01, 0.1, 1, 5, 10, 20, 50, 100


For each alpha, calculates:

* Train & Test MSE
* Train & Test RMSE
* Train & Test MAE

Selects the **best alpha** based on the lowest Test MSE.

### 6. Final Model

Trains the final Ridge model with the best alpha.
Saves the model as:

ridge.pkl


### 7. Model Evaluation

Calculates:

* MAE
* RMSE
* R² Score

### 8. Visualizations

The script automatically generates the following plots:

* MSE vs Alpha
* RMSE vs Alpha
* MAE vs Alpha
* Predicted vs Actual FWI

---

## How to Run

### Step 1: Install required libraries

```
pip install pandas numpy matplotlib scikit-learn joblib
```

### Step 2: Keep the dataset ready

Place `forestfire.csv` in the project directory.

### Step 3: Run the script

```
python script.py
```

---

## Saved Files

| File       | Description                          |
| ---------- | ------------------------------------ |
| scaler.pkl | Saved StandardScaler for deployment  |
| ridge.pkl  | Final trained Ridge Regression model |

---

## Future Improvements

* Add cross-validation
* Test more ML models (Random Forest, XGBoost)
* Build a web app for real-time prediction
* Add deeper documentation in a `/docs` folder

---

=======
# FWI-Prediction
Here is the repository created for evaluating the tasks for each of the milestones


This the repository where you should upload the files for the tasks

Hello everyone,

I have created the GitHub repository for your FWI predictor.
Please follow these steps to submit your assignments:

Fork the repository to your own GitHub account.
Create a folder named <your_name_milestone_1> inside the repo.
Upload:
Raw files
Python scripts (Data, preprocessing, obtaining histograms, correlation plots)
Any output files
A README explaining your work
Commit and push to your fork.
Create a Pull Request back to the main repository with the title: "Milestone 1 – "
I will review your submission through the Pull Request.
>>>>>>> e3eb0069d328eda9dab16da15b5c2f4d161377ec
