# Milestone 2 – Ridge Regression Model
**Author:** Jami Gayatri  

## Overview
This milestone builds a Ridge Regression model to predict **FWI (Fire Weather Index)** using the cleaned dataset from Milestone 1.

## Key Steps
- Loaded `clean_data.csv`
- Removed non-feature columns: day, month, year, Classes, Region
- Selected top correlated features with FWI
- Split data into train/test sets
- Applied StandardScaler
- Trained Ridge models across multiple alpha values
- Evaluated: MSE, RMSE, MAE, R²
- Selected the best model (lowest test MSE)

## Outputs (saved in `output/`)
- `scaler.pkl` — fitted StandardScaler  
- `ridge.pkl` — best Ridge model  
- `ridge_alpha_metrics.csv` — metrics for all alphas  
- Plots:  
  - RMSE vs Alpha  
  - MSE vs Alpha  
  - MAE vs Alpha  
  - Actual vs Predicted

## Conclusion
Ridge Regression was tuned across many alpha values, and the best-performing model was saved along with evaluation plots and metrics.
