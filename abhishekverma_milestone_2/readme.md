# Forest Fire Weather Index (FWI) Prediction – Ridge Regression

## 1) Project Overview
This project predicts the **Forest Fire Weather Index (FWI)** using meteorological and environmental features. A **Ridge Regression model** is implemented to handle multicollinearity and tuned for optimal performance using the hyperparameter alpha.

---

## 2) Project Features

### 2.1) Module 3: Feature Engineering & Scaling
- Selected input features most correlated with the target `FWI`.  
- Applied `StandardScaler` to normalize numerical features for consistent scale.  
- Split dataset into input features (`X`) and target (`y`).  
- Created training and test sets using an 80%-20% split.  
- Saved the fitted scaler as `scaler.pkl` for deployment consistency.  

### 2.2) Module 4: Model Training – Ridge Regression
- Trained Ridge Regression to reduce variance and handle multicollinearity.  
- Hyperparameter tuning performed on alpha values: `[0.001, 0.01, 0.1, 1, 10, 100, 200]`.  
- Computed performance metrics for each alpha on train and test data.  
- Selected **best alpha** based on minimum Test MSE.  
- Saved the best Ridge model as `ridge.pkl`.  

### 2.3) Module 5: Evaluation & Optimization
**Metrics computed:**
- Mean Squared Error (MSE)  
- Root Mean Squared Error (RMSE)  
- Mean Absolute Error (MAE)  
- R² Score  

**Plots generated:**
- Actual vs Predicted FWI  
- Alpha vs MSE
- Alpha vs RMSE
- Alpha vs MAE
- Alpha vs R²

**Overfitting/underfitting diagnosis:**
- Underfitting → High train & test error  
- Overfitting → Low train error & high test error  
- Good Fit → Train & test errors similar  

---

## 3) Hyperparameter Tuning – Alpha

Metrics tracked for multiple values of alpha:

| Alpha   | Train MSE | Test MSE | Train RMSE | Test RMSE | Train MAE | Test MAE | Train R²  | Test R²  |
|--------|-----------|----------|------------|-----------|-----------|----------|-----------|----------|
| 0.001  | 0.004444  | 0.006280 | 0.066667   | 0.079244  | 0.051540  | 0.067186 | 0.999555  | 0.999341 |
| 0.01   | 0.004506  | 0.006237 | 0.067128   | 0.078974  | 0.053221  | 0.067473 | 0.999549  | 0.999345 |
| 0.1    | 0.006976  | 0.007536 | 0.083524   | 0.086808  | 0.066502  | 0.069279 | 0.999302  | 0.999209 |
| 1.0    | 0.020310  | 0.020051 | 0.142513   | 0.141601  | 0.099442  | 0.086511 | 0.997969  | 0.997896 |
| 10.0   | 0.044952  | 0.043804 | 0.212020   | 0.209295  | 0.173814  | 0.175421 | 0.995504  | 0.995403 |
| 100.0  | 0.168569  | 0.133519 | 0.410571   | 0.365402  | 0.320241  | 0.277690 | 0.983141  | 0.985987 |
| 200.0  | 0.316688  | 0.259187 | 0.562750   | 0.509104  | 0.456688  | 0.404280 | 0.968326  | 0.972797 |

**Best alpha:** `0.01` – chosen for the **lowest Test MSE** and balanced train-test performance.  

---

## 4) Output Files

| File Name                     | Description                                                       |
| ----------------------------- | ----------------------------------------------------------------- |
| `Ridge_Model.py`              | Main Python script for training, evaluation, and saving the model |
| `ridge.pkl`                   | Saved Ridge Regression model (best alpha)                         |
| `scaler.pkl`                  | Saved StandardScaler used for feature normalization               |
| `RR(Actual vs Predicted).png` | Scatter plot showing Predicted vs Actual                          |
| `RR(Alpha vs MSE).png`        | Line plot showing Train & Test MSE vs Alpha                       |
| `RR(Alpha vs RMSE).png`       | Line plot showing Train & Test RMSE vs Alpha                      |
| `RR(Alpha vs MAE).png`        | Line plot showing Train & Test MAE vs Alpha                       |
| `RR(Alpha vs R²).png`         | Line plot showing Train & Test R² score vs Alpha                  |

---

## 5) Evaluation Metrics
| Metric | Description |
|--------|-------------|
| MAE    | Average absolute error |
| MSE    | Penalizes large errors |
| RMSE   | Square root of MSE |
| R² Score | Measures proportion of variance explained |

---

## 6) Underfitting & Overfitting Check
- **Underfitting:** High train & test errors  
- **Overfitting:** Low train error & high test error  
- **Good Fit:** Train & test errors are similar  

---

## 7) Conclusion
The Ridge Regression model with **alpha = 0.01** achieved excellent generalization, with minimal error on both training and testing sets. All evaluation metrics indicate a strong fit to the data, and the model is stable, robust, and ready for deployment.

---

