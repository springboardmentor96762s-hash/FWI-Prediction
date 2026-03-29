# 🌡️ Milestone 2 – Ridge Regression for Fire Weather Index (FWI) Prediction

This milestone focuses on developing a **Ridge Regression model** to predict the **Fire Weather Index (FWI)** using the cleaned dataset. The objective is to evaluate different regularization strengths, measure accuracy, diagnose overfitting/underfitting, and visualize model performance.

---

## 🔍 1. Overview

In this milestone, we:

- Loaded and preprocessed the dataset  
- Selected **FWI** as the prediction target  
- Removed irrelevant/categorical columns  
- Tested multiple **alpha** values for Ridge Regression  
- Chose the best model based on **test MSE**  
- Calculated evaluation metrics  
- Diagnosed overfitting/underfitting  
- Plotted MSE, RMSE, and MAE vs Alpha  

---

## 📊 2. Features Used

The following numerical features were used:

- Temperature  
- RH (Relative Humidity)  
- Wind Speed  
- Rain  
- FFMC  
- DMC  
- DC  
- ISI  
- BUI  

Removed (categorical / irrelevant):

- `day`, `month`, `year`  
- `Classes`  
- `Region`  

---

## 🧮 3. Model Used — Ridge Regression

Ridge Regression helps:

- Reduce overfitting  
- Smooth model coefficients  
- Improve generalization  

We tested 20 alpha values using:


The best alpha was selected using lowest **test MSE**.

---

## 📈 4. Performance Metrics

The final Ridge model displays:

- **Slope (coefficients)**  
- **Intercept**  
- **MSE (Mean Squared Error)**  
- **RMSE (Root Mean Squared Error)**  
- **MAE (Mean Absolute Error)**  
- **R² Score**  
- **List of input features**

These metrics represent the quality of the prediction model.

---

## 🧪 5. Overfitting / Underfitting Check

We compare **train MSE** and **test MSE**:

- If **train MSE << test MSE** → ❌ Overfitting  
- If **train MSE >> test MSE** → ❌ Underfitting  
- If MSE values are close → ✅ Well-fitted model  

The code prints the final diagnosis automatically.

---

## 📉 6. Graphs Generated

The following plots are generated:

1. **MSE vs Alpha**  
2. **RMSE vs Alpha**  
3. **MAE vs Alpha**

All plots use **log scale** for alpha for proper visualization.  
These graphs help understand how regularization affects model performance.

---

## 📝 7. How to Run the Code

Install necessary libraries:

```bash
pip install numpy pandas scikit-learn matplotlib




---

If you want, I can also create:

✅ README for the **main project**  
✅ README for **Milestone 1**  
✅ A **GitHub profile README**  

Just tell me!
