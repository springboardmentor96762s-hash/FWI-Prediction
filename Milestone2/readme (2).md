Ridge Regression – Alpha Tuning & Model Evaluation

This project builds a Ridge Regression model to predict the Fire Weather Index (FWI) and evaluates how different values of alpha (λ) affect the model’s performance. The goal is to identify the optimal alpha based on training and testing errors.

Module 3: Feature Engineering and Scaling
✔ Key Steps

Selected important input features that show strong correlation with the FWI target variable.

Normalized all numerical features using StandardScaler for consistent scale.

Split the dataset into input features (X) and target (y).

Performed a train–test split to evaluate generalization.

Saved the trained scaler as a .pkl file for deployment use.

Module 4: Model Training Using Ridge Regression
✔ Why Ridge Regression?

Ridge helps handle multicollinearity, stabilizing model weights by adding an L2 penalty.
This penalty strength is controlled by alpha (λ).

✔ What Was Done

Trained Ridge models using various alpha values.

Observed how alpha affects the bias–variance tradeoff.

Monitored performance on both training and test sets.

Saved the final trained model as ridge.pkl.

Documented all experiment settings.

Module 5: Evaluation and Optimization
✔ Evaluation Metrics

The model was evaluated using:

Mean Absolute Error (MAE) – measures average absolute difference.

Mean Squared Error (MSE) – penalizes larger errors more strongly.

Root Mean Squared Error (RMSE) – square-root of MSE, interpretable in original scale.

R² Score – measures how much variance is explained by the model.

✔ Alpha Tuning

Multiple alpha values were tested and plotted to compare:

Train MSE vs Alpha

Test MSE vs Alpha

Train RMSE vs Alpha

Test RMSE vs Alpha

Train MAE vs Alpha

Test MAE vs Alpha

✔ Optimal Alpha

The optimal alpha is the one that:

Gives the lowest test error (MSE, RMSE, MAE).

Shows balanced train–test performance (no overfitting or underfitting).

Summary

Data was preprocessed using scaling and feature selection.

Ridge Regression was trained while tuning alpha to achieve the best performance.

The model was evaluated using multiple error metrics to find the most reliable alpha.

Best model and scaler were saved for later use in deployment.