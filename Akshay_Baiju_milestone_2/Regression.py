import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

data = pd.read_csv("Algeria_Dataset_clean.csv")

feature_cols = [
    "temperature", "rh", "ws", "rain", "dmc", "ffmc", "dc", "isi",
]

X = data[feature_cols]
y = data["fwi"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#Linear Regression
linear_model = LinearRegression(fit_intercept=True, copy_X=True, positive=False)
linear_model.fit(X_train, y_train)

lr_train_pred = linear_model.predict(X_train)
lr_test_pred = linear_model.predict(X_test)

lr_train_mse = mean_squared_error(y_train, lr_train_pred)
lr_train_rmse = np.sqrt(lr_train_mse)
lr_train_mae = mean_absolute_error(y_train, lr_train_pred)
lr_train_r2 = r2_score(y_train, lr_train_pred)

lr_test_mse = mean_squared_error(y_test, lr_test_pred)
lr_test_rmse = np.sqrt(lr_test_mse)
lr_test_mae = mean_absolute_error(y_test, lr_test_pred)
lr_test_r2 = r2_score(y_test, lr_test_pred)

print("Training Set Linear Regression Results:")
print(f"MSE  : {lr_train_mse:.8f}")
print(f"RMSE : {lr_train_rmse:.8f}")
print(f"MAE  : {lr_train_mae:.8f}")
print(f"R²   : {lr_train_r2:.8f}")

print("\nTesting Set Linear Regression Results:")
print(f"MSE  : {lr_test_mse:.8f}")
print(f"RMSE : {lr_test_rmse:.8f}")
print(f"MAE  : {lr_test_mae:.8f}")
print(f"R²   : {lr_test_r2:.8f}")

#Ridge Regression
alphas = [ 0.0001,0.001,0.1 ,0.5, 1, 2, 5, 10, 20, 50, 75, 100]
alpha_mae_scores = []
best_alpha = None
best_mae = float("inf")

for alpha in alphas:
    ridge_model = Ridge(alpha=alpha)
    ridge_model.fit(X_train, y_train)

    ridge_train_pred = ridge_model.predict(X_train)
    ridge_train_mae = mean_absolute_error(y_train, ridge_train_pred)
    alpha_mae_scores.append(ridge_train_mae)

    if ridge_train_mae < best_mae:
        best_mae = ridge_train_mae
        best_alpha = alpha

print(f"\nBest Ridge alpha based on MAE: {best_alpha} (MAE={best_mae:.8f})")

best_ridge = Ridge(alpha=best_alpha)
best_ridge.fit(X_train, y_train)

ridge_train_pred = best_ridge.predict(X_train)
ridge_test_pred = best_ridge.predict(X_test)

ridge_train_mse = mean_squared_error(y_train, ridge_train_pred)
ridge_train_rmse = np.sqrt(ridge_train_mse)
ridge_train_mae = mean_absolute_error(y_train, ridge_train_pred)
ridge_train_r2 = r2_score(y_train, ridge_train_pred)

ridge_test_mse = mean_squared_error(y_test, ridge_test_pred)
ridge_test_rmse = np.sqrt(ridge_test_mse)
ridge_test_mae = mean_absolute_error(y_test, ridge_test_pred)
ridge_test_r2 = r2_score(y_test, ridge_test_pred)

print("\nBest Ridge Regression Results:")
print("Training:")
print(f"  MSE  : {ridge_train_mse:.8f}")
print(f"  RMSE : {ridge_train_rmse:.8f}")
print(f"  MAE  : {ridge_train_mae:.8f}")
print(f"  R²   : {ridge_train_r2:.8f}")

print("Testing:")
print(f"  MSE  : {ridge_test_mse:.8f}")
print(f"  RMSE : {ridge_test_rmse:.8f}")
print(f"  MAE  : {ridge_test_mae:.8f}")
print(f"  R²   : {ridge_test_r2:.8f}")

comparison_min = min(y_test.min(), lr_test_pred.min(), ridge_test_pred.min())
comparison_max = max(y_test.max(), lr_test_pred.max(), ridge_test_pred.max())
reference_line = np.linspace(comparison_min, comparison_max, 100)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(y_test, lr_test_pred, alpha=0.6, edgecolor="k")
plt.plot(reference_line, reference_line, color="red", linestyle="--", label="Ideal")
plt.xlabel("Actual FWI")
plt.ylabel("Predicted FWI")
plt.title("Linear Regression: Predicted FWI vs Actual FWI")
plt.legend()
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
plt.scatter(y_test, ridge_test_pred, alpha=0.6, edgecolor="k")
plt.plot(reference_line, reference_line, color="red", linestyle="--", label="Ideal")
plt.xlabel("Actual FWI")
plt.ylabel("Predicted FWI")
plt.title(f"Ridge (alpha={best_alpha}) Predicted FWI vs Actual FWI")
plt.legend()
plt.grid(alpha=0.3)

plt.figure(figsize=(8, 4))
plt.plot(alphas, alpha_mae_scores, marker="o")
plt.xscale("log")
plt.xlabel("Alpha")
plt.ylabel("MAE")
plt.title("Ridge Regression: MAE vs Alpha")
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()
