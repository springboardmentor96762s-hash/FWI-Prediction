import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle

from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)
from sklearn.preprocessing import StandardScaler

os.makedirs("output", exist_ok=True)


df = pd.read_csv("clean_data.csv")
target_col = "FWI"

# Columns to drop (non-features)
drop_cols = ["day", "month", "year", "Classes", "Region"]

# Keep only numeric columns for correlation and modeling
num_df = df.drop(columns=drop_cols, errors="ignore").select_dtypes(include="number")

corr = num_df.corr()[target_col].drop(target_col)
# pick top-k correlated features (you can change k)
k = min(8, len(corr))
top_features = corr.abs().sort_values(ascending=False).head(k).index.tolist()
print("Top correlated features with FWI:", top_features)

X = num_df[top_features]
y = num_df[target_col]

print("Final feature columns used:", list(X.columns))

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test = scaler.transform(X_test_raw)

with open("output/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
print("Scaler saved to output/scaler.pkl")

alphas = np.logspace(-3, 3, 20)

mse_train_list, mse_test_list = [], []
rmse_train_list, rmse_test_list = [], []
mae_train_list, mae_test_list = [], []
r2_train_list, r2_test_list = [], []
fit_diagnostics = []

for alpha in alphas:
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)

    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Train metrics
    mse_tr = mean_squared_error(y_train, y_train_pred)
    rmse_tr = np.sqrt(mse_tr)
    mae_tr = mean_absolute_error(y_train, y_train_pred)
    r2_tr = r2_score(y_train, y_train_pred)

    # Test metrics
    mse_te = mean_squared_error(y_test, y_test_pred)
    rmse_te = np.sqrt(mse_te)
    mae_te = mean_absolute_error(y_test, y_test_pred)
    r2_te = r2_score(y_test, y_test_pred)

    mse_train_list.append(mse_tr)
    mse_test_list.append(mse_te)
    rmse_train_list.append(rmse_tr)
    rmse_test_list.append(rmse_te)
    mae_train_list.append(mae_tr)
    mae_test_list.append(mae_te)
    r2_train_list.append(r2_tr)
    r2_test_list.append(r2_te)

    ratio = mse_tr / mse_te
    if ratio < 0.7:
        status = "OVERFITTING"
    elif ratio > 1.3:
        status = "UNDERFITTING"
    else:
        status = "WELL_FITTED"
    fit_diagnostics.append(status)

best_idx = np.argmin(mse_test_list)      # lowest test MSE
best_alpha = alphas[best_idx]

print("\n===== BEST RIDGE MODEL (BY TEST MSE) =====")
print(f"Best alpha: {best_alpha:.4f}")
print(f"Train MSE:  {mse_train_list[best_idx]:.4f}")
print(f"Test  MSE:  {mse_test_list[best_idx]:.4f}")
print(f"Train RMSE: {rmse_train_list[best_idx]:.4f}")
print(f"Test  RMSE: {rmse_test_list[best_idx]:.4f}")
print(f"Train MAE:  {mae_train_list[best_idx]:.4f}")
print(f"Test  MAE:  {mae_test_list[best_idx]:.4f}")
print(f"Train R2:   {r2_train_list[best_idx]:.4f}")
print(f"Test  R2:   {r2_test_list[best_idx]:.4f}")
print(f"Fit status: {fit_diagnostics[best_idx]}")

final_model = Ridge(alpha=best_alpha)
final_model.fit(X_train, y_train)

with open("output/ridge.pkl", "wb") as f:
    pickle.dump(final_model, f)
print("Final Ridge model saved to output/ridge.pkl")


metrics_df = pd.DataFrame({
    "alpha": alphas,
    "mse_train": mse_train_list,
    "mse_test": mse_test_list,
    "rmse_train": rmse_train_list,
    "rmse_test": rmse_test_list,
    "mae_train": mae_train_list,
    "mae_test": mae_test_list,
    "r2_train": r2_train_list,
    "r2_test": r2_test_list,
    "fit_status": fit_diagnostics
})

metrics_df.to_csv("output/ridge_alpha_metrics.csv", index=False)
print("All alpha metrics saved to output/ridge_alpha_metrics.csv")


plt.figure(figsize=(8, 6))
plt.plot(alphas, rmse_train_list, label="Train RMSE", marker="o")
plt.plot(alphas, rmse_test_list, label="Test RMSE", marker="o")
plt.xscale("log")
plt.xlabel("Alpha (log scale)")
plt.ylabel("RMSE")
plt.title("Ridge Regression: RMSE vs Alpha")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("output/ridge_rmse_vs_alpha.png")
plt.close()

plt.figure(figsize=(8, 6))
plt.plot(alphas, mse_train_list, label="Train MSE", marker="o")
plt.plot(alphas, mse_test_list, label="Test MSE", marker="o")
plt.xscale("log")
plt.xlabel("Alpha (log scale)")
plt.ylabel("MSE")
plt.title("Ridge Regression: MSE vs Alpha")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("output/ridge_mse_vs_alpha.png")
plt.close()

plt.figure(figsize=(8, 6))
plt.plot(alphas, mae_train_list, label="Train MAE", marker="o")
plt.plot(alphas, mae_test_list, label="Test MAE", marker="o")
plt.xscale("log")
plt.xlabel("Alpha (log scale)")
plt.ylabel("MAE")
plt.title("Ridge Regression: MAE vs Alpha")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("output/ridge_mae_vs_alpha.png")
plt.close()

print("Metric vs alpha plots saved in output/")

y_test_pred = final_model.predict(X_test)

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_test_pred, alpha=0.6, edgecolor="black")
low = min(min(y_test), min(y_test_pred))
high = max(max(y_test), max(y_test_pred))
plt.plot([low, high], [low, high], "r--")
plt.xlabel("Actual FWI")
plt.ylabel("Predicted FWI")
plt.title(f"Actual vs Predicted FWI (Ridge, alpha={best_alpha:.4f})")
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig("output/ridge_actual_vs_predicted.png")
plt.close()
print("Actual vs predicted plot saved to output/ridge_actual_vs_predicted.png")

