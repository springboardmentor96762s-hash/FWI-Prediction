import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import pickle
import os


df = pd.read_csv(r"C:\Users\HP\Desktop\InfosysInternFWI\FWI-Prediction\abhishekverma_milestone_2\cleaned_forest_fire_dataset.csv")

# Target Variable
y = df["FWI"]

# Features (all except target)
X = df.drop(columns=["FWI"])

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X) 

# Save scaler
current_dir = os.path.dirname(os.path.abspath(__file__))
scaler_path = os.path.join(current_dir, "scaler.pkl")
with open(scaler_path, "wb") as f:
    pickle.dump(scaler, f)
print("Scaler saved at:", scaler_path)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

print("\n== Ridge Regression ==")

alphas = [0.001, 0.01, 0.1, 1, 10, 100, 200]

mse_train_list = []
mse_test_list = []
rmse_train_list = []
rmse_test_list = []
mae_train_list = []
mae_test_list = []
r2_train_list = []
r2_test_list = []

ridge_models = {}

for alpha in alphas:
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)

    ridge_models[alpha] = model

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    mse_train = mean_squared_error(y_train, y_pred_train)
    mse_test = mean_squared_error(y_test, y_pred_test)
    rmse_train = np.sqrt(mse_train)
    rmse_test = np.sqrt(mse_test)
    mae_train = mean_absolute_error(y_train, y_pred_train)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)

    mse_train_list.append(mse_train)
    mse_test_list.append(mse_test)
    rmse_train_list.append(rmse_train)
    rmse_test_list.append(rmse_test)
    mae_train_list.append(mae_train)
    mae_test_list.append(mae_test)
    r2_train_list.append(r2_train)
    r2_test_list.append(r2_test)

    print(f"\nAlpha = {alpha}")
    print("Train MSE:", mse_train, " | Test MSE:", mse_test)
    print("Train RMSE:", rmse_train, " | Test RMSE:", rmse_test)
    print("Train MAE:", mae_train, " | Test MAE:", mae_test)
    print("Train R2:", r2_train, " | Test R2:", r2_test)

# Find Best Alpha
best_alpha = alphas[np.argmin(mse_test_list)]
print("\nBest Alpha =", best_alpha)
best_model = ridge_models[best_alpha]

#  Ridge Plots:
plt.plot(alphas, mse_train_list, marker='o', label="Train MSE")
plt.plot(alphas, mse_test_list, marker='o', label="test MSE")
plt.xscale("log")
plt.xlabel("Alpha (log scale)")
plt.ylabel("MSE")
plt.title("Ridge Regression: Alpha vs MSE")
plt.grid(True)
plt.legend()
plt.show()

plt.plot(alphas, rmse_train_list, marker='o', label="Train RMSE")
plt.plot(alphas, rmse_test_list, marker='o', label="Test RMSE")
plt.xscale("log")
plt.xlabel("Alpha (log scale)")
plt.ylabel("RMSE")
plt.title("Ridge Regression: Alpha vs RMSE")
plt.grid(True)
plt.legend()
plt.show()

plt.plot(alphas, mae_train_list, marker='o', label="Train MAE")
plt.plot(alphas, mae_test_list, marker='o', label="Test MAE")
plt.xscale("log")
plt.xlabel("Alpha (log scale)")
plt.ylabel("MAE")
plt.title("Ridge Regression: Alpha vs MAE")
plt.grid(True)
plt.legend()
plt.show()

plt.plot(alphas, r2_train_list, marker='o', label="Train R²")
plt.plot(alphas, r2_test_list, marker='o', label="Test R²")
plt.xscale("log")
plt.xlabel("Alpha (log scale)")
plt.ylabel("R² Score")
plt.title("Ridge Regression: Alpha vs R²")
plt.grid(True)
plt.legend()
plt.show()

#  Predicted vs Actual
y_pred_best_test = best_model.predict(X_test)
plt.scatter(y_test, y_pred_best_test)
plt.xlabel("Actual FWI")
plt.ylabel("Predicted FWI")
plt.title(f"Ridge Regression (Alpha={best_alpha}): Actual vs Predicted")
plt.grid(True)
plt.show()

# Overfitting / Underfitting Check:
print("\n== Ridge Model Diagnosis ==")
for i, alpha in enumerate(alphas):
    diff = mse_test_list[i] - mse_train_list[i]
    if diff > 10:
        print(f"Alpha {alpha}: Overfitting (Test MSE >> Train MSE)")
    elif diff < -10:
        print(f"Alpha {alpha}: Underfitting (Train MSE >> Test MSE)")
    else:
        print(f"Alpha {alpha}: Good Fit")

#  Ridge Metrics Table:
ridge_results_df = pd.DataFrame({
    "Alpha": alphas,
    "Train MSE": mse_train_list,
    "Test MSE": mse_test_list,
    "Train RMSE": rmse_train_list,
    "Test RMSE": rmse_test_list,
    "Train MAE": mae_train_list,
    "Test MAE": mae_test_list,
    "Train R2": r2_train_list,
    "Test R2": r2_test_list
})

print("\n Ridge Regression Results Table")
print(ridge_results_df)
print("\n")

# Save Best Ridge Model:
current_dir = os.path.dirname(os.path.abspath(__file__))
save_path = os.path.join(current_dir, "ridge.pkl")

with open(save_path, "wb") as f:
    pickle.dump(best_model, f)

print("Best Ridge Model Saved at:", save_path)
