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
X_scaled = scaler.fit_transform(X)  # Fit scaler

# Save scaler
current_dir = os.path.dirname(os.path.abspath(__file__))
scaler_path = os.path.join(current_dir, "linear_scaler.pkl")
with open(scaler_path, "wb") as f:
    pickle.dump(scaler, f)
print("Scaler saved at:", scaler_path)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)


print("\n== Linear Regression ==")

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Predictions
y_pred_lr_train = lr_model.predict(X_train)
y_pred_lr_test = lr_model.predict(X_test)

# Metrics
lr_mse_train = mean_squared_error(y_train, y_pred_lr_train)
lr_mse_test = mean_squared_error(y_test, y_pred_lr_test)
lr_rmse_test = np.sqrt(lr_mse_test)
lr_mae_test = mean_absolute_error(y_test, y_pred_lr_test)
lr_r2_test = r2_score(y_test, y_pred_lr_test)

# Print metrics
print("Linear Regression Train MSE:", lr_mse_train)
print("Linear Regression Test MSE:", lr_mse_test)
print("Linear Regression Test RMSE:", lr_rmse_test)
print("Linear Regression Test MAE:", lr_mae_test)
print("Linear Regression Test R2:", lr_r2_test)

# Overfitting / Underfitting check:
diff_lr = lr_mse_test - lr_mse_train
print("\n========== Linear Regression: Model Diagnosis ==========")
if diff_lr > 10:
    print("Overfitting: Test MSE is much higher than Train MSE")
elif diff_lr < -10:
    print("Underfitting: Train MSE is much higher than Test MSE")
else:
    print("Good Fit: Train/Test MSE are similar")

#  PLOT
plt.scatter(y_test, y_pred_lr_test)
plt.xlabel("Actual FWI")
plt.ylabel("Predicted FWI")
plt.title("Linear Regression: Actual vs Predicted")
plt.grid(True)
plt.show()

# SAVE MODEL
current_dir = os.path.dirname(os.path.abspath(__file__))
save_path = os.path.join(current_dir, "linear.pkl")

with open(save_path, "wb") as f:
    pickle.dump(lr_model, f)

print("\nLinear Regression Model Saved At:", save_path)