import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import joblib

# ---------------------------
# Load Data
# ---------------------------
df = pd.read_csv("cleaned_data.csv")

features = ['day','month','year','temperature','rh','ws','rain',
            'ffmc','dmc','dc','isi','bui','region']

X = df[features].values
y = df['fwi'].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------------------
# Try different alpha values
# ---------------------------
alphas = [0.001, 0.01, 0.1, 1, 10, 50, 100]

mse_train = []
mse_test = []
rmse_train = []
rmse_test = []
mae_train = []
mae_test = []

r2_train = []
r2_test = []

for a in alphas:
    model = Ridge(alpha=a)
    model.fit(X_train, y_train)

    # predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Metrics
    mse_train.append(mean_squared_error(y_train, y_train_pred))
    mse_test.append(mean_squared_error(y_test, y_test_pred))

    rmse_train.append(np.sqrt(mean_squared_error(y_train, y_train_pred)))
    rmse_test.append(np.sqrt(mean_squared_error(y_test, y_test_pred)))

    mae_train.append(mean_absolute_error(y_train, y_train_pred))
    mae_test.append(mean_absolute_error(y_test, y_test_pred))

    r2_train.append(r2_score(y_train, y_train_pred))
    r2_test.append(r2_score(y_test, y_test_pred))


# ---------------------------
# Best Alpha (lowest test MSE)
# ---------------------------
best_alpha = alphas[mse_test.index(min(mse_test))]
best_model = Ridge(alpha=best_alpha)
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

# ---------------------------
# Plot Predicted vs Actual
# ---------------------------
plt.scatter(y_test, y_pred)
plt.xlabel("Actual FWI")
plt.ylabel("Predicted FWI")
plt.title(f"Predicted vs Actual (Ridge, alpha={best_alpha})")
plt.show()

# ---------------------------
# Plot MSE vs Alpha
# ---------------------------
plt.plot(alphas, mse_train, label="Train MSE")
plt.xlabel("Alpha")
plt.ylabel("MSE")
plt.title("Train MSE vs Alpha (Ridge Regression)")
plt.xscale("log")
plt.legend()
plt.show()

# ---------------------------
# Plot RMSE vs Alpha (Train Only)
# ---------------------------
plt.plot(alphas, rmse_train, label="Train RMSE")
plt.xlabel("Alpha")
plt.ylabel("RMSE")
plt.title("Train RMSE vs Alpha (Ridge Regression)")
plt.xscale("log")
plt.legend()
plt.show()

# ---------------------------
# Plot MAE vs Alpha (Train Only)
# ---------------------------
plt.plot(alphas, mae_train, label="Train MAE")
plt.xlabel("Alpha")
plt.ylabel("MAE")
plt.title("Train MAE vs Alpha (Ridge Regression)")
plt.xscale("log")
plt.legend()
plt.show()

# ---------------------------
# Print Best Alpha
# ---------------------------
print("Best Alpha (lowest test MSE):", best_alpha)
print("Train MSE at best alpha:", mse_train[alphas.index(best_alpha)])
print("Test MSE at best alpha:", mse_test[alphas.index(best_alpha)])

# ---------------------------
# Print table including MAE
# ---------------------------
print("\nAlpha  | Train_RMSE | Test_RMSE | Train_MAE | Test_MAE | Train_R2  | Test_R2")
print("-------------------------------------------------------------------------------")

for i in range(len(alphas)):
    print(f"{alphas[i]:<6} | {rmse_train[i]:<10.4f} | {rmse_test[i]:<10.4f} | "
          f"{mae_train[i]:<10.4f} | {mae_test[i]:<10.4f} | "
          f"{r2_train[i]:<8.4f} | {r2_test[i]:<8.4f}")

# ---------------------------
# Check Overfitting / Underfitting
# ---------------------------
if mse_train[alphas.index(best_alpha)] < mse_test[alphas.index(best_alpha)] * 0.7:
    print("\nYour model is OVERFITTING")
elif mse_train[alphas.index(best_alpha)] > mse_test[alphas.index(best_alpha)] * 1.3:
    print("\nYour model is UNDERFITTING")
else:
    print("\nModel is GOOD — no major overfitting/underfitting")
    
joblib.dump(model, "fwi_model.pkl")
print("Model saved as fwi_model.pkl")

