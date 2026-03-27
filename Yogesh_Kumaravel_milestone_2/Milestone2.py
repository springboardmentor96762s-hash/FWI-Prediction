import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib

# -------------------------
# Load & Preprocess Data
# -------------------------
df = pd.read_csv("Algerian_forest_fires_cleaned_dataset.csv")

df['Classes'] = df['Classes'].str.strip().map({'fire': 1, 'not fire': 0})
df = df.drop(["day", "month", "year"], axis=1)

X = df.drop("FWI", axis=1)
y = df["FWI"]

# -------------------------
# Train-test split (80/20)
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# -------------------------
# RidgeCV (Auto Alpha Selection)
# -------------------------
alphas = [0.01, 0.1, 1, 5, 10, 50, 100]

ridge_cv = RidgeCV(alphas=alphas, cv=5)
ridge_cv.fit(X_train, y_train)

print("\nBest Alpha Selected:", ridge_cv.alpha_)

# -------------------------
# Test Set Evaluation
# -------------------------
y_pred_test = ridge_cv.predict(X_test)
r2 = r2_score(y_test, y_pred_test)
mse_test = mean_squared_error(y_test, y_pred_test)

print("\nRidgeCV Results:")
print("Test MSE:", mse_test)
print("Test R²:", r2)

# -------------------------
# Train Set Evaluation
# -------------------------
y_pred_train = ridge_cv.predict(X_train)
mse_train = mean_squared_error(y_train, y_pred_train)

print("\nTrain MSE:", mse_train)
print("Test MSE:", mse_test)

# -------------------------
# Diagnose Underfitting / Overfitting
# -------------------------
print("\nModel Diagnosis:")

if mse_train < mse_test * 0.5:
    print("OVERFITTING: Train MSE is much lower than Test MSE.")
elif mse_test > mse_train * 1.5:
    print("Possible OVERFITTING: Test MSE significantly higher.")
elif mse_train > 2 and mse_test > 2:
    print("UNDERFITTING: Both Train & Test MSE are high.")
else:
    print("GOOD FIT: Train & Test MSE are close. No major overfitting/underfitting.")

# -------------------------
# Print Coefficients
# -------------------------
print("\nModel Coefficients:")
for feature, coef in zip(X.columns, ridge_cv.coef_):
    print(f"{feature}: {coef}")

# -------------------------
# Alpha vs MSE Graph
# -------------------------
mse_scores = []
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for alpha in alphas:
    fold_mse = []
    for train_idx, val_idx in kf.split(X_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        model = Ridge(alpha=alpha)
        model.fit(X_tr, y_tr)
        pred = model.predict(X_val)
        fold_mse.append(mean_squared_error(y_val, pred))

    mse_scores.append(np.mean(fold_mse))

# Plot Alpha vs MSE
plt.figure(figsize=(8, 6))
plt.plot(alphas, mse_scores, marker='o')
plt.xlabel("Alpha")
plt.ylabel("Cross-Validated MSE")
plt.title("Ridge Regression: Alpha vs MSE")
plt.grid(True)
plt.show()

# -------------------------
# SAVE TRAINED MODEL AS .pkl
# -------------------------
model_filename = "ridge_fwi_model.pkl"
joblib.dump(ridge_cv, model_filename)

print(f"\nModel saved successfully as: {model_filename}")
