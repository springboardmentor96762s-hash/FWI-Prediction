import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

# ------------------------------------------------------
# 1. LOAD DATASET
# ------------------------------------------------------
# We load the cleaned dataset
df = pd.read_csv("clean_data.csv")
print("Columns:", df.columns.tolist())

# Our prediction target is FWI (Fire Weather Index)
target_column = "FWI"
print("Using target:", target_column)

# These columns are not useful for regression (non-numerical or categorical)
drop_cols = ["day", "month", "year", "Classes", "Region"]

# X = input features, y = output variable (FWI)
X = df.drop(columns=drop_cols + [target_column])
y = df[target_column]

print("Feature columns used:", list(X.columns))

# ------------------------------------------------------
# 2. TRAIN-TEST SPLIT
# ------------------------------------------------------
# Split the dataset: 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------------------------------------
# 3. TRY MULTIPLE ALPHAS FOR RIDGE REGRESSION
# ------------------------------------------------------
# Ridge Regression has a tuning parameter 'alpha'
# We test different alpha values ranging from very small to large
alphas = np.logspace(-3, 3, 20)

mse_train_list = []   # store training errors
mse_test_list = []    # store testing errors

# Try each alpha one by one
for alpha in alphas:
    model = Ridge(alpha=alpha)      # Build model
    model.fit(X_train, y_train)     # Train model

    # Predictions on train and test data
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Calculate MSE (mean squared error)
    mse_train = mean_squared_error(y_train, y_pred_train)
    mse_test = mean_squared_error(y_test, y_pred_test)

    mse_train_list.append(mse_train)
    mse_test_list.append(mse_test)

# Find the alpha value with the lowest test MSE
best_alpha_idx = np.argmin(mse_test_list)
best_alpha = alphas[best_alpha_idx]

print("\nBest alpha =", best_alpha)

# ------------------------------------------------------
# 4. TRAIN FINAL RIDGE MODEL WITH BEST ALPHA
# ------------------------------------------------------
best_model = Ridge(alpha=best_alpha)
best_model.fit(X_train, y_train)

# Predict on the test set
y_pred_test = best_model.predict(X_test)

# Calculate final performance metrics
mse = mean_squared_error(y_test, y_pred_test)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred_test)
r2 = best_model.score(X_test, y_test)


print("\n================ RIDGE REGRESSION OUTPUT ================")
print("Slope (coef):", list(best_model.coef_))   # feature weights
print("Intercept:", best_model.intercept_)        # bias term
print("MSE:", mse)
print("RMSE:", rmse)
print("MAE:", mae)
print("R2 Score:", r2)                            # accuracy measure
print("Features:", list(X.columns))
print("==========================================================")

# ------------------------------------------------------
# 6. UNDERFITTING / OVERFITTING CHECK
# ------------------------------------------------------
# Compare training vs testing errors
train_best = mse_train_list[best_alpha_idx]
test_best = mse_test_list[best_alpha_idx]

print("\n================ MODEL DIAGNOSIS ==================")

# Overfitting: model memorizes training data (train error much smaller)
if train_best < test_best * 0.7:
    print("❌ Model is OVERFITTING (train error << test error)")

# Underfitting: model too simple (train error much higher)
elif train_best > test_best * 1.3:
    print("❌ Model is UNDERFITTING (train error >> test error)")

# Well-fitted: balanced performance
else:
    print("✅ Model is WELL-FITTED (train and test errors close)")

print("Train MSE at best alpha:", train_best)
print("Test MSE at best alpha :", test_best)
print("=====================================================")
