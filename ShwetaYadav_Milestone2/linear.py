import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt


df = pd.read_csv("FWIdata_cleaned.csv")

# Selecting numeric columns 
df_numeric = df.select_dtypes(include=[np.number])

# Correlation selection
corr = df_numeric.corr()["FWI"].abs()
selected_features = corr[corr > 0.3].index.drop("FWI")  
print("\nSelected Features:", list(selected_features))

X = df_numeric[selected_features].values
y = df_numeric["FWI"].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Saving scaler
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("\nScaler saved as scaler.pkl")


# Ridge Model Training

alpha_values = [0.01, 0.1, 1, 10, 100]

mse_train_list = []
mse_test_list = []
mae_list = []
rmse_list = []

best_alpha = None
best_score = -np.inf

for alpha in alpha_values:
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    mse_train = mean_squared_error(y_train, y_train_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)
    mae = mean_absolute_error(y_test, y_test_pred)
    rmse = np.sqrt(mse_test)
    r2 = r2_score(y_test, y_test_pred)

    mse_train_list.append(mse_train)
    mse_test_list.append(mse_test)
    mae_list.append(mae)
    rmse_list.append(rmse)

    print(f"\nAlpha: {alpha}")
    print("Training MSE:", mse_train)
    print("Testing MSE:", mse_test)
    print("MAE:", mae)
    print("RMSE:", rmse)
    print("R²:", r2)

    if r2 > best_score:
        best_score = r2
        best_alpha = alpha
        best_model = model

# Saving best model
with open("ridge.pkl", "wb") as f:
    pickle.dump(best_model, f)

print("\n Best Alpha:", best_alpha)
print(" Ridge model saved as ridge.pkl")


#  Evaluation & Fit Diagnosis

mse_train_best = mse_train_list[alpha_values.index(best_alpha)]
mse_test_best = mse_test_list[alpha_values.index(best_alpha)]

print("\n--- MODEL FIT DIAGNOSIS ---")

if mse_train_best < mse_test_best * 0.5:
    print(" Overfitting detected (model memorizing training data)")
elif mse_train_best > mse_test_best * 1.5:
    print(" Underfitting detected (model too simple)")
else:
    print(" Good Fit (balanced bias-variance)")


# PLOTS


plt.figure()
plt.plot(alpha_values, mse_train_list, marker='o', label="Training MSE")
plt.plot(alpha_values, mse_test_list, marker='o', label="Testing MSE")
plt.xlabel("Alpha")
plt.ylabel("MSE")
plt.title("MSE vs Alpha")
plt.legend()
plt.show()

plt.figure()
plt.plot(alpha_values, mae_list, marker='o')
plt.xlabel("Alpha")
plt.ylabel("MAE")
plt.title("MAE vs Alpha")
plt.show()

plt.figure()
plt.plot(alpha_values, rmse_list, marker='o')
plt.xlabel("Alpha")
plt.ylabel("RMSE")
plt.title("RMSE vs Alpha")
plt.show()

# Predicted vs Actual Plot
y_pred_best = best_model.predict(X_test)

plt.figure()
plt.scatter(y_test, y_pred_best)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Predicted vs Actual")
plt.show()
