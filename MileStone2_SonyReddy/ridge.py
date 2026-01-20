import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error

df = pd.read_csv("forestfires_final_clean.csv")

X = df[['month','day','ffmc','dmc','dc','isi','temp', 'wind','bui']]
y = df['fwi']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

alphas = np.logspace(-3, 3, 20)   

train_mse = []
test_mse = []

train_rmse = []
test_rmse = []

train_mae = []
test_mae = []

for a in alphas:
    model = Ridge(alpha=a)
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # MSE
    train_mse.append(mean_squared_error(y_train, y_train_pred))
    test_mse.append(mean_squared_error(y_test, y_test_pred))

    # RMSE
    train_rmse.append(np.sqrt(mean_squared_error(y_train, y_train_pred)))
    test_rmse.append(np.sqrt(mean_squared_error(y_test, y_test_pred)))

    # MAE
    train_mae.append(mean_absolute_error(y_train, y_train_pred))
    test_mae.append(mean_absolute_error(y_test, y_test_pred))



plt.figure(figsize=(14, 10))

plt.subplot(3, 1, 1)
plt.plot(alphas, train_mse, marker='o', label="Train MSE")
plt.plot(alphas, test_mse, marker='s', label="Test MSE")
plt.xscale("log")
plt.xlabel("Alpha")
plt.ylabel("MSE")
plt.title("MSE vs Alpha (Ridge Regression)")
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(alphas, train_rmse, marker='o', label="Train RMSE")
plt.plot(alphas, test_rmse, marker='s', label="Test RMSE")
plt.xscale("log")
plt.xlabel("Alpha")
plt.ylabel("RMSE")
plt.title("RMSE vs Alpha (Ridge Regression)")
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(alphas, train_mae, marker='o', label="Train MAE")
plt.plot(alphas, test_mae, marker='s', label="Test MAE")
plt.xscale("log")
plt.xlabel("Alpha")
plt.ylabel("MAE")
plt.title("MAE vs Alpha (Ridge Regression)")
plt.legend()

plt.tight_layout()
plt.show()

best_alpha = alphas[np.argmin(test_mse)]
print("Best Alpha:", best_alpha)
print("Minimum Test MSE:", min(test_mse))
