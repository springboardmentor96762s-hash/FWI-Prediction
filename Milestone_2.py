#!/usr/bin/env python
# coding: utf-8

import pandas as pd
df = pd.read_csv(r"C:/Users/amark/Downloads/Portuguese_Forest_Fires_Dataset.csv")
print(df.head())


temp1 = df["temp"]
print(temp1.isna().any())


print("\n--- Dataset Info ---")
print(df.info())

print("\n--- Summary Statistics ---")
print(df.describe())

print("\n--- Missing Values ---")
print(df.isna().sum())


#HISTOGRAM
import matplotlib.pyplot as plt

numeric_cols = df.select_dtypes(include='number').columns

for col in numeric_cols:
    plt.figure(figsize=(6,4))
    plt.hist(df[col], bins=20)
    plt.title(f"Histogram of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.show()


#Correlation Matrix
import matplotlib.pyplot as plt

numeric_df = df.select_dtypes(include='number')  # only numeric columns

plt.figure(figsize=(10,8))
corr = numeric_df.corr()
plt.imshow(corr, cmap="coolwarm", interpolation="nearest")
plt.colorbar()
plt.xticks(range(len(corr)), corr.columns, rotation=90)
plt.yticks(range(len(corr)), corr.columns)
plt.title("Correlation Heatmap (Numeric Features Only)")
plt.show()


#Pair plot
import seaborn as sns
sns.pairplot(df[numeric_cols])
plt.show()


#DISTRIBUTION PLOTS:
for col in numeric_cols:
    plt.figure(figsize=(6,4))
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col}")
    plt.show()


p = df.isnull().values.any()
print(p)


df = df.drop(columns=["X","Y"], errors="ignore")

cols = ["FFMC", "DMC", "ISI", "temp", "wind", "FWI"]

df_selected = df[cols]

print("\nSelected Data:")
print(df_selected.head())


plt.figure(figsize =(14,10))
for i, column in enumerate (cols, 1):
    plt.subplot(3,2,1)
    sns.histplot(df_selected[column],kde = True,bins = 20)
    plt.title(f"Distribution of (column)",fontsize=12)
    plt.xlabel(column)
    plt.ylabel("count")
    plt.tight_layout()
    plt.show()


corr = df_selected.corr()
plt.figure(figsize=(8,6))
plt.title("correlation heatmap(FFMC,DMC,ISI,Temp,Wind,Fwi)",fontsize=14)
sns.heatmap(
    corr,
    annot=True,
    cmap = "coolwarm",
    fmt=".2f",
    linewidths=.5
)
plt.show()


import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

df = pd.read_csv(r"C:/Users/amark/Downloads/Portuguese_Forest_Fires_Dataset.csv")

print(df.shape)
print(df.head())


features = ['FFMC', 'DMC', 'ISI', 'temp', 'wind']
target = 'FWI'

X = df[features]
y = df[target]

print(X.shape)
print(y.shape)


for feature in features:
    slope, intercept, r_value, p_value, std_err = stats.linregress(df[feature], y)
    print(feature, slope, intercept, r_value**2, p_value)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

y_pred = lr_model.predict(X_test)

print(lr_model.coef_)
print(lr_model.intercept_)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(mse)
print(rmse)
print(r2)


X_with_intercept = np.column_stack([np.ones(len(X)), X])

try:
    coefficients = np.linalg.inv(X_with_intercept.T @ X_with_intercept) @ (X_with_intercept.T @ y)

    print(coefficients)

except np.linalg.LinAlgError:
    print("Error")


print(X.corr())
print(y.min(), y.max(), y.mean(), y.std())


plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.hist(y, bins=50)

plt.subplot(1, 2, 2)
plt.scatter(range(len(y)), y, alpha=0.6)

plt.tight_layout()
plt.show()


residuals = y_test - y_pred

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.scatter(y_pred, residuals, alpha=0.6)
plt.axhline(y=0, color='red', linestyle='--')

plt.subplot(1, 3, 2)
plt.hist(residuals, bins=20)

plt.subplot(1, 3, 3)
plt.scatter(y_pred, y_test, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'red', lw=2)

plt.tight_layout()
plt.show()


from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Assume X, y already created from previous cells
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


alpha_values = [0.01, 0.1, 1, 10, 50, 100, 200, 500, 1000]


results = {
    "alpha": [],
    "mse_train": [],
    "mse_test": [],
    "rmse_train": [],
    "rmse_test": [],
    "mae_train": [],
    "mae_test": []
}

for alpha in alpha_values:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train, y_train)

    y_train_pred = ridge.predict(X_train)
    y_test_pred = ridge.predict(X_test)

    mse_train = mean_squared_error(y_train, y_train_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)
    rmse_train = np.sqrt(mse_train)
    rmse_test = np.sqrt(mse_test)
    mae_train = mean_absolute_error(y_train, y_train_pred)
    mae_test = mean_absolute_error(y_test, y_test_pred)

    results["alpha"].append(alpha)
    results["mse_train"].append(mse_train)
    results["mse_test"].append(mse_test)
    results["rmse_train"].append(rmse_train)
    results["rmse_test"].append(rmse_test)
    results["mae_train"].append(mae_train)
    results["mae_test"].append(mae_test)

results


results_df = pd.DataFrame(results)
results_df


plt.plot(results_df["alpha"], results_df["mse_train"], marker='o')
plt.plot(results_df["alpha"], results_df["mse_test"], marker='o')
plt.xscale("log")
plt.xlabel("Alpha")
plt.ylabel("MSE")
plt.legend(["Train MSE", "Test MSE"])
plt.title("MSE vs Alpha (Ridge Regression)")
plt.show()


plt.plot(results_df["alpha"], results_df["rmse_train"], marker='o')
plt.plot(results_df["alpha"], results_df["rmse_test"], marker='o')
plt.xscale("log")
plt.xlabel("Alpha")
plt.ylabel("RMSE")
plt.legend(["Train RMSE", "Test RMSE"])
plt.title("RMSE vs Alpha (Ridge Regression)")
plt.show()


plt.plot(results_df["alpha"], results_df["mae_train"], marker='o')
plt.plot(results_df["alpha"], results_df["mae_test"], marker='o')
plt.xscale("log")
plt.xlabel("Alpha")
plt.ylabel("MAE")
plt.legend(["Train MAE", "Test MAE"])
plt.title("MAE vs Alpha (Ridge Regression)")
plt.show()


best_alpha = results_df.loc[results_df["mse_test"].idxmin()]["alpha"]
best_alpha


for i in range(len(alpha_values)):
    alpha = results_df["alpha"][i]
    train_mse = results_df["mse_train"][i]
    test_mse = results_df["mse_test"][i]

    if train_mse < test_mse and (test_mse - train_mse) > 0.5:
        print(alpha, "→ Overfitting")
    elif train_mse > test_mse:
        print(alpha, "→ Underfitting")
    else:
        print(alpha, "→ Good Fit")

