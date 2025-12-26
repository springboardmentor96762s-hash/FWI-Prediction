import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("Algerian_forest_fires_dataset.csv")

print("First 5 rows of the dataset:")
print(df.head())

print("\nDataset Shape (rows, columns):")
print(df.shape)

print("\nColumn Names:")
print(df.columns)

print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

df.drop_duplicates(inplace=True)

categorical_cols = df.select_dtypes(include=['object']).columns
print("\nCategorical Columns:", categorical_cols)

for col in categorical_cols:
    df[col] = df[col].astype('category').cat.codes

numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

print("\nSummary Statistics:")
print(df[numeric_cols].describe())

# Save cleaned dataset
df.to_csv("cleaned_fwi_dataset.csv", index=False)

print("\n--- AFTER CLEANING ---")
print(df.info())
print("\n Cleaned dataset saved as 'cleaned_fwi_dataset.csv'")

column_name = "Temperature"  

plt.figure(figsize=(8,5))
plt.hist(df[column_name], bins=20, edgecolor='black')
plt.title(f"Histogram of {column_name}", fontsize=14)
plt.xlabel(column_name)
plt.ylabel("Frequency")
plt.grid(axis='y', linestyle='--', alpha=0.7)


plt.figure(figsize=(12, 8))
corr = df.corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Heatmap", fontsize=14)


X = df[['Temperature', 'RH', 'Ws', 'Rain',
        'FFMC', 'DMC', 'DC', 'ISI', 'BUI']]
y = df['FWI']

print("\n Features and Target Selected")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

y_train_pred_lr = lr_model.predict(X_train)
y_test_pred_lr = lr_model.predict(X_test)

def print_errors(name, y_true, y_pred):
    print(f"\n{name} Errors:")
    print("MAE  :", mean_absolute_error(y_true, y_pred))
    print("MSE  :", mean_squared_error(y_true, y_pred))
    print("RMSE :", np.sqrt(mean_squared_error(y_true, y_pred)))
    print("R²   :", r2_score(y_true, y_pred))

print("\n  Linear Regression Results ")
print_errors("Training", y_train, y_train_pred_lr)
print_errors("Testing", y_test, y_test_pred_lr)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

joblib.dump(scaler, "scaler.pkl")
print("\n Scaler saved as scaler.pkl")

alphas = [0.01, 0.1, 1, 10, 100]

train_mse = []
test_mse = []

print("\n Ridge Regression Results ")
print("Alpha | Train MSE | Test MSE | Train RMSE | Test RMSE | Train MAE | Test MAE")

for alpha in alphas:
    ridge_model = Ridge(alpha=alpha)
    ridge_model.fit(X_train_scaled, y_train)

    y_train_pred = ridge_model.predict(X_train_scaled)
    y_test_pred = ridge_model.predict(X_test_scaled)

    mse_train_val = mean_squared_error(y_train, y_train_pred)
    mse_test_val = mean_squared_error(y_test, y_test_pred)

    rmse_train = np.sqrt(mse_train_val)
    rmse_test = np.sqrt(mse_test_val)

    mae_train = mean_absolute_error(y_train, y_train_pred)
    mae_test = mean_absolute_error(y_test, y_test_pred)

    train_mse.append(mse_train_val)
    test_mse.append(mse_test_val)

    print(f"{alpha} | {mse_train_val:.4f} | {mse_test_val:.4f} | {rmse_train:.4f} | {rmse_test:.4f} | {mae_train:.4f} | {mae_test:.4f}")

plt.figure()
plt.plot(alphas, train_mse)
plt.plot(alphas, test_mse)
plt.xlabel("Alpha")
plt.ylabel("MSE")
plt.title("MSE vs Alpha (Ridge Regression)")
plt.xscale("log")
plt.legend(["Train MSE", "Test MSE"])
plt.show()

best_ridge_model = Ridge(alpha=1)
best_ridge_model.fit(X_train_scaled, y_train)

joblib.dump(best_ridge_model, "ridge_model.pkl")

print("\n Ridge model saved as ridge_model.pkl")

