import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import joblib

# LOAD CLEANED DATASET
df = pd.read_csv("cleaned_data.csv")   

# SET FEATURES & TARGET
target = "fwi"
feature_cols = df.select_dtypes(include=['int64', 'float64']).columns
feature_cols = feature_cols.drop(target)

X = df[feature_cols]
y = df[target]

print("Features used:", list(feature_cols))
print("Target:", target)

# TRAIN–TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# TRAIN LINEAR REGRESSION MODEL
model = LinearRegression()
model.fit(X_train, y_train)

print("\nModel training completed successfully!")

# PREDICT ON TEST DATA
y_pred = model.predict(X_test)
y_pred_train = model.predict(X_train)

# MODEL EVALUATION
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mse_train = mean_squared_error(y_train, y_pred_train)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n===================")
print("MODEL PERFORMANCE")
print("===================")
print(f"MAE  : {mae:.4f}")
print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R² Score : {r2:.4f}")

# SAVE THE MODEL
joblib.dump(model, "fwi_model.pkl")
print("Model saved as fwi_model.pkl")


# LOAD MODEL
model = joblib.load("fwi_model.pkl")

# EXAMPLE INPUT
new_data = pd.DataFrame([{
    "day": 1,
    "month": 6,
    "year": 2012,
    "temperature": 29,
    "rh": 57,
    "ws": 18,
    "rain": 0.0,
    "ffmc": 65.7,
    "dmc": 3.4,
    "dc": 7.6,
    "isi": 1.3,
    "bui": 3.4,
    "region": 0
}])

# PREDICT FWI
prediction = model.predict(new_data)
print("Predicted FWI:", prediction[0])

