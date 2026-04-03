import numpy as np
from scipy import linalg
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("cleaned_data.csv")

# Features
features = ['day','month','year',
            'temperature','rh','ws','rain',
            'ffmc','dmc','dc','isi','bui','region']

X = df[features].values
y = df['fwi'].values

# Train-test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Add bias column
X_train_b = np.c_[np.ones((X_train.shape[0], 1)), X_train]
X_test_b = np.c_[np.ones((X_test.shape[0], 1)), X_test]

# Train model using SciPy lstsq
theta, _, _, _ = linalg.lstsq(X_train_b, y_train)

# Coefficients
intercept = theta[0]
coefficients = theta[1:]

print("Intercept:", intercept)
print("Coefficients:", coefficients)

# Prediction
y_pred = X_test_b @ theta

# Evaluation on test set
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R² Score:", r2_score(y_test, y_pred))
