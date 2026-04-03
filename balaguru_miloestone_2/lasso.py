import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("cleaned_data.csv")

# Features & Target
features = ['day','month','year','temperature','rh','ws','rain',
            'ffmc','dmc','dc','isi','bui','region']

X = df[features].values
y = df["fwi"].values

# Train–test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Lasso model
lasso = Lasso(alpha=0.1)   # alpha controls penalty strength
lasso.fit(X_train, y_train)

# Predictions
y_pred = lasso.predict(X_test)

# Metrics
print("===== LASSO REGRESSION =====")
print("Intercept:", lasso.intercept_)
print("Coefficients:", lasso.coef_)
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R² Score:", r2_score(y_test, y_pred))

import joblib

joblib.dump(lasso, "lasso_model.pkl")
print("Lasso model saved as lasso_model.pkl")
