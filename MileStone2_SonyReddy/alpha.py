import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import math

# Load dataset
df = pd.read_csv("forestfires_final_clean.csv")   

X = df[['month','day','ffmc','dmc','dc','isi','temp', 'wind','bui']]  
y = df['fwi']                                 

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Linear Regression
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Predictions and metrics for Linear
train_pred = linear_model.predict(X_train)
test_pred = linear_model.predict(X_test)

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge, Lasso

# Define alpha values to search
alpha_values = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]

# Ridge
ridge = Ridge()
ridge_cv = GridSearchCV(ridge, param_grid={'alpha': alpha_values}, scoring='neg_mean_squared_error', cv=5)
ridge_cv.fit(X_train, y_train)
best_ridge_alpha = ridge_cv.best_params_['alpha']
print("Best Ridge alpha:", best_ridge_alpha)

# Lasso
lasso = Lasso(max_iter=10000)
lasso_cv = GridSearchCV(lasso, param_grid={'alpha': alpha_values}, scoring='neg_mean_squared_error', cv=5)
lasso_cv.fit(X_train, y_train)
best_lasso_alpha = lasso_cv.best_params_['alpha']
print("Best Lasso alpha:", best_lasso_alpha)
