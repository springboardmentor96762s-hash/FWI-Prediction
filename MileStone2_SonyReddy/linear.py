import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
import matplotlib.pyplot as plt
import math
df = pd.read_csv("forestfires_final_clean.csv")   

X = df[['month','day','ffmc','dmc','dc','isi','temp', 'wind','bui']]  
y = df['fwi']                                 

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

train_mse = mean_squared_error(y_train, train_pred)
test_mse = mean_squared_error(y_test, test_pred)
train_mae=mean_absolute_error(y_train, train_pred)
test_mae=mean_absolute_error(y_test, test_pred)
train_root_mean_square=math.sqrt(train_mse)
test_root_mean_square=math.sqrt(test_mse)
print("Train MSE:", train_mse)
print("Test MSE:", test_mse)
print("Train MAE:",train_mae)
print("Test MAE:",test_mae)
print("Train root mean square: ",train_root_mean_square)
print("Test root mean square:",test_root_mean_square)

print("R2 Score (Train):", r2_score(y_train, train_pred))
print("R2 Score (Test):", r2_score(y_test, test_pred))

ridge = Ridge(alpha=0.001)
ridge.fit(X_train, y_train)

lasso = Lasso(alpha=0.001)
lasso.fit(X_train, y_train)

models = {"Linear": model, "Ridge": ridge, "Lasso": lasso}

for name, model in models.items():
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    print(f"\n{name} Regression:")
    print("Train MSE:", mean_squared_error(y_train, y_pred_train))
    print("Test MSE:", mean_squared_error(y_test, y_pred_test))
    print("Train MAE:",mean_absolute_error(y_train, y_pred_train))
    print("Test MAE",mean_absolute_error(y_test, y_pred_test))
    print("Train R2:", r2_score(y_train, y_pred_train))
    print("Test R2:", r2_score(y_test, y_pred_test))
    print("Train root mean square: ",math.sqrt(mean_squared_error(y_train,y_pred_train)))
    print("Test root mean square:",math.sqrt(mean_squared_error(y_test,y_pred_test)))

plt.figure(figsize=(10, 6))

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    plt.scatter(y_test, y_pred, s=30, alpha=0.7, label=name)
    plt.xlabel("Actual FWI")
    plt.ylabel("Predicted FWI")
    plt.title(model)
    plt.legend()
    plt.grid(True)
    plt.show()
