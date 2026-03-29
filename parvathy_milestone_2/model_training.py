import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle
df=pd.read_csv('Bejaia-Region-Dataset_Cleaned.csv')
X=df[['Temperature', 'RH', 'Ws', 'Rain', 'DMC', 'DC', 'ISI', 'BUI', 'FFMC']]
y=df['FWI']
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model=LinearRegression()
model.fit(X_train, y_train)
y_pred=model.predict(X_test)
mse=mean_squared_error(y_test, y_pred)
mae=mean_absolute_error(y_test, y_pred)
r2=r2_score(y_test, y_pred)
print(f'Mean Absolute Error: {mae}')
print(f'Mean Squared Error: {mse}')
print(f'R2 Score: {r2}')
ridge_model=Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)
y_ridge_pred=ridge_model.predict(X_test)
ridge_mse=mean_squared_error(y_test, y_ridge_pred)
ridge_mae=mean_absolute_error(y_test, y_ridge_pred)
ridge_r2=r2_score(y_test, y_ridge_pred)
print(f'Ridge Mean Squared Error: {ridge_mse}')
print(f'Ridge Mean Absolute Error: {ridge_mae}')
print(f'Ridge R2 Score: {ridge_r2}')

train_ridge_mse=[]
test_ridge_mse=[]
train_ridge_mae=[]
test_ridge_mae=[]
train_ridge_r2=[]
test_ridge_r2=[]
train_ridge_rmse=[]
test_ridge_rmse=[]

a=[0.1,0.2,1.0,2.0,5.0,10.0]

for alpha in a:
    ridge_model=Ridge(alpha=alpha)
    ridge_model.fit(X_train, y_train)
    y_train_ridge_pred=ridge_model.predict(X_train)
    y_test_ridge_pred=ridge_model.predict(X_test)
    
    train_mse=mean_squared_error(y_train, y_train_ridge_pred)
    test_mse=mean_squared_error(y_test, y_test_ridge_pred)
    
    train_mae=mean_absolute_error(y_train, y_train_ridge_pred)
    test_mae=mean_absolute_error(y_test, y_test_ridge_pred)
    
    train_r2=r2_score(y_train, y_train_ridge_pred)
    test_r2=r2_score(y_test, y_test_ridge_pred)

    train_rmse=np.sqrt(train_mse)
    test_rmse=np.sqrt(test_mse)
    
    train_ridge_mse.append(train_mse)
    test_ridge_mse.append(test_mse)
    
    train_ridge_mae.append(train_mae)
    test_ridge_mae.append(test_mae)
    
    train_ridge_r2.append(train_r2)
    test_ridge_r2.append(test_r2)

    train_ridge_rmse.append(train_rmse)
    test_ridge_rmse.append(test_rmse)
    
    print(f'Alpha: {alpha}')
    print(f'Train MSE: {train_mse}, Test MSE: {test_mse}')
    print(f'Train MAE: {train_mae}, Test MAE: {test_mae}')
    print(f'Train R2: {train_r2}, Test R2: {test_r2}')
    print(f'Train RMSE: {train_rmse}, Test RMSE: {test_rmse}')



plt.figure(figsize=(10, 6))
plt.plot(a,train_ridge_mse, marker='o', label='Train MSE', color='blue')
plt.plot(a,test_ridge_mse, marker='o', label='Test MSE', color='red')
plt.xlabel('Alpha')
plt.ylabel('Mean Squared Error')
plt.title('Ridge Regression: Alpha vs Mean Squared Error')
plt.legend()
plt.show()


plt.figure(figsize=(10, 6))
plt.plot(a,train_ridge_r2, marker='o', label='Train R2 Score', color='blue')
plt.plot(a,test_ridge_r2, marker='o', label='Test R2 Score', color='red')
plt.xlabel('Alpha')
plt.ylabel('R2 Score')
plt.title('Ridge Regression: Alpha vs R2 Score')
plt.legend()
plt.show()


plt.figure(figsize=(10, 6))
plt.plot(a,train_ridge_mae, marker='o', label='Train MAE', color='blue')
plt.plot(a,test_ridge_mae, marker='o', label='Test MAE', color='red')
plt.xlabel('Alpha') 
plt.ylabel('Mean Absolute Error')
plt.title('Ridge Regression: Alpha vs Mean Absolute Error')
plt.legend()
plt.show()


final_ridge_model=Ridge(alpha=0.1)
with open('ridge_model.pkl', 'wb') as f:
    pickle.dump(final_ridge_model, f)