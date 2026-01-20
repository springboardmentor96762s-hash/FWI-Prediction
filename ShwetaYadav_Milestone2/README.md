# Milestone 2 : Forest Fire Weather Index (FWI) Prediction using Ridge Regression
This project builds a Ridge Regression model to predict the Fire Weather Index (FWI) using real-world meteorological and fire-danger indicators such as temperature, humidity, windspeed, DC, DMC, ISI, BUI, etc.

# The workflow includes:
Feature Engineering

Data Scaling

Ridge Regression Training

Hyperparameter Tuning (alpha)

Model Evaluation

Overfitting/Underfitting Diagnosis

Saving the scaler and trained model

# steps achieved 
# module 3
1. Selected strongly correlated features
2. Divided dataset into Input (X) and Output (y)
3. Applied Train-Test Split
4. Normalized using StandardScaler
5. Saved the scaler as scaler.pkl

# module 4
1. Trained Ridge Regression Model
2. Tuned Alpha Parameter
3. Evaluated Training and Validation Performance
4. Saved the trained model as ridge.pkl

# module 5
1. Evaluation Metrics Used (MAE,MSE,RMSE,R² Score)
2. Plotted Graphs
MSE vs Alpha
MAE vs Alpha
RMSE vs Alpha
Predicted vs Actual scatter plot
3. Overfitting/Underfitting Checked 
