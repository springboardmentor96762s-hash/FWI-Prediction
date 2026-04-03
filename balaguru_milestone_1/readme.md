🌡️ Tempest FWI Predictor – Wildfire Risk Analysis
Infosys Springboard Internship Project

This project focuses on analyzing wildfire-related meteorological and environmental data to predict the Fire Weather Index (FWI) using data preprocessing, exploratory data analysis, and correlation-based insights.

🎯 Project Objectives

1)Clean and preprocess wildfire dataset
2)Handle missing values, duplicates, and formatting issues
3)Perform data exploration using histograms & heatmaps
4)Analyze correlation of meteorological factors with FWI
5)Prepare dataset for FWI regression modeling

📁 Dataset Overview

The dataset includes weather parameters, Canadian FWI system indices, and the FWI target value.

🌡️ Meteorological Variables

1)Temperature – Air temperature (°C)
2)RH – Relative Humidity (%)
3)Ws – Wind Speed
4)Rain – Rainfall amount (mm)

🔥 Fire Weather Indices (Canadian FWI System)

1)FFMC – Fine Fuel Moisture Code
2)DMC – Duff Moisture Code
3)DC – Drought Code
4)ISI – Initial Spread Index
5)BUI – Build Up Index
5)FWI – Fire Weather Index (📌 Target variable for prediction)

🏷️ Additional Field

1)Classes – fire / not fire (not used as target, used only for reference)
2)Region – Region code

🛠️ Data Preprocessing Steps

1️⃣ Data Quality Check

1)Verified missing values
2)Identified whitespace issues in columns
3)Inspected numeric data types
4)Checked for duplicates

2️⃣ Data Cleaning

1)Removed whitespace from column names
2)Cleaned Classes column only for reference
3)Converted all numeric columns to proper types
4)Ensured FWI column has no missing values

3️⃣ Exploratory Data Analysis

1)Histograms for all numerical features
2)Correlation heatmap for feature relationships
3)FWI correlation ranking to identify key influencers

📊 Key Insights

1)FFMC, ISI, DMC, DC, BUI show strong positive correlation with FWI
2)Humidity (RH) often shows negative correlation
3)Dataset is balanced and clean for regression modeling

🎯 Target of the Project

The main goal of this project is:
🔥 Predicting the Fire Weather Index (FWI)
FWI is a numerical rating of fire intensity based on environmental conditions.
This helps authorities understand how severe a fire could be under current weather conditions.

🚀 Next Steps

1)Build regression models: Linear Regression
2)Evaluate using R², RMSE, MAE
3)Feature importance analysis
4)Flask deployment

🛡️ Applications

Useful for:

🚒 Early wildfire risk prediction
🌲 Forest and environment monitoring
🔬 Climate research
🏞️ Disaster management planning

💻 Technologies Used

1)Python
2)Pandas
3)NumPy
4)Matplotlib
5)Seaborn
6)Scikit-learn