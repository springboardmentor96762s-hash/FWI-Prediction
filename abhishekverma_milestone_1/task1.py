# 1. Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 2. Load Dataset

data_path = "FWI-Prediction/abhishekverma_milestone_1/forest_fire_dataset.csv"
df = pd.read_csv(data_path)

print("=== Original Dataset ===")
print(df.head())

# 3. Check Missing Values

print("\n=== Missing Values Before Cleaning ===")
print(df.isna().sum())

# 4. Handle Missing Values

# If numeric columns have missing values  → fill NaN with mean

numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

# If categorical columns have missing values → fill NaN with mode

categorical_cols = df.select_dtypes(include='object').columns
df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])

# 5. Encode Categorical Columns

# Encode 'Classes' column: fire=1, not fire=0
df['Classes'] = df['Classes'].map({'fire': 1, 'not fire': 0})

# Encode 'Region' column using manual mapping

region_map = {"East": 0, "North": 1, "South": 2, "West": 3}
df['Region'] = df['Region'].map(region_map)
print("\n=== \n=== Dataset with Numeric 'Region' Column === ===")
print(df.head())

# 6. Verify Cleaning

print("\n=== Missing Values After Cleaning ===")
print(df.isna().sum())

print("\n=== Cleaned & Encoded Dataset Preview ===")
print(df.head())

# 7. Save Cleaned Dataset

cleaned_csv = "FWI-Prediction/abhishekverma_milestone_1/cleaned_forest_fire_dataset.csv"
df.to_csv(cleaned_csv, index=False)
print(f"\nCleaned CSV Saved as: {cleaned_csv}")

# 8. Exploratory Data Analysis

# 8a. Histograms of numeric columns

plt.figure(figsize=(12, 10))
df[numeric_cols].hist(bins=10, figsize=(12, 10))
plt.suptitle("Histogram of All Numeric Features", fontsize=14)
plt.tight_layout()
plt.show()

# 8b. Correlation Matrix & Heatmap
corr = df.select_dtypes(include=['number']).corr()
print("\n=== Correlation Matrix ===")
print(corr)

plt.figure(figsize=(10, 7))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()