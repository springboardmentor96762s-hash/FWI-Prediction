

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew

# --- Project Configuration ---
FILE_PATH = 'ff2.csv'
TARGET_VARIABLE = 'area' # The area burned is typically the ultimate target for severity prediction
FWI_COMPONENTS = ['FFMC', 'DMC', 'DC', 'ISI']

print(f"--- Starting Data Processing for {FILE_PATH} ---\n")

try:
    # Step 1: Loading the Dataset
    df = pd.read_csv(FILE_PATH)
    print(f"Successfully loaded {FILE_PATH}. Initial shape: {df.shape}")
except FileNotFoundError:
    print(f"ERROR: File not found at '{FILE_PATH}'. Please ensure the file is in the correct directory.")
    # Exit if file cannot be loaded
    exit()

# --- Step 2: Preprocessing the Data ---

## A. Check for Missing Values
print("\n[2A] Checking for Missing Values:")
missing_values = df.isnull().sum()
if missing_values.sum() == 0:
    print("✅ No missing values found in the dataset.")
else:
    print("⚠️ Missing values found. Handling them now...")
    # Example handling: Fill missing numerical data with the mean
    for col in df.columns:
        if df[col].dtype in ['float64', 'int64'] and missing_values[col] > 0:
            df[col].fillna(df[col].mean(), inplace=True)
            print(f"   Filled {missing_values[col]} missing values in '{col}' with the mean.")

## B. Handling Categorical Labels
print("\n[2B] Handling Categorical Labels ('month', 'day'):")
# Using One-Hot Encoding (pd.get_dummies)
df_processed = pd.get_dummies(df, columns=['month', 'day'], drop_first=False)
print(f"   Original categorical columns 'month' and 'day' have been converted to {len(df_processed.columns) - len(df.columns)} new binary columns.")
print(f"   Processed DataFrame shape: {df_processed.shape}")
print("\n--- Data Preprocessing Complete ---")

# --- Step 3: Analysing the Data ---

# 3A. Histograms for FWI Components and Target
print("\n[3A] Generating Histograms and Checking Skewness:")

# Set up the plotting environment
num_plots = len(FWI_COMPONENTS) + 1 # +1 for the target variable
fig, axes = plt.subplots(1, num_plots, figsize=(4 * num_plots, 5))
fig.suptitle('Distribution of Key Fire Weather Variables', fontsize=16)

all_features = FWI_COMPONENTS + [TARGET_VARIABLE]

for i, col in enumerate(all_features):
    # Log-transform the target variable 'area' for a better visual representation, 
    # as it's highly skewed (many zeros). Use np.log1p for handling zeros.
    if col == TARGET_VARIABLE:
        data_to_plot = np.log1p(df_processed[col])
        skewness = data_to_plot.skew()
        title = f'Log(1 + {col}) Distribution\nSkew: {skewness:.2f}'
        xlabel = f'Log(1 + {col})'
    else:
        data_to_plot = df_processed[col]
        skewness = data_to_plot.skew()
        title = f'{col} Distribution\nSkew: {skewness:.2f}'
        xlabel = col

    sns.histplot(data_to_plot, bins=20, kde=True, ax=axes[i], color='teal')
    axes[i].set_title(title, fontsize=10)
    axes[i].set_xlabel(xlabel)
    axes[i].set_ylabel('Frequency')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('analysis_histograms.png')
print("✅ Histograms saved to 'analysis_histograms.png'.")

# 3B. Covariance/Correlation Analysis
print("\n[3B] Calculating and Visualizing Correlation Matrix (Covariance/Correlation):")

# Select only the numerical FWI and weather features for the correlation map
correlation_features = FWI_COMPONENTS + ['temp', 'RH', 'wind', 'rain', TARGET_VARIABLE]
corr_matrix = df_processed[correlation_features].corr()

# Visualize the correlation matrix using a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, linecolor='black')
plt.title('Correlation Matrix of Fire Weather Index Components and Features')
plt.savefig('analysis_correlation_heatmap.png')
print("✅ Correlation Heatmap saved to 'analysis_correlation_heatmap.png'.")

print("\n--- All Milestone Steps Completed Successfully ---")