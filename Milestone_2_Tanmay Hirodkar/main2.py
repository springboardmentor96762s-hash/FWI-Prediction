import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

FILE_NAME = 'ff2.csv'

try:
    # 1. Load Data
    df = pd.read_csv(FILE_NAME)
except FileNotFoundError:
    print(f"Error: File '{FILE_NAME}' not found.")
    exit()

# --- 2. Preprocessing & Cleaning ---

# Force non-date/class columns to numeric, handling errors by converting to NaN
cols_to_convert = [col for col in df.columns if col not in ['day', 'month', 'year', 'Classes']]
for col in cols_to_convert:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Fill missing values (if any) with the column mean and remove 'Classes'
df = df.fillna(df.mean(numeric_only=True)).drop(columns=['Classes', 'day', 'month', 'year'])

# --- 3. Analysis ---

# Features for Analysis
features = df.columns.tolist()

# 3A. Histogram Analysis
print("Generating Histograms...")
df[features].hist(figsize=(12, 8), bins=20)
plt.suptitle('Distribution of Fire Weather Features', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('fwi_histograms_short.png')

# 3B. Correlation Analysis
print("Generating Correlation Heatmap...")
corr_matrix = df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix of FWI Features')
plt.savefig('fwi_correlation_short.png')

print("Analysis Complete. Check 'fwi_histograms_short.png' and 'fwi_correlation_short.png'.")