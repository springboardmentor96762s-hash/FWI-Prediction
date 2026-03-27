import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

 
# DATA PREPROCESSING
 
df = pd.read_csv("Algerian_forest_fires_cleaned_dataset.csv")

df['Classes'] = df['Classes'].str.strip()
df['Classes'] = df['Classes'].map({'fire': 1, 'not fire': 0})

df = df.drop("day", axis=1)
df = df.drop("month", axis=1)
df = df.drop("year", axis=1)

print(df.isna().sum())
print(df.describe())

 
# FIX: MULTI-PANEL HISTOGRAM (AUTO GRID)
 

num_cols = df.select_dtypes(include=['int64', 'float64']).columns
num_features = len(num_cols)

cols = 3                                  # number of columns in subplot grid
rows = (num_features // cols) + 1         # auto-calculate rows needed

fig, axes = plt.subplots(rows, cols, figsize=(16, rows * 4))
axes = axes.flatten()

for i, col in enumerate(num_cols):
    sns.histplot(df[col], kde=True, bins=15, color="skyblue",
                 edgecolor="black", ax=axes[i])
    axes[i].set_title(f"Distribution of {col}", fontsize=12)
    axes[i].set_xlabel("")
    axes[i].set_ylabel("")

# hide empty subplots
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
plt.show()


# CORRELATION HEATMAP


plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# CORRELATION WITH FWI

corr_fwi = df.corr()['FWI'].sort_values(ascending=False)
positive_corr = corr_fwi[corr_fwi > 0].drop('FWI')
negative_corr = corr_fwi[corr_fwi < 0]

print("\nPositively correlated with FWI:\n", positive_corr)
print("\nNegatively correlated with FWI:\n", negative_corr)
