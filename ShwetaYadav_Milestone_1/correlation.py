import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("FWIdata_cleaned.csv")

if "year" in df.columns:
    df = df.drop(columns=["year"])

corr_matrix = df.corr(numeric_only=True)

plt.figure(figsize=(10, 8))
plt.imshow(corr_matrix, cmap="coolwarm", interpolation="nearest")
plt.colorbar(label="Correlation Value")

plt.xticks(range(len(corr_matrix)), corr_matrix.columns, rotation=90)
plt.yticks(range(len(corr_matrix)), corr_matrix.columns)

for i in range(len(corr_matrix)):
    for j in range(len(corr_matrix)):
        value = corr_matrix.iloc[i, j]
        plt.text(j, i, f"{value:.2f}", ha='center', va='center', color='black')

plt.title("Correlation Heatmap (year column removed)")
plt.tight_layout()
plt.show()

