import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("FWIdata_cleaned.csv")

numeric_cols = df.select_dtypes(include="number").columns

num_plots = len(numeric_cols)
rows = (num_plots // 3) + 1  # 3 histograms per row
cols = 3

plt.figure(figsize=(15, rows * 4))

for i, col in enumerate(numeric_cols, 1):
    plt.subplot(rows, cols, i)
    plt.hist(df[col].dropna(), bins=30)
    plt.title(col)
    plt.xlabel("Value")
    plt.ylabel("Frequency")

plt.tight_layout()
plt.show()
