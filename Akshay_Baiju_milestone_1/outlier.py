import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("Algeria_dataset_clean.csv")
exclude_cols = ["day", "month", "year", "region", "class"]
numeric_cols = [col for col in data.columns 
                if col not in exclude_cols and data[col].dtype != 'object']

plt.figure(figsize=(10, 6))
plt.boxplot([data[col] for col in numeric_cols], labels=numeric_cols, vert=False)
plt.title("Boxplots of Attributes")
plt.xlabel("Value Range")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
