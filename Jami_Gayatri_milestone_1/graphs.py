import pandas as pd
import matplotlib.pyplot as plt
import os

# Create output folder
os.makedirs("output", exist_ok=True)

# Load cleaned CSV
df = pd.read_csv("clean_data.csv")

# Numeric columns, exclude day/month/year
exclude_cols = ["day", "month", "year"]
numeric_cols = [c for c in df.select_dtypes(include='number').columns if c not in exclude_cols]


# 1. Histograms of all numeric features

n_vars = len(numeric_cols)
n_cols = 2  # 2 plots per row
n_rows = (n_vars + n_cols - 1) // n_cols

plt.figure(figsize=(12, n_rows * 4))

for i, col in enumerate(numeric_cols):
    plt.subplot(n_rows, n_cols, i + 1)
    plt.hist(df[col], bins=20, alpha=0.75, color='skyblue', edgecolor='black')
    plt.title(f"Histogram of {col}")
    plt.xlabel("Value")
    plt.ylabel("Frequency")

plt.tight_layout()
plt.savefig("output/hist_all_features.png")
plt.close()
print("✅ Histograms of all numeric features saved as 'output/hist_all_features.png'.")

# 2. Correlation heatmap
corr = df[numeric_cols].corr()

plt.figure(figsize=(10,8))
im = plt.imshow(corr, cmap="viridis", interpolation="nearest")
plt.colorbar(im)

plt.xticks(range(len(corr.columns)), corr.columns, rotation=45)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.title("Correlation Heatmap (Numeric Features)", fontsize=14)

# Annotate correlation values
for i in range(len(corr.columns)):
    for j in range(len(corr.columns)):
        plt.text(j, i, f"{corr.iloc[i, j]:.2f}",
                 ha='center', va='center',
                 color='white' if abs(corr.iloc[i, j])>0.5 else 'black',
                 fontsize=8)

plt.tight_layout()
plt.savefig("output/correlation_heatmap.png")
plt.close()
print("✅ Correlation heatmap saved as 'output/correlation_heatmap.png'.")
