import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("dataset.csv")

# Create a copy for correlation analysis
df_corr = df.copy()

# Select only numeric columns
numeric_cols = df_corr.select_dtypes(include=['int64', 'float64']).columns

# HISTOGRAM OF ALL NUMERIC FEATURES
plt.figure(figsize=(14, 10))
df_corr[numeric_cols].hist(figsize=(14, 10), bins=20)
plt.suptitle("Overall Distribution of Numerical Features", fontsize=16)
plt.tight_layout()
plt.show()

# CORRELATION OF FEATURES WITH FWI
# Only numeric values
df_numeric = df_corr[numeric_cols]

# Correlation with FWI
corr_values = df_numeric.corr()['FWI'].sort_values(ascending=False)

print("\n============================")
print("Correlation of Variables with FWI")
print("============================\n")
print(corr_values)

# CORRELATION HEATMAP (ALL NUMERIC)
plt.figure(figsize=(10, 6))
sns.heatmap(df_numeric.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap of Numerical Variables")
plt.show()
