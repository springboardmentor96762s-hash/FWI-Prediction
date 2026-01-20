import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ===============================
# Milestone 2: EDA – FWI Predictor
# ===============================

# 1. Load dataset
df = pd.read_csv("ff2.csv")

print("\n--- Dataset Loaded ---")
print(df.head())
print("\nShape:", df.shape)

# 2. Basic info
print("\n--- Dataset Info ---")
print(df.info())

# 3. Encode target class (fire / not fire)
df['Classes'] = df['Classes'].astype(str).str.strip()
df['Classes_binary'] = np.where(df['Classes'] == 'fire', 1, 0)

# 4. Select numeric columns
numeric_cols = [
    'day','month','year','Temperature','RH','Ws','Rain',
    'FFMC','DMC','DC','ISI','BUI','FWI'
]

df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

# 5. Missing values check
print("\n--- Missing Values ---")
print(df[numeric_cols].isna().sum())

# 6. Statistical summary
print("\n--- Statistical Summary ---")
print(df[numeric_cols].describe())

# 7. Histograms
plt.figure(figsize=(18, 12))

for i, col in enumerate(numeric_cols, 1):
    plt.subplot(4, 4, i)
    plt.hist(df[col], bins=10, edgecolor='black')
    plt.title(col)
    plt.xlabel(col)
    plt.ylabel("Frequency")

plt.suptitle("Histograms of FWI Dataset Features", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# 8. Correlation matrix
corr_matrix = df[numeric_cols].corr()

print("\n--- Correlation Matrix ---")
print(corr_matrix.round(2))

# 9. Correlation heatmap (matplotlib only)
plt.figure(figsize=(12, 10))
plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
plt.colorbar(label="Correlation")

plt.xticks(range(len(numeric_cols)), numeric_cols, rotation=90)
plt.yticks(range(len(numeric_cols)), numeric_cols)

for i in range(len(numeric_cols)):
    for j in range(len(numeric_cols)):
        plt.text(j, i, f"{corr_matrix.iloc[i, j]:.2f}",
                 ha='center', va='center', fontsize=8)

plt.title("Correlation Heatmap – FWI Predictor")
plt.tight_layout()
plt.show()

# 10. Correlation with FWI
print("\n--- Correlation with FWI ---")
fwi_corr = corr_matrix['FWI'].sort_values(ascending=False)
print(fwi_corr.round(3))

# 11. Multicollinearity check
print("\n--- Highly Correlated Feature Pairs (|r| > 0.85) ---")
for i in range(len(numeric_cols)):
    for j in range(i+1, len(numeric_cols)):
        if abs(corr_matrix.iloc[i, j]) > 0.85:
            print(
                numeric_cols[i],
                "<->",
                numeric_cols[j],
                ":",
                round(corr_matrix.iloc[i, j], 3)
            )

# 12. Final EDA Insight
print("\n--- EDA Insights ---")
print("Top positive FWI drivers:", list(fwi_corr[fwi_corr > 0.6].index))
print("Negative impact on FWI:", list(fwi_corr[fwi_corr < 0].index))
