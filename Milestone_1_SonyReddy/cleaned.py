import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv("forestfires_cleaned.csv")
df = df.dropna()
if 'month' in df.columns:
    df['month'] = df['month'].astype('category').cat.codes
if 'day' in df.columns:
    df['day'] = df['day'].astype('category').cat.codes
df = df.apply(pd.to_numeric, errors='coerce')
df = df.dropna()
df.to_csv("forestfires_final_clean.csv", index=False)
print(df.head())
#checking if any missing values are there
p=df.isnull().values.any()
print(p)
df = df.drop(columns=["X", "Y"], errors="ignore")
cols = ["ffmc", "dmc", "isi", "temp", "wind", "fwi"]
df_selected = df[cols]

print("\nSelected Data:")
print(df_selected.head())

plt.figure(figsize=(14, 10))
for i, column in enumerate(cols, 1):
    plt.subplot(3, 2, i)
    sns.histplot(df_selected[column], kde=True, bins=20)
    plt.title(f"Distribution of {column}", fontsize=12)
    plt.xlabel(column)
    plt.ylabel("Count")
    plt.tight_layout()

plt.show()

corr = df_selected.corr()

plt.figure(figsize=(8, 6))
plt.title("Correlation Heatmap (FFMC, DMC, ISI, Temp, Wind, FWI)", fontsize=14)

sns.heatmap(
    corr,
    annot=True,        # Show values
    cmap="coolwarm",
    fmt=".2f",         # Two decimal places
    linewidths=.5
)

plt.show()