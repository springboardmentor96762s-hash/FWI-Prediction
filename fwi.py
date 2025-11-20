import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Algerian_forest_fires_dataset.csv")

print("First 5 rows of the dataset:")
print(df.head())

print("\n Dataset Shape (rows, columns):")
print(df.shape)

print("\n Column Names:")
print(df.columns)

print("\n Dataset Info:")
print(df.info())

print("\n Missing Values in Each Column:")
print(df.isnull().sum())

df.drop_duplicates(inplace=True)

categorical_cols = df.select_dtypes(include=['object']).columns
print("\nCategorical Columns:", categorical_cols)

for col in categorical_cols:
    df[col] = df[col].astype('category').cat.codes

numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

print("\nSummary Statistics:")
print(df[numeric_cols].describe())

df.to_csv("cleaned_fwi_dataset.csv", index=False)

print("\n---AFTER CLEANING ---")
print(df.info())
print("\nCleaned dataset saved as 'cleaned_fwi_dataset.csv'")
# histogram
column_name = "Temperature"   # change here to RH, Ws, FWI, etc.

plt.figure(figsize=(8,5))
plt.hist(df[column_name], bins=20, edgecolor='black')
plt.title(f"Histogram of {column_name.capitalize()}", fontsize=14)
plt.xlabel(column_name.capitalize(), fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)


# correlation
plt.figure(figsize=(12, 8))
corr = df.corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Heatmap", fontsize=14)
plt.show()
