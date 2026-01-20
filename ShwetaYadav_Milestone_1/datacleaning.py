import pandas as pd 

df = pd.read_csv("FWIdata.csv")

print("\nOriginal Data Info:")
print(df.info())
print("\nMissing values per column:")
print(df.isna().sum())

numeric_fix_cols = ["DC", "FWI"]

for col in numeric_fix_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.infer_objects(copy=False)


num_cols = df.select_dtypes(include=['number']).columns
df[num_cols] = df[num_cols].interpolate(method='cubic', limit_direction='both')

df_clean = df.copy()

df_clean["Classes"] = (
    df_clean["Classes"]
    .astype(str)
    .str.strip()
    .str.lower()
)

df_clean["Classes"] = df_clean["Classes"].map({
    "fire": 1,
    "not fire": 0       
})

print("\nClass value counts after conversion:")
print(df_clean["Classes"].value_counts(dropna=False))

print("\nCleaned Data Info:")
print(df_clean.info())

print("\nPreview of Cleaned Data:")
print(df_clean.head())

df_clean.to_csv("FWIdata_cleaned.csv", index=False)
print("\nCleaned file saved as FWIdata_cleaned.csv")
