import pandas as pd

data = pd.read_csv("Dataset_Algeria.csv").dropna(how="all")
data.columns = data.columns.str.strip().str.lower()


data["day_numeric"] = pd.to_numeric(data["day"], errors="coerce")
data = data[data["day_numeric"].notna()].drop(columns=["day_numeric"])

for col in data.select_dtypes(include="object").columns:
    data[col] = data[col].str.strip()

region1 = data.iloc[:122].copy()
region2 = data.iloc[122:].copy()
region1["region"] = "bejaia"
region2["region"] = "sidi_bel_abbes"
data = pd.concat([region1, region2], ignore_index=True)

print("\nNull values:\n", data.isnull().sum())


numeric_cols = ["day","month","year","temperature","rh","ws","rain","ffmc","dmc","dc","isi","bui","fwi"]
data[numeric_cols] = data[numeric_cols].apply(pd.to_numeric, errors="coerce")
data = data.dropna(subset=numeric_cols)

data["class"] = data["classes"].str.lower().map({"not fire": 0, "fire": 1})
data = data.drop(columns=["classes"])
data = data.dropna(subset=["class"])
data["class"] = data["class"].astype(int)

print(data.head())
print("\nColumns:", data.columns.tolist())

data.to_csv("Algeria_dataset_clean.csv", index=False)
