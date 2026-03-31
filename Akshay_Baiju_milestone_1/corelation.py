import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("Algeria_dataset_clean.csv")


data = data.drop(columns=["class"], errors="ignore")
data = data.drop(columns=["day"], errors="ignore")
data = data.drop(columns=["month"], errors="ignore")

numeric_data = data.select_dtypes(include=["float64", "int64"])


corr = numeric_data.corr()

fwi_corr = corr["fwi"].drop("fwi")  

fwi_corr = fwi_corr.sort_values(key=abs)


plt.figure(figsize=(10, 7))
plt.barh(fwi_corr.index, fwi_corr.values, color="orange", edgecolor="black")

plt.title("Correlation of Features with FWI")
plt.xlabel("Correlation Value")
plt.ylabel("Features")
plt.grid(axis="x", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.show()
