import pandas as pd
df = pd.read_csv(r"C:/Users/amark/Downloads/Portuguese_Forest_Fires_Dataset.csv")
print(df.head())

temp1 = df["temp"]
print(temp1.isna().any())

print("\n--- Dataset Info ---")
print(df.info())
print("\n--- Summary Statistics ---")
print(df.describe())
print("\n--- Missing Values ---")
print(df.isna().sum())

#HISTOGRAM
import matplotlib.pyplot as plt
numeric_cols = df.select_dtypes(include='number').columns
for col in numeric_cols:
    plt.figure(figsize=(6,4))
    plt.hist(df[col], bins=20)
    plt.title(f"Histogram of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.show()

#Correlation Matrix
import matplotlib.pyplot as plt
numeric_df = df.select_dtypes(include='number')  # only numeric columns
plt.figure(figsize=(10,8))
corr = numeric_df.corr()
plt.imshow(corr, cmap="coolwarm", interpolation="nearest")
plt.colorbar()
plt.xticks(range(len(corr)), corr.columns, rotation=90)
plt.yticks(range(len(corr)), corr.columns)
plt.title("Correlation Heatmap (Numeric Features Only)")
plt.show()

#Pair plot
import seaborn as sns
sns.pairplot(df[numeric_cols])
plt.show()

#DISTRIBUTION PLOTS:
for col in numeric_cols:
    plt.figure(figsize=(6,4))
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col}")
    plt.show()

p = df.isnull().values.any()
print(p)

df = df.drop(columns=["X","Y"], errors="ignore")
cols = ["FFMC", "DMC", "ISI", "temp", "wind", "FWI"]
df_selected = df[cols]
print("\nSelected Data:")
print(df_selected.head())


plt.figure(figsize =(14,10))
for i, column in enumerate (cols, 1):
    plt.subplot(3,2,1)
    sns.histplot(df_selected[column],kde = True,bins = 20)
    plt.title(f"Distribution of (column)",fontsize=12)
    plt.xlabel(column)
    plt.ylabel("count")
    plt.tight_layout()
    plt.show()

corr = df_selected.corr()
plt.figure(figsize=(8,6))
plt.title("correlation heatmap(FFMC,DMC,ISI,Temp,Wind,Fwi)",fontsize=14)
sns.heatmap(
    corr,
    annot=True,
    cmap = "coolwarm",
    fmt=".2f",
    linewidths=.5
)
plt.show()