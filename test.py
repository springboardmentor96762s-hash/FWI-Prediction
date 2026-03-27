import pandas as pd
import numpy as np
my_data=pd.read_csv("forestfires.csv")

my_data=pd.read_csv("forestfires.csv")
                    
print(my_data)
# check for nan values(replace with 0) and convert categorical data into numerical data if any numerical
temp1=my_data["temp"]
print(temp1)

df = pd.read_csv('forestfires.csv')


# analysis.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data from CSV
df = pd.read_csv('forestfires.csv')

# Show the first few rows to confirm correct import
print(df.head())

# Encode categorical variables ("month", "day")
df['month'] = df['month'].astype('category').cat.codes
df['day'] = df['day'].astype('category').cat.codes

# Handle missing values (if any)
# Here, we fill numeric columns with the mean
for col in df.columns:
    if df[col].dtype in ['float64', 'int64']:
        df[col].fillna(df[col].mean(), inplace=True)

# Print if any missing values remain
print("Missing values in each column:")
print(df.isnull().sum())

# Calculate and print the correlation matrix (all features)
corr = df.corr()
print("Correlation matrix:")
print(corr)

# Plot histograms for all numeric columns
df.hist(bins=15, figsize=(15, 12))
plt.suptitle('Histograms of Features')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# Plot feature correlation heatmap
plt.figure(figsize=(15, 12))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title('Correlation Heatmap')
plt.show()

# Optionally, save the cleaned/encoded dataframe for further analysis
df.to_csv("forestfires_processed.csv", index=False)



