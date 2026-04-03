import pandas as pd
import numpy as np

# Load data
df = pd.read_csv("dataset.csv")

# View basic info
print(df.head())
print(df.info())
print(df.describe())

#Remove duplicate rows
df = df.drop_duplicates()

# Handle missing values
# Fill missing numeric values with mean
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

# Option C: Fill missing categorical values with mode
cat_cols = df.select_dtypes(include=['object']).columns
df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])

# Remove unwanted spaces 
df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

# Convert categorical labels to proper format
df = df.apply(lambda x: x.str.lower() if x.dtype == "object" else x)

# Fix wrong data types
df['day'] = df['day'].astype(int)
df['month'] = df['month'].astype(int)

# Rename columns for consistent format
num_cols = ['Temperature','RH','Ws','Rain','FFMC','DMC','DC','ISI','BUI','FWI','Region']
df.columns = df.columns.str.strip().str.lower()

#Remove outliers using IQR
# Get numeric columns automatically
num_cols = df.select_dtypes(include=['int64','float64']).columns

# Calculate IQR only if numeric columns exist
if len(num_cols) > 0:
    Q1 = df[num_cols].quantile(0.25)
    Q3 = df[num_cols].quantile(0.75)
    IQR = Q3 - Q1
    df = df[~((df[num_cols] < (Q1 - 1.5 * IQR)) | 
              (df[num_cols] > (Q3 + 1.5 * IQR))).any(axis=1)]

# Final cleaned data shape
print("Cleaned Data Shape:", df.shape)

# Save cleaned file
df.to_csv("cleaned_data.csv", index=False)

