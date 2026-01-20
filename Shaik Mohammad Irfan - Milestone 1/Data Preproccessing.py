import pandas as pd
import numpy as np

df = pd.read_csv('/content/drive/MyDrive/FWI_DATASET/FWI_UPDATE.csv', skiprows=1)

df.columns = [col.strip() for col in df.columns]

numeric_columns = ['day', 'month', 'year', 'Temperature', 'RH', 'Ws', 'Rain', 'FFMC', 'DMC', 'DC', 'ISI', 'BUI', 'FWI']
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna(subset=['day']).reset_index(drop=True)

df.insert(0, 'Region', 'Bejaia')
df.loc[122:, 'Region'] = 'Sidi-Bel'

df['Classes'] = df['Classes'].astype(str).str.strip()
df['Classes'] = np.where(df['Classes'] == 'fire', 1, 0)  

# Handle missing values using linear interpolation (mean of neighbors)
df[numeric_columns] = df[numeric_columns].interpolate(method='linear', limit_direction='both')

df['day'] = df['day'].astype(int)
df['month'] = df['month'].astype(int)
df['year'] = df['year'] = df['year'].astype(int)
df['Temperature'] = df['Temperature'].astype(int)
df['RH'] = df['RH'].astype(int)
df['Ws'] = df['Ws'].astype(int)
df['Region'] = df['Region'].map({'Bejaia': 0, 'Sidi-Bel': 1}).astype(int)

df.to_csv('\content\Cleaned_FWI_dataset.csv', index=False)
print("Cleaned dataset saved as Cleaned_FWI_dataset.csv")
df.info()