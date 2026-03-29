import pandas as pd
df=pd.read_csv('Bejaia Region Dataset.csv')
print(df.head())
print(df.info())
print(df.isnull().sum())
df.columns = df.columns.str.strip()
df['Classes']=df['Classes'].map({'fire':1,'not fire':0})
print(df['Classes'].value_counts())
df.to_csv('Bejaia-Region-Dataset_Cleaned.csv', index=False)