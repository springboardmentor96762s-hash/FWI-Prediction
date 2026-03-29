import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv('Bejaia Region Dataset.csv')
df.columns = df.columns.str.strip()
df['Classes'] = df['Classes'].map({'not fire': 0, 'fire': 1})
corrmatrix=df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corrmatrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()
print(df.cov())