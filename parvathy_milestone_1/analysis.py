import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
df=pd.read_csv('Bejaia Region Dataset.csv')
df.columns = df.columns.str.strip()
features = ['Temperature', 'RH', 'Ws', 'Rain', 'FWI', 'DMC', 'DC', 'ISI', 'BUI', 'FFMC']
hist_df = df[features]
hist_df.hist(figsize=(15, 10))
plt.title('FWI Analysis')
plt.show()