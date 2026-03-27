import pandas as pd
df = pd.read_csv("forestfire.csv")
df = df.dropna()
df = df.select_dtypes(include=['float64', 'int64'])
df.to_csv("forestfire.csv", index=False)
