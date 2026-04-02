import pandas as pd
import matplotlib.pyplot as plt
import os

os.makedirs("output", exist_ok=True)

df = pd.read_csv("forestfire.csv")
df = df[(df != 0).all(axis=1)]
df = df[df['Temperature'] > 0]
df = df[df['RH'] > 0]
df = df[df['FFMC'] > 0]
df.hist()
plt.savefig("histograms.png")
import seaborn as sns
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True)
plt.savefig("output/correlation.png")
