import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('/content/Cleaned_FWI_dataset.csv')

df.hist(bins=30, figsize=(20, 15), color='skyblue', edgecolor='black')
plt.suptitle('Histograms of All Features', fontsize=20)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('histograms.png', dpi=300, bbox_inches='tight')
plt.show()

corr = df.select_dtypes(include='number').corr()

fig, ax = plt.subplots(figsize=(14, 12))
cax = ax.matshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax, shrink=0.8)

ticks = np.arange(0, len(corr.columns), 1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(corr.columns, rotation=90)
ax.set_yticklabels(corr.columns)

for i in range(len(corr.columns)):
    for j in range(len(corr.columns)):
        ax.text(j, i, f'{corr.iloc[i, j]:.2f}',
                ha='center', va='center', color='black', fontsize=9)

plt.title('Correlation Heatmap', fontsize=20, pad=20)
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nTop correlations with FWI (your target):")
print(corr['FWI'].sort_values(ascending=False).round(3))

print("\nTop covariance with FWI:")
print(df.cov(numeric_only=True)['FWI'].sort_values(ascending=False).round(3))