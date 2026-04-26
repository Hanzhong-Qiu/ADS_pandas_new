
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

                                               
SAVE_DIR = '/home/qqq/ADS_pandas_new/graphs/exploration'

df = pd.read_csv(f'{SAVE_DIR}/exp2_differenced_full.csv', index_col='date', parse_dates=True)

                                 
CORE = ['sentiment_mean', 'sentiment_volatility', 'tweet_volume',
        'StringencyIndex_Average', 'daily_new_cases', 'daily_new_deaths']

df_core = df[CORE].dropna()
print(f"差分数据: {len(df_core)} 天, {len(CORE)} 个变量")

                                               
corr = df_core.corr()

              
print("\n差分后 Pearson 相关性矩阵:")
print(corr.round(3).to_string())

                   
print("\n相关性排名（绝对值从高到低）:")
pairs = []
for i in range(len(CORE)):
    for j in range(i + 1, len(CORE)):
        r = corr.iloc[i, j]
        pairs.append((CORE[i], CORE[j], r))
pairs.sort(key=lambda x: abs(x[2]), reverse=True)

for v1, v2, r in pairs:
    sig = "⭐" if abs(r) > 0.1 else ""
    print(f"  {v1:28s} ↔ {v2:28s}  r = {r:+.4f}  {sig}")

                                               
short_names = ['Sentiment\nMean', 'Sentiment\nVolatility', 'Tweet\nVolume',
               'Stringency\nIndex', 'Daily New\nCases', 'Daily New\nDeaths']

fig, ax = plt.subplots(figsize=(9, 7))

mask = np.triu(np.ones_like(corr, dtype=bool), k=1)          

sns.heatmap(corr, mask=mask, annot=True, fmt='.3f', cmap='RdBu_r',
            center=0, vmin=-0.5, vmax=0.5, square=True,
            linewidths=0.5, linecolor='white',
            xticklabels=short_names, yticklabels=short_names,
            annot_kws={'size': 11}, ax=ax)

ax.set_title('Correlation Matrix: Differenced Core Variables\n'
             '(Values close to 0 = weak linear association)',
             fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/exp3_correlation_heatmap.png', dpi=150)
plt.close()
print(f"\n✅ 图已保存: {SAVE_DIR}/exp3_correlation_heatmap.png")