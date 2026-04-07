"""
步骤 2：平稳性检验（ADF Test）& 差分

前置条件：enriched_research_data.csv 在同一目录下
依赖：pip install pandas numpy matplotlib statsmodels
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
import os


# --- 加载数据，截取分析窗口 ---
df = pd.read_csv('/home/qqq/ADS_pandas_new/.csv/enriched_research_data.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

mask = (df['date'] >= '2020-03-19') & (df['date'] <= '2022-03-31')
df_analysis = df.loc[mask].copy().set_index('date')

VARS = ['sentiment_mean', 'sentiment_volatility',
        'tweet_volume', 'StringencyIndex_Average']
df_analysis = df_analysis[VARS].dropna()

print(f"分析窗口: {df_analysis.index.min().date()} 至 {df_analysis.index.max().date()}")
print(f"有效天数: {len(df_analysis)}")


# =============================================
# ADF 检验
# =============================================
print("\n" + "=" * 60)
print("ADF 平稳性检验")
print("=" * 60)

def run_adf(series, name):
    """
    ADF检验的零假设: 序列存在单位根（= 非平稳）
    p < 0.05 → 拒绝零假设 → 平稳
    p >= 0.05 → 不能拒绝 → 非平稳，需要差分
    """
    result = adfuller(series.dropna(), autolag='AIC')
    return {
        'Variable':      name,
        'ADF Statistic': round(result[0], 4),
        'p-value':       round(result[1], 6),
        'Lags Used':     result[2],
        'Stationary':    'Yes' if result[1] < 0.05 else 'No'
    }

# 原始序列
print("\n原始序列:")
results_level = []
for var in VARS:
    res = run_adf(df_analysis[var], var)
    results_level.append(res)
    mark = "✅ 平稳" if res['Stationary'] == 'Yes' else "❌ 非平稳"
    print(f"  {var:30s}  ADF={res['ADF Statistic']:8.4f}  p={res['p-value']:.6f}  {mark}")

# 一阶差分
df_diff = df_analysis[VARS].diff().dropna()

print("\n一阶差分后:")
results_diff = []
for var in VARS:
    res = run_adf(df_diff[var], f"Δ{var}")
    results_diff.append(res)
    mark = "✅ 平稳" if res['Stationary'] == 'Yes' else "❌ 非平稳"
    print(f"  Δ{var:28s}  ADF={res['ADF Statistic']:8.4f}  p={res['p-value']:.6f}  {mark}")

all_ok = all(r['Stationary'] == 'Yes' for r in results_diff)
print(f"\n{'✅ 全部平稳，后续使用差分数据' if all_ok else '⚠️ 仍有非平稳变量'}")

# 保存结果表
adf_all = pd.DataFrame(results_level + results_diff)
adf_all.to_csv('/home/qqq/ADS_pandas_new/graphs/modelling/step2_adf_results.csv', index=False)
print("表格已保存: figures/step2_adf_results.csv")


# =============================================
# 对比图：原始 vs 差分
# =============================================
labels = ['Sentiment Mean', 'Sentiment Volatility',
          'Tweet Volume', 'Stringency Index']
colors = ['#2196F3', '#FF9800', '#4CAF50', '#E91E63']

fig, axes = plt.subplots(4, 2, figsize=(14, 12))

for i, (var, label) in enumerate(zip(VARS, labels)):
    # 左列：原始
    axes[i][0].plot(df_analysis.index, df_analysis[var], color=colors[i], linewidth=0.8)
    axes[i][0].set_ylabel(label, fontsize=9)
    p_lv = results_level[i]['p-value']
    axes[i][0].set_title(f"p={p_lv:.4f} → {'Stationary' if p_lv<0.05 else 'Non-stationary'}",
                         fontsize=9, color='green' if p_lv<0.05 else 'red')
    axes[i][0].grid(True, alpha=0.3)

    # 右列：差分
    axes[i][1].plot(df_diff.index, df_diff[var], color=colors[i], linewidth=0.8)
    axes[i][1].axhline(0, color='black', linewidth=0.5, linestyle='--')
    p_df = results_diff[i]['p-value']
    axes[i][1].set_title(f"p={p_df:.6f} → Stationary", fontsize=9, color='green')
    axes[i][1].grid(True, alpha=0.3)

axes[0][0].set_title('Original (Level)\n' + axes[0][0].get_title())
axes[0][1].set_title('First Difference (Δ)\n' + axes[0][1].get_title())
plt.suptitle('ADF Stationarity Test: Original vs Differenced',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('/home/qqq/ADS_pandas_new/graphs/modelling/step2_level_vs_diff.png', dpi=150)
plt.close()
print("图已保存: figures/step2_level_vs_diff.png")

# 保存差分数据供后续使用
df_diff.to_csv('/home/qqq/ADS_pandas_new/graphs/modelling/step2_differenced_data.csv')
print("差分数据已保存: figures/step2_differenced_data.csv")