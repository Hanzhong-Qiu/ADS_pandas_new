"""
Exploration 第二步：ADF平稳性检验 + 差分

对 full_analysis_data.csv 中的所有核心变量做ADF检验，
不平稳的做一阶差分，保存差分后的数据供后续使用。

前置条件：full_analysis_data.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
import os

os.makedirs('figures', exist_ok=True)

# =============================================
# 加载数据
# =============================================
df = pd.read_csv('/home/qqq/ADS_pandas_new/.csv/new/full_analysis_data.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').set_index('date')

# 要检验的变量（6个核心 + 7个子指标 = 13个）
CORE_VARS = ['sentiment_mean', 'sentiment_volatility', 'tweet_volume',
             'StringencyIndex_Average', 'daily_new_cases', 'daily_new_deaths']

SUB_VARS = ['C1_School_closing', 'C2_Workplace_closing',
            'C4_Restrictions_gatherings', 'C6_Stay_at_home',
            'C7_Internal_movement', 'C8_International_travel',
            'H6_Facial_coverings']

ALL_VARS = CORE_VARS + SUB_VARS
print(f"数据: {len(df)} 天, 检验 {len(ALL_VARS)} 个变量")

# =============================================
# ADF 检验
# =============================================
def run_adf(series, name):
    result = adfuller(series.dropna(), autolag='AIC')
    return {
        'Variable': name,
        'ADF Statistic': round(result[0], 4),
        'p-value': round(result[1], 6),
        'Stationary': 'Yes' if result[1] < 0.05 else 'No'
    }

print("\n" + "=" * 60)
print("原始序列 ADF 检验")
print("=" * 60)

results_level = []
for var in ALL_VARS:
    res = run_adf(df[var], var)
    results_level.append(res)
    mark = "✅" if res['Stationary'] == 'Yes' else "❌"
    print(f"  {mark} {var:35s}  ADF={res['ADF Statistic']:8.4f}  p={res['p-value']:.6f}")

# 一阶差分
df_diff = df[ALL_VARS].diff().dropna()

print("\n" + "=" * 60)
print("一阶差分后 ADF 检验")
print("=" * 60)

results_diff = []
for var in ALL_VARS:
    res = run_adf(df_diff[var], f"Δ{var}")
    results_diff.append(res)
    mark = "✅" if res['Stationary'] == 'Yes' else "❌"
    print(f"  {mark} Δ{var:33s}  ADF={res['ADF Statistic']:8.4f}  p={res['p-value']:.6f}")

all_ok = all(r['Stationary'] == 'Yes' for r in results_diff)
print(f"\n{'✅ 全部平稳' if all_ok else '⚠️ 仍有非平稳变量'}")

# 保存结果表
adf_all = pd.DataFrame(results_level + results_diff)
adf_all.to_csv('/home/qqq/ADS_pandas_new/graphs/exploration/exp2_adf_results.csv', index=False)

# =============================================
# 对比图：只画6个核心变量（子指标太多会挤）
# =============================================
labels = ['Sentiment Mean', 'Sentiment Volatility', 'Tweet Volume',
          'Stringency Index', 'Daily New Cases', 'Daily New Deaths']
colors = ['#2196F3', '#FF9800', '#4CAF50', '#E91E63', '#9C27B0', '#795548']

fig, axes = plt.subplots(6, 2, figsize=(14, 16))

for i, (var, label, color) in enumerate(zip(CORE_VARS, labels, colors)):
    # 左列：原始
    axes[i][0].plot(df.index, df[var], color=color, linewidth=0.7)
    p_lv = results_level[i]['p-value']
    tag = 'Stationary' if p_lv < 0.05 else 'Non-stationary'
    axes[i][0].set_title(f"p={p_lv:.4f} → {tag}",
                         fontsize=8, color='green' if p_lv < 0.05 else 'red')
    axes[i][0].set_ylabel(label, fontsize=8)
    axes[i][0].grid(True, alpha=0.3)

    # 右列：差分
    axes[i][1].plot(df_diff.index, df_diff[var], color=color, linewidth=0.7)
    axes[i][1].axhline(0, color='black', linewidth=0.4, linestyle='--')
    p_df = results_diff[i]['p-value']
    axes[i][1].set_title(f"p={p_df:.6f} → Stationary",
                         fontsize=8, color='green')
    axes[i][1].grid(True, alpha=0.3)

axes[0][0].set_title('Original (Level)\n' + axes[0][0].get_title())
axes[0][1].set_title('First Difference (Δ)\n' + axes[0][1].get_title())
plt.suptitle('ADF Stationarity Test: Original vs Differenced (6 Core Variables)',
             fontsize=13, fontweight='bold', y=1.0)
plt.tight_layout()
plt.savefig('/home/qqq/ADS_pandas_new/graphs/exploration/exp2_adf_comparison.png', dpi=150)
plt.close()
print("\n✅ 图已保存: figures/exp2_adf_comparison.png")

# 保存差分数据
df_diff.to_csv('/home/qqq/ADS_pandas_new/graphs/exploration/exp2_differenced_full.csv')
print("✅ 差分数据已保存: figures/exp2_differenced_full.csv")
print(f"   {len(df_diff)} 行 × {len(df_diff.columns)} 列")