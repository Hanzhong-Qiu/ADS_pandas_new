"""
步骤 5：Granger因果检验

检验政策变化是否在统计意义上"先导"情感变化（以及反方向）。

前置条件：
  - figures/step2_differenced_data.csv（步骤2输出）
  - figures/step4_chosen_lag.txt（步骤4输出）
依赖：pip install pandas numpy matplotlib statsmodels
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import grangercausalitytests
import os

os.makedirs('figures', exist_ok=True)

# =============================================
# 加载数据和滞后设置
# =============================================
df = pd.read_csv('/home/qqq/ADS_pandas_new/graphs/modelling/step2_differenced_data.csv', index_col='date', parse_dates=True)

with open('figures/step4_chosen_lag.txt', 'r') as f:
    CHOSEN_LAG = int(f.read().strip())

VARS = ['sentiment_mean', 'sentiment_volatility',
        'tweet_volume', 'StringencyIndex_Average']

print(f"差分数据: {len(df)} 天")
print(f"选定滞后: {CHOSEN_LAG}")

# =============================================
# Granger 检验函数
# =============================================
def granger_test(data, cause, effect, max_lag):
    """
    检验 cause 是否 Granger-cause effect。

    statsmodels 要求输入的 DataFrame 第一列是 effect，第二列是 cause。
    返回每个滞后的 F检验 p值。
    """
    test_df = data[[effect, cause]].dropna()
    p_values = {}

    try:
        results = grangercausalitytests(test_df, maxlag=max_lag, verbose=False)
        for lag in range(1, max_lag + 1):
            p_values[lag] = round(results[lag][0]['ssr_ftest'][1], 6)
    except Exception as e:
        print(f"  ⚠️ {cause} → {effect} 检验失败: {e}")
        for lag in range(1, max_lag + 1):
            p_values[lag] = np.nan

    return p_values


# =============================================
# 核心分析：Stringency ↔ Sentiment（双向 × 多滞后）
# =============================================
print("\n" + "=" * 60)
print("核心分析：Stringency ↔ Sentiment Mean")
print("=" * 60)

SCAN_MAX = 14  # 扫描lag 1到14

# 方向1：政策 → 情感
print(f"\n方向1: StringencyIndex → sentiment_mean")
print(f"  问: 过去的政策变化能否帮助预测今天的情感变化？")
p_str_to_sent = granger_test(df, 'StringencyIndex_Average', 'sentiment_mean', SCAN_MAX)

for lag, p in p_str_to_sent.items():
    sig = "***" if p < 0.01 else "** " if p < 0.05 else "*  " if p < 0.1 else "   "
    print(f"  Lag {lag:2d}: p = {p:.6f} {sig}")

# 方向2：情感 → 政策
print(f"\n方向2: sentiment_mean → StringencyIndex")
print(f"  问: 过去的情感变化能否帮助预测今天的政策变化？")
p_sent_to_str = granger_test(df, 'sentiment_mean', 'StringencyIndex_Average', SCAN_MAX)

for lag, p in p_sent_to_str.items():
    sig = "***" if p < 0.01 else "** " if p < 0.05 else "*  " if p < 0.1 else "   "
    print(f"  Lag {lag:2d}: p = {p:.6f} {sig}")

# 在选定滞后下的结论
p1 = p_str_to_sent[CHOSEN_LAG]
p2 = p_sent_to_str[CHOSEN_LAG]
print(f"\n在选定滞后 lag={CHOSEN_LAG} 下:")
print(f"  Stringency → Sentiment: p = {p1:.4f} → {'显著 ✅' if p1 < 0.05 else '不显著 ❌'}")
print(f"  Sentiment → Stringency: p = {p2:.4f} → {'显著 ✅' if p2 < 0.05 else '不显著 ❌'}")


# =============================================
# 完整 4×4 Granger矩阵
# =============================================
print(f"\n" + "=" * 60)
print(f"完整Granger因果矩阵 (lag = {CHOSEN_LAG})")
print("=" * 60)
print("读法: 格子里的数字是p值。p < 0.05 说明【列变量】Granger-cause【行变量】")

matrix = pd.DataFrame(index=VARS, columns=VARS, dtype=float)

for cause in VARS:
    for effect in VARS:
        if cause == effect:
            matrix.loc[effect, cause] = np.nan
        else:
            result = granger_test(df, cause, effect, CHOSEN_LAG)
            matrix.loc[effect, cause] = result[CHOSEN_LAG]

# 打印矩阵
print()
for effect in VARS:
    row_str = f"  {effect:28s}"
    for cause in VARS:
        val = matrix.loc[effect, cause]
        if np.isnan(val):
            row_str += "     -    "
        else:
            sig = "***" if val < 0.01 else "** " if val < 0.05 else "*  " if val < 0.1 else "   "
            row_str += f"  {val:.4f}{sig}"
    print(row_str)

# 找出所有显著的关系
print(f"\n显著的Granger因果关系 (p < 0.05):")
found_any = False
for cause in VARS:
    for effect in VARS:
        if cause == effect:
            continue
        val = matrix.loc[effect, cause]
        if not np.isnan(val) and val < 0.05:
            print(f"  ✅ {cause} → {effect} (p = {val:.4f})")
            found_any = True
if not found_any:
    print("  ❌ 没有发现显著的Granger因果关系")

matrix.to_csv('/home/qqq/ADS_pandas_new/graphs/modelling/step5_granger_matrix.csv')


# =============================================
# 图1：双向Granger p值随滞后变化
# =============================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

lags_list = list(range(1, SCAN_MAX + 1))

# 方向1
axes[0].plot(lags_list, [p_str_to_sent[l] for l in lags_list],
             'o-', color='#E91E63', markersize=6)
axes[0].axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='p = 0.05')
axes[0].axhline(y=0.01, color='darkred', linestyle=':', alpha=0.7, label='p = 0.01')
axes[0].axvline(x=CHOSEN_LAG, color='green', linestyle='--', alpha=0.5,
                label=f'Chosen lag = {CHOSEN_LAG}')
axes[0].set_xlabel('Lag (days)')
axes[0].set_ylabel('p-value')
axes[0].set_title('Stringency → Sentiment Mean', fontweight='bold')
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)
axes[0].set_xticks(lags_list)

# 方向2
axes[1].plot(lags_list, [p_sent_to_str[l] for l in lags_list],
             's-', color='#2196F3', markersize=6)
axes[1].axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='p = 0.05')
axes[1].axhline(y=0.01, color='darkred', linestyle=':', alpha=0.7, label='p = 0.01')
axes[1].axvline(x=CHOSEN_LAG, color='green', linestyle='--', alpha=0.5,
                label=f'Chosen lag = {CHOSEN_LAG}')
axes[1].set_xlabel('Lag (days)')
axes[1].set_ylabel('p-value')
axes[1].set_title('Sentiment Mean → Stringency', fontweight='bold')
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)
axes[1].set_xticks(lags_list)

plt.suptitle('Granger Causality: p-values Across Lags\n'
             '(Below red dashed line = statistically significant)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('/home/qqq/ADS_pandas_new/graphs/modelling/step5_granger_by_lag.png', dpi=150)
plt.close()
print("\n✅ 图已保存: figures/step5_granger_by_lag.png")


# =============================================
# 图2：4×4 Granger矩阵热力图
# =============================================
fig, ax = plt.subplots(figsize=(8, 7))

matrix_numeric = matrix.astype(float).values

# 用颜色表示显著程度：p越小颜色越深
# 用 -log10(p) 变换，这样0.05对应1.3，0.01对应2，越显著数值越大
log_matrix = np.where(np.isnan(matrix_numeric), np.nan,
                      -np.log10(np.clip(matrix_numeric, 1e-10, 1)))

im = ax.imshow(log_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=3)
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('-log10(p-value)\nHigher = More Significant', fontsize=9)

# 在格子里标注p值和显著性星号
short_names = ['Sent.\nMean', 'Sent.\nVolatility', 'Tweet\nVolume', 'Stringency\nIndex']

for i in range(4):
    for j in range(4):
        if i == j:
            ax.text(j, i, '-', ha='center', va='center', fontsize=11, color='gray')
        else:
            val = matrix_numeric[i, j]
            sig = '***' if val < 0.01 else '**' if val < 0.05 else '*' if val < 0.1 else ''
            text_color = 'white' if log_matrix[i, j] > 1.5 else 'black'
            ax.text(j, i, f'{val:.3f}\n{sig}',
                    ha='center', va='center', fontsize=9, color=text_color)

ax.set_xticks(range(4))
ax.set_xticklabels(short_names, fontsize=9)
ax.set_yticks(range(4))
ax.set_yticklabels(short_names, fontsize=9)
ax.set_xlabel('Cause →', fontsize=11)
ax.set_ylabel('← Effect', fontsize=11)
ax.set_title(f'Granger Causality Matrix (Lag = {CHOSEN_LAG})\n'
             f'*** p<0.01  ** p<0.05  * p<0.1',
             fontweight='bold', fontsize=12)

plt.tight_layout()
plt.savefig('/home/qqq/ADS_pandas_new/graphs/modelling/step5_granger_matrix.png', dpi=150)
plt.close()
print("✅ 图已保存: figures/step5_granger_matrix.png")