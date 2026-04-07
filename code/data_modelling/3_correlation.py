"""
步骤 3：交叉相关分析（Cross-Correlation）

读取步骤2输出的差分数据，计算变量对之间的交叉相关，
找出是否存在先导-滞后关系。

前置条件：运行过 step2_adf.py，生成了 figures/step2_differenced_data.csv
依赖：pip install pandas numpy matplotlib scipy
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


# =============================================
# 加载差分数据
# =============================================
df = pd.read_csv('/home/qqq/ADS_pandas_new/graphs/modelling/step2_differenced_data.csv', index_col='date', parse_dates=True)
print(f"差分数据: {len(df)} 天, 列: {list(df.columns)}")


# =============================================
# 定义交叉相关函数
# =============================================
def cross_corr(x, y, max_lag=14):
    """
    计算 x 和 y 之间在不同滞后下的Pearson相关系数。

    lag > 0 : x 领先 y（x先变化，lag天后y才跟着变）
    lag < 0 : y 领先 x（y先变化，|lag|天后x才跟着变）
    lag = 0 : 同一天的相关
    """
    x_arr = (x - x.mean()) / x.std()
    y_arr = (y - y.mean()) / y.std()
    n = len(x_arr)

    lags = list(range(-max_lag, max_lag + 1))
    corrs = []

    for lag in lags:
        if lag >= 0:
            r = np.corrcoef(x_arr[:n - lag], y_arr[lag:])[0, 1]
        else:
            r = np.corrcoef(x_arr[-lag:], y_arr[:n + lag])[0, 1]
        corrs.append(r)

    return lags, corrs


# =============================================
# 计算四组核心变量对
# =============================================
# 每一组的含义：
#   (x变量, y变量, 图表标题, 我们希望回答的问题)
pairs = [
    ('StringencyIndex_Average', 'sentiment_mean',
     'Stringency vs Sentiment Mean',
     '政策变化是否领先于情感变化？'),

    ('StringencyIndex_Average', 'sentiment_volatility',
     'Stringency vs Sentiment Volatility',
     '政策变化是否导致舆论分裂加剧？'),

    ('StringencyIndex_Average', 'tweet_volume',
     'Stringency vs Tweet Volume',
     '政策变化是否引发更多讨论？'),

    ('sentiment_mean', 'tweet_volume',
     'Sentiment Mean vs Tweet Volume',
     '情感变化是否驱动讨论量？'),
]

MAX_LAG = 14
n = len(df)
ci = 1.96 / np.sqrt(n)  # 95%置信区间

print(f"\n样本量: {n}")
print(f"95%置信区间: ±{ci:.4f}")
print(f"超过此范围的相关系数才算统计显著\n")


# =============================================
# 绘图
# =============================================
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
axes = axes.flatten()

for idx, (x_var, y_var, title, question) in enumerate(pairs):
    lags, corrs = cross_corr(df[x_var].values, df[y_var].values, MAX_LAG)

    ax = axes[idx]

    # 画柱状图
    bar_colors = ['#E91E63' if abs(c) > ci else '#90CAF9' for c in corrs]
    ax.bar(lags, corrs, width=0.7, color=bar_colors, alpha=0.8)

    # 置信区间线
    ax.axhline(y=ci, color='red', linestyle='--', linewidth=0.8, alpha=0.6)
    ax.axhline(y=-ci, color='red', linestyle='--', linewidth=0.8, alpha=0.6)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.axvline(x=0, color='gray', linewidth=0.5, linestyle=':')

    # 找峰值（绝对值最大的那个lag）
    peak_idx = np.argmax(np.abs(corrs))
    peak_lag = lags[peak_idx]
    peak_r = corrs[peak_idx]

    # 标注峰值
    ax.annotate(
        f'Peak: lag={peak_lag}, r={peak_r:.4f}',
        xy=(peak_lag, peak_r),
        xytext=(peak_lag + (4 if peak_lag < 5 else -8), peak_r * 1.3),
        arrowprops=dict(arrowstyle='->', color='darkred', lw=1.5),
        fontsize=9, color='darkred', fontweight='bold'
    )

    ax.set_xlabel('Lag (days)      ← Y leads | X leads →')
    ax.set_ylabel('Correlation')
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlim(-MAX_LAG - 1, MAX_LAG + 1)
    ax.grid(True, alpha=0.2)

    # 打印结果
    sig = "显著 ✅" if abs(peak_r) > ci else "不显著 ❌"
    if peak_lag > 0:
        direction = f"{x_var.split('_')[0]} 领先 {y_var.split('_')[0]} {peak_lag}天"
    elif peak_lag < 0:
        direction = f"{y_var.split('_')[0]} 领先 {x_var.split('_')[0]} {abs(peak_lag)}天"
    else:
        direction = "同步变化"

    print(f"{title}:")
    print(f"  峰值: lag={peak_lag}, r={peak_r:.4f} ({sig})")
    print(f"  含义: {direction}")
    print(f"  问题: {question}")
    print()

plt.suptitle(
    f'Cross-Correlation Analysis (Differenced Series, 95% CI = ±{ci:.3f})\n'
    f'Pink bars = statistically significant',
    fontsize=13, fontweight='bold'
)
plt.tight_layout()
plt.savefig('/home/qqq/ADS_pandas_new/graphs/modelling/step3_cross_correlation.png', dpi=150)
plt.close()
print("✅ 图已保存: figures/step3_cross_correlation.png")