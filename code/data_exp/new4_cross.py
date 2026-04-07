"""
Exploration 第四步：交叉相关分析

第三步发现同期相关全部极弱。现在加入时间维度——
看是否存在"滞后几天"后才出现的关联。

重点对比：Stringency vs Sentiment 和 Cases/Deaths vs Sentiment，
看哪个的滞后关联更强。
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

SAVE_DIR = '/home/qqq/ADS_pandas_new/graphs/exploration'

# =============================================
# 加载差分数据
# =============================================
df = pd.read_csv(f'{SAVE_DIR}/exp2_differenced_full.csv', index_col='date', parse_dates=True)
print(f"差分数据: {len(df)} 天")

# =============================================
# 交叉相关函数
# =============================================
def cross_corr(x, y, max_lag=14):
    """
    lag > 0 : x 领先 y（x先变，lag天后y跟着变）
    lag < 0 : y 领先 x
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
# 核心变量对
# =============================================
pairs = [
    ('StringencyIndex_Average', 'sentiment_mean',
     'Stringency → Sentiment Mean'),
    ('daily_new_cases', 'sentiment_mean',
     'Daily New Cases → Sentiment Mean'),
    ('daily_new_deaths', 'sentiment_mean',
     'Daily New Deaths → Sentiment Mean'),
    ('StringencyIndex_Average', 'sentiment_volatility',
     'Stringency → Sentiment Volatility'),
    ('daily_new_cases', 'sentiment_volatility',
     'Daily New Cases → Sentiment Volatility'),
    ('daily_new_deaths', 'sentiment_volatility',
     'Daily New Deaths → Sentiment Volatility'),
]

MAX_LAG = 14
n = len(df)
ci = 1.96 / np.sqrt(n)

print(f"样本量: {n}")
print(f"95%置信区间: ±{ci:.4f}\n")

# =============================================
# 绘图：3行2列（左列=Sentiment Mean，右列=Volatility）
# =============================================
fig, axes = plt.subplots(3, 2, figsize=(14, 11))

for idx, (x_var, y_var, title) in enumerate(pairs):
    row = idx % 3
    col = idx // 3
    ax = axes[row][col]

    lags, corrs = cross_corr(df[x_var].values, df[y_var].values, MAX_LAG)

    # 柱子颜色：显著的粉红，不显著的浅蓝
    bar_colors = ['#E91E63' if abs(c) > ci else '#90CAF9' for c in corrs]
    ax.bar(lags, corrs, width=0.7, color=bar_colors, alpha=0.8)

    # 置信区间
    ax.axhline(y=ci, color='red', linestyle='--', linewidth=0.7, alpha=0.5)
    ax.axhline(y=-ci, color='red', linestyle='--', linewidth=0.7, alpha=0.5)
    ax.axhline(y=0, color='black', linewidth=0.4)
    ax.axvline(x=0, color='gray', linewidth=0.4, linestyle=':')

    # 找峰值
    peak_idx = np.argmax(np.abs(corrs))
    peak_lag = lags[peak_idx]
    peak_r = corrs[peak_idx]

    ax.annotate(f'Peak: lag={peak_lag}\nr={peak_r:.4f}',
                xy=(peak_lag, peak_r),
                xytext=(peak_lag + (4 if peak_lag < 5 else -8), peak_r * 1.4),
                arrowprops=dict(arrowstyle='->', color='darkred', lw=1.2),
                fontsize=8, color='darkred', fontweight='bold')

    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_xlim(-MAX_LAG - 1, MAX_LAG + 1)
    ax.grid(True, alpha=0.2)

    if row == 2:
        ax.set_xlabel('Lag (days)    ← Y leads | X leads →')
    if col == 0:
        ax.set_ylabel('Correlation')

    # 打印
    sig = "显著 ✅" if abs(peak_r) > ci else "不显著 ❌"
    print(f"{title}:")
    print(f"  峰值: lag={peak_lag}, r={peak_r:.4f} ({sig})")

    # 也找第二大峰值（可能在反方向）
    corrs_copy = list(corrs)
    corrs_copy[peak_idx] = 0
    peak2_idx = np.argmax(np.abs(corrs_copy))
    peak2_lag = lags[peak2_idx]
    peak2_r = corrs[peak2_idx]
    if abs(peak2_r) > ci:
        print(f"  第二峰: lag={peak2_lag}, r={peak2_r:.4f} (显著 ✅)")
    print()

plt.suptitle(f'Cross-Correlation Analysis (Differenced, 95% CI = ±{ci:.3f})\n'
             f'Left: vs Sentiment Mean | Right: vs Sentiment Volatility',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/exp4_cross_correlation.png', dpi=150)
plt.close()
print(f"✅ 图已保存: {SAVE_DIR}/exp4_cross_correlation.png")