"""
Exploration 第五步：滚动窗口相关曲线

全局交叉相关几乎全部不显著。现在检验：
是否存在某些特定时段，关系特别强？

用30天滑动窗口计算相关系数随时间的变化。
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

SAVE_DIR = '/home/qqq/ADS_pandas_new/graphs/exploration'

# =============================================
# 加载差分数据
# =============================================
df = pd.read_csv(f'{SAVE_DIR}/exp2_differenced_full.csv', index_col='date', parse_dates=True)
print(f"差分数据: {len(df)} 天")

# =============================================
# 滚动窗口相关
# =============================================
WINDOW = 30  # 30天窗口

# 我们关注的三对：哪个变量和 sentiment_mean 的关系最强？
pairs = [
    ('StringencyIndex_Average', 'sentiment_mean', 'Stringency', '#E91E63'),
    ('daily_new_cases',         'sentiment_mean', 'Daily New Cases', '#9C27B0'),
    ('daily_new_deaths',        'sentiment_mean', 'Daily New Deaths', '#795548'),
]

# 关键事件标注
EVENTS = [
    ('2020-03-23', 'UK first lockdown'),
    ('2020-12-08', 'UK vaccination starts'),
    ('2021-01-06', 'UK third lockdown'),
    ('2021-07-19', 'UK Freedom Day'),
    ('2022-02-24', 'UK lifts all restrictions'),
]

# =============================================
# 计算滚动相关
# =============================================
rolling_corrs = {}
for x_var, y_var, label, _ in pairs:
    rc = df[x_var].rolling(WINDOW).corr(df[y_var])
    rolling_corrs[label] = rc
    
    # 打印统计
    print(f"\n{label} ↔ Sentiment Mean ({WINDOW}-day rolling):")
    print(f"  均值: {rc.mean():.4f}")
    print(f"  最大值: {rc.max():.4f} ({rc.idxmax().date()})")
    print(f"  最小值: {rc.min():.4f} ({rc.idxmin().date()})")
    
    # 找出 |r| > 0.3 的时段
    strong = rc[rc.abs() > 0.3]
    if len(strong) > 0:
        print(f"  |r| > 0.3 的天数: {len(strong)}")
        # 找连续强相关的起止日期
        strong_dates = strong.index
        gaps = (strong_dates[1:] - strong_dates[:-1]).days
        segments = []
        seg_start = strong_dates[0]
        for k in range(len(gaps)):
            if gaps[k] > 3:  # 超过3天间隔视为新段
                segments.append((seg_start, strong_dates[k]))
                seg_start = strong_dates[k + 1]
        segments.append((seg_start, strong_dates[-1]))
        
        print(f"  强相关时段:")
        for s, e in segments:
            avg_r = rc[s:e].mean()
            print(f"    {s.date()} ~ {e.date()} (avg r = {avg_r:.3f})")
    else:
        print(f"  没有 |r| > 0.3 的时段")

# =============================================
# 图1：三条滚动相关曲线叠在一起
# =============================================
fig, ax = plt.subplots(figsize=(14, 5))

for x_var, y_var, label, color in pairs:
    ax.plot(rolling_corrs[label].index, rolling_corrs[label],
            color=color, linewidth=1.2, label=label, alpha=0.8)

# 显著性参考线
ax.axhline(y=0, color='black', linewidth=0.5)
ax.axhline(y=0.3, color='gray', linestyle=':', linewidth=0.7, alpha=0.5, label='|r| = 0.3')
ax.axhline(y=-0.3, color='gray', linestyle=':', linewidth=0.7, alpha=0.5)

# 事件标注
for date_str, event_label in EVENTS:
    event_date = pd.to_datetime(date_str)
    ax.axvline(x=event_date, color='gray', linestyle='--', linewidth=0.6, alpha=0.4)
    ax.text(event_date, ax.get_ylim()[1] * 0.9, event_label,
            rotation=90, fontsize=7, color='dimgray', ha='right', va='top')

ax.set_ylabel('Pearson Correlation', fontsize=11)
ax.set_title(f'{WINDOW}-Day Rolling Correlation with Sentiment Mean\n'
             f'(Higher absolute value = stronger association in that period)',
             fontsize=13, fontweight='bold')
ax.legend(loc='lower right', fontsize=9)
ax.grid(True, alpha=0.2)
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/exp5_rolling_correlation.png', dpi=150)
plt.close()
print(f"\n✅ 图已保存: {SAVE_DIR}/exp5_rolling_correlation.png")

# =============================================
# 图2：分开画三行，每行一对，更清晰
# =============================================
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

for i, (x_var, y_var, label, color) in enumerate(pairs):
    ax = axes[i]
    rc = rolling_corrs[label]
    
    # 正相关填充红色，负相关填充蓝色
    ax.fill_between(rc.index, rc, 0, where=(rc > 0), color=color, alpha=0.3)
    ax.fill_between(rc.index, rc, 0, where=(rc < 0), color='steelblue', alpha=0.3)
    ax.plot(rc.index, rc, color=color, linewidth=1, alpha=0.8)
    
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.axhline(y=0.3, color='red', linestyle=':', linewidth=0.7, alpha=0.5)
    ax.axhline(y=-0.3, color='red', linestyle=':', linewidth=0.7, alpha=0.5)
    ax.set_ylabel(f'{label}\nvs Sentiment', fontsize=9)
    ax.set_ylim(-0.6, 0.6)
    ax.grid(True, alpha=0.2)
    
    # 事件标注
    for date_str, event_label in EVENTS:
        event_date = pd.to_datetime(date_str)
        ax.axvline(x=event_date, color='gray', linestyle='--', linewidth=0.6, alpha=0.4)

axes[0].set_title(f'{WINDOW}-Day Rolling Correlation with Sentiment Mean (by Variable)',
                  fontsize=13, fontweight='bold')
axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/exp5_rolling_correlation_detail.png', dpi=150)
plt.close()
print(f"✅ 图已保存: {SAVE_DIR}/exp5_rolling_correlation_detail.png")