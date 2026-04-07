"""
Exploration 第五步优化：多窗口滚动相关对比

对 Stringency vs Sentiment Mean 分别用 15天、30天、45天 窗口
计算滚动相关，观察哪些高相关时段在不同窗口下都成立。
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
# 三种窗口
# =============================================
WINDOWS = [15, 30, 45]
colors = ['#2196F3', '#E91E63', '#4CAF50']
x_var = 'StringencyIndex_Average'
y_var = 'sentiment_mean'

# 关键事件
EVENTS = [
    ('2020-03-23', 'UK first\nlockdown'),
    ('2020-12-08', 'Vaccination\nstarts'),
    ('2021-01-06', 'UK third\nlockdown'),
    ('2021-07-19', 'Freedom\nDay'),
    ('2022-02-24', 'Restrictions\nlifted'),
]

# =============================================
# 计算并统计
# =============================================
rolling_results = {}

for w in WINDOWS:
    rc = df[x_var].rolling(w).corr(df[y_var])
    rolling_results[w] = rc

    strong = rc[rc.abs() > 0.3].dropna()
    print(f"\n窗口 = {w} 天:")
    print(f"  均值: {rc.mean():.4f}")
    print(f"  最大: {rc.max():.3f} ({rc.idxmax().date()})")
    print(f"  最小: {rc.min():.3f} ({rc.idxmin().date()})")
    print(f"  |r| > 0.3 的天数: {len(strong)}")
    print(f"  |r| > 0.4 的天数: {len(rc[rc.abs() > 0.4])}")

# =============================================
# 绘图：三条曲线叠在一起
# =============================================
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

for i, (w, color) in enumerate(zip(WINDOWS, colors)):
    ax = axes[i]
    rc = rolling_results[w]

    ax.fill_between(rc.index, rc, 0, where=(rc > 0), color=color, alpha=0.3)
    ax.fill_between(rc.index, rc, 0, where=(rc < 0), color='steelblue', alpha=0.3)
    ax.plot(rc.index, rc, color=color, linewidth=1, alpha=0.8)

    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.axhline(y=0.3, color='red', linestyle=':', linewidth=0.7, alpha=0.5)
    ax.axhline(y=-0.3, color='red', linestyle=':', linewidth=0.7, alpha=0.5)
    ax.set_ylabel(f'{w}-day\nwindow', fontsize=10, fontweight='bold')
    ax.set_ylim(-0.7, 0.7)
    ax.grid(True, alpha=0.2)

    for date_str, label in EVENTS:
        event_date = pd.to_datetime(date_str)
        ax.axvline(x=event_date, color='gray', linestyle='--', linewidth=0.6, alpha=0.4)
        if i == 0:
            ax.text(event_date, 0.65, label, fontsize=7, color='dimgray',
                    ha='center', va='top')

axes[0].set_title('Stringency vs Sentiment Mean: Rolling Correlation at Different Windows\n'
                  '(Red dotted = |r| > 0.3 threshold)',
                  fontsize=13, fontweight='bold')
axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/exp5b_multi_window.png', dpi=150)
plt.close()
print(f"\n✅ 图已保存: {SAVE_DIR}/exp5b_multi_window.png")

# =============================================
# 找出三个窗口都一致的强相关时段
# =============================================
print("\n" + "=" * 60)
print("三个窗口都 |r| > 0.3 的时段（最稳健的发现）")
print("=" * 60)

# 对齐三个窗口的结果
aligned = pd.DataFrame({
    f'r_{w}d': rolling_results[w] for w in WINDOWS
}).dropna()

# 三个都超过0.3的天
all_strong = aligned[(aligned.abs() > 0.3).all(axis=1)]
print(f"\n三个窗口同时 |r| > 0.3 的天数: {len(all_strong)}")

if len(all_strong) > 0:
    # 找连续段
    dates = all_strong.index
    segments = []
    seg_start = dates[0]
    for k in range(1, len(dates)):
        if (dates[k] - dates[k-1]).days > 3:
            segments.append((seg_start, dates[k-1]))
            seg_start = dates[k]
    segments.append((seg_start, dates[-1]))

    print("\n稳健的强相关时段:")
    for s, e in segments:
        avg_rs = [aligned.loc[s:e, f'r_{w}d'].mean() for w in WINDOWS]
        duration = (e - s).days + 1
        print(f"  {s.date()} ~ {e.date()} ({duration}天)")
        for w, avg_r in zip(WINDOWS, avg_rs):
            print(f"    {w}d窗口 avg r = {avg_r:.3f}")
else:
    # 放宽条件：至少两个窗口都超过0.3
    two_strong = aligned[(aligned.abs() > 0.3).sum(axis=1) >= 2]
    print(f"\n至少两个窗口 |r| > 0.3 的天数: {len(two_strong)}")
    if len(two_strong) > 0:
        dates = two_strong.index
        segments = []
        seg_start = dates[0]
        for k in range(1, len(dates)):
            if (dates[k] - dates[k-1]).days > 3:
                segments.append((seg_start, dates[k-1]))
                seg_start = dates[k]
        segments.append((seg_start, dates[-1]))

        print("\n较稳健的强相关时段（至少2个窗口一致）:")
        for s, e in segments[:10]:  # 最多显示10段
            duration = (e - s).days + 1
            avg_rs = [aligned.loc[s:e, f'r_{w}d'].mean() for w in WINDOWS]
            print(f"  {s.date()} ~ {e.date()} ({duration}天)")
            for w, avg_r in zip(WINDOWS, avg_rs):
                print(f"    {w}d窗口 avg r = {avg_r:.3f}")