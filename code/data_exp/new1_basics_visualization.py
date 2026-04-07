"""
Exploration 第一步：各变量时间序列总览图

"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

os.makedirs('figures', exist_ok=True)

# =============================================
# 加载数据
# =============================================
df = pd.read_csv('/home/qqq/ADS_pandas_new/.csv/new/full_analysis_data.csv')
df['date'] = pd.to_datetime(df['date'])
print(f"数据: {len(df)} 天, {df['date'].min().date()} 至 {df['date'].max().date()}")

# =============================================
# 定义关键事件（用于标注垂直线）
# =============================================
# 你可以根据需要增减事件
EVENTS = [
    ('2020-03-11', 'WHO declares\npandemic'),
    ('2020-03-23', 'UK first\nlockdown'),
    ('2020-12-08', 'UK starts\nvaccination'),
    ('2021-01-06', 'UK third\nlockdown'),
    ('2021-07-19', 'UK "Freedom\nDay"'),
    ('2022-02-24', 'UK lifts\nall restrictions'),
]

# =============================================
# 绘图
# =============================================
variables = [
    ('sentiment_mean',        'Sentiment Mean',      '#2196F3'),
    ('sentiment_volatility',  'Sentiment Volatility', '#FF9800'),
    ('tweet_volume',          'Tweet Volume',         '#4CAF50'),
    ('StringencyIndex_Average', 'Stringency Index',   '#E91E63'),
    ('daily_new_cases',       'Daily New Cases',      '#9C27B0'),
    ('daily_new_deaths',      'Daily New Deaths',     '#795548'),
]

fig, axes = plt.subplots(6, 1, figsize=(15, 18), sharex=True)

for i, (col, label, color) in enumerate(variables):
    ax = axes[i]

    # 原始数据（浅色细线）
    ax.plot(df['date'], df[col], color=color, alpha=0.35, linewidth=0.6)

    # 7天移动平均（深色粗线）
    ma7 = df[col].rolling(7).mean()
    ax.plot(df['date'], ma7, color=color, linewidth=1.8, label='7-day MA')

    ax.set_ylabel(label, fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.2)
    ax.legend(loc='upper right', fontsize=8)

    # 标注关键事件
    for date_str, event_label in EVENTS:
        event_date = pd.to_datetime(date_str)
        if event_date >= df['date'].min() and event_date <= df['date'].max():
            ax.axvline(x=event_date, color='gray', linestyle='--',
                       linewidth=0.7, alpha=0.6)
            # 只在第一行图上标注文字（避免重复）
            if i == 0:
                ax.text(event_date, ax.get_ylim()[1], event_label,
                        rotation=0, fontsize=7, color='dimgray',
                        ha='center', va='bottom')

# x轴格式
axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.xticks(rotation=45)

fig.suptitle('COVID-19 Time Series Overview: Sentiment, Policy & Pandemic Indicators',
             fontsize=14, fontweight='bold', y=1.0)

plt.tight_layout()
plt.savefig('/home/qqq/ADS_pandas_new/graphs/exploration/exp1_time_series_overview.png', dpi=150)
plt.close()
print("✅ 图已保存: figures/exp1_time_series_overview.png")