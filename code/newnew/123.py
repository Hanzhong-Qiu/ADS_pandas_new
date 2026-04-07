"""
COVID-19 Social Media Sentiment vs Government Policy Analysis
=============================================================
Step 1: Phase-based Correlation Analysis
Step 2: Cross-Correlation Analysis
Step 3: Granger Causality Test
"""

import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.tsa.stattools import ccf, adfuller, grangercausalitytests
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import warnings
import os

warnings.filterwarnings('ignore')

# ============================================================
# 配置
# ============================================================
DATA_PATH = '/home/qqq/ADS_pandas_new/.csv/new/full_analysis_data.csv'
OUTPUT_DIR = '/home/qqq/ADS_pandas_new/graphs/newnew'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 读取数据
df = pd.read_csv(DATA_PATH, parse_dates=['date'])
df = df.sort_values('date').reset_index(drop=True)
df = df.dropna(subset=['sentiment_mean', 'StringencyIndex_Average'])

print("=" * 70)
print("DATA OVERVIEW")
print("=" * 70)
print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
print(f"Total rows: {len(df)}")
print(f"\nColumns: {list(df.columns)}")
print(f"\nBasic stats:")
print(df[['sentiment_mean', 'sentiment_volatility', 'tweet_volume',
          'StringencyIndex_Average', 'daily_new_cases', 'daily_new_deaths']].describe().round(4))

# ============================================================
# STEP 1: 分阶段相关分析 (Phase-based Correlation)
# ============================================================
print("\n" + "=" * 70)
print("STEP 1: PHASE-BASED CORRELATION ANALYSIS")
print("=" * 70)

phases = {
    'Phase 1: Outbreak & Lockdown': ('2020-03-01', '2020-06-30'),
    'Phase 2: First Relaxation': ('2020-07-01', '2020-10-31'),
    'Phase 3: Second Wave': ('2020-11-01', '2021-03-31'),
    'Phase 4: Vaccine Rollout': ('2021-04-01', '2021-10-31'),
    'Phase 5: Omicron & Reopening': ('2021-11-01', '2022-06-30'),
}

phase_results = []

for name, (start, end) in phases.items():
    mask = (df['date'] >= start) & (df['date'] <= end)
    phase_data = df[mask].dropna(subset=['sentiment_mean', 'StringencyIndex_Average', 'daily_new_cases'])

    if len(phase_data) < 10:
        print(f"\n{name}: Not enough data (N={len(phase_data)}), skipping.")
        continue

    r_str, p_str = stats.pearsonr(phase_data['sentiment_mean'], phase_data['StringencyIndex_Average'])
    r_cas, p_cas = stats.pearsonr(phase_data['sentiment_mean'], phase_data['daily_new_cases'])
    r_vol, p_vol = stats.pearsonr(phase_data['sentiment_mean'], phase_data['tweet_volume']) if phase_data['tweet_volume'].notna().sum() > 10 else (np.nan, np.nan)
    r_dea, p_dea = stats.pearsonr(phase_data['sentiment_mean'], phase_data['daily_new_deaths']) if phase_data['daily_new_deaths'].notna().sum() > 10 else (np.nan, np.nan)

    # Spearman（非线性关系）
    rs_str, ps_str = stats.spearmanr(phase_data['sentiment_mean'], phase_data['StringencyIndex_Average'])

    phase_results.append({
        'Phase': name,
        'N': len(phase_data),
        'Sentiment_Mean': phase_data['sentiment_mean'].mean(),
        'Stringency_Mean': phase_data['StringencyIndex_Average'].mean(),
        'r_Stringency': r_str,
        'p_Stringency': p_str,
        'r_Spearman_Stringency': rs_str,
        'r_Cases': r_cas,
        'p_Cases': p_cas,
        'r_Volume': r_vol,
        'p_Volume': p_vol,
        'r_Deaths': r_dea,
        'p_Deaths': p_dea,
    })

    sig_str = "***" if p_str < 0.001 else "**" if p_str < 0.01 else "*" if p_str < 0.05 else "n.s."
    sig_cas = "***" if p_cas < 0.001 else "**" if p_cas < 0.01 else "*" if p_cas < 0.05 else "n.s."

    print(f"\n{name} (N={len(phase_data)})")
    print(f"  Avg Sentiment: {phase_data['sentiment_mean'].mean():.4f}")
    print(f"  Avg Stringency: {phase_data['StringencyIndex_Average'].mean():.1f}")
    print(f"  Sentiment vs Stringency:  Pearson r={r_str:.4f} (p={p_str:.4f}) {sig_str}")
    print(f"                            Spearman r={rs_str:.4f}")
    print(f"  Sentiment vs New Cases:   Pearson r={r_cas:.4f} (p={p_cas:.4f}) {sig_cas}")
    if not np.isnan(r_vol):
        sig_vol = "***" if p_vol < 0.001 else "**" if p_vol < 0.01 else "*" if p_vol < 0.05 else "n.s."
        print(f"  Sentiment vs Tweet Vol:   Pearson r={r_vol:.4f} (p={p_vol:.4f}) {sig_vol}")
    if not np.isnan(r_dea):
        sig_dea = "***" if p_dea < 0.001 else "**" if p_dea < 0.01 else "*" if p_dea < 0.05 else "n.s."
        print(f"  Sentiment vs Deaths:      Pearson r={r_dea:.4f} (p={p_dea:.4f}) {sig_dea}")

phase_df = pd.DataFrame(phase_results)
phase_df.to_csv(os.path.join(OUTPUT_DIR, 'phase_correlation_results.csv'), index=False)

# --- 图1: 分阶段相关系数对比柱状图 ---
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(phase_df))
width = 0.35
bars1 = ax.bar(x - width/2, phase_df['r_Stringency'], width, label='vs Stringency Index', color='#e74c3c', alpha=0.8)
bars2 = ax.bar(x + width/2, phase_df['r_Cases'], width, label='vs Daily New Cases', color='#3498db', alpha=0.8)

ax.set_ylabel('Pearson Correlation Coefficient (r)', fontsize=12)
ax.set_title('Sentiment Correlation with Stringency & Cases Across Pandemic Phases', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([p.replace(': ', ':\n') for p in phase_df['Phase']], fontsize=9)
ax.legend(fontsize=11)
ax.axhline(y=0, color='black', linewidth=0.8)
ax.set_ylim(-0.6, 0.6)
ax.grid(axis='y', alpha=0.3)

for bar_group, p_col in [(bars1, 'p_Stringency'), (bars2, 'p_Cases')]:
    for bar, p_val in zip(bar_group, phase_df[p_col]):
        sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''
        if sig:
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02 * np.sign(bar.get_height()),
                    sig, ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '1_phase_correlation_bars.png'), dpi=200)
plt.close()
print(f"\nSaved: 1_phase_correlation_bars.png")

# --- 图2: 分阶段的散点图矩阵 ---
fig, axes = plt.subplots(1, len(phases), figsize=(20, 4), sharey=True)
colors = ['#e74c3c', '#e67e22', '#f1c40f', '#2ecc71', '#3498db']

for i, (name, (start, end)) in enumerate(phases.items()):
    mask = (df['date'] >= start) & (df['date'] <= end)
    phase_data = df[mask].dropna(subset=['sentiment_mean', 'StringencyIndex_Average'])
    if len(phase_data) < 2:
        continue
    ax = axes[i]
    ax.scatter(phase_data['StringencyIndex_Average'], phase_data['sentiment_mean'],
               alpha=0.4, s=15, color=colors[i])
    # 回归线
    if len(phase_data) > 5:
        z = np.polyfit(phase_data['StringencyIndex_Average'], phase_data['sentiment_mean'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(phase_data['StringencyIndex_Average'].min(), phase_data['StringencyIndex_Average'].max(), 100)
        ax.plot(x_line, p(x_line), color=colors[i], linewidth=2, linestyle='--')
    ax.set_title(name.split(': ')[1], fontsize=10, fontweight='bold')
    ax.set_xlabel('Stringency Index', fontsize=9)
    if i == 0:
        ax.set_ylabel('Sentiment Mean', fontsize=10)
    ax.grid(alpha=0.3)

fig.suptitle('Sentiment vs Stringency by Pandemic Phase', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '2_phase_scatter_plots.png'), dpi=200, bbox_inches='tight')
plt.close()
print("Saved: 2_phase_scatter_plots.png")


# ============================================================
# STEP 2: CROSS-CORRELATION ANALYSIS
# ============================================================
print("\n" + "=" * 70)
print("STEP 2: CROSS-CORRELATION ANALYSIS")
print("=" * 70)

max_lag = 30

# --- 2a: Sentiment vs Stringency ---
sent = df['sentiment_mean'].values
stri = df['StringencyIndex_Average'].values

# 用 scipy 的 correlate 做双向
from scipy.signal import correlate

def compute_cross_correlation(x, y, max_lag):
    """计算标准化的双向cross-correlation"""
    x_norm = (x - np.mean(x)) / (np.std(x) * len(x))
    y_norm = (y - np.mean(y)) / np.std(y)
    cc = correlate(x_norm, y_norm, mode='full')
    mid = len(cc) // 2
    lags = np.arange(-max_lag, max_lag + 1)
    cc_subset = cc[mid - max_lag: mid + max_lag + 1]
    return lags, cc_subset

lags_str, cc_str = compute_cross_correlation(sent, stri, max_lag)
lags_cas, cc_cas = compute_cross_correlation(sent, df['daily_new_cases'].fillna(0).values, max_lag)
lags_dea, cc_dea = compute_cross_correlation(sent, df['daily_new_deaths'].fillna(0).values, max_lag)

# 找峰值
peak_lag_str = lags_str[np.argmax(np.abs(cc_str))]
peak_cc_str = cc_str[np.argmax(np.abs(cc_str))]
peak_lag_cas = lags_cas[np.argmax(np.abs(cc_cas))]
peak_cc_cas = cc_cas[np.argmax(np.abs(cc_cas))]
peak_lag_dea = lags_dea[np.argmax(np.abs(cc_dea))]
peak_cc_dea = cc_dea[np.argmax(np.abs(cc_dea))]

print(f"\nSentiment vs Stringency:  Peak at lag={peak_lag_str}, r={peak_cc_str:.4f}")
print(f"  (Negative lag = Stringency leads; Positive lag = Sentiment leads)")
print(f"Sentiment vs New Cases:   Peak at lag={peak_lag_cas}, r={peak_cc_cas:.4f}")
print(f"Sentiment vs New Deaths:  Peak at lag={peak_lag_dea}, r={peak_cc_dea:.4f}")

# --- 图3: Cross-Correlation 图 ---
fig, axes = plt.subplots(3, 1, figsize=(12, 10))

# 95% CI bounds
n = len(df)
ci_bound = 1.96 / np.sqrt(n)

for ax, lags_data, cc_data, title, color, peak_lag, peak_cc in [
    (axes[0], lags_str, cc_str, 'Sentiment vs Stringency Index', '#e74c3c', peak_lag_str, peak_cc_str),
    (axes[1], lags_cas, cc_cas, 'Sentiment vs Daily New Cases', '#3498db', peak_lag_cas, peak_cc_cas),
    (axes[2], lags_dea, cc_dea, 'Sentiment vs Daily New Deaths', '#8e44ad', peak_lag_dea, peak_cc_dea),
]:
    ax.bar(lags_data, cc_data, color=color, alpha=0.7, width=0.8)
    ax.axhline(y=0, color='black', linewidth=0.8)
    ax.axhline(y=ci_bound, color='grey', linewidth=0.8, linestyle='--', label='95% CI')
    ax.axhline(y=-ci_bound, color='grey', linewidth=0.8, linestyle='--')
    ax.axvline(x=0, color='black', linewidth=0.5, linestyle=':')
    # 标注峰值
    ax.annotate(f'Peak: lag={peak_lag}, r={peak_cc:.3f}',
                xy=(peak_lag, peak_cc), fontsize=10, fontweight='bold',
                xytext=(peak_lag + 5, peak_cc + 0.02 * np.sign(peak_cc)),
                arrowprops=dict(arrowstyle='->', color='black'))
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_ylabel('Cross-Correlation')
    ax.set_xlim(-max_lag - 1, max_lag + 1)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

axes[2].set_xlabel('Lag (days)\n← Stringency/Cases/Deaths leads | Sentiment leads →', fontsize=11)
fig.suptitle('Cross-Correlation Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '3_cross_correlation.png'), dpi=200)
plt.close()
print("Saved: 3_cross_correlation.png")


# ============================================================
# STEP 3: STATIONARITY TEST & GRANGER CAUSALITY
# ============================================================
print("\n" + "=" * 70)
print("STEP 3: STATIONARITY & GRANGER CAUSALITY")
print("=" * 70)

# --- 3a: ADF 平稳性检验 ---
print("\n--- ADF Stationarity Test (original series) ---")
adf_results = {}
for col in ['sentiment_mean', 'StringencyIndex_Average', 'daily_new_cases']:
    series = df[col].dropna()
    result = adfuller(series, autolag='AIC')
    stationary = "YES (stationary)" if result[1] < 0.05 else "NO (non-stationary, need differencing)"
    adf_results[col] = {'statistic': result[0], 'p_value': result[1], 'stationary': result[1] < 0.05}
    print(f"  {col}: ADF={result[0]:.4f}, p={result[1]:.6f} → {stationary}")

# 差分
df['sentiment_diff'] = df['sentiment_mean'].diff()
df['stringency_diff'] = df['StringencyIndex_Average'].diff()
df['cases_diff'] = df['daily_new_cases'].diff()

print("\n--- ADF Stationarity Test (differenced series) ---")
for col in ['sentiment_diff', 'stringency_diff', 'cases_diff']:
    series = df[col].dropna()
    result = adfuller(series, autolag='AIC')
    stationary = "YES" if result[1] < 0.05 else "NO"
    print(f"  {col}: ADF={result[0]:.4f}, p={result[1]:.6f} → {stationary}")

# --- 3b: Granger Causality ---
print("\n--- Granger Causality Tests ---")
print("(Using differenced series if original is non-stationary)\n")

# 决定用原始还是差分
sent_col = 'sentiment_mean' if adf_results['sentiment_mean']['stationary'] else 'sentiment_diff'
stri_col = 'StringencyIndex_Average' if adf_results['StringencyIndex_Average']['stationary'] else 'stringency_diff'
case_col = 'daily_new_cases' if adf_results['daily_new_cases']['stationary'] else 'cases_diff'

print(f"Using: {sent_col}, {stri_col}, {case_col}\n")

max_granger_lag = 14
granger_results = {}

tests = [
    ("Stringency → Sentiment", sent_col, stri_col),
    ("Sentiment → Stringency", stri_col, sent_col),
    ("Cases → Sentiment", sent_col, case_col),
    ("Sentiment → Cases", case_col, sent_col),
    ("Cases → Stringency", stri_col, case_col),
    ("Stringency → Cases", case_col, stri_col),
]

for test_name, y_col, x_col in tests:
    print(f"\n{'='*50}")
    print(f"Testing: {test_name}")
    print(f"  (Does {x_col} Granger-cause {y_col}?)")
    print(f"{'='*50}")

    data_gc = df[[y_col, x_col]].dropna()

    if len(data_gc) < max_granger_lag * 3:
        print(f"  Not enough data (N={len(data_gc)}), skipping.")
        continue

    try:
        gc_result = grangercausalitytests(data_gc, maxlag=max_granger_lag, verbose=False)

        best_lag = None
        best_p = 1.0
        lag_summary = []

        for lag in range(1, max_granger_lag + 1):
            f_test = gc_result[lag][0]['ssr_ftest']
            p_val = f_test[1]
            f_stat = f_test[0]
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
            lag_summary.append({'lag': lag, 'F': f_stat, 'p': p_val, 'sig': sig})
            if p_val < best_p:
                best_p = p_val
                best_lag = lag

        # 打印摘要表
        print(f"\n  {'Lag':>4} | {'F-stat':>10} | {'p-value':>10} | Sig")
        print(f"  {'-'*4}-+-{'-'*10}-+-{'-'*10}-+----")
        for row in lag_summary:
            print(f"  {row['lag']:>4} | {row['F']:>10.4f} | {row['p']:>10.6f} | {row['sig']}")

        significant = best_p < 0.05
        print(f"\n  → Best lag: {best_lag} (p={best_p:.6f})")
        print(f"  → Conclusion: {'SIGNIFICANT - ' + test_name + ' is supported' if significant else 'NOT SIGNIFICANT'}")

        granger_results[test_name] = {
            'best_lag': best_lag,
            'best_p': best_p,
            'significant': significant,
            'all_lags': lag_summary
        }

    except Exception as e:
        print(f"  Error: {e}")

# --- 图4: Granger结果汇总 ---
if granger_results:
    fig, ax = plt.subplots(figsize=(10, 5))

    test_names = list(granger_results.keys())
    best_ps = [-np.log10(granger_results[t]['best_p']) for t in test_names]
    colors_gc = ['#e74c3c' if granger_results[t]['significant'] else '#95a5a6' for t in test_names]

    bars = ax.barh(range(len(test_names)), best_ps, color=colors_gc, alpha=0.8)
    ax.axvline(x=-np.log10(0.05), color='black', linestyle='--', linewidth=1.5, label='p=0.05 threshold')
    ax.axvline(x=-np.log10(0.01), color='grey', linestyle=':', linewidth=1, label='p=0.01 threshold')

    ax.set_yticks(range(len(test_names)))
    ax.set_yticklabels(test_names, fontsize=10)
    ax.set_xlabel('-log10(p-value)\n(Higher = more significant)', fontsize=11)
    ax.set_title('Granger Causality Test Results (Best Lag)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(axis='x', alpha=0.3)

    for i, (bar, name) in enumerate(zip(bars, test_names)):
        lag = granger_results[name]['best_lag']
        p = granger_results[name]['best_p']
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                f'lag={lag}, p={p:.4f}', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '4_granger_causality_summary.png'), dpi=200)
    plt.close()
    print("\nSaved: 4_granger_causality_summary.png")

# 保存Granger结果
granger_df = pd.DataFrame([
    {'Test': k, 'Best_Lag': v['best_lag'], 'Best_p': v['best_p'], 'Significant': v['significant']}
    for k, v in granger_results.items()
])
granger_df.to_csv(os.path.join(OUTPUT_DIR, 'granger_results.csv'), index=False)


# ============================================================
# STEP 4: 全时段相关矩阵热力图
# ============================================================
print("\n" + "=" * 70)
print("STEP 4: CORRELATION HEATMAP")
print("=" * 70)

corr_cols = ['sentiment_mean', 'StringencyIndex_Average', 'daily_new_cases', 'daily_new_deaths',
             'tweet_volume', 'C1_School_closing', 'C2_Workplace_closing',
             'C4_Restrictions_gatherings', 'C6_Stay_at_home',
             'C7_Internal_movement', 'C8_International_travel', 'H6_Facial_coverings']

# 只保留存在的列
corr_cols = [c for c in corr_cols if c in df.columns]
corr_matrix = df[corr_cols].corr()

print("\nCorrelation with sentiment_mean:")
print(corr_matrix['sentiment_mean'].sort_values().round(4))

# --- 图5: 热力图 ---
fig, ax = plt.subplots(figsize=(12, 10))
short_labels = {
    'sentiment_mean': 'Sentiment',
    'StringencyIndex_Average': 'Stringency',
    'daily_new_cases': 'New Cases',
    'daily_new_deaths': 'New Deaths',
    'tweet_volume': 'Tweet Volume',
    'C1_School_closing': 'School Close',
    'C2_Workplace_closing': 'Workplace Close',
    'C4_Restrictions_gatherings': 'Gathering Restrict',
    'C6_Stay_at_home': 'Stay at Home',
    'C7_Internal_movement': 'Internal Movement',
    'C8_International_travel': 'Intl Travel',
    'H6_Facial_coverings': 'Face Coverings',
}
display_labels = [short_labels.get(c, c) for c in corr_cols]

mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, vmin=-1, vmax=1, square=True,
            xticklabels=display_labels, yticklabels=display_labels,
            linewidths=0.5, ax=ax)
ax.set_title('Correlation Matrix: Sentiment, Policy & Pandemic Indicators', fontsize=13, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '5_correlation_heatmap.png'), dpi=200)
plt.close()
print("Saved: 5_correlation_heatmap.png")


# ============================================================
# STEP 5: 带事件标注的时间序列图（优化版）
# ============================================================
print("\n" + "=" * 70)
print("STEP 5: ANNOTATED TIMELINE")
print("=" * 70)

key_events = [
    ('2020-03-11', 'WHO Declares\nPandemic'),
    ('2020-03-23', 'UK First\nLockdown'),
    ('2020-12-08', 'UK Starts\nVaccination'),
    ('2021-01-06', 'UK Third\nLockdown'),
    ('2021-07-19', 'UK "Freedom\nDay"'),
    ('2021-11-26', 'Omicron\nDetected'),
    ('2022-02-24', 'UK Lifts\nRestrictions'),
]

fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)

# 7-day MA
df['sentiment_7d'] = df['sentiment_mean'].rolling(7, center=True).mean()
df['stringency_7d'] = df['StringencyIndex_Average'].rolling(7, center=True).mean()
df['cases_7d'] = df['daily_new_cases'].rolling(7, center=True).mean()

# Panel 1: Sentiment
ax1 = axes[0]
ax1.plot(df['date'], df['sentiment_mean'], alpha=0.2, color='#3498db', linewidth=0.5)
ax1.plot(df['date'], df['sentiment_7d'], color='#2980b9', linewidth=1.5, label='7-day MA')
ax1.set_ylabel('Sentiment Mean', fontsize=11)
ax1.legend(loc='upper right')
ax1.grid(alpha=0.3)

# 阶段背景色
phase_colors = ['#ffcccc', '#ffe0cc', '#ffffcc', '#ccffcc', '#cce0ff']
for (name, (start, end)), color in zip(phases.items(), phase_colors):
    ax1.axvspan(pd.Timestamp(start), pd.Timestamp(end), alpha=0.15, color=color)

# Panel 2: Stringency
ax2 = axes[1]
ax2.plot(df['date'], df['StringencyIndex_Average'], alpha=0.2, color='#e74c3c', linewidth=0.5)
ax2.plot(df['date'], df['stringency_7d'], color='#c0392b', linewidth=1.5, label='7-day MA')
ax2.set_ylabel('Stringency Index', fontsize=11)
ax2.legend(loc='upper right')
ax2.grid(alpha=0.3)

for (name, (start, end)), color in zip(phases.items(), phase_colors):
    ax2.axvspan(pd.Timestamp(start), pd.Timestamp(end), alpha=0.15, color=color)

# Panel 3: Cases
ax3 = axes[2]
ax3.plot(df['date'], df['daily_new_cases'], alpha=0.2, color='#8e44ad', linewidth=0.5)
ax3.plot(df['date'], df['cases_7d'], color='#6c3483', linewidth=1.5, label='7-day MA')
ax3.set_ylabel('Daily New Cases', fontsize=11)
ax3.legend(loc='upper right')
ax3.grid(alpha=0.3)

for (name, (start, end)), color in zip(phases.items(), phase_colors):
    ax3.axvspan(pd.Timestamp(start), pd.Timestamp(end), alpha=0.15, color=color)

# 事件标注
for date_str, label in key_events:
    event_date = pd.Timestamp(date_str)
    if event_date >= df['date'].min() and event_date <= df['date'].max():
        for ax in axes:
            ax.axvline(x=event_date, color='grey', linewidth=0.8, linestyle='--', alpha=0.7)
        axes[0].annotate(label, xy=(event_date, axes[0].get_ylim()[1]),
                         fontsize=7, ha='center', va='bottom', rotation=0,
                         bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='grey', alpha=0.8))

axes[2].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
axes[2].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.xticks(rotation=45, ha='right')
fig.suptitle('COVID-19 Timeline: Sentiment, Policy & Cases (English-speaking countries avg.)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '6_annotated_timeline.png'), dpi=200)
plt.close()
print("Saved: 6_annotated_timeline.png")


# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("SUMMARY OF ALL OUTPUTS")
print("=" * 70)
print(f"\nAll outputs saved to: {OUTPUT_DIR}")
print(f"\nGenerated files:")
for f in sorted(os.listdir(OUTPUT_DIR)):
    filepath = os.path.join(OUTPUT_DIR, f)
    size = os.path.getsize(filepath)
    print(f"  {f} ({size/1024:.1f} KB)")

print("\n" + "=" * 70)
print("NEXT STEPS")
print("=" * 70)
print("""
Based on these results, the next analysis steps are:
1. Regression modelling (linear + random forest) with feature importance
2. Change point detection on sentiment time series
3. Phase-based sub-analysis for the Omicron decoupling phenomenon
4. Generate all final visualisations for the report

Please share the terminal output and generated plots for framework update.
""")