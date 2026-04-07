"""
Step 5c: Cross-Correlation on High-Correlation Window
Purpose: Find optimal lag between Stringency and Sentiment
Window: 2020-07-15 to 2020-10-15 (90-day high-correlation period)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

SAVE_DIR = '/home/qqq/ADS_pandas_new/graphs/exploration'
# ── Load data ──
df = pd.read_csv(f'{SAVE_DIR}/exp2_differenced_full.csv', index_col='date', parse_dates=True)

# ── Subset to high-correlation window ──
window_start = '2020-07-15'
window_end = '2020-10-15'
window = df.loc[window_start:window_end].copy()

print(f"Window: {window_start} to {window_end}")
print("Columns:", window.columns.tolist())
# print(f"Data points: {len(window)}")

# # ── Differencing (for stationarity) ──
# window['d_sentiment'] = window['sentiment_mean'].diff()
# window['d_stringency'] = window['StringencyIndex'].diff()
# window = window.dropna(subset=['d_sentiment', 'd_stringency'])

# ── Cross-correlation function ──
max_lag = 21  # test up to 21 days

lags = range(-max_lag, max_lag + 1)
ccf_values = []
p_values = []

for lag in lags:
    if lag < 0:
        # negative lag: stringency LEADS sentiment by |lag| days
        x = window['d_stringency'].iloc[:lag].values
        y = window['d_sentiment'].iloc[-lag:].values
    elif lag > 0:
        # positive lag: sentiment LEADS stringency by lag days
        x = window['d_stringency'].iloc[lag:].values
        y = window['d_sentiment'].iloc[:-lag].values
    else:
        x = window['d_stringency'].values
        y = window['d_sentiment'].values
    
    r, p = stats.pearsonr(x, y)
    ccf_values.append(r)
    p_values.append(p)

ccf_values = np.array(ccf_values)
p_values = np.array(p_values)
lags = np.array(list(range(-max_lag, max_lag + 1)))

# ── Find optimal lag ──
best_idx = np.argmax(np.abs(ccf_values))
best_lag = lags[best_idx]
best_r = ccf_values[best_idx]
best_p = p_values[best_idx]

print(f"\n{'='*50}")
print(f"Optimal lag: {best_lag} days")
print(f"  r = {best_r:.4f}, p = {best_p:.4f}")
if best_lag < 0:
    print(f"  Interpretation: Stringency LEADS Sentiment by {abs(best_lag)} day(s)")
elif best_lag > 0:
    print(f"  Interpretation: Sentiment LEADS Stringency by {best_lag} day(s)")
else:
    print(f"  Interpretation: Synchronous (no lead/lag)")

# ── Print all significant lags ──
print(f"\nAll lags with |r| > 0.15:")
print(f"{'Lag':>6} {'r':>8} {'p':>8}  Interpretation")
print(f"{'-'*50}")
for i, lag in enumerate(lags):
    if abs(ccf_values[i]) > 0.15:
        if lag < 0:
            interp = f"Stringency leads by {abs(lag)}d"
        elif lag > 0:
            interp = f"Sentiment leads by {lag}d"
        else:
            interp = "Synchronous"
        sig = "*" if p_values[i] < 0.05 else ""
        print(f"{lag:>6} {ccf_values[i]:>8.4f} {p_values[i]:>8.4f}  {interp} {sig}")

# ── Plot ──
fig, ax = plt.subplots(figsize=(12, 6))

# Color bars by sign
colors = ['#8B1A1A' if v < 0 else '#2E5090' for v in ccf_values]
ax.bar(lags, ccf_values, color=colors, alpha=0.7, width=0.8)

# Significance threshold (approximate 95% CI)
n = len(window)
sig_threshold = 1.96 / np.sqrt(n)
ax.axhline(y=sig_threshold, color='grey', linestyle='--', alpha=0.5, label=f'95% CI (±{sig_threshold:.3f})')
ax.axhline(y=-sig_threshold, color='grey', linestyle='--', alpha=0.5)
ax.axhline(y=0, color='black', linewidth=0.5)

# Mark the best lag
ax.bar(best_lag, best_r, color='gold', edgecolor='black', linewidth=2, width=0.8, 
       label=f'Best lag = {best_lag}d (r={best_r:.3f})')

# Labels
ax.set_xlabel('Lag (days)\n← Stringency leads Sentiment | Sentiment leads Stringency →', fontsize=11)
ax.set_ylabel('Cross-Correlation (r)', fontsize=11)
ax.set_title(f'Cross-Correlation: ΔStringency vs ΔSentiment\nWindow: {window_start} to {window_end}', fontsize=13)
ax.legend(fontsize=10)
ax.set_xlim(-max_lag - 1, max_lag + 1)

plt.tight_layout()
plt.savefig('f{SAVE_DIR}/step5c_cross_correlation_window.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\nFigure saved to graphs/exploration/step5c_cross_correlation_window.png")