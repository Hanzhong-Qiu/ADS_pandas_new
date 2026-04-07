"""
Modelling Step 1: Granger Causality Testing

Tests whether policy changes Granger-cause sentiment changes (and vice versa).
- Global test across full dataset
- Phase-specific tests across 3 pandemic stages

Uses differenced (stationary) data from exploration phase.
Tests multiple lags (1-7 days) to find optimal lag.
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests
import warnings
warnings.filterwarnings('ignore')

SAVE_DIR = '/home/qqq/ADS_pandas_new/graphs/exploration'

# =============================================
# Load differenced data
# =============================================
df = pd.read_csv(f'{SAVE_DIR}/exp2_differenced_full.csv', index_col='date', parse_dates=True)
df = df[['sentiment_mean', 'StringencyIndex_Average']].dropna()
print(f"Data loaded: {len(df)} days")
print(f"Date range: {df.index.min().date()} to {df.index.max().date()}\n")

# =============================================
# Define pandemic phases
# =============================================
phases = {
    'Global (full dataset)': (df.index.min(), df.index.max()),
    'Phase 1: Initial outbreak & lockdowns\n(2020-03 ~ 2020-09)': ('2020-03-01', '2020-09-30'),
    'Phase 2: Second wave & vaccine rollout\n(2020-10 ~ 2021-06)': ('2020-10-01', '2021-06-30'),
    'Phase 3: Reopening & late pandemic\n(2021-07 ~ 2022-12)': ('2021-07-01', '2022-12-31'),
}

max_lag = 7

# =============================================
# Run Granger tests
# =============================================
print("=" * 70)
print("GRANGER CAUSALITY TEST RESULTS")
print("=" * 70)

results_summary = []

for phase_name, (start, end) in phases.items():
    sub = df.loc[start:end].dropna()
    n = len(sub)
    
    print(f"\n{'─' * 70}")
    print(f"{phase_name.replace(chr(10), ' ')}")
    print(f"  Sample: {n} days ({sub.index.min().date()} to {sub.index.max().date()})")
    print(f"{'─' * 70}")
    
    if n < 50:
        print("  ⚠ Insufficient data, skipping.")
        continue
    
    # Direction 1: Stringency -> Sentiment
    # (Column order for grangercausalitytests: [effect, cause])
    print(f"\n  Direction: Stringency → Sentiment")
    print(f"  (Does past stringency help predict sentiment?)")
    data_1 = sub[['sentiment_mean', 'StringencyIndex_Average']].values
    
    try:
        gc_1 = grangercausalitytests(data_1, maxlag=max_lag, verbose=False)
        print(f"  {'Lag':>4s}  {'F-stat':>8s}  {'p-value':>8s}  {'Significant?':>12s}")
        best_p1 = 1.0
        best_lag1 = 0
        for lag in range(1, max_lag + 1):
            f_stat = gc_1[lag][0]['ssr_ftest'][0]
            p_val = gc_1[lag][0]['ssr_ftest'][1]
            sig = '✓ YES' if p_val < 0.05 else ''
            print(f"  {lag:>4d}  {f_stat:>8.3f}  {p_val:>8.4f}  {sig:>12s}")
            if p_val < best_p1:
                best_p1 = p_val
                best_lag1 = lag
        print(f"  → Best: lag={best_lag1}, p={best_p1:.4f} {'*** SIGNIFICANT' if best_p1 < 0.05 else '(not significant)'}")
    except Exception as e:
        print(f"  Error: {e}")
        best_p1 = None
        best_lag1 = None
    
    # Direction 2: Sentiment -> Stringency
    print(f"\n  Direction: Sentiment → Stringency")
    print(f"  (Does past sentiment help predict stringency?)")
    data_2 = sub[['StringencyIndex_Average', 'sentiment_mean']].values
    
    try:
        gc_2 = grangercausalitytests(data_2, maxlag=max_lag, verbose=False)
        print(f"  {'Lag':>4s}  {'F-stat':>8s}  {'p-value':>8s}  {'Significant?':>12s}")
        best_p2 = 1.0
        best_lag2 = 0
        for lag in range(1, max_lag + 1):
            f_stat = gc_2[lag][0]['ssr_ftest'][0]
            p_val = gc_2[lag][0]['ssr_ftest'][1]
            sig = '✓ YES' if p_val < 0.05 else ''
            print(f"  {lag:>4d}  {f_stat:>8.3f}  {p_val:>8.4f}  {sig:>12s}")
            if p_val < best_p2:
                best_p2 = p_val
                best_lag2 = lag
        print(f"  → Best: lag={best_lag2}, p={best_p2:.4f} {'*** SIGNIFICANT' if best_p2 < 0.05 else '(not significant)'}")
    except Exception as e:
        print(f"  Error: {e}")
        best_p2 = None
        best_lag2 = None
    
    # Summary
    results_summary.append({
        'phase': phase_name.replace(chr(10), ' '),
        'n': n,
        'str_to_sent_p': best_p1,
        'str_to_sent_lag': best_lag1,
        'sent_to_str_p': best_p2,
        'sent_to_str_lag': best_lag2,
    })

# =============================================
# Final summary table
# =============================================
print(f"\n\n{'=' * 70}")
print("SUMMARY TABLE")
print(f"{'=' * 70}")
print(f"{'Phase':<45s} {'n':>4s}  {'Str→Sent':>10s}  {'Sent→Str':>10s}  {'Conclusion'}")
print(f"{'─' * 100}")

for r in results_summary:
    s2s = f"p={r['str_to_sent_p']:.3f}" if r['str_to_sent_p'] is not None else 'N/A'
    s2s_sig = r['str_to_sent_p'] is not None and r['str_to_sent_p'] < 0.05
    
    s2r = f"p={r['sent_to_str_p']:.3f}" if r['sent_to_str_p'] is not None else 'N/A'
    s2r_sig = r['sent_to_str_p'] is not None and r['sent_to_str_p'] < 0.05
    
    if s2s_sig and s2r_sig:
        conclusion = 'Bidirectional ↔'
    elif s2s_sig:
        conclusion = 'Policy → Sentiment'
    elif s2r_sig:
        conclusion = 'Sentiment → Policy'
    else:
        conclusion = 'No Granger causality'
    
    print(f"{r['phase']:<45s} {r['n']:>4d}  {s2s:>10s}  {s2r:>10s}  {conclusion}")

print(f"\nSignificance level: α = 0.05")
print(f"Test: F-test (SSR-based)")
print(f"Data: first-differenced (stationary) series")