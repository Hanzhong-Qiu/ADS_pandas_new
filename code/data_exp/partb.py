import sys
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import ruptures as rpt

sys.path.append(str(Path(__file__).resolve().parents[2]))
from analysis_common import DATA_OUTPUT_DIR, PLOTS_OUTPUT_DIR, ensure_output_dirs

ensure_output_dirs()

# ── 1. LOAD DATA ────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_OUTPUT_DIR / 'enriched_research_data.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

# Auto-detect column names
policy_col   = [c for c in df.columns if 'stringency' in c.lower()][0]
vol_col      = 'sentiment_volatility'
volume_col   = 'tweet_volume'
sentiment_col = 'sentiment_mean'

print(f"✅ Data loaded: {df.shape[0]} rows | {df['date'].min().date()} → {df['date'].max().date()}")
print(f"   Policy col : {policy_col}")

# ── 2. MANUALLY DEFINED PHASE BOUNDARIES (your current segmentation) ────────
MANUAL_PHASES = [
    ("Phase 1: Initial Outbreak",    "2020-03-01", "2020-08-31"),
    ("Phase 2: Second Wave",         "2020-09-01", "2021-02-28"),
    ("Phase 3: Vaccination Era",     "2021-03-01", "2021-09-30"),
    ("Phase 4: Omicron",             "2021-10-01", "2022-03-31"),
    ("Phase 5: Post-Restrictions",   "2022-04-01", "2022-12-31"),
]

PHASE_COLORS = ['#3b82f6', '#f59e0b', '#10b981', '#ef4444', '#8b5cf6']

# ── 3. SIGNALS TO RUN CHANGEPOINT DETECTION ON ──────────────────────────────
signals = {
    'Tweet Volume':          df[volume_col].fillna(method='ffill').values,
    'Sentiment Volatility':  df[vol_col].fillna(method='ffill').values,
    'Stringency Index':      df[policy_col].fillna(method='ffill').values,
}

# ── 4. PELT CHANGEPOINT DETECTION ───────────────────────────────────────────
# PELT minimises a cost function (rbf kernel) with a penalty to avoid overfitting
# n_bkps = 4 because 5 phases = 4 internal breakpoints

N_BREAKPOINTS = 4   # adjust if you want more/fewer
PENALTY       = 10  # higher = fewer breakpoints (less sensitive)

pelt_results = {}

print("\n📍 PELT Changepoint Detection Results:")
print("─" * 50)

for signal_name, signal_values in signals.items():
    # Normalise signal for fair comparison
    signal_norm = (signal_values - signal_values.mean()) / signal_values.std()

    algo = rpt.Pelt(model="rbf").fit(signal_norm)
    breakpoints = algo.predict(pen=PENALTY)

    # Convert index positions → dates (exclude the last point which is always len(df))
    bp_dates = [df['date'].iloc[bp - 1].date() for bp in breakpoints if bp < len(df)]

    pelt_results[signal_name] = bp_dates
    print(f"\n  {signal_name}:")
    for d in bp_dates:
        print(f"    → {d}")

# ── 5. BINSEG CHANGEPOINT DETECTION (Bai-Perron style) ──────────────────────
# Binary segmentation — faster, hypothesis-testing flavour
# Finds exactly N_BREAKPOINTS structural breaks

binseg_results = {}

print("\n\n📍 BinSeg (Bai-Perron style) Changepoint Detection Results:")
print("─" * 50)

for signal_name, signal_values in signals.items():
    signal_norm = (signal_values - signal_values.mean()) / signal_values.std()

    algo = rpt.Binseg(model="l2").fit(signal_norm)
    breakpoints = algo.predict(n_bkps=N_BREAKPOINTS)

    bp_dates = [df['date'].iloc[bp - 1].date() for bp in breakpoints if bp < len(df)]

    binseg_results[signal_name] = bp_dates
    print(f"\n  {signal_name}:")
    for d in bp_dates:
        print(f"    → {d}")

# ── 6. COMPARISON: DETECTED vs MANUAL BOUNDARIES ────────────────────────────
print("\n\n📊 Comparison: Detected Breakpoints vs Manual Phase Boundaries:")
print("─" * 60)
print(f"{'Manual Boundary':<30} {'PELT (Vol)':<15} {'BinSeg (Vol)':<15} {'Match?'}")
print("─" * 60)

manual_boundaries = [pd.Timestamp(end).date() for _, _, end in MANUAL_PHASES[:-1]]
pelt_vol_dates    = pelt_results.get('Tweet Volume', [])
binseg_vol_dates  = binseg_results.get('Tweet Volume', [])

for mb in manual_boundaries:
    # Check if any detected breakpoint is within 30 days of manual boundary
    pelt_close   = [d for d in pelt_vol_dates   if abs((pd.Timestamp(d) - pd.Timestamp(mb)).days) <= 30]
    binseg_close = [d for d in binseg_vol_dates if abs((pd.Timestamp(d) - pd.Timestamp(mb)).days) <= 30]
    match = "✅" if pelt_close or binseg_close else "❌"
    p_str = str(pelt_close[0])   if pelt_close   else "—"
    b_str = str(binseg_close[0]) if binseg_close else "—"
    print(f"{str(mb):<30} {p_str:<15} {b_str:<15} {match}")

# ── 7. PLOT: ALL SIGNALS WITH DETECTED BREAKPOINTS ──────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
fig.suptitle('Changepoint Detection vs Manual Phase Boundaries', fontsize=16, fontweight='bold', y=0.98)

signal_list = list(signals.items())

for ax_idx, (signal_name, signal_values) in enumerate(signal_list):
    ax = axes[ax_idx]

    # Plot raw signal
    ax.plot(df['date'], signal_values, color='#94a3b8', linewidth=1, alpha=0.8, label=signal_name)

    # Shade manual phases
    for i, (phase_name, start, end) in enumerate(MANUAL_PHASES):
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end),
                   alpha=0.08, color=PHASE_COLORS[i], label=phase_name if ax_idx == 0 else "")

    # Manual phase boundary lines
    for _, _, end in MANUAL_PHASES[:-1]:
        ax.axvline(pd.Timestamp(end), color='gray', linestyle='--', linewidth=1, alpha=0.5)

    # PELT breakpoints
    for bp_date in pelt_results.get(signal_name, []):
        ax.axvline(pd.Timestamp(bp_date), color='#ef4444', linestyle='-', linewidth=2,
                   label='PELT Breakpoint' if bp_date == pelt_results[signal_name][0] else "")

    # BinSeg breakpoints
    for bp_date in binseg_results.get(signal_name, []):
        ax.axvline(pd.Timestamp(bp_date), color='#f59e0b', linestyle=':', linewidth=2,
                   label='BinSeg Breakpoint' if bp_date == binseg_results[signal_name][0] else "")

    ax.set_ylabel(signal_name, fontsize=10)
    ax.grid(axis='y', alpha=0.2)
    ax.legend(loc='upper right', fontsize=8, ncol=2)

axes[-1].set_xlabel('Date', fontsize=11)
plt.tight_layout()
plt.savefig(PLOTS_OUTPUT_DIR / 'changepoint_detection.png', dpi=300)
plt.close()
print("\n✅ Saved: changepoint_detection.png")

# ── 8. PLOT: SUMMARY COMPARISON CHART ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 5))
ax.set_title('Breakpoint Comparison: Manual vs PELT vs BinSeg\n(All Signals)', fontsize=14, fontweight='bold')

y_positions = {'Tweet Volume': 3, 'Sentiment Volatility': 2, 'Stringency Index': 1}
y_labels    = {3: 'Tweet Volume', 2: 'Sentiment Volatility', 1: 'Stringency Index'}

# Manual boundaries
for mb in manual_boundaries:
    ax.axvline(pd.Timestamp(mb), color='gray', linestyle='--', linewidth=1.5, alpha=0.6)

# Plot detected breakpoints per signal
for signal_name, y_pos in y_positions.items():
    # PELT
    for bp in pelt_results.get(signal_name, []):
        ax.scatter(pd.Timestamp(bp), y_pos + 0.15, color='#ef4444', s=120, zorder=5, marker='v')
    # BinSeg
    for bp in binseg_results.get(signal_name, []):
        ax.scatter(pd.Timestamp(bp), y_pos - 0.15, color='#f59e0b', s=120, zorder=5, marker='^')

ax.set_yticks([1, 2, 3])
ax.set_yticklabels([y_labels[i] for i in [1, 2, 3]], fontsize=11)
ax.set_xlabel('Date', fontsize=11)
ax.set_xlim(df['date'].min(), df['date'].max())
ax.set_ylim(0.5, 3.8)
ax.grid(axis='x', alpha=0.2)

# Phase labels along top
for i, (phase_name, start, end) in enumerate(MANUAL_PHASES):
    mid = pd.Timestamp(start) + (pd.Timestamp(end) - pd.Timestamp(start)) / 2
    ax.text(mid, 3.6, f"P{i+1}", ha='center', fontsize=9, color=PHASE_COLORS[i], fontweight='bold')

# Legend
legend_elements = [
    mpatches.Patch(color='gray',    alpha=0.5, label='Manual Phase Boundary'),
    plt.Line2D([0], [0], marker='v', color='w', markerfacecolor='#ef4444', markersize=10, label='PELT Breakpoint'),
    plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='#f59e0b', markersize=10, label='BinSeg Breakpoint'),
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

plt.tight_layout()
plt.savefig(PLOTS_OUTPUT_DIR / 'changepoint_comparison.png', dpi=300)
plt.close()
print("✅ Saved: changepoint_comparison.png")

# ── 9. FINAL SUMMARY ────────────────────────────────────────────────────────
print("\n" + "═" * 60)
print("📋 CHANGEPOINT DETECTION SUMMARY")
print("═" * 60)
for signal_name in signals:
    print(f"\n  {signal_name}:")
    print(f"    PELT   → {pelt_results.get(signal_name, [])}")
    print(f"    BinSeg → {binseg_results.get(signal_name, [])}")

print(f"\n  Manual boundaries : {manual_boundaries}")
print("\n  Interpretation:")
print("  ✅ Match (±30 days) = phase boundary statistically validated")
print("  ❌ No match         = consider revising that phase boundary")
print("═" * 60)
