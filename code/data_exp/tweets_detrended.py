"""
Detrended Cross-Correlation Analysis
======================================
1. Global cross-correlation on differenced (detrended) data
2. Phase-segmented cross-correlation on raw data per pandemic wave

USAGE:
    python cross_corr_detrended.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats
import os
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# PATHS - hardcoded as requested
# ============================================================
INPUT_DIFF = "/home/qqq/ADS_pandas_new/graphs/exploration/exp2_differenced_full.csv"
INPUT_RAW = "/home/qqq/ADS_pandas_new/.csv/new/full_analysis_data.csv"
OUTPUT_DIR = "/home/qqq/ADS_pandas_new/graphs/exploration"

# ============================================================
# COLUMN CONFIG
# ============================================================
DATE_COL = "date"
TWEET_VOL_COL = "tweet_volume"
CASES_COL = "daily_new_cases"
DEATHS_COL = "daily_new_deaths"
STRINGENCY_COL = "StringencyIndex_Average"

MAX_LAG = 30

# Pandemic phases for segmented analysis
PHASES = {
    "Phase 1: Initial Outbreak\n(Mar-Aug 2020)": ("2020-03-19", "2020-08-31"),
    "Phase 2: Second Wave\n(Sep 2020-Feb 2021)": ("2020-09-01", "2021-02-28"),
    "Phase 3: Vaccination Era\n(Mar-Sep 2021)": ("2021-03-01", "2021-09-30"),
    "Phase 4: Omicron\n(Oct 2021-Mar 2022)": ("2021-10-01", "2022-03-31"),
    "Phase 5: Post-Restrictions\n(Apr-Dec 2022)": ("2022-04-01", "2022-12-31"),
}

# Key events for annotation
EVENTS = {
    "2020-03-23": "UK lockdown",
    "2020-12-08": "Vaccination",
    "2021-01-06": "3rd lockdown",
    "2021-07-19": "Freedom Day",
    "2022-02-24": "Restrictions end",
}


def compute_xcorr(x, y, max_lag=30):
    """Cross-correlation for lags in [-max_lag, +max_lag]."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    # Remove NaN pairs
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    if len(x) < 20:
        return np.arange(-max_lag, max_lag + 1), np.full(2 * max_lag + 1, np.nan)

    x = (x - x.mean()) / x.std()
    y = (y - y.mean()) / y.std()
    n = len(x)
    lags = np.arange(-max_lag, max_lag + 1)
    corrs = np.zeros(len(lags))
    for i, lag in enumerate(lags):
        if lag >= 0:
            if n - lag < 10:
                corrs[i] = np.nan
            else:
                corrs[i] = np.corrcoef(x[:n - lag], y[lag:])[0, 1]
        else:
            if n + lag < 10:
                corrs[i] = np.nan
            else:
                corrs[i] = np.corrcoef(x[-lag:], y[:n + lag])[0, 1]
    return lags, corrs


def find_peak(lags, corrs):
    """Find lag with max absolute correlation, ignoring NaN."""
    valid = ~np.isnan(corrs)
    if not valid.any():
        return 0, 0.0
    idx = np.nanargmax(np.abs(corrs))
    return lags[idx], corrs[idx]


def sig_threshold(n):
    from scipy.stats import norm
    return norm.ppf(0.975) / np.sqrt(n)


# ==================================================================
# PART 1: DIFFERENCED (DETRENDED) GLOBAL CROSS-CORRELATION
# ==================================================================
def run_differenced_analysis():
    print("=" * 70)
    print("PART 1: DIFFERENCED (DETRENDED) CROSS-CORRELATION")
    print("=" * 70)

    df = pd.read_csv(INPUT_DIFF)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)

    print(f"Loaded differenced data: {len(df)} rows")
    print(f"Columns: {list(df.columns)}")

    # Try to find differenced column names
    # Common patterns: diff_tweet_volume, tweet_volume_diff, d_tweet_volume
    def find_col(base, df_cols):
        candidates = [
            f"diff_{base}", f"{base}_diff", f"d_{base}",
            f"delta_{base}", f"{base}_delta", f"{base}_change",
            base,  # might already be differenced
        ]
        for c in candidates:
            if c in df_cols:
                return c
        # Fuzzy match
        for c in df_cols:
            if base.replace("_", "") in c.replace("_", "").lower():
                return c
        return None

    cols = list(df.columns)
    tv_col = find_col(TWEET_VOL_COL, cols)
    cases_col = find_col(CASES_COL, cols)
    deaths_col = find_col(DEATHS_COL, cols)
    str_col = find_col(STRINGENCY_COL, cols)

    print(f"\nDetected columns:")
    print(f"  Tweet volume: {tv_col}")
    print(f"  Cases:        {cases_col}")
    print(f"  Deaths:       {deaths_col}")
    print(f"  Stringency:   {str_col}")

    if tv_col is None or cases_col is None:
        print("\nERROR: Cannot find required columns in differenced data.")
        print(f"Available columns: {cols}")
        print("Please update column names at the top of this script.")
        return None

    n = len(df)
    sig = sig_threshold(n)
    print(f"Significance threshold: ±{sig:.4f}")

    # Compute cross-correlations
    pairs = []
    if cases_col:
        pairs.append((cases_col, "Daily New Cases (differenced)", "steelblue"))
    if deaths_col:
        pairs.append((deaths_col, "Daily New Deaths (differenced)", "firebrick"))
    if str_col:
        pairs.append((str_col, "Stringency Index (differenced)", "deeppink"))

    results = {}
    for col, label, color in pairs:
        lags, corrs = compute_xcorr(df[tv_col].values, df[col].values, MAX_LAG)
        peak_lag, peak_corr = find_peak(lags, corrs)
        results[col] = {
            "lags": lags, "corrs": corrs,
            "peak_lag": peak_lag, "peak_corr": peak_corr,
            "label": label, "color": color,
        }
        print(f"\n  {label}:")
        print(f"    Peak: lag={peak_lag}, r={peak_corr:.4f}")
        at_boundary = abs(peak_lag) >= MAX_LAG - 1
        print(f"    At boundary: {'YES ⚠️' if at_boundary else 'No ✓'}")

    # ---- PLOT ----
    n_plots = len(pairs)
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 6))
    if n_plots == 1:
        axes = [axes]
    fig.suptitle("Cross-Correlation on Differenced (Detrended) Data",
                 fontsize=14, fontweight="bold")

    for i, (col, label, color) in enumerate(pairs):
        ax = axes[i]
        r = results[col]
        ax.bar(r["lags"], r["corrs"], color=color, alpha=0.7, width=0.8)
        ax.axhline(sig, color="gray", linestyle=":", alpha=0.7, label=f"95% sig (±{sig:.3f})")
        ax.axhline(-sig, color="gray", linestyle=":", alpha=0.7)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.axvline(0, color="black", linewidth=0.5, alpha=0.3)

        # Mark peak
        ax.plot(r["peak_lag"], r["peak_corr"], "v", color="black", markersize=12, zorder=5)
        ax.annotate(f"lag={r['peak_lag']}\nr={r['peak_corr']:.3f}",
                    xy=(r["peak_lag"], r["peak_corr"]),
                    xytext=(r["peak_lag"] + 5, r["peak_corr"] + 0.03 * np.sign(r["peak_corr"])),
                    fontsize=9, fontweight="bold",
                    arrowprops=dict(arrowstyle="->"),
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))

        ax.set_xlabel("Lag (days)\n← Tweet vol leads | Tweet vol lags →")
        ax.set_ylabel("Correlation")
        ax.set_title(f"Tweet Volume vs\n{label}", fontsize=11)
        ax.legend(loc="upper right", fontsize=8)
        ax.set_xlim(-MAX_LAG - 1, MAX_LAG + 1)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "05_xcorr_differenced.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {path}")
    plt.close()

    return results


# ==================================================================
# PART 2: PHASE-SEGMENTED CROSS-CORRELATION
# ==================================================================
def run_phase_analysis():
    print("\n" + "=" * 70)
    print("PART 2: PHASE-SEGMENTED CROSS-CORRELATION")
    print("=" * 70)

    df = pd.read_csv(INPUT_RAW)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)

    # 7-day MA
    for col in [TWEET_VOL_COL, CASES_COL, DEATHS_COL]:
        df[f"{col}_7d"] = df[col].rolling(7, center=True).mean()
    if STRINGENCY_COL in df.columns:
        df[f"{STRINGENCY_COL}_7d"] = df[STRINGENCY_COL].rolling(7, center=True).mean()

    print(f"Loaded raw data: {len(df)} rows")

    targets = [
        (f"{CASES_COL}_7d", "Cases", "steelblue"),
        (f"{DEATHS_COL}_7d", "Deaths", "firebrick"),
    ]

    phase_results = {}

    for phase_name, (start, end) in PHASES.items():
        mask = (df[DATE_COL] >= start) & (df[DATE_COL] <= end)
        phase_df = df[mask].copy()
        n = len(phase_df)

        if n < 30:
            print(f"\n{phase_name}: Only {n} rows, skipping")
            continue

        print(f"\n{phase_name} ({start} to {end}): {n} days")
        phase_results[phase_name] = {"n": n, "start": start, "end": end}

        for col, label, color in targets:
            # Detrend within phase using differencing
            tv_diff = phase_df[f"{TWEET_VOL_COL}_7d"].diff().dropna().values
            var_diff = phase_df[col].diff().dropna().values
            min_len = min(len(tv_diff), len(var_diff))
            tv_diff = tv_diff[:min_len]
            var_diff = var_diff[:min_len]

            lags, corrs = compute_xcorr(tv_diff, var_diff, min(MAX_LAG, n // 3))
            peak_lag, peak_corr = find_peak(lags, corrs)

            phase_results[phase_name][label] = {
                "lags": lags, "corrs": corrs,
                "peak_lag": peak_lag, "peak_corr": peak_corr,
            }
            print(f"  vs {label}: peak lag={peak_lag}, r={peak_corr:.4f}")

    # ---- PLOT: Phase-segmented cross-correlation ----
    n_phases = len(phase_results)
    fig, axes = plt.subplots(n_phases, 2, figsize=(14, 4 * n_phases))
    fig.suptitle("Phase-Segmented Cross-Correlation (Differenced Within Phase)",
                 fontsize=14, fontweight="bold", y=1.01)

    for i, (phase_name, pdata) in enumerate(phase_results.items()):
        for j, (label, color) in enumerate([("Cases", "steelblue"), ("Deaths", "firebrick")]):
            ax = axes[i, j] if n_phases > 1 else axes[j]
            if label not in pdata:
                continue
            r = pdata[label]
            ax.bar(r["lags"], r["corrs"], color=color, alpha=0.7, width=0.8)

            sig = sig_threshold(pdata["n"])
            ax.axhline(sig, color="gray", linestyle=":", alpha=0.5)
            ax.axhline(-sig, color="gray", linestyle=":", alpha=0.5)
            ax.axhline(0, color="black", linewidth=0.5)
            ax.axvline(0, color="black", linewidth=0.5, alpha=0.3)

            ax.plot(r["peak_lag"], r["peak_corr"], "v", color="black", markersize=10)
            ax.set_title(f"{phase_name}\nvs {label} (peak: lag={r['peak_lag']}, r={r['peak_corr']:.3f})",
                         fontsize=9)
            ax.set_xlim(-MAX_LAG - 1, MAX_LAG + 1)

            if i == n_phases - 1:
                ax.set_xlabel("Lag (days)")
            if j == 0:
                ax.set_ylabel("Correlation")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "06_xcorr_by_phase.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {path}")
    plt.close()

    # ---- PLOT: Summary heatmap ----
    phases_list = list(phase_results.keys())
    targets_list = ["Cases", "Deaths"]

    peak_lag_matrix = np.zeros((len(phases_list), len(targets_list)))
    peak_corr_matrix = np.zeros((len(phases_list), len(targets_list)))

    for i, phase in enumerate(phases_list):
        for j, target in enumerate(targets_list):
            if target in phase_results[phase]:
                peak_lag_matrix[i, j] = phase_results[phase][target]["peak_lag"]
                peak_corr_matrix[i, j] = phase_results[phase][target]["peak_corr"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Phase-Segmented Summary", fontsize=14, fontweight="bold")

    # Peak lag heatmap
    ax = axes[0]
    im = ax.imshow(peak_lag_matrix, cmap="RdBu_r", aspect="auto",
                   vmin=-MAX_LAG, vmax=MAX_LAG)
    ax.set_xticks(range(len(targets_list)))
    ax.set_xticklabels(targets_list)
    ax.set_yticks(range(len(phases_list)))
    ax.set_yticklabels([p.replace("\n", " ") for p in phases_list], fontsize=8)
    ax.set_title("Peak Lag (days)\nBlue=tweet leads, Red=tweet lags")
    for i in range(len(phases_list)):
        for j in range(len(targets_list)):
            ax.text(j, i, f"{peak_lag_matrix[i,j]:.0f}",
                    ha="center", va="center", fontsize=11, fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.8)

    # Peak correlation heatmap
    ax = axes[1]
    im = ax.imshow(peak_corr_matrix, cmap="RdYlGn", aspect="auto",
                   vmin=-0.5, vmax=0.5)
    ax.set_xticks(range(len(targets_list)))
    ax.set_xticklabels(targets_list)
    ax.set_yticks(range(len(phases_list)))
    ax.set_yticklabels([p.replace("\n", " ") for p in phases_list], fontsize=8)
    ax.set_title("Peak Correlation (r)")
    for i in range(len(phases_list)):
        for j in range(len(targets_list)):
            ax.text(j, i, f"{peak_corr_matrix[i,j]:.3f}",
                    ha="center", va="center", fontsize=11, fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "07_phase_summary_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()

    return phase_results


# ==================================================================
# PART 3: INTERPRETATION
# ==================================================================
def interpret(diff_results, phase_results):
    print("\n" + "=" * 70)
    print("COMBINED INTERPRETATION")
    print("=" * 70)

    if diff_results:
        print("\n--- Differenced (Detrended) Global Results ---")
        for col, r in diff_results.items():
            at_boundary = abs(r["peak_lag"]) >= MAX_LAG - 1
            print(f"  {r['label']}:")
            print(f"    Peak: lag={r['peak_lag']}, r={r['peak_corr']:.4f}")
            if at_boundary:
                print(f"    ⚠️  Still at boundary — relationship may be very weak globally")
            elif abs(r["peak_corr"]) < 0.1:
                print(f"    ⚠️  Very weak correlation — limited practical significance")
            else:
                if r["peak_lag"] > 0:
                    print(f"    ✓ Tweet volume lags {r['label'].split('(')[0].strip()} by {r['peak_lag']} days")
                elif r["peak_lag"] < 0:
                    print(f"    ✓ Tweet volume leads {r['label'].split('(')[0].strip()} by {-r['peak_lag']} days")
                else:
                    print(f"    ✓ Simultaneous response")

    if phase_results:
        print("\n--- Phase-Segmented Results ---")
        for phase, pdata in phase_results.items():
            phase_short = phase.replace("\n", " ")
            print(f"\n  {phase_short}:")
            for target in ["Cases", "Deaths"]:
                if target in pdata:
                    r = pdata[target]
                    strength = "strong" if abs(r["peak_corr"]) > 0.3 else \
                               "moderate" if abs(r["peak_corr"]) > 0.15 else "weak"
                    print(f"    vs {target}: lag={r['peak_lag']}, r={r['peak_corr']:.3f} ({strength})")

        # Check for fatigue pattern
        print("\n--- Fatigue Effect Check ---")
        for target in ["Cases", "Deaths"]:
            corrs = []
            for phase, pdata in phase_results.items():
                if target in pdata:
                    corrs.append(abs(pdata[target]["peak_corr"]))
            if len(corrs) >= 3:
                if corrs[0] > corrs[-1]:
                    print(f"  {target}: correlation decreases over time "
                          f"({corrs[0]:.3f} → {corrs[-1]:.3f}) — suggests fatigue ✓")
                else:
                    print(f"  {target}: correlation does NOT decrease "
                          f"({corrs[0]:.3f} → {corrs[-1]:.3f}) — no fatigue signal")

    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print("1. If differenced analysis shows clear non-boundary peaks:")
    print("   → Proceed to Granger causality on differenced data")
    print("2. If phase analysis shows varying correlation strength:")
    print("   → Explore fatigue effect with regression (RQ3)")
    print("3. If specific phases show strong signal:")
    print("   → Focus detailed modelling on those phases")


# ==================================================================
# MAIN
# ==================================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Part 1
    diff_results = run_differenced_analysis()

    # Part 2
    phase_results = run_phase_analysis()

    # Part 3
    interpret(diff_results, phase_results)

    # Save summary table
    rows = []
    if phase_results:
        for phase, pdata in phase_results.items():
            for target in ["Cases", "Deaths"]:
                if target in pdata:
                    rows.append({
                        "phase": phase.replace("\n", " "),
                        "target": target,
                        "peak_lag": pdata[target]["peak_lag"],
                        "peak_corr": pdata[target]["peak_corr"],
                        "n_days": pdata["n"],
                    })
    if rows:
        summary = pd.DataFrame(rows)
        path = os.path.join(OUTPUT_DIR, "phase_xcorr_summary.csv")
        summary.to_csv(path, index=False)
        print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()