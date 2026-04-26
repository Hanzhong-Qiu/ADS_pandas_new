"""
Cross-Correlation Analysis: Tweet Volume vs COVID Cases/Deaths
===============================================================
Step 1 of the analysis chain.
Explores the temporal relationship between tweet volume and pandemic indicators.

USAGE:
    python cross_correlation.py --input /path/to/merged_daily_data.csv --output ./cross_corr_output

INPUT: Your merged daily CSV with columns including:
    - date
    - tweet_volume (or tweet_count)
    - daily_new_cases
    - daily_new_deaths
    - StringencyIndex_Average (optional, for bonus analysis)

Adjust COLUMN CONFIG below if your column names differ.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats, signal
import argparse
import os
import sys
import warnings
warnings.filterwarnings("ignore")


# ============================================================
# COLUMN CONFIG - UPDATE THESE TO MATCH YOUR DATA
# ============================================================
DATE_COL = "date"
TWEET_VOL_COL = "tweet_volume"          # daily tweet count
CASES_COL = "daily_new_cases"            # daily new confirmed cases
DEATHS_COL = "daily_new_deaths"          # daily new deaths
STRINGENCY_COL = "StringencyIndex_Average"  # Oxford stringency index
# ============================================================

MAX_LAG = 30  # check lags from -30 to +30 days


def load_and_prepare(filepath):
    """Load merged daily data and prepare for analysis."""
    print(f"Loading: {filepath}")
    df = pd.read_csv(filepath)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)

    # Check columns
    required = [DATE_COL, TWEET_VOL_COL, CASES_COL, DEATHS_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"ERROR: Missing columns: {missing}")
        print(f"Available: {list(df.columns)}")
        sys.exit(1)

    print(f"Rows: {len(df)}, Date range: {df[DATE_COL].min().date()} to {df[DATE_COL].max().date()}")
    print(f"\nBasic stats:")
    for col in [TWEET_VOL_COL, CASES_COL, DEATHS_COL]:
        print(f"  {col}: mean={df[col].mean():.1f}, std={df[col].std():.1f}, "
              f"NaN={df[col].isna().sum()}")

    # Apply 7-day rolling mean to smooth weekly cycles
    for col in [TWEET_VOL_COL, CASES_COL, DEATHS_COL]:
        df[f"{col}_7d"] = df[col].rolling(7, center=True).mean()

    if STRINGENCY_COL in df.columns:
        df[f"{STRINGENCY_COL}_7d"] = df[STRINGENCY_COL].rolling(7, center=True).mean()

    df = df.dropna(subset=[f"{TWEET_VOL_COL}_7d", f"{CASES_COL}_7d", f"{DEATHS_COL}_7d"])
    print(f"After smoothing & dropping NaN: {len(df)} rows")

    return df


def compute_cross_correlation(x, y, max_lag=30):
    """
    Compute normalized cross-correlation between x and y for lags in [-max_lag, +max_lag].

    Positive lag means y is shifted forward (x leads y).
    Negative lag means y is shifted backward (y leads x).

    Returns: lags array, correlation array
    """
    x = (x - x.mean()) / x.std()
    y = (y - y.mean()) / y.std()
    n = len(x)
    lags = np.arange(-max_lag, max_lag + 1)
    corrs = np.zeros(len(lags))

    for i, lag in enumerate(lags):
        if lag >= 0:
            corrs[i] = np.corrcoef(x[:n - lag], y[lag:])[0, 1]
        else:
            corrs[i] = np.corrcoef(x[-lag:], y[:n + lag])[0, 1]

    return lags, corrs


def find_peak_lag(lags, corrs):
    """Find the lag with maximum absolute correlation."""
    idx = np.argmax(np.abs(corrs))
    return lags[idx], corrs[idx]


def bootstrap_confidence(x, y, max_lag, n_boot=1000, ci=0.95):
    """Bootstrap confidence interval for cross-correlation at each lag."""
    n = len(x)
    lags = np.arange(-max_lag, max_lag + 1)
    boot_corrs = np.zeros((n_boot, len(lags)))

    for b in range(n_boot):
        # Block bootstrap to preserve temporal structure
        block_size = 14
        n_blocks = n // block_size + 1
        indices = []
        for _ in range(n_blocks):
            start = np.random.randint(0, n - block_size)
            indices.extend(range(start, start + block_size))
        indices = np.array(indices[:n])

        x_boot = x.values[indices]
        y_boot = y.values[indices]
        x_boot = (x_boot - x_boot.mean()) / x_boot.std()
        y_boot = (y_boot - y_boot.mean()) / y_boot.std()

        for i, lag in enumerate(lags):
            if lag >= 0:
                boot_corrs[b, i] = np.corrcoef(x_boot[:n - lag], y_boot[lag:])[0, 1]
            else:
                boot_corrs[b, i] = np.corrcoef(x_boot[-lag:], y_boot[:n + lag])[0, 1]

    alpha = (1 - ci) / 2
    ci_lower = np.percentile(boot_corrs, alpha * 100, axis=0)
    ci_upper = np.percentile(boot_corrs, (1 - alpha) * 100, axis=0)

    return ci_lower, ci_upper


def significance_threshold(n, ci=0.95):
    """Approximate significance threshold for cross-correlation."""
    from scipy.stats import norm
    z = norm.ppf((1 + ci) / 2)
    return z / np.sqrt(n)


def plot_cross_correlations(results, df, output_dir):
    """
    Main visualization: cross-correlation plots for all variable pairs.
    """
    # ----------------------------------------------------------------
    # FIGURE 1: Cross-correlation plots side by side
    # ----------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Cross-Correlation: Tweet Volume vs Pandemic Indicators (7-day MA)",
                 fontsize=14, fontweight="bold")

    pairs = [
        (f"{CASES_COL}_7d", "Daily New Cases", "steelblue"),
        (f"{DEATHS_COL}_7d", "Daily New Deaths", "firebrick"),
    ]

    for i, (var, label, color) in enumerate(pairs):
        ax = axes[i]
        r = results[var]
        lags, corrs = r["lags"], r["corrs"]
        peak_lag, peak_corr = r["peak_lag"], r["peak_corr"]
        ci_lo, ci_hi = r["ci_lower"], r["ci_upper"]

        # Plot correlation curve
        ax.plot(lags, corrs, color=color, linewidth=2, label="Cross-correlation")

        # Confidence interval
        ax.fill_between(lags, ci_lo, ci_hi, alpha=0.15, color=color,
                        label="95% CI (bootstrap)")

        # Significance threshold
        sig = r["sig_threshold"]
        ax.axhline(sig, color="gray", linestyle=":", alpha=0.5)
        ax.axhline(-sig, color="gray", linestyle=":", alpha=0.5)

        # Mark peak
        ax.axvline(peak_lag, color="black", linestyle="--", alpha=0.5)
        ax.plot(peak_lag, peak_corr, "o", color="black", markersize=10, zorder=5)
        ax.annotate(f"Peak: lag={peak_lag}, r={peak_corr:.3f}",
                    xy=(peak_lag, peak_corr),
                    xytext=(peak_lag + 3, peak_corr + 0.05),
                    fontsize=10, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="black"),
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))

        ax.axhline(0, color="black", linewidth=0.5)
        ax.axvline(0, color="black", linewidth=0.5, alpha=0.3)
        ax.set_xlabel("Lag (days)\n← Tweet volume leads | Tweet volume lags →")
        ax.set_ylabel("Pearson Correlation")
        ax.set_title(f"Tweet Volume vs {label}")
        ax.legend(loc="upper right", fontsize=9)
        ax.set_xlim(-MAX_LAG, MAX_LAG)

    plt.tight_layout()
    path = os.path.join(output_dir, "01_cross_correlation_cases_deaths.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()

    # ----------------------------------------------------------------
    # FIGURE 2: Overlay comparison
    # ----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 6))
    for var, label, color in pairs:
        r = results[var]
        ax.plot(r["lags"], r["corrs"], color=color, linewidth=2, label=label)
        ax.plot(r["peak_lag"], r["peak_corr"], "o", color=color, markersize=10, zorder=5)

    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.5, alpha=0.3)
    ax.set_xlabel("Lag (days)\n← Tweet volume leads | Tweet volume lags →")
    ax.set_ylabel("Pearson Correlation")
    ax.set_title("Cross-Correlation Comparison: Cases vs Deaths", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_xlim(-MAX_LAG, MAX_LAG)

    plt.tight_layout()
    path = os.path.join(output_dir, "02_cross_correlation_overlay.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()

    # ----------------------------------------------------------------
    # FIGURE 3: Time-aligned visualization at optimal lag
    # ----------------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    fig.suptitle("Time Series Aligned at Optimal Lag", fontsize=14, fontweight="bold")

    for i, (var, label, color) in enumerate(pairs):
        ax = axes[i]
        r = results[var]
        lag = r["peak_lag"]

        # Normalize for visual comparison
        tv = df[f"{TWEET_VOL_COL}_7d"].values
        other = df[var].values
        tv_norm = (tv - np.nanmean(tv)) / np.nanstd(tv)
        other_norm = (other - np.nanmean(other)) / np.nanstd(other)

        dates = df[DATE_COL].values

        ax.plot(dates, tv_norm, color="green", linewidth=1.5,
                label="Tweet Volume (normalized)", alpha=0.8)

        if lag >= 0:
            # Shift other forward by lag days to align
            shifted_dates = dates[lag:]
            shifted_vals = other_norm[:len(dates) - lag]
            ax.plot(shifted_dates, shifted_vals, color=color, linewidth=1.5,
                    label=f"{label} (shifted {lag} days forward)", alpha=0.8)
            r_aligned = np.corrcoef(
                tv_norm[lag:],
                other_norm[:len(dates) - lag]
            )[0, 1]
        else:
            shifted_dates = dates[:len(dates) + lag]
            shifted_vals = other_norm[-lag:]
            ax.plot(shifted_dates, shifted_vals, color=color, linewidth=1.5,
                    label=f"{label} (shifted {-lag} days backward)", alpha=0.8)
            r_aligned = np.corrcoef(
                tv_norm[:len(dates) + lag],
                other_norm[-lag:]
            )[0, 1]

        ax.set_ylabel("Normalized Value (z-score)")
        ax.set_title(f"Tweet Volume vs {label} — aligned at lag={lag} (r={r_aligned:.3f})")
        ax.legend(loc="upper right")

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    plt.tight_layout()

    path = os.path.join(output_dir, "03_time_aligned_at_optimal_lag.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()

    # ----------------------------------------------------------------
    # FIGURE 4: Stringency Index cross-correlation (if available)
    # ----------------------------------------------------------------
    stringency_key = f"{STRINGENCY_COL}_7d"
    if stringency_key in results:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle("Cross-Correlation: Tweet Volume & Stringency vs Cases",
                     fontsize=14, fontweight="bold")

        # Tweet vol vs cases (repeat for context)
        r1 = results[f"{CASES_COL}_7d"]
        axes[0].plot(r1["lags"], r1["corrs"], color="steelblue", linewidth=2)
        axes[0].plot(r1["peak_lag"], r1["peak_corr"], "o", color="steelblue", markersize=10)
        axes[0].axhline(0, color="black", linewidth=0.5)
        axes[0].axvline(0, color="black", linewidth=0.5, alpha=0.3)
        axes[0].set_xlabel("Lag (days)")
        axes[0].set_ylabel("Correlation")
        axes[0].set_title(f"Tweet Volume vs Cases\n(peak lag={r1['peak_lag']}, r={r1['peak_corr']:.3f})")

        # Stringency vs cases
        r2 = results[stringency_key]
        axes[1].plot(r2["lags"], r2["corrs"], color="deeppink", linewidth=2)
        axes[1].plot(r2["peak_lag"], r2["peak_corr"], "o", color="deeppink", markersize=10)
        axes[1].axhline(0, color="black", linewidth=0.5)
        axes[1].axvline(0, color="black", linewidth=0.5, alpha=0.3)
        axes[1].set_xlabel("Lag (days)")
        axes[1].set_ylabel("Correlation")
        axes[1].set_title(f"Stringency vs Cases\n(peak lag={r2['peak_lag']}, r={r2['peak_corr']:.3f})")

        plt.tight_layout()
        path = os.path.join(output_dir, "04_tweet_vs_stringency_response_speed.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
        plt.close()


def print_interpretation(results):
    """Print interpretation guide based on results."""
    print("\n" + "=" * 70)
    print("CROSS-CORRELATION RESULTS & INTERPRETATION")
    print("=" * 70)

    cases_r = results[f"{CASES_COL}_7d"]
    deaths_r = results[f"{DEATHS_COL}_7d"]

    print(f"\n1. Tweet Volume vs Daily New Cases:")
    print(f"   Peak correlation: r = {cases_r['peak_corr']:.4f} at lag = {cases_r['peak_lag']} days")
    if cases_r["peak_lag"] > 0:
        print(f"   → Tweet volume LAGS cases by {cases_r['peak_lag']} days")
        print(f"     (Public discussion reacts to case reports)")
    elif cases_r["peak_lag"] < 0:
        print(f"   → Tweet volume LEADS cases by {-cases_r['peak_lag']} days")
        print(f"     (Social media may act as early warning signal)")
    else:
        print(f"   → Tweet volume responds SIMULTANEOUSLY with cases")

    print(f"\n2. Tweet Volume vs Daily New Deaths:")
    print(f"   Peak correlation: r = {deaths_r['peak_corr']:.4f} at lag = {deaths_r['peak_lag']} days")
    if deaths_r["peak_lag"] > 0:
        print(f"   → Tweet volume LAGS deaths by {deaths_r['peak_lag']} days")
    elif deaths_r["peak_lag"] < 0:
        print(f"   → Tweet volume LEADS deaths by {-deaths_r['peak_lag']} days")
    else:
        print(f"   → Tweet volume responds SIMULTANEOUSLY with deaths")

    print(f"\n3. Comparison:")
    if abs(cases_r["peak_corr"]) > abs(deaths_r["peak_corr"]):
        print(f"   Cases show STRONGER correlation with tweet volume "
              f"({abs(cases_r['peak_corr']):.3f} vs {abs(deaths_r['peak_corr']):.3f})")
    else:
        print(f"   Deaths show STRONGER correlation with tweet volume "
              f"({abs(deaths_r['peak_corr']):.3f} vs {abs(cases_r['peak_corr']):.3f})")

    lag_diff = cases_r["peak_lag"] - deaths_r["peak_lag"]
    if lag_diff != 0:
        print(f"   Lag difference: {abs(lag_diff)} days "
              f"({'cases responded faster' if cases_r['peak_lag'] < deaths_r['peak_lag'] else 'deaths responded faster'})")

    # Stringency comparison
    stringency_key = f"{STRINGENCY_COL}_7d"
    if stringency_key in results:
        str_r = results[stringency_key]
        print(f"\n4. Response Speed Comparison (Tweet Volume vs Stringency):")
        print(f"   Tweet volume peak lag vs cases: {cases_r['peak_lag']} days")
        print(f"   Stringency peak lag vs cases:   {str_r['peak_lag']} days")
        speed_diff = str_r["peak_lag"] - cases_r["peak_lag"]
        if speed_diff > 0:
            print(f"   → Social media responds {speed_diff} days FASTER than policy")
        elif speed_diff < 0:
            print(f"   → Policy responds {-speed_diff} days FASTER than social media")
        else:
            print(f"   → Both respond at similar speed")

    # Decide next step
    print(f"\n{'=' * 70}")
    print("RECOMMENDED NEXT STEP")
    print("=" * 70)

    has_clear_peak = abs(cases_r["peak_corr"]) > cases_r["sig_threshold"]

    if has_clear_peak:
        print("✓ Clear cross-correlation signal detected.")
        print("  → Proceed to Step 2A: Granger causality test")
        print(f"     Use lag range around {max(1, abs(cases_r['peak_lag']) - 5)} "
              f"to {abs(cases_r['peak_lag']) + 5} days")
    else:
        print("✗ No clear global cross-correlation peak.")
        print("  → Proceed to Step 2B: Phase-segmented cross-correlation")
        print("    (The relationship may vary across pandemic phases)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True,
                        help="Path to merged daily CSV")
    parser.add_argument("--output", "-o", default="./cross_corr_output",
                        help="Output directory")
    parser.add_argument("--bootstrap", action="store_true",
                        help="Run bootstrap CI (slower, ~2 min)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Load data
    df = load_and_prepare(args.input)
    n = len(df)

    # Significance threshold
    sig_thresh = significance_threshold(n)
    print(f"\nSignificance threshold (95%): ±{sig_thresh:.4f}")

    # Compute cross-correlations
    results = {}
    pairs = [
        (f"{CASES_COL}_7d", "Daily New Cases"),
        (f"{DEATHS_COL}_7d", "Daily New Deaths"),
    ]

    # Also do stringency vs cases for response speed comparison
    if STRINGENCY_COL in df.columns:
        pairs.append((f"{STRINGENCY_COL}_7d", "Stringency Index"))

    for var, label in pairs:
        print(f"\nComputing cross-correlation: Tweet Volume vs {label}...")

        # For stringency, correlate against cases instead of tweet volume
        if "Stringency" in var:
            x = df[f"{CASES_COL}_7d"]
            y = df[var]
        else:
            x = df[f"{TWEET_VOL_COL}_7d"]
            y = df[var]

        lags, corrs = compute_cross_correlation(x, y, MAX_LAG)
        peak_lag, peak_corr = find_peak_lag(lags, corrs)

        result = {
            "lags": lags,
            "corrs": corrs,
            "peak_lag": peak_lag,
            "peak_corr": peak_corr,
            "sig_threshold": sig_thresh,
            "ci_lower": np.full_like(corrs, -sig_thresh),
            "ci_upper": np.full_like(corrs, sig_thresh),
        }

        # Optional bootstrap CI
        if args.bootstrap:
            print(f"  Running bootstrap (1000 iterations)...")
            ci_lo, ci_hi = bootstrap_confidence(x, y, MAX_LAG, n_boot=1000)
            result["ci_lower"] = ci_lo
            result["ci_upper"] = ci_hi

        print(f"  Peak: lag={peak_lag}, r={peak_corr:.4f}")
        results[var] = result

    # Save numerical results
    summary_rows = []
    for var, label in pairs:
        r = results[var]
        summary_rows.append({
            "variable": label,
            "peak_lag": r["peak_lag"],
            "peak_correlation": r["peak_corr"],
            "significant": "Yes" if abs(r["peak_corr"]) > sig_thresh else "No",
        })
        # Save full correlation curve
        curve_df = pd.DataFrame({"lag": r["lags"], "correlation": r["corrs"]})
        curve_df.to_csv(
            os.path.join(args.output, f"xcorr_{var.replace('_7d', '')}.csv"),
            index=False
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(args.output, "cross_correlation_summary.csv"), index=False)
    print(f"\nSaved summary: {os.path.join(args.output, 'cross_correlation_summary.csv')}")

    # Plots
    print("\nGenerating plots...")
    plot_cross_correlations(results, df, args.output)

    # Interpretation
    print_interpretation(results)


if __name__ == "__main__":
    main()