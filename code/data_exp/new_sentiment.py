"""
Zero-Sentiment Impact Test
===========================
Tests whether removing zero-sentiment tweets reveals real sentiment changes.

USAGE:
    python zero_test.py --input /path/to/all_covid_data --output ./zero_test_output
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import sys
import argparse
import glob


DATE_COL = "date"
SENTIMENT_COL = "sentiment"
HIST_BINS = 200
HIST_RANGE = (-1.0, 1.0)


def load_data(filepath):
    """Load raw tweet data from single CSV or directory of CSVs."""
    if os.path.isdir(filepath):
        print(f"Loading CSVs from directory: {filepath}")
        files = sorted(glob.glob(os.path.join(filepath, "*.csv")))
        print(f"Found {len(files)} CSV files")
        chunks = []
        skipped = 0
        for i, f in enumerate(files):
            try:
                df = pd.read_csv(f, header=None, names=["tweet_id", SENTIMENT_COL])
                # Extract date from tweet_id (Snowflake ID) or from filename
                # Try parsing sentiment as numeric
                df[SENTIMENT_COL] = pd.to_numeric(df[SENTIMENT_COL], errors="coerce")
                df = df.dropna(subset=[SENTIMENT_COL])
                chunks.append(df)
            except Exception:
                skipped += 1
            if (i + 1) % 100 == 0:
                print(f"  Processed {i+1}/{len(files)}")
        print(f"Loaded {len(chunks)} files, skipped {skipped}")
        df = pd.concat(chunks, ignore_index=True)
    else:
        df = pd.read_csv(filepath, low_memory=False)
        if DATE_COL not in df.columns and "tweet_id" not in df.columns:
            df = pd.read_csv(
                filepath,
                header=None,
                names=["tweet_id", SENTIMENT_COL],
                usecols=[0, 1],
                low_memory=False,
            )
    
    # Try to get date column
    if DATE_COL not in df.columns:
        # If there's a tweet_id column, try Snowflake ID date extraction
        if "tweet_id" in df.columns:
            print("Extracting dates from tweet IDs (Snowflake)...")
            df["tweet_id"] = pd.to_numeric(df["tweet_id"], errors="coerce")
            df = df.dropna(subset=["tweet_id"])
            df["tweet_id"] = df["tweet_id"].astype(np.int64)
            # Twitter Snowflake: (id >> 22) + 1288834974657
            timestamps_ms = (df["tweet_id"] // (2 ** 22)) + 1288834974657
            df[DATE_COL] = pd.to_datetime(timestamps_ms, unit="ms", errors="coerce")
            df = df.dropna(subset=[DATE_COL])
        else:
            print(f"ERROR: No '{DATE_COL}' column found. Columns: {list(df.columns)}")
            sys.exit(1)
    else:
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
        df = df.dropna(subset=[DATE_COL])
    
    df[SENTIMENT_COL] = pd.to_numeric(df[SENTIMENT_COL], errors="coerce")
    df = df.dropna(subset=[SENTIMENT_COL])
    
    print(f"Total rows: {len(df):,}")
    print(f"Date range: {df[DATE_COL].min()} to {df[DATE_COL].max()}")
    return df


def compute_metrics_from_series(sentiment_series):
    """Compute one row of sentiment metrics from a numeric series."""
    sentiment_series = pd.to_numeric(sentiment_series, errors="coerce").dropna()
    return {
        "mean": sentiment_series.mean(),
        "std": sentiment_series.std(),
        "median": sentiment_series.median(),
        "count": int(sentiment_series.count()),
        "negative_ratio": (sentiment_series < 0).mean(),
        "strong_neg_ratio": (sentiment_series < -0.5).mean(),
        "positive_ratio": (sentiment_series > 0).mean(),
        "strong_pos_ratio": (sentiment_series > 0.5).mean(),
        "skewness": sentiment_series.skew(),
        "polarization": ((sentiment_series < -0.5) | (sentiment_series > 0.5)).mean(),
        "q10": sentiment_series.quantile(0.10),
        "q90": sentiment_series.quantile(0.90),
    }


def finalize_daily_metrics(daily, label="all"):
    """Sort daily metrics and add rolling averages."""
    daily = daily.copy()
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values("date").reset_index(drop=True)

    for col in ["mean", "negative_ratio", "strong_neg_ratio", "skewness",
                 "polarization", "q10", "q90", "std"]:
        daily[f"{col}_7d"] = daily[col].rolling(7, center=True).mean()

    daily["label"] = label
    return daily


def compute_metrics(df, label="all"):
    """Compute daily sentiment metrics for an in-memory dataframe."""
    daily_rows = []
    for day, sentiment_series in df.groupby(df[DATE_COL].dt.floor("D"))[SENTIMENT_COL]:
        row = compute_metrics_from_series(sentiment_series)
        row["date"] = day
        daily_rows.append(row)
    return finalize_daily_metrics(pd.DataFrame(daily_rows), label=label)


def load_directory_analysis(input_dir):
    """Process a directory of raw tweet CSVs one file at a time."""
    print(f"Loading CSVs from directory: {input_dir}")
    files = sorted(glob.glob(os.path.join(input_dir, "*.csv")))
    print(f"Found {len(files)} CSV files")

    if not files:
        print("No CSV files found.")
        sys.exit(1)

    hist_counts = np.zeros(HIST_BINS, dtype=np.int64)
    daily_rows_all = []
    daily_rows_nonzero = []
    skipped = 0
    total = 0
    exact_zero = 0
    near_zero_01 = 0
    near_zero_05 = 0
    near_zero_10 = 0
    min_date = None
    max_date = None

    for i, filepath in enumerate(files, start=1):
        try:
            df = pd.read_csv(
                filepath,
                header=None,
                names=["tweet_id", SENTIMENT_COL],
                usecols=[0, 1],
                low_memory=False,
            )
        except Exception:
            skipped += 1
            continue

        df["tweet_id"] = pd.to_numeric(df["tweet_id"], errors="coerce")
        df[SENTIMENT_COL] = pd.to_numeric(df[SENTIMENT_COL], errors="coerce")
        df = df.dropna(subset=["tweet_id", SENTIMENT_COL])
        if df.empty:
            skipped += 1
            continue

        df["tweet_id"] = df["tweet_id"].astype(np.int64)
        timestamps_ms = (df["tweet_id"] // (2 ** 22)) + 1288834974657
        df[DATE_COL] = pd.to_datetime(timestamps_ms, unit="ms", errors="coerce")
        df = df.dropna(subset=[DATE_COL])
        if df.empty:
            skipped += 1
            continue

        sentiments = df[SENTIMENT_COL].to_numpy(dtype=float, copy=False)
        hist_counts += np.histogram(np.clip(sentiments, *HIST_RANGE), bins=HIST_BINS, range=HIST_RANGE)[0]

        total += len(df)
        exact_zero += int((df[SENTIMENT_COL] == 0).sum())
        near_zero_01 += int(((df[SENTIMENT_COL] >= -0.01) & (df[SENTIMENT_COL] <= 0.01)).sum())
        near_zero_05 += int(((df[SENTIMENT_COL] >= -0.05) & (df[SENTIMENT_COL] <= 0.05)).sum())
        near_zero_10 += int(((df[SENTIMENT_COL] >= -0.10) & (df[SENTIMENT_COL] <= 0.10)).sum())

        file_min = df[DATE_COL].min()
        file_max = df[DATE_COL].max()
        min_date = file_min if min_date is None else min(min_date, file_min)
        max_date = file_max if max_date is None else max(max_date, file_max)

        for day, sentiment_series in df.groupby(df[DATE_COL].dt.floor("D"))[SENTIMENT_COL]:
            all_row = compute_metrics_from_series(sentiment_series)
            all_row["date"] = day
            daily_rows_all.append(all_row)

            nonzero = sentiment_series[sentiment_series != 0]
            if not nonzero.empty:
                nonzero_row = compute_metrics_from_series(nonzero)
                nonzero_row["date"] = day
                daily_rows_nonzero.append(nonzero_row)

        if i % 100 == 0:
            print(f"  Processed {i}/{len(files)}")

    print(f"Loaded {len(daily_rows_all)} day-level slices, skipped {skipped} files")
    print(f"Total rows: {total:,}")
    print(f"Date range: {min_date} to {max_date}")

    zero_summary = {
        "hist_counts": hist_counts,
        "hist_edges": np.linspace(HIST_RANGE[0], HIST_RANGE[1], HIST_BINS + 1),
        "total": total,
        "exact_zero": exact_zero,
        "near_zero_01": near_zero_01,
        "near_zero_05": near_zero_05,
        "near_zero_10": near_zero_10,
    }

    daily_all = finalize_daily_metrics(pd.DataFrame(daily_rows_all), label="all")
    daily_nonzero = finalize_daily_metrics(pd.DataFrame(daily_rows_nonzero), label="non-zero")
    return zero_summary, daily_all, daily_nonzero


def plot_zero_distribution_from_summary(zero_summary, output_dir):
    """Show the distribution of sentiment scores and zero proportion."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    centers = (zero_summary["hist_edges"][:-1] + zero_summary["hist_edges"][1:]) / 2
    widths = np.diff(zero_summary["hist_edges"])
    ax.bar(centers, zero_summary["hist_counts"], width=widths, color="steelblue", alpha=0.7, edgecolor="none")
    ax.axvline(0, color="red", linewidth=1.5, linestyle="--", label="Zero")
    ax.set_xlabel("Sentiment Score")
    ax.set_ylabel("Count")
    ax.set_title("Sentiment Score Distribution (Full Data)")
    ax.legend()
    ax.set_yscale("log")

    ax = axes[1]
    total = zero_summary["total"]
    exact_zero = zero_summary["exact_zero"]
    near_zero_01 = zero_summary["near_zero_01"]
    near_zero_05 = zero_summary["near_zero_05"]
    near_zero_10 = zero_summary["near_zero_10"]

    categories = ["Exact 0", "|s| ≤ 0.01", "|s| ≤ 0.05", "|s| ≤ 0.10"]
    counts = [exact_zero, near_zero_01, near_zero_05, near_zero_10]
    pcts = [c / total * 100 for c in counts]

    bars = ax.bar(categories, pcts, color=["#d32f2f", "#ef5350", "#ef9a9a", "#ffcdd2"])
    for bar, pct, cnt in zip(bars, pcts, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{pct:.1f}%\n({cnt:,.0f})", ha="center", fontsize=10)
    ax.set_ylabel("% of Total Tweets")
    ax.set_title(f"Zero/Near-Zero Sentiment Proportion\n(Total: {total:,.0f})")
    
    plt.tight_layout()
    path = os.path.join(output_dir, "00_zero_distribution.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()
    
    print(f"\n--- Zero/Near-Zero Stats ---")
    print(f"Total tweets:        {total:,}")
    print(f"Exact zero:          {exact_zero:,} ({exact_zero/total*100:.1f}%)")
    print(f"|sentiment| ≤ 0.01:  {near_zero_01:,} ({near_zero_01/total*100:.1f}%)")
    print(f"|sentiment| ≤ 0.05:  {near_zero_05:,} ({near_zero_05/total*100:.1f}%)")
    print(f"|sentiment| ≤ 0.10:  {near_zero_10:,} ({near_zero_10/total*100:.1f}%)")

    return exact_zero, total


def plot_comparison(daily_all, daily_nonzero, output_dir):
    """Compare all tweets vs non-zero tweets across key metrics."""
    
    events = {
        "2020-03-11": "WHO pandemic",
        "2020-03-23": "UK lockdown",
        "2020-12-08": "Vaccination",
        "2021-01-06": "3rd lockdown",
        "2021-07-19": "Freedom Day",
        "2022-02-24": "Restrictions end",
    }
    
    def add_events(ax):
        for d, lbl in events.items():
            dt = pd.to_datetime(d)
            if daily_all["date"].min() <= dt <= daily_all["date"].max():
                ax.axvline(dt, color="gray", alpha=0.3, linestyle="--", linewidth=0.7)
    
    metrics = [
        ("mean_7d", "Sentiment Mean (7d MA)", "Mean"),
        ("negative_ratio_7d", "Negative Tweet % (7d MA)", "Neg %"),
        ("strong_neg_ratio_7d", "Strong Negative % (7d MA)", "Strong Neg %"),
        ("skewness_7d", "Skewness (7d MA)", "Skewness"),
        ("polarization_7d", "Polarization (7d MA)", "Polarization"),
        ("q10", "10th Percentile", "Q10"),
    ]
    
    fig, axes = plt.subplots(len(metrics), 1, figsize=(16, 4 * len(metrics)), sharex=True)
    fig.suptitle("Sentiment Comparison: All Tweets vs Non-Zero Tweets",
                 fontsize=16, fontweight="bold", y=1.01)
    
    colors = {"all": "steelblue", "non-zero": "crimson"}
    
    for i, (col, title, ylabel) in enumerate(metrics):
        ax = axes[i]
        
        if col in daily_all.columns:
            ax.plot(daily_all["date"], daily_all[col],
                    color=colors["all"], linewidth=2, label="All tweets", alpha=0.8)
        if col in daily_nonzero.columns:
            ax.plot(daily_nonzero["date"], daily_nonzero[col],
                    color=colors["non-zero"], linewidth=2, label="Non-zero only",
                    alpha=0.8, linestyle="-")
        
        add_events(ax)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11)
        ax.legend(loc="upper right", fontsize=9)
    
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    path = os.path.join(output_dir, "01_all_vs_nonzero_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()
    
    # ---- Difference plot ----
    merged = daily_all[["date"]].copy()
    for col_base in ["mean_7d", "negative_ratio_7d", "skewness_7d"]:
        if col_base in daily_all.columns and col_base in daily_nonzero.columns:
            merged_tmp = pd.merge(
                daily_all[["date", col_base]].rename(columns={col_base: f"{col_base}_all"}),
                daily_nonzero[["date", col_base]].rename(columns={col_base: f"{col_base}_nz"}),
                on="date", how="inner"
            )
            merged = pd.merge(merged, merged_tmp, on="date", how="left")
    
    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
    fig.suptitle("Difference: Non-Zero minus All Tweets (7d MA)",
                 fontsize=14, fontweight="bold")
    
    diff_metrics = [
        ("mean_7d", "Mean Difference"),
        ("negative_ratio_7d", "Neg% Difference"),
        ("skewness_7d", "Skewness Difference"),
    ]
    
    for i, (col_base, label) in enumerate(diff_metrics):
        col_all = f"{col_base}_all"
        col_nz = f"{col_base}_nz"
        if col_all in merged.columns and col_nz in merged.columns:
            diff = merged[col_nz] - merged[col_all]
            axes[i].plot(merged["date"], diff, color="darkgreen", linewidth=1.5)
            axes[i].axhline(0, color="black", linewidth=0.5)
            axes[i].set_ylabel(label)
            add_events(axes[i])
    
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    path = os.path.join(output_dir, "02_difference_plot.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()


def print_summary(daily_all, daily_nonzero):
    """Print summary statistics for comparison."""
    print("\n" + "=" * 70)
    print("SUMMARY COMPARISON")
    print("=" * 70)
    
    def stats_row(daily, label):
        return {
            "Dataset": label,
            "Days": len(daily),
            "Mean sentiment": daily["mean"].mean(),
            "Std of daily mean": daily["mean"].std(),
            "Avg neg%": daily["negative_ratio"].mean(),
            "Std of neg%": daily["negative_ratio"].std(),
            "Avg skewness": daily["skewness"].mean(),
            "Std of skewness": daily["skewness"].std(),
            "Avg polarization": daily["polarization"].mean(),
        }
    
    rows = [stats_row(daily_all, "All tweets"), stats_row(daily_nonzero, "Non-zero only")]
    summary = pd.DataFrame(rows)
    
    print(summary.to_string(index=False))
    
    print("\n--- Key Question: Does removing zeros increase variability? ---")
    mean_std_all = daily_all["mean"].std()
    mean_std_nz = daily_nonzero["mean"].std()
    neg_std_all = daily_all["negative_ratio"].std()
    neg_std_nz = daily_nonzero["negative_ratio"].std()
    
    print(f"Daily mean std:   All={mean_std_all:.5f}  NonZero={mean_std_nz:.5f}  "
          f"Ratio={mean_std_nz/mean_std_all:.2f}x")
    print(f"Daily neg% std:   All={neg_std_all:.5f}  NonZero={neg_std_nz:.5f}  "
          f"Ratio={neg_std_nz/neg_std_all:.2f}x")
    
    if mean_std_nz / mean_std_all > 1.5:
        print("\n→ Removing zeros SIGNIFICANTLY increases variability.")
        print("  Sentiment may carry useful signal after all. Re-examine!")
    else:
        print("\n→ Removing zeros does NOT substantially change variability.")
        print("  Sentiment is genuinely stable. Safe to deprioritize it.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True)
    parser.add_argument("--output", "-o", default="./zero_test_output")
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)

    if os.path.isdir(args.input):
        zero_summary, daily_all, daily_nonzero = load_directory_analysis(args.input)
        exact_zero = zero_summary["exact_zero"]
        total = zero_summary["total"]
        remaining = total - exact_zero
        exact_zero, total = plot_zero_distribution_from_summary(zero_summary, args.output)
    else:
        df = load_data(args.input)
        exact_zero, total = plot_zero_distribution_from_summary(
            {
                "hist_counts": np.histogram(
                    np.clip(df[SENTIMENT_COL].to_numpy(dtype=float, copy=False), *HIST_RANGE),
                    bins=HIST_BINS,
                    range=HIST_RANGE,
                )[0],
                "hist_edges": np.linspace(HIST_RANGE[0], HIST_RANGE[1], HIST_BINS + 1),
                "total": len(df),
                "exact_zero": int((df[SENTIMENT_COL] == 0).sum()),
                "near_zero_01": int(((df[SENTIMENT_COL] >= -0.01) & (df[SENTIMENT_COL] <= 0.01)).sum()),
                "near_zero_05": int(((df[SENTIMENT_COL] >= -0.05) & (df[SENTIMENT_COL] <= 0.05)).sum()),
                "near_zero_10": int(((df[SENTIMENT_COL] >= -0.10) & (df[SENTIMENT_COL] <= 0.10)).sum()),
            },
            args.output,
        )

        print("\nComputing metrics for ALL tweets...")
        daily_all = compute_metrics(df, label="all")

        df_nonzero = df[df[SENTIMENT_COL] != 0].copy()
        remaining = len(df_nonzero)
        print("\nComputing metrics for NON-ZERO tweets...")
        daily_nonzero = compute_metrics(df_nonzero, label="non-zero")

    print(f"\nRemoved {exact_zero:,} zero-sentiment tweets "
          f"({exact_zero/total*100:.1f}%)")
    print(f"Remaining: {remaining:,} tweets")

    # Save metrics
    daily_all.to_csv(os.path.join(args.output, "daily_metrics_all.csv"), index=False)
    daily_nonzero.to_csv(os.path.join(args.output, "daily_metrics_nonzero.csv"), index=False)
    
    # Plot comparison
    print("\nGenerating comparison plots...")
    plot_comparison(daily_all, daily_nonzero, args.output)
    
    # Summary
    print_summary(daily_all, daily_nonzero)


if __name__ == "__main__":
    main()
