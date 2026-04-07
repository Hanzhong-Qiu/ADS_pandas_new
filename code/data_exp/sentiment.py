"""
Daily sentiment metrics from raw COVID tweet CSV files.

Calculates:
1. Percentage of negative tweets
2. Skewness of sentiment distribution
3. Degree of sentiment polarization

Usage:
    python3 code/data_exp/sentiment.py --input ./all_covid_data
    python3 code/data_exp/sentiment.py --input ./all_covid_data --output ./graphs/exploration/sentiment_output
"""

import argparse
import glob
import os
import sys

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATE_COL = "date"
SENTIMENT_COL = "sentiment"


def tweet_id_to_timestamp(tweet_ids):
    """Decode Twitter snowflake IDs into pandas timestamps."""
    tweet_ids = pd.to_numeric(tweet_ids, errors="coerce")
    timestamp_ms = (tweet_ids // (2 ** 22)) + 1288834974657
    return pd.to_datetime(timestamp_ms, unit="ms", errors="coerce")


def compute_required_metrics(sentiment_series):
    """Return the three requested sentiment metrics for one day."""
    return {
        "tweet_count": int(sentiment_series.count()),
        "negative_tweet_pct": float((sentiment_series < 0).mean() * 100.0),
        "skewness": float(sentiment_series.skew()),
        "polarization": float(((sentiment_series < -0.5) | (sentiment_series > 0.5)).mean()),
    }


def finalize_daily_metrics(daily):
    """Sort daily metrics and add 7-day smoothed versions."""
    daily = daily.copy()
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values("date").reset_index(drop=True)

    for col in ["negative_tweet_pct", "skewness", "polarization"]:
        daily[f"{col}_7d"] = daily[col].rolling(7, center=True).mean()

    return daily


def load_daily_metrics_from_directory(input_dir):
    """Process raw tweet CSVs one file at a time to avoid high memory use."""
    print(f"Loading data from {input_dir}...")
    csv_files = sorted(glob.glob(os.path.join(input_dir, "*.csv")))
    if not csv_files:
        print(f"No CSV files found in directory: {input_dir}")
        sys.exit(1)

    print(f"Found {len(csv_files):,} CSV files in directory")

    daily_rows = []
    skipped = 0
    total_rows = 0
    min_date = None
    max_date = None

    for i, csv_path in enumerate(csv_files, start=1):
        try:
            df = pd.read_csv(
                csv_path,
                header=None,
                names=["tweet_id", SENTIMENT_COL],
                usecols=[0, 1],
                low_memory=False,
            )
        except Exception as exc:
            skipped += 1
            print(f"Skipping {os.path.basename(csv_path)}: {exc}")
            continue

        df["tweet_id"] = pd.to_numeric(df["tweet_id"], errors="coerce")
        df[SENTIMENT_COL] = pd.to_numeric(df[SENTIMENT_COL], errors="coerce")
        df = df.dropna(subset=["tweet_id", SENTIMENT_COL])
        if df.empty:
            skipped += 1
            continue

        df[DATE_COL] = tweet_id_to_timestamp(df["tweet_id"])
        df = df.dropna(subset=[DATE_COL])
        if df.empty:
            skipped += 1
            continue

        day_values = df[DATE_COL].dt.floor("D")
        for day, sentiment_series in df.groupby(day_values)[SENTIMENT_COL]:
            row = compute_required_metrics(sentiment_series)
            row["date"] = day
            daily_rows.append(row)

        total_rows += len(df)
        file_min = df[DATE_COL].min()
        file_max = df[DATE_COL].max()
        min_date = file_min if min_date is None else min(min_date, file_min)
        max_date = file_max if max_date is None else max(max_date, file_max)

        if i % 50 == 0:
            print(f"Processed {i:,}/{len(csv_files):,} files")

    if not daily_rows:
        print("Could not parse any input CSV files.")
        sys.exit(1)

    daily = finalize_daily_metrics(pd.DataFrame(daily_rows))

    print(f"Loaded {total_rows:,} tweet rows")
    print(f"Date range: {min_date} to {max_date}")
    if skipped:
        print(f"Skipped {skipped} files that could not be parsed")

    return daily


def load_daily_metrics_from_csv(input_file):
    """Load a single CSV with date and sentiment columns and compute daily metrics."""
    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file, low_memory=False)

    if SENTIMENT_COL not in df.columns or DATE_COL not in df.columns:
        print(f"Input CSV must contain '{DATE_COL}' and '{SENTIMENT_COL}' columns.")
        print(f"Found columns: {list(df.columns)}")
        sys.exit(1)

    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df[SENTIMENT_COL] = pd.to_numeric(df[SENTIMENT_COL], errors="coerce")
    df = df.dropna(subset=[DATE_COL, SENTIMENT_COL])

    daily_rows = []
    for day, sentiment_series in df.groupby(df[DATE_COL].dt.floor("D"))[SENTIMENT_COL]:
        row = compute_required_metrics(sentiment_series)
        row["date"] = day
        daily_rows.append(row)

    if not daily_rows:
        print("No valid rows found in input CSV.")
        sys.exit(1)

    return finalize_daily_metrics(pd.DataFrame(daily_rows))


def plot_metrics(daily, output_dir):
    """Create one plot for the three requested metrics."""
    fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
    fig.suptitle("Daily Tweet Sentiment Metrics", fontsize=16, fontweight="bold")

    configs = [
        ("negative_tweet_pct", "negative_tweet_pct_7d", "crimson", "Negative Tweets (%)"),
        ("skewness", "skewness_7d", "darkorange", "Skewness"),
        ("polarization", "polarization_7d", "purple", "Polarization"),
    ]

    for ax, (raw_col, smooth_col, color, ylabel) in zip(axes, configs):
        ax.plot(daily["date"], daily[raw_col], color=color, alpha=0.2)
        ax.plot(daily["date"], daily[smooth_col], color=color, linewidth=2)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.2)

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    output_path = os.path.join(output_dir, "sentiment_metrics.png")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Compute daily tweet sentiment metrics")
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to a directory of raw tweet CSVs or a single CSV with date/sentiment columns",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="./sentiment_output",
        help="Output directory (default: ./sentiment_output)",
    )
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    if os.path.isdir(args.input):
        daily = load_daily_metrics_from_directory(args.input)
    else:
        daily = load_daily_metrics_from_csv(args.input)

    daily_path = os.path.join(args.output, "daily_sentiment_metrics.csv")
    daily.to_csv(daily_path, index=False)
    print(f"Saved metrics CSV: {daily_path}")

    plot_metrics(daily, args.output)

    print("\nSummary")
    print(f"Days analysed: {len(daily):,}")
    print(f"Date range: {daily['date'].min().date()} to {daily['date'].max().date()}")
    print(f"Average negative tweet %: {daily['negative_tweet_pct'].mean():.2f}")
    print(f"Average skewness: {daily['skewness'].mean():.4f}")
    print(f"Average polarization: {daily['polarization'].mean():.4f}")


if __name__ == "__main__":
    main()
