import datetime
import os
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / ".csv"
input_path = BASE_DIR / "all_covid_data"
output_file = DATA_DIR / "daily_sentiment_volatility.csv"
DATA_DIR.mkdir(exist_ok=True)
all_files = sorted(input_path.rglob("*.csv")) if input_path.exists() else []

print(f"Found {len(all_files)} files. Starting aggregation...")

if not input_path.exists():
    if output_file.exists():
        print(
            f"Input directory not found: {input_path}\n"
            f"Using existing aggregated file instead: {output_file}"
        )
        raise SystemExit(0)
    raise SystemExit(
        f"Input directory not found: {input_path}\n"
        "Place the raw tweet CSV files inside 'all_covid_data/' and run this script again."
    )

if not all_files:
    if output_file.exists():
        print(
            f"No CSV files found under: {input_path}\n"
            f"Using existing aggregated file instead: {output_file}"
        )
        raise SystemExit(0)
    raise SystemExit(
        f"No CSV files found under: {input_path}\n"
        "Make sure the raw tweet files exist there before running fusion.py."
    )

# The dataset has no headers, Column 0: Tweet ID; Column 1: Sentiment Score
daily_stats = []

for i, filename in enumerate(all_files):
    try:
        # We read the file without headers
        # Use only column 1 (sentiment) to save massive amounts of RAM
        # We use column 0 (ID) just for the first row to get the date
        df = pd.read_csv(filename, header=None, names=['tweet_id', 'sentiment'], low_memory=False)

        if df.empty:
            print(f"⚠️ Skipping empty file: {filename}")
            continue
        
        # Twitter encodes the timestamp in the ID.
        example_id = int(df['tweet_id'].iloc[0])
        timestamp = (example_id >> 22) + 1288834974657
        date = datetime.datetime.fromtimestamp(timestamp/1000.0).date()
        
        stats = {
            'date': date,
            'sentiment_mean': df['sentiment'].mean(),
            'sentiment_volatility': df['sentiment'].std(),
            'tweet_volume': len(df),
            'source_file': os.path.basename(filename)
        }
        
        daily_stats.append(stats)
        
        if i % 50 == 0:
            print(f"✅ Processed {i} files. Current Date: {date}")
            
    except Exception as e:
        print(f"⚠️ Skipping {filename}: {e}")

print("🔗 Fusing daily aggregates...")
fused_df = pd.DataFrame(daily_stats)

if fused_df.empty:
    raise SystemExit(
        "No usable daily records were created from the raw files.\n"
        "Check whether the files are empty or whether they contain valid tweet IDs and sentiment values."
    )

# If volatility is NaN (happens if a file has only 1 tweet), we drop or impute.
fused_df = fused_df.dropna(subset=['sentiment_volatility'])

# Save the "Cleaned" Fused Dataset
fused_df.to_csv(output_file, index=False)

print(f"Fused metrics saved as {output_file}")
print(f"Final Dataset Shape: {fused_df.shape}")
