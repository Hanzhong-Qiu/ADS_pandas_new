import pandas as pd
import glob
import os
import datetime

input_path = './all_covid_data' 
output_file = 'daily_sentiment_volatility.csv'
all_files = glob.glob(os.path.join(input_path, "*.csv"))

print(f"Found {len(all_files)} files. Starting aggregation...")

# The dataset has no headers, Column 0: Tweet ID; Column 1: Sentiment Score
daily_stats = []

for i, filename in enumerate(all_files):
    try:
        # We read the file without headers
        # Use only column 1 (sentiment) to save massive amounts of RAM
        # We use column 0 (ID) just for the first row to get the date
        df = pd.read_csv(filename, header=None, names=['tweet_id', 'sentiment'], low_memory=False)
        
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

# If volatility is NaN (happens if a file has only 1 tweet), we drop or impute.
fused_df = fused_df.dropna(subset=['sentiment_volatility'])

# Save the "Cleaned" Fused Dataset
fused_df.to_csv(output_file, index=False)

print(f"Fused metrics saved as {output_file}")
print(f"Final Dataset Shape: {fused_df.shape}")