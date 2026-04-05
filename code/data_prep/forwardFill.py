import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / ".csv"
INPUT_FILE = DATA_DIR / "daily_sentiment_volatility.csv"
OUTPUT_FILE = DATA_DIR / "cleaned_sentiment_data.csv"

# 1. Load the data
df = pd.read_csv(INPUT_FILE)
df['date'] = pd.to_datetime(df['date'])

print(f"📊 Rows before deduplication: {len(df)}")

# 2. FIX: Aggregate duplicate dates
# We take the mean of sentiment and volatility, but SUM the tweet volume.
# This ensures each date appears exactly once.
df = df.groupby('date').agg({
    'sentiment_mean': 'mean',
    'sentiment_volatility': 'mean',
    'tweet_volume': 'sum'
}).sort_index()

print(f"📊 Rows after deduplication (Unique days): {len(df)}")

# 3. Now reindex is safe because the index is unique
full_range = pd.date_range(start=df.index.min(), end=df.index.max())
df_complete = df.reindex(full_range).ffill()

# 4. Reset index so 'date' is a column (essential for the next ML step)
df_complete = df_complete.reset_index().rename(columns={'index': 'date'})

# 5. Save the final "Golden Dataset"
df_complete.to_csv(OUTPUT_FILE, index=False)

print(f"✅ Success! Continuous timeline saved to 'cleaned_sentiment_data.csv'")
print(f"📊 Final count: {len(df_complete)} consecutive days.")
