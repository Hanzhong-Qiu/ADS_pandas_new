import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / ".csv"
INPUT_FILE = DATA_DIR / "daily_sentiment_volatility.csv"
OUTPUT_FILE = DATA_DIR / "cleaned_sentiment_data.csv"

                  
df = pd.read_csv(INPUT_FILE)
df['date'] = pd.to_datetime(df['date'])

print(f"📊 Rows before deduplication: {len(df)}")

                                   
df = df.groupby('date').agg({
    'sentiment_mean': 'mean',
    'sentiment_volatility': 'mean',
    'tweet_volume': 'sum'
}).sort_index()

print(f"📊 Rows after deduplication (Unique days): {len(df)}")

                                                    
full_range = pd.date_range(start=df.index.min(), end=df.index.max())
df_complete = df.reindex(full_range).ffill()

                                                                       
df_complete = df_complete.reset_index().rename(columns={'index': 'date'})

                                    
df_complete.to_csv(OUTPUT_FILE, index=False)

print(f"✅ Success! Continuous timeline saved to 'cleaned_sentiment_data.csv'")
print(f"📊 Final count: {len(df_complete)} consecutive days.")
