import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / ".csv"

                             
df = pd.read_csv(DATA_DIR / 'daily_sentiment_volatility.csv')
df['date'] = pd.to_datetime(df['date'])

                                                  
full_range = pd.date_range(start=df['date'].min(), end=df['date'].max())

                       
missing_days = full_range.difference(df['date'])

print(f"Dataset Span: {df['date'].min().date()} to {df['date'].max().date()}")
print(f"Number of missing days: {len(missing_days)}")

if len(missing_days) > 0:
    print("\n📅 First 10 Missing Dates:")
    print(missing_days[:10].date)
