from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
TWEETS_FILE = BASE_DIR / "cleaned_sentiment_data.csv"
OXFORD_FILE = BASE_DIR / "OxCGRT_compact_national_v1.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
DATA_OUTPUT_DIR = OUTPUT_DIR / "data"
PLOTS_OUTPUT_DIR = OUTPUT_DIR / "plots"


def ensure_output_dirs():
    DATA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_inputs():
    df_tweets = pd.read_csv(TWEETS_FILE)
    df_tweets["date"] = pd.to_datetime(df_tweets["date"])

    df_oxford = pd.read_csv(OXFORD_FILE, low_memory=False)
    df_oxford["date"] = pd.to_datetime(df_oxford["Date"], format="%Y%m%d")

    stringency_col = [
        col for col in df_oxford.columns
        if "StringencyIndex" in col and "Average" in col
    ][0]
    return df_tweets, df_oxford, stringency_col


def build_merged_dataset(countries, cutoff_date=None):
    df_tweets, df_oxford, stringency_col = load_inputs()

    df_temp = df_oxford[df_oxford["CountryName"].isin(countries)]
    df_index = df_temp.groupby("date")[stringency_col].mean().reset_index()
    df_merged = pd.merge(df_tweets, df_index, on="date", how="inner").sort_values("date")

    if cutoff_date is not None:
        df_merged = df_merged[df_merged["date"] <= cutoff_date]

    return df_merged, stringency_col


def calculate_lag_correlations(df, stringency_col, value_col="sentiment_volatility", lags=range(-7, 8)):
    lag_values = list(lags)
    lag_corrs = [df[value_col].corr(df[stringency_col].shift(lag)) for lag in lag_values]
    best_idx = max(range(len(lag_corrs)), key=lambda idx: lag_corrs[idx])
    return lag_values, lag_corrs, lag_values[best_idx], lag_corrs[best_idx]
