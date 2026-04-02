import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. SETUP
groups = {
    "Anglosphere_No_India": ['United States', 'United Kingdom', 'Canada', 'Australia', 'New Zealand', 'Ireland'],
    "Anglosphere_With_India": ['United States', 'United Kingdom', 'Canada', 'Australia', 'New Zealand', 'Ireland', 'India']
}

df_tweets = pd.read_csv('cleaned_sentiment_data.csv')
df_tweets['date'] = pd.to_datetime(df_tweets['date'])
df_oxford = pd.read_csv('OxCGRT_compact_national_v1.csv', low_memory=False)
df_oxford['date'] = pd.to_datetime(df_oxford['Date'], format='%Y%m%d')

stringency_col = [c for c in df_oxford.columns if 'StringencyIndex' in c and 'Average' in c][0]

# 2. THE LOOP (Scenarios)
for name, countries in groups.items():
    print(f"📊 Processing Scenario: {name} (Data truncated to March 2022)...")
    
    # Merge Logic
    df_temp = df_oxford[df_oxford['CountryName'].isin(countries)]
    df_index = df_temp.groupby('date')[stringency_col].mean().reset_index()
    df_merged = pd.merge(df_tweets, df_index, on='date', how='inner').sort_values('date')
    
    # --- CRITICAL: DATE FILTER ---
    # We truncate the data to see the correlation before the "fatigue" era
    df_merged = df_merged[df_merged['date'] <= '2022-03-31']
    # -----------------------------

    # --- LAG ANALYSIS BLOCK ---
    lags = range(-7, 8)
    lag_corrs = []
    for l in lags:
        c = df_merged['sentiment_volatility'].corr(df_merged[stringency_col].shift(l))
        lag_corrs.append(c)
    
    best_r = max(lag_corrs)
    best_lag = lags[lag_corrs.index(best_r)]
    print(f"🚀 Best Lag for {name}: {best_lag} days (r = {best_r:.3f})")

    # Plot 1: Lag Analysis Bar Chart
    plt.figure(figsize=(10, 5))
    colors = ['tab:red' if r == best_r else 'tab:gray' for r in lag_corrs]
    plt.bar(lags, lag_corrs, color=colors)
    plt.axvline(0, color='black', linestyle='--', alpha=0.5)
    plt.title(f"Lag Analysis (Pre-Apr 2022): {name}\nPeak Correlation at {best_lag} days")
    plt.xlabel("Days Shifted (Policy relative to Mood)")
    plt.ylabel("Correlation (r)")
    plt.savefig(f"{name}_pre2022_lag_chart.png", dpi=300)
    plt.close()

    # 7-day Rolling Averages for Time Series
    df_merged['vol_smooth'] = df_merged['sentiment_volatility'].rolling(window=7).mean()
    df_merged['str_smooth'] = df_merged[stringency_col].rolling(window=7).mean()
    
    # Plot 2: Time Series
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Sentiment Volatility', color='#1f77b4', fontweight='bold')
    ax1.plot(df_merged['date'], df_merged['vol_smooth'], color='#1f77b4', label='Anxiety')
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('Lockdown Stringency', color='#d62728', fontweight='bold')
    ax2.plot(df_merged['date'], df_merged['str_smooth'], color='#d62728', alpha=0.5, label='Policy')
    
    plt.title(f"Pre-Apr 2022: Policy vs. Mood ({name})\n(Optimized r = {best_r:.3f} at {best_lag}d lag)")
    plt.savefig(f"{name}_pre2022_timeseries.png", dpi=300)
    plt.close()

    # Plot 3: Heatmap
    plt.figure(figsize=(8, 6))
    cols_to_check = ['sentiment_volatility', 'sentiment_mean', stringency_col, 'tweet_volume']
    sns.heatmap(df_merged[cols_to_check].corr(), annot=True, cmap='RdYlGn', center=0)
    plt.title(f"Correlation Heatmap: {name} (Pre-Apr 2022)")
    plt.savefig(f"{name}_pre2022_heatmap.png", dpi=300)
    plt.close()

    # 3. SAVE THE FINAL CSV
    if "With_India" in name:
        df_merged.to_csv('enriched_research_data_pre2022.csv', index=False)
        print("💾 Truncated CSV 'enriched_research_data_pre2022.csv' saved.")

print("\n🚀 Truncated analysis complete. Compare these r-values to your 1,000-day results!")