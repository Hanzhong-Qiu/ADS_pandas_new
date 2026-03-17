import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import grangercausalitytests

# 1. Load Data
df = pd.read_csv('enriched_research_data.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')

# Identify the core columns
policy_col = [c for c in df.columns if 'stringency' in c.lower()][0]
vol_col = 'sentiment_volatility'

print("🧪 Running Advanced Statistical Suite...")

# --- ANALYSIS 1: GRANGER CAUSALITY ---
# Does Mood (X) predict Policy (Y)? 
# We test up to the 7-day lag we identified earlier.
print("\n1. Testing Granger Causality (Does Mood predict Policy?)...")
# Granger test requires a 2D array [Target, Predictor]
data_gc = df[[policy_col, vol_col]].dropna()
gc_results = grangercausalitytests(data_gc, maxlag=7, verbose=False)

# Get the p-value for the 6-day lag specifically
p_val_6d = gc_results[6][0]['ssr_ftest'][1]
print(f"👉 P-Value at 6-day lag: {p_val_6d:.10f}")
# If p < 0.05, we say it "Granger-causes" it.

# --- ANALYSIS 2: Z-SCORE ANOMALY DETECTION ---
df['z_score'] = (df[vol_col] - df[vol_col].mean()) / df[vol_col].std()
anomalies = df[df['z_score'].abs() > 3] # Standard threshold for outliers

print(f"\n2. Detected {len(anomalies)} 'Black Swan' mood events (Z > 3).")
print("Top 5 Dates for your Report:")
print(anomalies[['date', 'z_score']].sort_values('z_score', ascending=False).head(5))

# Plotting Anomalies
plt.figure(figsize=(12, 5))
plt.plot(df['date'], df['z_score'], color='gray', alpha=0.5, label='Volatility Z-Score')
plt.scatter(anomalies['date'], anomalies['z_score'], color='red', s=20, label='Anomalies')
plt.axhline(3, color='black', linestyle='--')
plt.title('Z-Score Anomaly Detection: Identifying Pandemic "Black Swan" Events')
plt.legend()
plt.savefig('anomaly_detection.png')

# --- ANALYSIS 3: DECAY ANALYSIS (STRESS RECOVERY) ---
# We look at how many days it takes for a shock to drop by 50%
print("\n3. Analyzing Volatility Half-Life (Stress Recovery)...")
autocorr = df[vol_col].autocorr(lag=1)
half_life = -np.log(2) / np.log(abs(autocorr))
print(f"👉 Estimated 'Emotional Half-Life': {half_life:.2f} days")