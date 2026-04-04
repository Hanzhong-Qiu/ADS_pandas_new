import pandas as pd
import matplotlib.pyplot as plt
from analysis_common import DATA_OUTPUT_DIR, PLOTS_OUTPUT_DIR, ensure_output_dirs


ensure_output_dirs()

# 1. LOAD DATA
# Ensure you are using the merged file
df = pd.read_csv(DATA_OUTPUT_DIR / 'enriched_research_data.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')

# Identify the core columns
# (Adjust these names if they differ in your CSV)
policy_col = [c for c in df.columns if 'stringency' in c.lower()][0]
vol_col = 'sentiment_volatility'

# 2. LAG ADJUSTMENT
# Based on our analysis, Mood leads Policy by 6 days.
# We shift Policy by -6 to align the 'Cause' (Lockdown) with the 'Effect' (Mood).
df['policy_shifted'] = df[policy_col].shift(-6)

# 3. CALCULATE ROLLING CORRELATIONS
# We compare the standard (0-lag) vs. the Optimized (-6 lag)
window = 90 # 90-day window to observe seasonal/quarterly fatigue
df['r_standard'] = df[vol_col].rolling(window=window).corr(df[policy_col])
df['r_optimized'] = df[vol_col].rolling(window=window).corr(df['policy_shifted'])

# 4. VISUALIZATION
plt.figure(figsize=(14, 7))

# Plot both to show the difference alignment makes
plt.plot(df['date'], df['r_standard'], color='gray', alpha=0.3, label='Standard Rolling r (No Lag)')
plt.plot(df['date'], df['r_optimized'], color='#8e44ad', linewidth=2.5, label='Optimized Rolling r (-6 Day Lag)')

# Reference Lines
plt.axhline(0, color='black', linestyle='--', alpha=0.4)
plt.axhline(0.2, color='green', linestyle=':', alpha=0.3, label='Significant Threshold')

# Aesthetic Styling
plt.title('The "Pandemic Fatigue" Curve: Evolution of Policy Impact', fontsize=16, fontweight='bold')
plt.ylabel('Correlation Coefficient (r)', fontsize=12)
plt.xlabel('Date', fontsize=12)
plt.ylim(-0.5, 0.8) # Focus on the positive correlation range
plt.grid(axis='y', alpha=0.2)
plt.legend(loc='upper right')

# Highlight the "Decoupling" Phase
plt.annotate('Highest Sensitivity (2020)', xy=(pd.Timestamp('2020-07-01'), 0.4), 
             xytext=(pd.Timestamp('2020-03-01'), 0.6),
             arrowprops=dict(facecolor='black', shrink=0.05))

plt.tight_layout()
plt.savefig(PLOTS_OUTPUT_DIR / 'fatigue_analysis_rolling.png', dpi=300)
plt.close()

# 5. SUMMARY PRINT
print("📊 Rolling Correlation Insights:")
print(f"Mean Optimized Correlation: {df['r_optimized'].mean():.3f}")
print(f"Maximum Sensitivity Reached: {df['r_optimized'].max():.3f}")
