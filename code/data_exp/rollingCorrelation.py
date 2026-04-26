import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parents[2]))
from analysis_common import DATA_OUTPUT_DIR, PLOTS_OUTPUT_DIR, ensure_output_dirs


ensure_output_dirs()

              
df = pd.read_csv(DATA_OUTPUT_DIR / 'enriched_research_data.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')

                           
policy_col = [c for c in df.columns if 'stringency' in c.lower()][0]
vol_col = 'sentiment_volatility'

                   
df['policy_shifted'] = df[policy_col].shift(-6)

                                   
window = 90                                                      
df['r_standard'] = df[vol_col].rolling(window=window).corr(df[policy_col])
df['r_optimized'] = df[vol_col].rolling(window=window).corr(df['policy_shifted'])

                  
plt.figure(figsize=(14, 7))

                                                  
plt.plot(df['date'], df['r_standard'], color='gray', alpha=0.3, label='Standard Rolling r (No Lag)')
plt.plot(df['date'], df['r_optimized'], color='#8e44ad', linewidth=2.5, label='Optimized Rolling r (-6 Day Lag)')

                 
plt.axhline(0, color='black', linestyle='--', alpha=0.4)
plt.axhline(0.2, color='green', linestyle=':', alpha=0.3, label='Significant Threshold')

                   
plt.title('The "Pandemic Fatigue" Curve: Evolution of Policy Impact', fontsize=16, fontweight='bold')
plt.ylabel('Correlation Coefficient (r)', fontsize=12)
plt.xlabel('Date', fontsize=12)
plt.ylim(-0.5, 0.8)                                          
plt.grid(axis='y', alpha=0.2)
plt.legend(loc='upper right')

                                  
plt.annotate('Highest Sensitivity (2020)', xy=(pd.Timestamp('2020-07-01'), 0.4), 
             xytext=(pd.Timestamp('2020-03-01'), 0.6),
             arrowprops=dict(facecolor='black', shrink=0.05))

plt.tight_layout()
plt.savefig(PLOTS_OUTPUT_DIR / 'fatigue_analysis_rolling.png', dpi=300)
plt.close()

                  
print("📊 Rolling Correlation Insights:")
print(f"Mean Optimized Correlation: {df['r_optimized'].mean():.3f}")
print(f"Maximum Sensitivity Reached: {df['r_optimized'].max():.3f}")
