import matplotlib.pyplot as plt
import seaborn as sns
from analysis_common import DATA_OUTPUT_DIR, PLOTS_OUTPUT_DIR, build_merged_dataset, calculate_lag_correlations, ensure_output_dirs
from analysis_config import FINAL_COUNTRIES, FINAL_SCENARIO_NAME


ensure_output_dirs()

print(f"📊 Processing Scenario: {FINAL_SCENARIO_NAME}...")

df_merged, stringency_col = build_merged_dataset(FINAL_COUNTRIES)
lags, lag_corrs, best_lag, best_r = calculate_lag_correlations(df_merged, stringency_col)

print(f"🚀 Best Lag for {FINAL_SCENARIO_NAME}: {best_lag} days (r = {best_r:.3f})")

plt.figure(figsize=(10, 5))
colors = ["tab:red" if r == best_r else "tab:gray" for r in lag_corrs]
plt.bar(lags, lag_corrs, color=colors)
plt.axvline(0, color="black", linestyle="--", alpha=0.5)
plt.title(f"Lag Analysis: {FINAL_SCENARIO_NAME}\nPeak Correlation at {best_lag} days")
plt.xlabel("Days Shifted (Policy relative to Mood)")
plt.ylabel("Correlation (r)")
plt.savefig(PLOTS_OUTPUT_DIR / f"{FINAL_SCENARIO_NAME}_lag_chart.png", dpi=300)
plt.close()

df_merged["vol_smooth"] = df_merged["sentiment_volatility"].rolling(window=7).mean()
df_merged["str_smooth"] = df_merged[stringency_col].rolling(window=7).mean()

fig, ax1 = plt.subplots(figsize=(12, 6))
ax1.set_xlabel("Date")
ax1.set_ylabel("Sentiment Volatility", color="#1f77b4", fontweight="bold")
ax1.plot(df_merged["date"], df_merged["vol_smooth"], color="#1f77b4", label="Anxiety")

ax2 = ax1.twinx()
ax2.set_ylabel("Lockdown Stringency", color="#d62728", fontweight="bold")
ax2.plot(df_merged["date"], df_merged["str_smooth"], color="#d62728", alpha=0.5, label="Policy")

plt.title(
    f"Policy vs. Mood: {FINAL_SCENARIO_NAME}\n(Optimized r = {best_r:.3f} at {best_lag}d lag)"
)
plt.savefig(PLOTS_OUTPUT_DIR / f"{FINAL_SCENARIO_NAME}_timeseries.png", dpi=300)
plt.close()

plt.figure(figsize=(8, 6))
cols_to_check = ["sentiment_volatility", "sentiment_mean", stringency_col, "tweet_volume"]
sns.heatmap(df_merged[cols_to_check].corr(), annot=True, cmap="RdYlGn", center=0)
plt.title(f"Correlation Heatmap: {FINAL_SCENARIO_NAME}")
plt.savefig(PLOTS_OUTPUT_DIR / f"{FINAL_SCENARIO_NAME}_heatmap.png", dpi=300)
plt.close()

df_merged.to_csv(DATA_OUTPUT_DIR / "enriched_research_data.csv", index=False)
with open(DATA_OUTPUT_DIR / "lag_results.txt", "w", encoding="utf-8") as f:
    f.write(f"Scenario: {FINAL_SCENARIO_NAME}\n")
    f.write(f"Best Lag: {best_lag}\n")
    f.write(f"Best R: {best_r}")

print("💾 Final CSV and lag_results.txt saved.")
print("\n🚀 All analysis complete. Check your folder for PNGs and the final CSV.")
