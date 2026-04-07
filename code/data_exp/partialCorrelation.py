"""
Partial Correlation Analysis: Testing for Confounding.

Research question:
Is the sentiment/stringency relationship genuine, or is it an artefact of
both variables responding to COVID cases/deaths?

Method:
1. Regress sentiment_mean on daily_new_cases and daily_new_deaths.
2. Regress StringencyIndex_Average on daily_new_cases and daily_new_deaths.
3. Correlate the two residual series.

The analysis uses differenced/stationary data to reduce spurious time-series
correlations. If the differenced file is missing, the full level dataset is
loaded and differenced inside this script.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression


BASE_DIR = Path("/home/mohsin/ADS/ADS_pandas_new")
DIFF_DATA_PATH = BASE_DIR / "graphs" / "exploration" / "exp2_differenced_full.csv"
FULL_DATA_PATH = BASE_DIR / ".csv" / "new" / "full_analysis_data.csv"
OUTPUT_DIR = BASE_DIR / "graphs" / "partial_correlation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SENTIMENT = "sentiment_mean"
STRINGENCY = "StringencyIndex_Average"
CASES = "daily_new_cases"
DEATHS = "daily_new_deaths"
REQUIRED = [SENTIMENT, STRINGENCY, CASES, DEATHS]


def load_stationary_data():
    try:
        df = pd.read_csv(DIFF_DATA_PATH, index_col="date", parse_dates=True)
        print(f"Loaded differenced data: {DIFF_DATA_PATH} ({len(df)} rows)")
        already_differenced = True
    except FileNotFoundError:
        print("Differenced file not found, loading full data and differencing...")
        df = pd.read_csv(FULL_DATA_PATH, parse_dates=["date"]).sort_values("date").set_index("date")
        already_differenced = False

    missing = [col for col in REQUIRED if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Available: {list(df.columns)}")

    if not already_differenced:
        print("Applying first-order differencing for stationarity...")
        df = df[REQUIRED].diff().dropna()
    else:
        df = df[REQUIRED].dropna()

    print(f"Analysis data: {len(df)} observations")
    print(f"Date range: {df.index.min().date()} to {df.index.max().date()}")
    return df


def residualize(target, controls):
    model = LinearRegression().fit(controls, target)
    residuals = target - model.predict(controls)
    return residuals, model.score(controls, target)


def find_peak_xcorr(x, y, max_lag=14):
    x_arr = (x - x.mean()) / x.std()
    y_arr = (y - y.mean()) / y.std()
    n = len(x_arr)
    best_lag, best_r = 0, 0
    all_results = []

    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            r_value = np.corrcoef(x_arr[: n - lag], y_arr[lag:])[0, 1]
        else:
            r_value = np.corrcoef(x_arr[-lag:], y_arr[: n + lag])[0, 1]
        all_results.append((lag, r_value))
        if abs(r_value) > abs(best_r):
            best_lag, best_r = lag, r_value

    return best_lag, best_r, all_results


def plot_partial_scatter(df, resid_sentiment, resid_stringency, r_raw, p_raw, r_partial, p_partial):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    ax = axes[0]
    ax.scatter(df[SENTIMENT], df[STRINGENCY], alpha=0.3, s=8, color="#e74c3c")
    z = np.polyfit(df[SENTIMENT], df[STRINGENCY], 1)
    x_line = np.linspace(df[SENTIMENT].min(), df[SENTIMENT].max(), 100)
    ax.plot(x_line, np.poly1d(z)(x_line), color="black", linewidth=2, linestyle="--")
    ax.set_xlabel("Differenced sentiment_mean")
    ax.set_ylabel("Differenced StringencyIndex_Average")
    ax.set_title(f"Raw correlation\nr = {r_raw:.4f}, p = {p_raw:.4f}", fontweight="bold")
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.scatter(resid_sentiment, resid_stringency, alpha=0.3, s=8, color="#3498db")
    z2 = np.polyfit(resid_sentiment, resid_stringency, 1)
    x_line2 = np.linspace(resid_sentiment.min(), resid_sentiment.max(), 100)
    ax.plot(x_line2, np.poly1d(z2)(x_line2), color="black", linewidth=2, linestyle="--")
    ax.set_xlabel("Residual sentiment after cases/deaths")
    ax.set_ylabel("Residual stringency after cases/deaths")
    ax.set_title(f"Partial correlation\nr = {r_partial:.4f}, p = {p_partial:.4f}", fontweight="bold")
    ax.grid(alpha=0.3)

    fig.suptitle(
        "Partial Correlation Test: Is Sentiment/Stringency Confounded by COVID Cases and Deaths?",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "partial_correlation_scatter.png", dpi=200, bbox_inches="tight")
    plt.close()


def plot_lag_chain(pairs, lag_results, n_obs):
    fig, ax = plt.subplots(figsize=(10, 5))
    pair_labels = [label for _, _, label in pairs]
    peak_lags = [lag_results[label]["lag"] for label in pair_labels]
    peak_rs = [lag_results[label]["r"] for label in pair_labels]
    bar_colors = ["#e74c3c", "#e67e22", "#8e44ad", "#9b59b6", "#3498db"]

    bars = ax.bar(range(len(pair_labels)), peak_lags, color=bar_colors, alpha=0.8, edgecolor="black")
    ax.set_xticks(range(len(pair_labels)))
    ax.set_xticklabels(pair_labels, fontsize=9, rotation=15, ha="right")
    ax.set_ylabel("Peak lag (days)")
    ax.set_title("Lag Chain: Who Reacts First to What?", fontweight="bold")
    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.grid(axis="y", alpha=0.3)

    for bar, r_value in zip(bars, peak_rs):
        offset = 0.2 if bar.get_height() >= 0 else -0.5
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + offset,
            f"r={r_value:.3f}",
            ha="center",
            va="bottom" if bar.get_height() >= 0 else "top",
            fontsize=9,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "lag_chain_comparison.png", dpi=200, bbox_inches="tight")
    plt.close()

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    axes = axes.flatten()
    ci = 1.96 / np.sqrt(n_obs)

    for idx, (x_col, y_col, label) in enumerate(pairs):
        ax = axes[idx]
        all_r = lag_results[label]["all"]
        lags = [lag for lag, _ in all_r]
        corrs = [corr for _, corr in all_r]
        colors = ["#e74c3c" if abs(corr) > ci else "#bdc3c7" for corr in corrs]

        ax.bar(lags, corrs, color=colors, alpha=0.8, width=0.7)
        ax.axhline(y=ci, color="red", linestyle="--", linewidth=0.7, alpha=0.6)
        ax.axhline(y=-ci, color="red", linestyle="--", linewidth=0.7, alpha=0.6)
        ax.axhline(y=0, color="black", linewidth=0.5)
        ax.axvline(x=0, color="gray", linewidth=0.5, linestyle=":")

        best = lag_results[label]
        ax.annotate(
            f"peak: lag={best['lag']}, r={best['r']:.3f}",
            xy=(best["lag"], best["r"]),
            xytext=(best["lag"] + 4, best["r"] * 1.3),
            arrowprops={"arrowstyle": "->", "color": "darkred"},
            fontsize=9,
            fontweight="bold",
            color="darkred",
        )

        ax.set_title(label, fontweight="bold")
        ax.set_xlim(-15, 15)
        ax.set_ylabel("r")
        ax.grid(alpha=0.2)
        if idx >= 4:
            ax.set_xlabel("Lag (days)")

    axes[5].set_visible(False)
    fig.suptitle(f"Cross-Correlation Lag Chain Analysis (95% CI = +/- {ci:.3f})", fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "lag_chain_cross_correlations.png", dpi=200, bbox_inches="tight")
    plt.close()


def main():
    df = load_stationary_data()

    print("\n" + "=" * 65)
    print("TEST 1: ZERO-ORDER CORRELATION")
    print("=" * 65)
    r_raw, p_raw = stats.pearsonr(df[SENTIMENT], df[STRINGENCY])
    print(f"  Sentiment <-> Stringency: r = {r_raw:.4f}, p = {p_raw:.6f}")

    print("\n" + "=" * 65)
    print("TEST 2: PARTIAL CORRELATION CONTROLLING CASES AND DEATHS")
    print("=" * 65)
    controls = df[[CASES, DEATHS]].values
    resid_sentiment, r2_sent = residualize(df[SENTIMENT].values, controls)
    resid_stringency, r2_str = residualize(df[STRINGENCY].values, controls)
    r_partial, p_partial = stats.pearsonr(resid_sentiment, resid_stringency)
    print(f"  Cases/deaths explain {r2_sent * 100:.2f}% of sentiment variance")
    print(f"  Cases/deaths explain {r2_str * 100:.2f}% of stringency variance")
    print(f"  Partial r = {r_partial:.4f}, p = {p_partial:.6f}")

    drop = abs(r_raw) - abs(r_partial)
    drop_pct = (drop / abs(r_raw)) * 100 if r_raw != 0 else 0
    print(f"  Absolute-r change: {drop:.4f} ({drop_pct:.1f}%)")

    print("\n" + "=" * 65)
    print("TEST 2b: SINGLE-CONFOUNDER PARTIAL CORRELATIONS")
    print("=" * 65)
    single_rows = []
    for confounder_name, confounder_col in [("Cases only", CASES), ("Deaths only", DEATHS)]:
        controls_single = df[[confounder_col]].values
        resid_s, _ = residualize(df[SENTIMENT].values, controls_single)
        resid_p, _ = residualize(df[STRINGENCY].values, controls_single)
        r_pc, p_pc = stats.pearsonr(resid_s, resid_p)
        single_rows.append({"control": confounder_name, "r_partial": r_pc, "p_value": p_pc})
        print(f"  {confounder_name}: r_partial = {r_pc:.4f}, p = {p_pc:.6f}")

    print("\n" + "=" * 65)
    print("TEST 3: LAG CHAIN")
    print("=" * 65)
    pairs = [
        (CASES, SENTIMENT, "Cases -> Sentiment"),
        (CASES, STRINGENCY, "Cases -> Stringency"),
        (DEATHS, SENTIMENT, "Deaths -> Sentiment"),
        (DEATHS, STRINGENCY, "Deaths -> Stringency"),
        (SENTIMENT, STRINGENCY, "Sentiment -> Stringency"),
    ]
    lag_results = {}
    lag_rows = []
    for x_col, y_col, label in pairs:
        best_lag, best_r, all_r = find_peak_xcorr(df[x_col].values, df[y_col].values, max_lag=14)
        lag_results[label] = {"lag": best_lag, "r": best_r, "all": all_r}
        lag_rows.append({"pair": label, "peak_lag": best_lag, "peak_r": best_r})
        print(f"  {label:24s} peak lag = {best_lag:+3d}, r = {best_r:+.4f}")

    plot_partial_scatter(df, resid_sentiment, resid_stringency, r_raw, p_raw, r_partial, p_partial)
    plot_lag_chain(pairs, lag_results, len(df))

    pd.DataFrame(
        [
            {"metric": "raw_correlation", "r": r_raw, "p_value": p_raw, "r2_sentiment": np.nan, "r2_stringency": np.nan},
            {
                "metric": "partial_correlation_cases_deaths",
                "r": r_partial,
                "p_value": p_partial,
                "r2_sentiment": r2_sent,
                "r2_stringency": r2_str,
            },
        ]
    ).to_csv(OUTPUT_DIR / "partial_correlation_summary.csv", index=False)
    pd.DataFrame(single_rows).to_csv(OUTPUT_DIR / "single_confounder_partial_correlations.csv", index=False)
    pd.DataFrame(lag_rows).to_csv(OUTPUT_DIR / "lag_chain_results.csv", index=False)

    print("\n" + "=" * 65)
    print("FINAL SUMMARY FOR REPORT")
    print("=" * 65)
    if abs(r_raw) < 0.05 and p_raw >= 0.05:
        print("  No detectable zero-order differenced relationship was found.")
        print("  Confounding is therefore not needed to explain the weak sentiment/stringency link.")
    elif abs(r_partial) < 0.05:
        print("  The relationship is largely removed after controlling for cases/deaths.")
    elif drop_pct > 50:
        print("  The relationship is substantially reduced after controlling for cases/deaths.")
    else:
        print("  The relationship remains after controls, suggesting only partial confounding.")

    print("\nSaved outputs:")
    for filename in [
        "partial_correlation_scatter.png",
        "lag_chain_comparison.png",
        "lag_chain_cross_correlations.png",
        "partial_correlation_summary.csv",
        "single_confounder_partial_correlations.csv",
        "lag_chain_results.csv",
    ]:
        print(f"  {OUTPUT_DIR / filename}")


if __name__ == "__main__":
    main()
