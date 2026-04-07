"""
Proportionality Hypothesis analysis.

Question:
Does public sentiment respond differently to government stringency depending
on whether COVID cases or deaths are rising versus falling?

The correlation/regression analysis uses differenced data. Regime labels are
derived from level data using a 7-day moving-average gradient.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.linear_model import LinearRegression


BASE_DIR = Path("/home/mohsin/ADS/ADS_pandas_new")
DIFF_DATA_PATH = BASE_DIR / "graphs" / "exploration" / "exp2_differenced_full.csv"
FULL_DATA_PATH = BASE_DIR / ".csv" / "new" / "full_analysis_data.csv"
OUTPUT_DIR = BASE_DIR / "graphs" / "proportionality"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SENTIMENT = "sentiment_mean"
STRINGENCY = "StringencyIndex_Average"
CASES = "daily_new_cases"
DEATHS = "daily_new_deaths"
REQUIRED = [SENTIMENT, STRINGENCY, CASES, DEATHS]


def significance_stars(p_value):
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


def load_data():
    if not FULL_DATA_PATH.exists():
        raise FileNotFoundError(f"Level data required for regime classification was not found: {FULL_DATA_PATH}")

    level_df = pd.read_csv(FULL_DATA_PATH, parse_dates=["date"]).sort_values("date").set_index("date")

    if DIFF_DATA_PATH.exists():
        diff_df = pd.read_csv(DIFF_DATA_PATH, parse_dates=["date"]).sort_values("date").set_index("date")
        print(f"Loaded differenced data: {DIFF_DATA_PATH} ({len(diff_df)} rows)")
    else:
        print(f"Differenced data not found. Loading full data and differencing: {FULL_DATA_PATH}")
        diff_df = level_df[REQUIRED].diff().dropna()

    missing_level = [col for col in REQUIRED if col not in level_df.columns]
    missing_diff = [col for col in REQUIRED if col not in diff_df.columns]
    if missing_level:
        raise ValueError(f"Missing required columns in level data: {missing_level}")
    if missing_diff:
        raise ValueError(f"Missing required columns in differenced data: {missing_diff}")

    return level_df[REQUIRED].dropna(), diff_df[REQUIRED].dropna()


def classify_regime(level_df, signal_col):
    ma_7d = level_df[signal_col].rolling(window=7, min_periods=3).mean()
    gradient = ma_7d.diff()
    return pd.Series(
        np.where(gradient > 0, "rising", "falling"),
        index=level_df.index,
        name=f"{signal_col}_regime",
    )


def attach_regime(diff_df, level_df, signal_col):
    regime = classify_regime(level_df, signal_col)
    df = diff_df.copy()
    df["regime"] = regime.reindex(df.index).ffill().fillna("falling")
    df["regime_dummy"] = (df["regime"] == "rising").astype(int)
    return df.dropna(subset=[SENTIMENT, STRINGENCY, "regime", "regime_dummy"])


def pearson_by_regime(df, signal_name):
    rows = []
    for regime_name in ["rising", "falling"]:
        subset = df[df["regime"] == regime_name]
        if len(subset) < 3:
            r_value, p_value = np.nan, np.nan
        else:
            r_value, p_value = stats.pearsonr(subset[STRINGENCY], subset[SENTIMENT])
        rows.append(
            {
                "signal": signal_name,
                "regime": regime_name,
                "n": len(subset),
                "r": r_value,
                "p_value": p_value,
            }
        )
    return rows


def fisher_z_test(r1, n1, r2, n2):
    if min(n1, n2) <= 3 or any(pd.isna(value) for value in [r1, r2]):
        return np.nan, np.nan
    r1 = np.clip(r1, -0.999999, 0.999999)
    r2 = np.clip(r2, -0.999999, 0.999999)
    z_score = (np.arctanh(r1) - np.arctanh(r2)) / np.sqrt((1 / (n1 - 3)) + (1 / (n2 - 3)))
    p_value = 2 * stats.norm.sf(abs(z_score))
    return z_score, p_value


def run_interaction_regression(df, signal_name):
    model_df = df[[SENTIMENT, STRINGENCY, "regime_dummy"]].rename(
        columns={SENTIMENT: "sentiment_diff", STRINGENCY: "stringency_diff"}
    )
    model = smf.ols("sentiment_diff ~ stringency_diff * regime_dummy", data=model_df).fit()
    interaction_name = "stringency_diff:regime_dummy"
    return {
        "signal": signal_name,
        "interaction_coef": model.params.get(interaction_name, np.nan),
        "interaction_p_value": model.pvalues.get(interaction_name, np.nan),
        "model": model,
    }


def plot_regime_split(results_df):
    plot_df = results_df[results_df["result_type"] == "split_correlation"].copy()
    fig, ax = plt.subplots(figsize=(10, 5.5))
    labels = [f"{row.signal}\n{row.regime}" for row in plot_df.itertuples()]
    colors = ["#e74c3c" if row.regime == "rising" else "#27ae60" for row in plot_df.itertuples()]

    bars = ax.bar(range(len(plot_df)), plot_df["r"], color=colors, edgecolor="black", alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(plot_df)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Pearson r")
    ax.set_title("Proportionality Hypothesis: Correlation by Rising/Falling Regime", fontweight="bold")
    ax.grid(axis="y", alpha=0.25)

    for bar, row in zip(bars, plot_df.itertuples()):
        text_y = row.r + (0.015 if row.r >= 0 else -0.035)
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            text_y,
            f"r={row.r:.3f}{significance_stars(row.p_value)}\nN={row.n}",
            ha="center",
            va="bottom" if row.r >= 0 else "top",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "regime_split_correlation.png", dpi=200, bbox_inches="tight")
    plt.close()


def plot_regime_scatter(case_df, death_df):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    panels = [
        ("Cases rising", case_df[case_df["regime"] == "rising"], axes[0, 0], "#e74c3c"),
        ("Cases falling", case_df[case_df["regime"] == "falling"], axes[0, 1], "#27ae60"),
        ("Deaths rising", death_df[death_df["regime"] == "rising"], axes[1, 0], "#e74c3c"),
        ("Deaths falling", death_df[death_df["regime"] == "falling"], axes[1, 1], "#27ae60"),
    ]

    for title, subset, ax, color in panels:
        ax.scatter(subset[STRINGENCY], subset[SENTIMENT], alpha=0.35, s=12, color=color)
        if len(subset) >= 3:
            model = LinearRegression().fit(subset[[STRINGENCY]], subset[SENTIMENT])
            x_line = np.linspace(subset[STRINGENCY].min(), subset[STRINGENCY].max(), 100)
            y_line = model.predict(pd.DataFrame({STRINGENCY: x_line}))
            ax.plot(x_line, y_line, color="black", linestyle="--", linewidth=2)
            r_value, p_value = stats.pearsonr(subset[STRINGENCY], subset[SENTIMENT])
            subtitle = f"r={r_value:.3f}, p={p_value:.3f}, N={len(subset)}"
        else:
            subtitle = f"N={len(subset)}"

        ax.axhline(0, color="gray", linewidth=0.5)
        ax.axvline(0, color="gray", linewidth=0.5)
        ax.set_title(f"{title}\n{subtitle}", fontweight="bold")
        ax.set_xlabel("Differenced Stringency")
        ax.set_ylabel("Differenced Sentiment Mean")
        ax.grid(alpha=0.25)

    fig.suptitle("Regime Scatter: Sentiment Response to Stringency", fontweight="bold", fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "regime_scatter.png", dpi=200, bbox_inches="tight")
    plt.close()


def shade_regimes(ax, df):
    start_date = df.index[0]
    current_regime = df["regime"].iloc[0]
    previous_date = df.index[0]

    for date, regime in df["regime"].iloc[1:].items():
        if regime != current_regime:
            color = "#ffcccc" if current_regime == "rising" else "#d9f2d9"
            ax.axvspan(start_date, previous_date, color=color, alpha=0.35, linewidth=0)
            start_date = date
            current_regime = regime
        previous_date = date

    color = "#ffcccc" if current_regime == "rising" else "#d9f2d9"
    ax.axvspan(start_date, previous_date, color=color, alpha=0.35, linewidth=0)


def plot_rolling_with_regimes(case_df):
    rolling_corr = case_df[STRINGENCY].rolling(window=30).corr(case_df[SENTIMENT])
    fig, ax = plt.subplots(figsize=(13, 6))
    shade_regimes(ax, case_df)
    ax.plot(rolling_corr.index, rolling_corr, color="#2c3e50", linewidth=2, label="30-day rolling r")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("30-Day Rolling Sentiment/Stringency Correlation with Case Regimes", fontweight="bold")
    ax.set_ylabel("Rolling Pearson r")
    ax.set_xlabel("Date")
    ax.grid(alpha=0.25)

    rising_patch = plt.Rectangle((0, 0), 1, 1, fc="#ffcccc", alpha=0.6)
    falling_patch = plt.Rectangle((0, 0), 1, 1, fc="#d9f2d9", alpha=0.6)
    ax.legend([ax.lines[0], rising_patch, falling_patch], ["30-day rolling r", "Cases rising", "Cases falling"])
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "rolling_with_regimes.png", dpi=200, bbox_inches="tight")
    plt.close()


def analyze_signal(diff_df, level_df, signal_col, signal_name):
    df = attach_regime(diff_df, level_df, signal_col)
    split_rows = pearson_by_regime(df, signal_name)
    rising = next(row for row in split_rows if row["regime"] == "rising")
    falling = next(row for row in split_rows if row["regime"] == "falling")
    fisher_z, fisher_p = fisher_z_test(rising["r"], rising["n"], falling["r"], falling["n"])
    regression = run_interaction_regression(df, signal_name)

    print("\n" + "=" * 72)
    print(f"PROPORTIONALITY TEST USING {signal_name.upper()} REGIMES")
    print("=" * 72)
    for row in split_rows:
        print(f"  {row['regime']:7s}: r = {row['r']:+.4f}, p = {row['p_value']:.6f}, N = {row['n']}")
    print(f"  Fisher z difference: z = {fisher_z:+.4f}, p = {fisher_p:.6f}")
    print(
        "  Interaction term: "
        f"coef = {regression['interaction_coef']:+.6f}, p = {regression['interaction_p_value']:.6f}"
    )

    result_rows = []
    for row in split_rows:
        result_rows.append(
            {
                "signal": signal_name,
                "result_type": "split_correlation",
                "regime": row["regime"],
                "n": row["n"],
                "r": row["r"],
                "p_value": row["p_value"],
                "fisher_z": np.nan,
                "fisher_p_value": np.nan,
                "interaction_coef": regression["interaction_coef"],
                "interaction_p_value": regression["interaction_p_value"],
            }
        )
    result_rows.append(
        {
            "signal": signal_name,
            "result_type": "fisher_difference",
            "regime": "rising_vs_falling",
            "n": len(df),
            "r": np.nan,
            "p_value": np.nan,
            "fisher_z": fisher_z,
            "fisher_p_value": fisher_p,
            "interaction_coef": regression["interaction_coef"],
            "interaction_p_value": regression["interaction_p_value"],
        }
    )
    return df, regression, result_rows


def main():
    level_df, diff_df = load_data()
    case_df, case_regression, case_rows = analyze_signal(diff_df, level_df, CASES, "cases")
    death_df, death_regression, death_rows = analyze_signal(diff_df, level_df, DEATHS, "deaths")

    results_df = pd.DataFrame(case_rows + death_rows)
    results_df.to_csv(OUTPUT_DIR / "proportionality_results.csv", index=False)

    with open(OUTPUT_DIR / "interaction_regression_summary.txt", "w", encoding="utf-8") as f:
        f.write("PROPORTIONALITY HYPOTHESIS: INTERACTION REGRESSION SUMMARY\n")
        f.write("=" * 72 + "\n\n")
        f.write("CASES REGIME MODEL\n")
        f.write(case_regression["model"].summary().as_text())
        f.write("\n\n" + "=" * 72 + "\n\n")
        f.write("DEATHS REGIME MODEL\n")
        f.write(death_regression["model"].summary().as_text())

    plot_regime_split(results_df)
    plot_regime_scatter(case_df, death_df)
    plot_rolling_with_regimes(case_df)

    print("\n" + "=" * 72)
    print("FINAL INTERPRETATION")
    print("=" * 72)
    if case_regression["interaction_p_value"] < 0.05:
        print("Cases regime: significant interaction; sentiment/stringency relationship differs by case trend.")
    else:
        print("Cases regime: no significant interaction; no clear evidence of a case-trend sign switch.")
    if death_regression["interaction_p_value"] < 0.05:
        print("Deaths regime: significant interaction; sentiment/stringency relationship differs by death trend.")
    else:
        print("Deaths regime: no significant interaction; no clear evidence of a death-trend sign switch.")

    print("\nSaved outputs:")
    for filename in [
        "regime_split_correlation.png",
        "interaction_regression_summary.txt",
        "rolling_with_regimes.png",
        "regime_scatter.png",
        "proportionality_results.csv",
    ]:
        print(f"  {OUTPUT_DIR / filename}")


if __name__ == "__main__":
    main()
