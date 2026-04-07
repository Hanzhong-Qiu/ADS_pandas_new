"""
Data-driven pandemic shock coupling analysis.

This script detects pandemic "shock events" objectively from the data itself
and tests whether those shocks coincide with elevated sentiment-policy
coupling.

Inputs:
- graphs/exploration/exp2_differenced_full.csv for stationary rolling
  sentiment/stringency correlation.
- .csv/new/full_analysis_data.csv for level-data shock detection and context.
"""

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


BASE_DIR = Path("/home/mohsin/ADS/ADS_pandas_new")
DIFF_DATA_PATH = BASE_DIR / "graphs" / "exploration" / "exp2_differenced_full.csv"
FULL_DATA_PATH = BASE_DIR / ".csv" / "new" / "full_analysis_data.csv"
OUTPUT_DIR = BASE_DIR / "graphs" / "shock_coupling"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SENTIMENT = "sentiment_mean"
STRINGENCY = "StringencyIndex_Average"
CASES = "daily_new_cases"
DEATHS = "daily_new_deaths"
REQUIRED = [SENTIMENT, STRINGENCY, CASES, DEATHS]

ROLLING_WINDOW = 30
COUPLING_THRESHOLD = 0.30
SHOCK_QUANTILE = 0.95
MERGE_GAP_DAYS = 3
NEAR_WINDOW_DAYS = 7
COMBINED_WINDOW_DAYS = 3
IMPULSE_OFFSETS = np.array([0, 7, 14, 21])
N_PERMUTATIONS = 1000
RANDOM_SEED = 42

SHOCK_COLORS = {
    "policy": "#d73027",
    "cases": "#7b3294",
    "deaths": "#8c510a",
    "any": "#2c3e50",
}


def load_data():
    if not FULL_DATA_PATH.exists():
        raise FileNotFoundError(f"Level data not found: {FULL_DATA_PATH}")

    level_df = pd.read_csv(FULL_DATA_PATH, parse_dates=["date"]).sort_values("date").set_index("date")
    missing_level = [col for col in REQUIRED if col not in level_df.columns]
    if missing_level:
        raise ValueError(f"Missing required columns in level data: {missing_level}")

    if DIFF_DATA_PATH.exists():
        diff_df = pd.read_csv(DIFF_DATA_PATH, parse_dates=["date"]).sort_values("date").set_index("date")
        print(f"Loaded differenced data: {DIFF_DATA_PATH} ({len(diff_df)} rows)")
    else:
        print(f"Differenced data not found. Differencing level data from: {FULL_DATA_PATH}")
        diff_df = level_df[REQUIRED].diff().dropna()

    missing_diff = [col for col in REQUIRED if col not in diff_df.columns]
    if missing_diff:
        raise ValueError(f"Missing required columns in differenced data: {missing_diff}")

    level_df = level_df[REQUIRED].dropna()
    diff_df = diff_df[REQUIRED].dropna()
    common_dates = diff_df.index.intersection(level_df.index)
    level_df = level_df.loc[common_dates].copy()
    diff_df = diff_df.loc[common_dates].copy()

    print(f"Analysis range: {common_dates.min().date()} to {common_dates.max().date()}")
    print(f"Differenced rows: {len(diff_df)} | Level rows: {len(level_df)}")
    return diff_df, level_df


def compute_rolling_correlation(diff_df):
    rolling_r = diff_df[SENTIMENT].rolling(ROLLING_WINDOW).corr(diff_df[STRINGENCY]).dropna()
    rolling_r.name = "rolling_r"
    print("\nComputed rolling correlation:")
    print(f"  Window: {ROLLING_WINDOW} days")
    print(f"  Max r: {rolling_r.max():+.4f} on {rolling_r.idxmax().date()}")
    print(f"  Min r: {rolling_r.min():+.4f} on {rolling_r.idxmin().date()}")
    print(f"  Days with |r| > {COUPLING_THRESHOLD}: {(rolling_r.abs() > COUPLING_THRESHOLD).sum()}")
    return rolling_r


def merge_nearby_events(candidates, max_gap_days=MERGE_GAP_DAYS):
    if candidates.empty:
        return candidates

    candidates = candidates.sort_index()
    merged_rows = []
    cluster = [(candidates.index[0], candidates.iloc[0])]

    for date, row in candidates.iloc[1:].iterrows():
        previous_date = cluster[-1][0]
        if (date - previous_date).days <= max_gap_days:
            cluster.append((date, row))
        else:
            merged_rows.append(select_largest_event(cluster))
            cluster = [(date, row)]
    merged_rows.append(select_largest_event(cluster))

    return pd.DataFrame(merged_rows).set_index("date").sort_index()


def select_largest_event(cluster):
    best_date, best_row = max(cluster, key=lambda item: abs(item[1]["magnitude"]))
    row = best_row.to_dict()
    row["date"] = best_date
    return row


def detect_policy_shocks(level_df):
    delta = level_df[STRINGENCY].diff().dropna()
    threshold = delta.abs().quantile(SHOCK_QUANTILE)
    candidates = pd.DataFrame({"magnitude": delta[delta.abs() > threshold]})
    shocks = merge_nearby_events(candidates)
    shocks["shock_type"] = "policy"
    shocks["description"] = shocks["magnitude"].map(lambda value: f"Stringency changed by {value:+.2f} points")
    print(f"\nPolicy shocks: threshold |Delta Stringency| > {threshold:.4f}; detected {len(shocks)} merged events")
    return shocks


def detect_growth_shocks(level_df, column, shock_type):
    ma_7d = level_df[column].rolling(7, min_periods=3).mean()
    pct_change = ma_7d.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).dropna()
    pct_change = pct_change[pct_change.abs() < 20]
    threshold = pct_change.abs().quantile(SHOCK_QUANTILE)
    candidates = pd.DataFrame({"magnitude": pct_change[pct_change.abs() > threshold]})
    shocks = merge_nearby_events(candidates)
    shocks["shock_type"] = shock_type
    label = "cases" if shock_type == "cases" else "deaths"
    shocks["description"] = shocks["magnitude"].map(lambda value: f"7-day MA {label} changed by {value * 100:+.1f}%")
    print(f"{shock_type.title()} shocks: threshold |7-day MA pct change| > {threshold * 100:.2f}%; detected {len(shocks)} merged events")
    return shocks


def detect_all_shocks(level_df):
    policy = detect_policy_shocks(level_df)
    cases = detect_growth_shocks(level_df, CASES, "cases")
    deaths = detect_growth_shocks(level_df, DEATHS, "deaths")

    detected = pd.concat([policy, cases, deaths], axis=0)
    detected = detected.reset_index()[["date", "shock_type", "magnitude", "description"]]
    detected["date"] = pd.to_datetime(detected["date"])
    detected = detected.sort_values(["date", "shock_type"]).reset_index(drop=True)

    print("\nDetected shock events:")
    if detected.empty:
        print("  No shocks detected.")
    else:
        print(detected.to_string(index=False))

    detected.to_csv(OUTPUT_DIR / "detected_shocks.csv", index=False)
    return detected


def build_shock_indicators(index, detected_shocks):
    indicators = pd.DataFrame(index=index)
    for shock_type in ["policy", "cases", "deaths"]:
        indicators[f"{shock_type}_shock_nearby"] = 0
        indicators[f"{shock_type}_shock_window3"] = 0

    for row in detected_shocks.itertuples():
        shock_type = row.shock_type
        date = pd.Timestamp(row.date)
        near_mask = (indicators.index >= date - pd.Timedelta(days=NEAR_WINDOW_DAYS)) & (
            indicators.index <= date + pd.Timedelta(days=NEAR_WINDOW_DAYS)
        )
        window3_mask = (indicators.index >= date - pd.Timedelta(days=COMBINED_WINDOW_DAYS)) & (
            indicators.index <= date + pd.Timedelta(days=COMBINED_WINDOW_DAYS)
        )
        indicators.loc[near_mask, f"{shock_type}_shock_nearby"] = 1
        indicators.loc[window3_mask, f"{shock_type}_shock_window3"] = 1

    indicators["any_shock_nearby"] = indicators[
        ["policy_shock_nearby", "cases_shock_nearby", "deaths_shock_nearby"]
    ].max(axis=1)
    indicators["any_shock_window3"] = indicators[
        ["policy_shock_window3", "cases_shock_window3", "deaths_shock_window3"]
    ].max(axis=1)
    return indicators


def cohens_d(group_a, group_b):
    group_a = np.asarray(group_a, dtype=float)
    group_b = np.asarray(group_b, dtype=float)
    if len(group_a) < 2 or len(group_b) < 2:
        return np.nan
    pooled_var = (
        ((len(group_a) - 1) * np.var(group_a, ddof=1) + (len(group_b) - 1) * np.var(group_b, ddof=1))
        / (len(group_a) + len(group_b) - 2)
    )
    if pooled_var <= 0:
        return np.nan
    return (np.mean(group_a) - np.mean(group_b)) / np.sqrt(pooled_var)


def run_mann_whitney_tests(analysis_df):
    rows = []
    shock_specs = [
        ("policy", "policy_shock_nearby"),
        ("cases", "case_shock_nearby"),
        ("deaths", "death_shock_nearby"),
        ("any", "any_shock_nearby"),
    ]

    print("\nMann-Whitney U tests on |rolling r|:")
    for shock_type, indicator_col in shock_specs:
        shock_values = analysis_df.loc[analysis_df[indicator_col] == 1, "abs_rolling_r"].dropna()
        nonshock_values = analysis_df.loc[analysis_df[indicator_col] == 0, "abs_rolling_r"].dropna()

        if len(shock_values) == 0 or len(nonshock_values) == 0:
            u_stat, p_value = np.nan, np.nan
        else:
            u_stat, p_value = stats.mannwhitneyu(shock_values, nonshock_values, alternative="two-sided")

        row = {
            "shock_type": shock_type,
            "n_shock_days": len(shock_values),
            "n_non_shock_days": len(nonshock_values),
            "median_r_shock": shock_values.median(),
            "median_r_nonshock": nonshock_values.median(),
            "U_statistic": u_stat,
            "p_value": p_value,
            "cohens_d": cohens_d(shock_values, nonshock_values),
            "coupling_proportion_shock": (shock_values > COUPLING_THRESHOLD).mean() if len(shock_values) else np.nan,
            "coupling_proportion_nonshock": (nonshock_values > COUPLING_THRESHOLD).mean() if len(nonshock_values) else np.nan,
        }
        rows.append(row)
        print(
            f"  {shock_type:6s}: median shock={row['median_r_shock']:.4f}, "
            f"median non-shock={row['median_r_nonshock']:.4f}, p={row['p_value']:.6f}, "
            f"d={row['cohens_d']:.3f}"
        )

    results = pd.DataFrame(rows)
    results.to_csv(OUTPUT_DIR / "shock_coupling_test.csv", index=False)
    return results


def run_regression(analysis_df):
    model_df = analysis_df[
        ["abs_rolling_r", "policy_shock_nearby", "case_shock_nearby", "death_shock_nearby"]
    ].dropna()
    x = sm.add_constant(model_df[["policy_shock_nearby", "case_shock_nearby", "death_shock_nearby"]])
    y = model_df["abs_rolling_r"]
    model = sm.OLS(y, x).fit()

    with open(OUTPUT_DIR / "regression_summary.txt", "w", encoding="utf-8") as handle:
        handle.write("OLS: |rolling_r| ~ policy_shock_nearby + case_shock_nearby + death_shock_nearby\n")
        handle.write("=" * 88 + "\n\n")
        handle.write(model.summary().as_text())

    print("\nOLS regression on |rolling r|:")
    print(f"  R-squared: {model.rsquared:.4f}")
    for name in ["policy_shock_nearby", "case_shock_nearby", "death_shock_nearby"]:
        print(f"  {name}: coef={model.params[name]:+.4f}, p={model.pvalues[name]:.6f}")

    return model


def extract_impulse_values(abs_rolling_r, event_dates):
    rows = []
    used_dates = []
    for date in pd.to_datetime(event_dates):
        values = []
        complete = True
        for offset in IMPULSE_OFFSETS:
            target = date + pd.Timedelta(days=int(offset))
            if target not in abs_rolling_r.index or pd.isna(abs_rolling_r.loc[target]):
                complete = False
                break
            values.append(abs_rolling_r.loc[target])
        if complete:
            rows.append(values)
            used_dates.append(date)
    return pd.DatetimeIndex(used_dates), np.array(rows, dtype=float)


def impulse_response_by_shock_type(abs_rolling_r, detected_shocks, analysis_df):
    rng = np.random.default_rng(RANDOM_SEED)
    valid_dates = pd.DatetimeIndex([date for date in abs_rolling_r.dropna().index if date + pd.Timedelta(days=21) in abs_rolling_r.index])
    nonshock_dates = analysis_df.loc[analysis_df["any_shock_nearby"] == 0].index.intersection(valid_dates)
    responses = {}

    print("\nCoupling impulse response by shock type:")
    for shock_type in ["policy", "cases", "deaths"]:
        raw_dates = detected_shocks.loc[detected_shocks["shock_type"] == shock_type, "date"]
        used_dates, event_windows = extract_impulse_values(abs_rolling_r, raw_dates)
        if len(event_windows) == 0:
            responses[shock_type] = {"used_dates": used_dates, "mean": np.full(len(IMPULSE_OFFSETS), np.nan), "low": np.full(len(IMPULSE_OFFSETS), np.nan), "high": np.full(len(IMPULSE_OFFSETS), np.nan)}
            print(f"  {shock_type:6s}: no complete event windows")
            continue

        mean_response = event_windows.mean(axis=0)
        random_means = []
        for _ in range(N_PERMUTATIONS):
            sampled_dates = rng.choice(nonshock_dates, size=len(used_dates), replace=True)
            _, sampled_windows = extract_impulse_values(abs_rolling_r, pd.DatetimeIndex(sampled_dates))
            if len(sampled_windows) == len(used_dates):
                random_means.append(sampled_windows.mean(axis=0))

        random_means = np.array(random_means)
        low = np.percentile(random_means, 2.5, axis=0)
        high = np.percentile(random_means, 97.5, axis=0)
        responses[shock_type] = {"used_dates": used_dates, "mean": mean_response, "low": low, "high": high}
        print(
            f"  {shock_type:6s}: N={len(used_dates)}, "
            f"mean |r| at day 0={mean_response[0]:.4f}, day +21={mean_response[-1]:.4f}"
        )

    return responses


def plot_shock_timeline(rolling_r, level_df, detected_shocks):
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(16, 10),
        sharex=True,
        gridspec_kw={"height_ratios": [2.5, 1.4, 1.6]},
    )
    ax_top, ax_mid, ax_bottom = axes

    ax_top.plot(rolling_r.index, rolling_r, color="#1f2933", linewidth=2.0, label="30-day rolling r")
    ax_top.axhline(COUPLING_THRESHOLD, color="#b2182b", linestyle=":", linewidth=1.2, label="|r| = 0.30")
    ax_top.axhline(-COUPLING_THRESHOLD, color="#2166ac", linestyle=":", linewidth=1.2)
    ax_top.axhline(0, color="black", linewidth=0.8)
    ax_top.set_ylabel("Rolling Pearson r")
    ax_top.set_title("Data-Detected Pandemic Shocks and Sentiment-Policy Coupling", fontsize=15, fontweight="bold")
    ax_top.grid(alpha=0.25)
    ax_top.legend(loc="lower left")

    lane_y = {"policy": 3, "cases": 2, "deaths": 1}
    lane_labels = {"policy": "Policy", "cases": "Cases", "deaths": "Deaths"}
    for shock_type, group in detected_shocks.groupby("shock_type"):
        max_magnitude = group["magnitude"].abs().max()
        for row in group.itertuples():
            height = 0.15 + 0.75 * (abs(row.magnitude) / max_magnitude if max_magnitude else 0)
            y0 = lane_y[shock_type] - 0.35
            y1 = y0 + height
            ax_mid.vlines(row.date, y0, y1, color=SHOCK_COLORS[shock_type], linewidth=2.0, alpha=0.9)
            ax_mid.scatter(row.date, y1, color=SHOCK_COLORS[shock_type], s=28, zorder=3)

    ax_mid.set_yticks([lane_y["deaths"], lane_y["cases"], lane_y["policy"]])
    ax_mid.set_yticklabels([lane_labels["deaths"], lane_labels["cases"], lane_labels["policy"]])
    ax_mid.set_ylim(0.4, 3.8)
    ax_mid.set_ylabel("Shock type")
    ax_mid.set_title("Objectively Detected Shock Events", fontsize=11, fontweight="bold")
    ax_mid.grid(axis="x", alpha=0.2)

    ax_bottom.plot(level_df.index, level_df[STRINGENCY], color="#d35400", linewidth=1.9, label="Stringency Index")
    ax_bottom.set_ylabel("Stringency")
    ax_bottom.set_title("Level Stringency Context", fontsize=11, fontweight="bold")
    ax_bottom.grid(alpha=0.25)
    ax_bottom.legend(loc="upper right")

    ax_bottom.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax_bottom.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax_bottom.get_xticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "shock_timeline.png", dpi=220, bbox_inches="tight")
    plt.close()


def plot_mann_whitney_results(analysis_df, test_results):
    specs = [
        ("policy", "policy_shock_nearby"),
        ("cases", "case_shock_nearby"),
        ("deaths", "death_shock_nearby"),
        ("any", "any_shock_nearby"),
    ]

    data = []
    positions = []
    colors = []
    labels = []
    pos = 1
    for shock_type, indicator_col in specs:
        shock = analysis_df.loc[analysis_df[indicator_col] == 1, "abs_rolling_r"].dropna()
        nonshock = analysis_df.loc[analysis_df[indicator_col] == 0, "abs_rolling_r"].dropna()
        data.extend([shock, nonshock])
        positions.extend([pos, pos + 0.45])
        colors.extend([SHOCK_COLORS.get(shock_type, "#333333"), "#bdbdbd"])
        labels.append((pos + 0.225, shock_type))
        pos += 1.4

    fig, ax = plt.subplots(figsize=(13, 6))
    box = ax.boxplot(data, positions=positions, widths=0.35, patch_artist=True, showfliers=False)
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.78)

    ax.set_xticks([x for x, _ in labels])
    ax.set_xticklabels([label for _, label in labels])
    ax.set_ylabel("|30-day rolling r|")
    ax.set_title("Shock-Adjacent vs Non-Shock Coupling Strength", fontweight="bold")
    ax.grid(axis="y", alpha=0.25)

    for center, shock_type in labels:
        p_value = test_results.loc[test_results["shock_type"] == shock_type, "p_value"].iloc[0]
        y_pos = ax.get_ylim()[1] * 0.94
        ax.text(center, y_pos, f"p={p_value:.3g}", ha="center", va="top", fontsize=10, fontweight="bold")

    legend_patches = [
        plt.Rectangle((0, 0), 1, 1, color="#555555", alpha=0.78),
        plt.Rectangle((0, 0), 1, 1, color="#bdbdbd", alpha=0.78),
    ]
    ax.legend(legend_patches, ["Shock-adjacent", "Non-shock"], loc="upper right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "mann_whitney_results.png", dpi=220, bbox_inches="tight")
    plt.close()


def plot_impulse_responses(responses):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax, shock_type in zip(axes, ["policy", "cases", "deaths"]):
        response = responses[shock_type]
        ax.fill_between(IMPULSE_OFFSETS, response["low"], response["high"], color="#d9d9d9", alpha=0.8, label="95% random band")
        ax.plot(
            IMPULSE_OFFSETS,
            response["mean"],
            color=SHOCK_COLORS[shock_type],
            marker="o",
            linewidth=2.2,
            label=f"{shock_type} shocks",
        )
        ax.axhline(COUPLING_THRESHOLD, color="#666666", linestyle=":", linewidth=1.0, label="|r| = 0.30")
        ax.set_title(f"{shock_type.title()} shocks\nN={len(response['used_dates'])}", fontweight="bold")
        ax.set_xlabel("Days after shock")
        ax.set_xticks(IMPULSE_OFFSETS)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Mean |rolling r|")
    axes[-1].legend(loc="upper right")
    fig.suptitle("Coupling Impulse Response by Shock Type", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "coupling_impulse_by_shock_type.png", dpi=220, bbox_inches="tight")
    plt.close()


def main():
    print("=" * 90)
    print("DATA-DRIVEN SHOCK COUPLING ANALYSIS")
    print("=" * 90)

    diff_df, level_df = load_data()
    rolling_r = compute_rolling_correlation(diff_df)
    detected_shocks = detect_all_shocks(level_df)

    indicators = build_shock_indicators(rolling_r.index, detected_shocks)
    analysis_df = pd.DataFrame({"rolling_r": rolling_r, "abs_rolling_r": rolling_r.abs()}, index=rolling_r.index)
    analysis_df = analysis_df.join(indicators, how="left").fillna(0)
    analysis_df = analysis_df.rename(columns={"cases_shock_nearby": "case_shock_nearby", "deaths_shock_nearby": "death_shock_nearby"})

    test_results = run_mann_whitney_tests(analysis_df)
    regression_model = run_regression(analysis_df)
    responses = impulse_response_by_shock_type(analysis_df["abs_rolling_r"], detected_shocks, analysis_df)

    plot_shock_timeline(rolling_r, level_df, detected_shocks)
    plot_mann_whitney_results(analysis_df, test_results)
    plot_impulse_responses(responses)

    print("\n" + "=" * 90)
    print("FINAL INTERPRETATION")
    print("=" * 90)
    any_row = test_results[test_results["shock_type"] == "any"].iloc[0]
    if any_row["p_value"] < 0.05:
        print("Data-detected shocks coincide with significantly elevated sentiment-policy coupling.")
        print(
            f"Median |r| near any shock = {any_row['median_r_shock']:.4f}, "
            f"vs {any_row['median_r_nonshock']:.4f} away from shocks."
        )
    else:
        print("Data-detected shocks do not show a statistically significant elevation in coupling overall.")

    best_predictor = regression_model.params.drop("const").idxmax()
    print(
        f"In the OLS model, the largest positive coefficient is {best_predictor} "
        f"(coef={regression_model.params[best_predictor]:+.4f}, p={regression_model.pvalues[best_predictor]:.6f})."
    )

    print("\nSaved outputs:")
    for filename in [
        "detected_shocks.csv",
        "shock_coupling_test.csv",
        "shock_timeline.png",
        "mann_whitney_results.png",
        "coupling_impulse_by_shock_type.png",
        "regression_summary.txt",
    ]:
        print(f"  {OUTPUT_DIR / filename}")


if __name__ == "__main__":
    main()
