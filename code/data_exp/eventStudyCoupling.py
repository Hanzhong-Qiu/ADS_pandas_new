"""
Event Study Coupling Analysis.

Purpose:
Identify windows where sentiment and policy stringency are unusually coupled,
then map those windows to real-world pandemic events.

Inputs:
- graphs/exploration/exp2_differenced_full.csv for stationary rolling correlation
- .csv/new/full_analysis_data.csv for level context and event-study policy shocks
"""

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


BASE_DIR = Path("/home/mohsin/ADS/ADS_pandas_new")
DIFF_DATA_PATH = BASE_DIR / "graphs" / "exploration" / "exp2_differenced_full.csv"
FULL_DATA_PATH = BASE_DIR / ".csv" / "new" / "full_analysis_data.csv"
OUTPUT_DIR = BASE_DIR / "graphs" / "event_study"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SENTIMENT = "sentiment_mean"
STRINGENCY = "StringencyIndex_Average"
CASES = "daily_new_cases"
DEATHS = "daily_new_deaths"
REQUIRED = [SENTIMENT, STRINGENCY, CASES, DEATHS]

ROLLING_WINDOW = 30
ACTIVE_THRESHOLD = 0.30
MERGE_GAP_DAYS = 7
POLICY_SHOCK_THRESHOLD = 5
EVENT_WINDOW = 7
N_PERMUTATIONS = 1000
RANDOM_SEED = 42

KNOWN_EVENTS = {
    "2020-03-23": "UK first national lockdown",
    "2020-06-01": "UK lockdown easing begins",
    "2020-09-22": "UK rule of six introduced",
    "2020-11-05": "UK second national lockdown",
    "2020-12-08": "UK begins vaccination rollout",
    "2020-12-19": "UK third lockdown announced (London Tier 4)",
    "2021-01-06": "UK third national lockdown begins",
    "2021-02-22": "UK roadmap out of lockdown announced",
    "2021-04-12": "UK non-essential retail reopens",
    "2021-05-17": "UK indoor hospitality reopens",
    "2021-06-18": "Delta variant declared dominant in UK",
    "2021-07-19": 'UK "Freedom Day" most restrictions lifted',
    "2021-11-26": "Omicron variant first reported globally",
    "2021-12-08": "UK Plan B restrictions announced",
    "2021-12-13": "UK booster programme accelerated",
    "2022-01-27": "UK Plan B restrictions lifted",
    "2022-02-24": 'UK "Living with COVID" plan announced',
    "2022-04-01": "Free COVID testing ends in UK",
}
KNOWN_EVENTS = {pd.Timestamp(date): label for date, label in KNOWN_EVENTS.items()}


def load_data():
    if not FULL_DATA_PATH.exists():
        raise FileNotFoundError(f"Level data not found: {FULL_DATA_PATH}")

    level_df = pd.read_csv(FULL_DATA_PATH, parse_dates=["date"]).sort_values("date").set_index("date")

    if DIFF_DATA_PATH.exists():
        diff_df = pd.read_csv(DIFF_DATA_PATH, parse_dates=["date"]).sort_values("date").set_index("date")
        print(f"Loaded differenced data: {DIFF_DATA_PATH} ({len(diff_df)} rows)")
    else:
        print(f"Differenced data not found. Differencing level data from: {FULL_DATA_PATH}")
        diff_df = level_df[REQUIRED].diff().dropna()

    missing_level = [col for col in REQUIRED if col not in level_df.columns]
    missing_diff = [col for col in REQUIRED if col not in diff_df.columns]
    if missing_level:
        raise ValueError(f"Missing columns in level data: {missing_level}")
    if missing_diff:
        raise ValueError(f"Missing columns in differenced data: {missing_diff}")

    level_df = level_df[REQUIRED].dropna()
    diff_df = diff_df[REQUIRED].dropna()
    common_dates = diff_df.index.intersection(level_df.index)
    return level_df.loc[common_dates].copy(), diff_df.loc[common_dates].copy()


def compute_rolling_correlation(diff_df):
    return diff_df[SENTIMENT].rolling(ROLLING_WINDOW).corr(diff_df[STRINGENCY]).dropna()


def group_episode_dates(active_dates):
    if len(active_dates) == 0:
        return []

    dates = pd.DatetimeIndex(active_dates).sort_values()
    episodes = []
    start_date = dates[0]
    previous_date = dates[0]

    for date in dates[1:]:
        gap_days = (date - previous_date).days
        if gap_days > MERGE_GAP_DAYS:
            episodes.append((start_date, previous_date))
            start_date = date
        previous_date = date

    episodes.append((start_date, previous_date))
    return episodes


def matched_events_for_episode(start_date, end_date):
    match_start = start_date - pd.Timedelta(days=7)
    labels = []
    for event_date, label in KNOWN_EVENTS.items():
        if match_start <= event_date <= end_date:
            labels.append(f"{event_date.date()}: {label}")
    return "; ".join(labels) if labels else "No known event matched"


def get_window_change(level_df, start_date, end_date, column):
    window = level_df.loc[start_date:end_date, column].dropna()
    if window.empty:
        return np.nan
    return window.iloc[-1] - window.iloc[0]


def build_episode_table(rolling_corr, level_df):
    active_dates = rolling_corr[rolling_corr.abs() > ACTIVE_THRESHOLD].index
    episode_ranges = group_episode_dates(active_dates)
    rows = []

    for episode_id, (start_date, end_date) in enumerate(episode_ranges, start=1):
        episode_corr = rolling_corr.loc[start_date:end_date].dropna()
        level_window = level_df.loc[start_date:end_date].dropna()
        if episode_corr.empty or level_window.empty:
            continue

        peak_date = episode_corr.abs().idxmax()
        peak_r = episode_corr.loc[peak_date]

        rows.append(
            {
                "episode_id": episode_id,
                "start_date": start_date.date(),
                "end_date": end_date.date(),
                "duration_days": (end_date - start_date).days + 1,
                "mean_r": episode_corr.mean(),
                "peak_r": peak_r,
                "peak_r_date": peak_date.date(),
                "stringency_change": get_window_change(level_df, start_date, end_date, STRINGENCY),
                "cases_change": get_window_change(level_df, start_date, end_date, CASES),
                "deaths_change": get_window_change(level_df, start_date, end_date, DEATHS),
                "mean_cases": level_window[CASES].mean(),
                "max_cases": level_window[CASES].max(),
                "mean_deaths": level_window[DEATHS].mean(),
                "max_deaths": level_window[DEATHS].max(),
                "mean_stringency": level_window[STRINGENCY].mean(),
                "max_stringency": level_window[STRINGENCY].max(),
                "matched_events": matched_events_for_episode(start_date, end_date),
            }
        )

    return pd.DataFrame(rows)


def episode_mask(index, episode_table):
    mask = pd.Series(False, index=index)
    for row in episode_table.itertuples():
        start_date = pd.Timestamp(row.start_date)
        end_date = pd.Timestamp(row.end_date)
        mask.loc[start_date:end_date] = True
    return mask


def permutation_test(rolling_corr, episode_table):
    if episode_table.empty:
        return np.nan, np.nan, np.array([])

    rng = np.random.default_rng(RANDOM_SEED)
    observed = episode_table["mean_r"].abs().mean()
    n_episodes = len(episode_table)
    average_duration = int(round(episode_table["duration_days"].mean()))
    average_duration = max(1, average_duration)

    mask = episode_mask(rolling_corr.index, episode_table)
    eligible_starts = []
    for i in range(0, len(rolling_corr) - average_duration + 1):
        window_mask = mask.iloc[i : i + average_duration]
        window_values = rolling_corr.iloc[i : i + average_duration]
        if not window_mask.any() and window_values.notna().all():
            eligible_starts.append(i)

    if len(eligible_starts) == 0:
        return observed, np.nan, np.array([])

    null_values = []
    eligible_starts = np.array(eligible_starts)
    for _ in range(N_PERMUTATIONS):
        chosen = rng.choice(eligible_starts, size=n_episodes, replace=True)
        sample_values = [
            rolling_corr.iloc[start : start + average_duration].abs().mean()
            for start in chosen
        ]
        null_values.append(np.mean(sample_values))

    null_values = np.array(null_values)
    p_value = (np.sum(null_values >= observed) + 1) / (len(null_values) + 1)
    return observed, p_value, null_values


def extract_event_windows(series, event_dates, window=EVENT_WINDOW):
    offsets = np.arange(-window, window + 1)
    rows = []
    for event_date in event_dates:
        values = []
        for offset in offsets:
            date = event_date + pd.Timedelta(days=int(offset))
            values.append(series.loc[date] if date in series.index else np.nan)
        rows.append(values)
    if not rows:
        return offsets, pd.DataFrame(columns=offsets)
    return offsets, pd.DataFrame(rows, columns=offsets, index=event_dates)


def impulse_response(diff_df, level_df):
    stringency_delta = level_df[STRINGENCY].diff()
    event_dates = stringency_delta[stringency_delta.abs() > POLICY_SHOCK_THRESHOLD].index
    offsets, event_windows = extract_event_windows(diff_df[SENTIMENT], event_dates)
    mean_response = event_windows.mean(axis=0)

    rng = np.random.default_rng(RANDOM_SEED)
    valid_dates = diff_df.index[EVENT_WINDOW : len(diff_df) - EVENT_WINDOW]
    event_exclusion = pd.Series(False, index=diff_df.index)
    for event_date in event_dates:
        event_exclusion.loc[event_date - pd.Timedelta(days=EVENT_WINDOW) : event_date + pd.Timedelta(days=EVENT_WINDOW)] = True
    non_event_dates = [date for date in valid_dates if not event_exclusion.loc[date]]

    random_means = []
    if len(non_event_dates) > 0 and len(event_dates) > 0:
        non_event_dates = np.array(non_event_dates)
        for _ in range(N_PERMUTATIONS):
            random_dates = rng.choice(non_event_dates, size=len(event_dates), replace=True)
            _, random_windows = extract_event_windows(diff_df[SENTIMENT], pd.DatetimeIndex(random_dates))
            random_means.append(random_windows.mean(axis=0).values)

    if random_means:
        random_means = np.array(random_means)
        lower = np.percentile(random_means, 2.5, axis=0)
        upper = np.percentile(random_means, 97.5, axis=0)
    else:
        lower = np.full(len(offsets), np.nan)
        upper = np.full(len(offsets), np.nan)

    return {
        "event_dates": event_dates,
        "offsets": offsets,
        "mean_response": mean_response.values,
        "lower": lower,
        "upper": upper,
    }


def plot_rolling_correlation(rolling_corr, level_df, episode_table):
    fig, (ax_corr, ax_context) = plt.subplots(
        2,
        1,
        figsize=(16, 9),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    for row in episode_table.itertuples():
        start_date = pd.Timestamp(row.start_date)
        end_date = pd.Timestamp(row.end_date)
        color = "#ffb3b3" if row.peak_r >= 0 else "#b3d1ff"
        ax_corr.axvspan(start_date, end_date, color=color, alpha=0.45, linewidth=0)

    ax_corr.plot(rolling_corr.index, rolling_corr, color="#1f2933", linewidth=2.1, label="30-day rolling r")
    ax_corr.axhline(0, color="black", linewidth=0.8)
    ax_corr.axhline(ACTIVE_THRESHOLD, color="#8b0000", linestyle=":", linewidth=1.1, label="|r| > 0.30 threshold")
    ax_corr.axhline(-ACTIVE_THRESHOLD, color="#004c99", linestyle=":", linewidth=1.1)
    ax_corr.set_ylabel("Rolling Pearson r")
    ax_corr.set_title(
        "Active Coupling Episodes Between Sentiment and Stringency",
        fontsize=15,
        fontweight="bold",
    )
    ax_corr.grid(alpha=0.22)

    y_top = ax_corr.get_ylim()[1]
    for event_date, label in KNOWN_EVENTS.items():
        if rolling_corr.index.min() <= event_date <= rolling_corr.index.max():
            ax_corr.axvline(event_date, color="gray", linestyle="--", linewidth=0.8, alpha=0.55)
            ax_corr.text(
                event_date,
                y_top,
                label,
                rotation=90,
                fontsize=7,
                color="dimgray",
                ha="right",
                va="top",
            )

    positive_patch = plt.Rectangle((0, 0), 1, 1, fc="#ffb3b3", alpha=0.55)
    negative_patch = plt.Rectangle((0, 0), 1, 1, fc="#b3d1ff", alpha=0.55)
    ax_corr.legend(
        [ax_corr.lines[0], positive_patch, negative_patch],
        ["30-day rolling r", "Positive active coupling", "Negative active coupling"],
        loc="lower right",
    )

    ax_context.plot(level_df.index, level_df[STRINGENCY], color="#7f1d1d", linewidth=1.8, label="Stringency index")
    ax_context.set_ylabel("Stringency")
    ax_context.grid(alpha=0.22)
    ax_context.legend(loc="upper right")
    ax_context.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax_context.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax_context.set_xlabel("Date")
    plt.setp(ax_context.get_xticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "rolling_correlation_with_events.png", dpi=220, bbox_inches="tight")
    plt.close()


def plot_episode_summary(episode_table):
    if episode_table.empty:
        return

    plot_df = episode_table.sort_values("start_date", ascending=True).copy()
    labels = [
        f"Episode {row.episode_id}: {row.start_date} to {row.end_date}\n{row.matched_events}"
        for row in plot_df.itertuples()
    ]
    colors = ["#d73027" if peak >= 0 else "#4575b4" for peak in plot_df["peak_r"]]

    fig_height = max(6, 0.65 * len(plot_df))
    fig, ax = plt.subplots(figsize=(14, fig_height))
    bars = ax.barh(labels, plot_df["duration_days"], color=colors, alpha=0.85, edgecolor="black")

    for bar, row in zip(bars, plot_df.itertuples()):
        ax.text(
            bar.get_width() + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f"peak r={row.peak_r:+.3f}",
            va="center",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_xlabel("Duration (days)")
    ax.set_title("Detected Active Coupling Episodes", fontsize=14, fontweight="bold")
    ax.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "episode_summary.png", dpi=220, bbox_inches="tight")
    plt.close()


def plot_impulse_response(response):
    offsets = response["offsets"]

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.fill_between(offsets, response["lower"], response["upper"], color="lightgray", alpha=0.75, label="95% random-date band")
    ax.plot(offsets, response["mean_response"], color="#c0392b", linewidth=2.4, marker="o", label="Mean sentiment response")
    ax.axvline(0, color="black", linestyle="--", linewidth=1.0, label="Large stringency change")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Days relative to |Delta Stringency| > 5 event")
    ax.set_ylabel("Differenced sentiment_mean")
    ax.set_title("Event Study: Sentiment Impulse Response to Large Stringency Changes", fontweight="bold")
    ax.grid(alpha=0.25)
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "impulse_response.png", dpi=220, bbox_inches="tight")
    plt.close()


def plot_permutation_test(observed, p_value, null_values):
    fig, ax = plt.subplots(figsize=(10, 6))
    if len(null_values) > 0:
        ax.hist(null_values, bins=35, color="#95a5a6", edgecolor="white", alpha=0.9)
    ax.axvline(observed, color="#c0392b", linewidth=2.5, label=f"Observed mean |r| = {observed:.3f}")
    ax.set_xlabel("Mean |rolling r| from random windows")
    ax.set_ylabel("Permutation count")
    ax.set_title("Permutation Test for Active Coupling Episodes", fontweight="bold")
    ax.text(
        0.97,
        0.95,
        f"p = {p_value:.4f}" if not pd.isna(p_value) else "p = NA",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=12,
        fontweight="bold",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )
    ax.legend()
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "permutation_test.png", dpi=220, bbox_inches="tight")
    plt.close()


def print_episode_table(episode_table):
    if episode_table.empty:
        print("No active coupling episodes detected.")
        return

    display_cols = [
        "episode_id",
        "start_date",
        "end_date",
        "duration_days",
        "mean_r",
        "peak_r",
        "peak_r_date",
        "stringency_change",
        "cases_change",
        "max_cases",
        "max_deaths",
        "matched_events",
    ]
    print("\nDetected active coupling episodes:")
    print(episode_table[display_cols].to_string(index=False))


def main():
    print("=" * 78)
    print("EVENT STUDY COUPLING ANALYSIS")
    print("=" * 78)

    level_df, diff_df = load_data()
    print(f"Analysis range: {diff_df.index.min().date()} to {diff_df.index.max().date()}")
    print(f"Rolling window: {ROLLING_WINDOW} days; active threshold: |r| > {ACTIVE_THRESHOLD}")

    rolling_corr = compute_rolling_correlation(diff_df)
    episode_table = build_episode_table(rolling_corr, level_df)
    episode_table.to_csv(OUTPUT_DIR / "episode_table.csv", index=False)
    print_episode_table(episode_table)
    print(f"\nSaved episode table: {OUTPUT_DIR / 'episode_table.csv'}")

    observed, permutation_p, null_values = permutation_test(rolling_corr, episode_table)
    print("\nPermutation validation:")
    print(f"  Episodes detected: {len(episode_table)}")
    print(f"  Observed mean |r|: {observed:.4f}" if not pd.isna(observed) else "  Observed mean |r|: NA")
    print(f"  Permutation p-value: {permutation_p:.4f}" if not pd.isna(permutation_p) else "  Permutation p-value: NA")

    response = impulse_response(diff_df, level_df)
    print("\nImpulse response event study:")
    print(f"  Large |Delta Stringency| > {POLICY_SHOCK_THRESHOLD} events: {len(response['event_dates'])}")
    if len(response["event_dates"]) > 0:
        print("  Event dates:")
        print("  " + ", ".join(date.strftime("%Y-%m-%d") for date in response["event_dates"]))

    plot_rolling_correlation(rolling_corr, level_df, episode_table)
    plot_episode_summary(episode_table)
    plot_impulse_response(response)
    plot_permutation_test(observed, permutation_p, null_values)

    print("\nSaved plots:")
    for filename in [
        "rolling_correlation_with_events.png",
        "episode_summary.png",
        "impulse_response.png",
        "permutation_test.png",
    ]:
        print(f"  {OUTPUT_DIR / filename}")

    print("\nInterpretation:")
    if episode_table.empty:
        print("  No |r| > 0.30 active coupling episodes were detected.")
    elif not pd.isna(permutation_p) and permutation_p < 0.05:
        print("  Active coupling episodes are stronger than expected from random non-episode windows.")
    elif not pd.isna(permutation_p):
        print("  Active coupling episodes were detected, but they are not clearly stronger than random windows.")
    else:
        print("  Active coupling episodes were detected, but the permutation test could not be evaluated.")


if __name__ == "__main__":
    main()
