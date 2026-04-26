"""
Unsupervised Phase Discovery Test
==================================
Validates researcher-defined pandemic phases against data-driven regimes
using three complementary methods:

    1. K-means (k=5) - direct comparison with researcher's 5 phases
    2. K-means silhouette sweep (k=2..10) - does the data prefer 5?
    3. PELT change-point detection - time-series-appropriate breakpoints
    4. HDBSCAN - density-based, discovers its own cluster count

Outputs:
    - Agreement metrics (Adjusted Rand Index, NMI)
    - Change-point proximity to researcher boundaries (in days)
    - Timeline figure overlaying discovered boundaries on researcher phases
    - Silhouette curve (how many regimes does the data really want?)

Dependencies:
    pip install pandas numpy matplotlib scikit-learn scipy
    pip install ruptures hdbscan     # optional but recommended

USAGE:
    python phase_discovery.py
"""

import os
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    silhouette_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)

warnings.filterwarnings("ignore")

# Optional: PELT change-point detection
try:
    import ruptures as rpt
    RUPTURES_OK = True
except ImportError:
    RUPTURES_OK = False
    print("[info] `ruptures` not installed; skipping PELT. Install with: pip install ruptures")

# Optional: HDBSCAN
try:
    import hdbscan
    HDBSCAN_OK = True
except ImportError:
    HDBSCAN_OK = False
    print("[info] `hdbscan` not installed; skipping HDBSCAN. Install with: pip install hdbscan")


# ============================================================
# CONFIG
# ============================================================
INPUT_PATH = "/home/mohsin/ADS/ADS_pandas_new/.csv/new/full_analysis_data.csv"
OUTPUT_DIR = "/home/mohsin/ADS/ADS_pandas_new/graphs/phase_discovery"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATE_COL = "date"

# Features defining pandemic "state" on each day
FEATURES = [
    "daily_new_cases",
    "daily_new_deaths",
    "StringencyIndex_Average",
    "tweet_volume",
    "sentiment_mean",
]

SMOOTHING_WINDOW = 14  # days — smooths day-to-day noise before clustering
K_TARGET = 5           # match researcher's 5 phases
K_SWEEP = range(2, 11) # silhouette scan
RANDOM_STATE = 42

# Researcher-defined phases (what we're validating against)
RESEARCHER_PHASES = [
    ("P1: Initial Outbreak",   "2020-03-19", "2020-08-31"),
    ("P2: Second Wave",        "2020-09-01", "2021-02-28"),
    ("P3: Vaccination Era",    "2021-03-01", "2021-09-30"),
    ("P4: Omicron",            "2021-10-01", "2022-03-31"),
    ("P5: Post-Restrictions",  "2022-04-01", "2022-12-31"),
]

# The 4 internal boundaries we want to compare discovered change-points to
RESEARCHER_BOUNDARIES = [
    pd.Timestamp("2020-09-01"),
    pd.Timestamp("2021-03-01"),
    pd.Timestamp("2021-10-01"),
    pd.Timestamp("2022-04-01"),
]

# ============================================================
# PHASE-ADJUSTED XCORR CONFIG
# ============================================================
# Re-runs phase-segmented cross-correlation using PELT-validated boundaries
# and compares to the original hand-drawn phases.
#   P1→P2: 2020-09-01 → 2020-11-01  (+61 days; PELT CP at Nov 4, 2020)
#   P2→P3: 2021-03-01 unchanged     (PELT CP within 2 days)
#   P3→P4: 2021-10-01 → 2021-11-26  (+56 days; WHO Omicron announcement)
#   P4→P5: 2022-04-01 unchanged     (PELT CP within 6 days)
PHASE_ADJUSTED_DIR = "/home/mohsin/ADS/ADS_pandas_new/graphs/phase_adjusted"

TWEET_VOL_COL = "tweet_volume"
CASES_COL = "daily_new_cases"
DEATHS_COL = "daily_new_deaths"
MAX_LAG = 30

PHASES_ORIGINAL = {
    "Phase 1: Initial Outbreak\n(Mar-Aug 2020)":     ("2020-03-19", "2020-08-31"),
    "Phase 2: Second Wave\n(Sep 2020-Feb 2021)":     ("2020-09-01", "2021-02-28"),
    "Phase 3: Vaccination Era\n(Mar-Sep 2021)":      ("2021-03-01", "2021-09-30"),
    "Phase 4: Omicron\n(Oct 2021-Mar 2022)":         ("2021-10-01", "2022-03-31"),
    "Phase 5: Post-Restrictions\n(Apr-Dec 2022)":    ("2022-04-01", "2022-12-31"),
}

PHASES_ADJUSTED = {
    "Phase 1: Initial Outbreak\n(Mar-Oct 2020)":     ("2020-03-19", "2020-10-31"),
    "Phase 2: Second Wave\n(Nov 2020-Feb 2021)":     ("2020-11-01", "2021-02-28"),
    "Phase 3: Vaccination Era\n(Mar-Nov 2021)":      ("2021-03-01", "2021-11-25"),
    "Phase 4: Omicron\n(Nov 2021-Mar 2022)":         ("2021-11-26", "2022-03-31"),
    "Phase 5: Post-Restrictions\n(Apr-Dec 2022)":    ("2022-04-01", "2022-12-31"),
}


# ============================================================
# DATA PREP
# ============================================================
def load_and_prepare():
    df = pd.read_csv(INPUT_PATH, parse_dates=[DATE_COL]).sort_values(DATE_COL).reset_index(drop=True)
    print(f"Loaded: {len(df)} days, {df[DATE_COL].min().date()} to {df[DATE_COL].max().date()}")

    # Trim to researcher's full window for apples-to-apples comparison
    start = pd.Timestamp(RESEARCHER_PHASES[0][1])
    end = pd.Timestamp(RESEARCHER_PHASES[-1][2])
    df = df[(df[DATE_COL] >= start) & (df[DATE_COL] <= end)].reset_index(drop=True)
    print(f"After trimming to researcher window: {len(df)} days")

    # Smooth
    for col in FEATURES:
        df[f"{col}_smooth"] = df[col].rolling(SMOOTHING_WINDOW, center=True, min_periods=1).mean()

    feat_cols = [f"{c}_smooth" for c in FEATURES]
    df_clean = df.dropna(subset=feat_cols).reset_index(drop=True)
    print(f"After smoothing/NA drop: {len(df_clean)} days, features: {FEATURES}")

    X = StandardScaler().fit_transform(df_clean[feat_cols].values)

    # Assign researcher phase label (1..5) to each day for ARI comparison
    labels_researcher = np.full(len(df_clean), -1, dtype=int)
    for i, (_, s, e) in enumerate(RESEARCHER_PHASES):
        mask = (df_clean[DATE_COL] >= pd.Timestamp(s)) & (df_clean[DATE_COL] <= pd.Timestamp(e))
        labels_researcher[mask] = i

    return df_clean, X, labels_researcher


# ============================================================
# METHOD 1: K-MEANS, k=5 (forced)
# ============================================================
def run_kmeans_k5(X):
    km = KMeans(n_clusters=K_TARGET, random_state=RANDOM_STATE, n_init=20).fit(X)
    return km.labels_


# ============================================================
# METHOD 2: SILHOUETTE SWEEP
# ============================================================
def silhouette_sweep(X):
    scores = {}
    for k in K_SWEEP:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20).fit(X)
        scores[k] = silhouette_score(X, km.labels_)
    best_k = max(scores, key=scores.get)
    return scores, best_k


# ============================================================
# METHOD 3: PELT CHANGE-POINT DETECTION
# ============================================================
def run_pelt(X, n_bkps=4):
    """Finds n_bkps change-points using L2-cost PELT. n_bkps=4 gives 5 segments."""
    if not RUPTURES_OK:
        return None
    algo = rpt.Pelt(model="l2", min_size=30).fit(X)
    # Use penalty-based: pen controls how many breakpoints are found
    # We do both: n_bkps-forced and penalty-based (data-driven count)
    try:
        forced = rpt.Dynp(model="l2", min_size=30).fit(X).predict(n_bkps=n_bkps)
    except Exception:
        forced = None

    # Data-driven: let penalty decide the count
    # Penalty heuristic: c * log(n) * d  — common BIC-like choice
    n, d = X.shape
    pen = np.log(n) * d
    data_driven = algo.predict(pen=pen)

    return {"forced": forced, "data_driven": data_driven}


# ============================================================
# METHOD 4: HDBSCAN
# ============================================================
def run_hdbscan(X):
    if not HDBSCAN_OK:
        return None
    clusterer = hdbscan.HDBSCAN(min_cluster_size=30, min_samples=10)
    return clusterer.fit_predict(X)


# ============================================================
# EVALUATION
# ============================================================
def evaluate_labels(labels_discovered, labels_researcher, method_name):
    # Filter noise (-1) from HDBSCAN before scoring
    mask = labels_discovered >= 0
    if mask.sum() < 10:
        print(f"  {method_name}: too few labeled points")
        return
    ari = adjusted_rand_score(labels_researcher[mask], labels_discovered[mask])
    nmi = normalized_mutual_info_score(labels_researcher[mask], labels_discovered[mask])
    n_clusters = len(set(labels_discovered[mask]))
    print(f"  {method_name:<30s} ARI={ari:+.3f}  NMI={nmi:.3f}  n_clusters={n_clusters}")
    return ari, nmi


def boundary_proximity(discovered_breaks_idx, dates):
    """For each researcher boundary, find distance in days to nearest discovered breakpoint."""
    discovered_dates = [dates.iloc[i] for i in discovered_breaks_idx if i < len(dates)]
    distances = []
    for b in RESEARCHER_BOUNDARIES:
        if not discovered_dates:
            distances.append(None)
            continue
        closest = min(discovered_dates, key=lambda d: abs((d - b).days))
        distances.append(abs((closest - b).days))
    return distances


# ============================================================
# PLOTS
# ============================================================
def plot_timeline(df, labels_k5, pelt_result, labels_hdbscan, output_dir):
    fig, axes = plt.subplots(4, 1, figsize=(15, 10), sharex=True)

    phase_colors = ["#ffcccc", "#ffe0cc", "#ffffcc", "#ccffcc", "#cce0ff"]

    # Panel 1: researcher phases
    ax = axes[0]
    for (name, s, e), c in zip(RESEARCHER_PHASES, phase_colors):
        ax.axvspan(pd.Timestamp(s), pd.Timestamp(e), color=c, alpha=0.6)
        ax.text(pd.Timestamp(s) + (pd.Timestamp(e) - pd.Timestamp(s)) / 2, 0.5,
                name.split(":")[0], ha="center", va="center", fontsize=9, fontweight="bold")
    ax.set_title("Researcher-defined phases (hand-drawn)", fontsize=11, loc="left")
    ax.set_yticks([]); ax.set_ylim(0, 1)

    # Panel 2: K-means k=5
    ax = axes[1]
    # Recolor labels so adjacent cluster IDs get distinct colors
    cmap = plt.cm.tab10
    for i in range(len(df) - 1):
        ax.axvspan(df[DATE_COL].iloc[i], df[DATE_COL].iloc[i + 1],
                   color=cmap(labels_k5[i] % 10), alpha=0.6)
    ax.set_title(f"K-means (k={K_TARGET}) — does it carve the same blocks?", fontsize=11, loc="left")
    ax.set_yticks([]); ax.set_ylim(0, 1)

    # Panel 3: PELT
    ax = axes[2]
    if pelt_result is not None:
        for bk in pelt_result["data_driven"]:
            if bk < len(df):
                ax.axvline(df[DATE_COL].iloc[bk - 1], color="darkred", linewidth=2)
        # also show forced 4-breakpoint solution in lighter color
        if pelt_result["forced"] is not None:
            for bk in pelt_result["forced"]:
                if bk < len(df):
                    ax.axvline(df[DATE_COL].iloc[bk - 1], color="navy", linewidth=1, linestyle="--", alpha=0.7)
        ax.set_title("PELT change-points  (red = data-driven count, dashed navy = forced 4 breaks)",
                     fontsize=11, loc="left")
    else:
        ax.text(0.5, 0.5, "PELT not available (install `ruptures`)", ha="center", va="center", transform=ax.transAxes)
    # Mark researcher boundaries for visual comparison
    for b in RESEARCHER_BOUNDARIES:
        ax.axvline(b, color="black", linestyle=":", alpha=0.5)
    ax.set_yticks([]); ax.set_ylim(0, 1)

    # Panel 4: HDBSCAN
    ax = axes[3]
    if labels_hdbscan is not None:
        for i in range(len(df) - 1):
            lab = labels_hdbscan[i]
            color = "lightgray" if lab == -1 else cmap(lab % 10)
            ax.axvspan(df[DATE_COL].iloc[i], df[DATE_COL].iloc[i + 1], color=color, alpha=0.6)
        n_found = len(set(labels_hdbscan)) - (1 if -1 in labels_hdbscan else 0)
        ax.set_title(f"HDBSCAN (found {n_found} clusters, grey = noise)", fontsize=11, loc="left")
    else:
        ax.text(0.5, 0.5, "HDBSCAN not available (install `hdbscan`)", ha="center", va="center", transform=ax.transAxes)
    ax.set_yticks([]); ax.set_ylim(0, 1)

    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(axes[-1].get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    path = os.path.join(output_dir, "phase_discovery_timeline.png")
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_silhouette(scores, best_k, output_dir):
    fig, ax = plt.subplots(figsize=(8, 4))
    ks = list(scores.keys())
    vals = [scores[k] for k in ks]
    bars = ax.bar(ks, vals, color=["firebrick" if k == best_k else "steelblue" for k in ks],
                  edgecolor="black", alpha=0.85)
    ax.axvline(K_TARGET, color="black", linestyle="--", alpha=0.7, label=f"Researcher k={K_TARGET}")
    ax.set_xlabel("k (number of clusters)")
    ax.set_ylabel("Silhouette score")
    ax.set_title(f"Silhouette sweep — data prefers k={best_k} (researcher chose k={K_TARGET})")
    ax.legend()
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.002, f"{v:.3f}",
                ha="center", fontsize=8)
    plt.tight_layout()
    path = os.path.join(output_dir, "silhouette_sweep.png")
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ============================================================
# PHASE-ADJUSTED XCORR: HELPERS
# ============================================================
def compute_xcorr(x, y, max_lag=30):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    if len(x) < 20:
        return np.arange(-max_lag, max_lag + 1), np.full(2 * max_lag + 1, np.nan)
    x = (x - x.mean()) / x.std()
    y = (y - y.mean()) / y.std()
    n = len(x)
    lags = np.arange(-max_lag, max_lag + 1)
    corrs = np.zeros(len(lags))
    for i, lag in enumerate(lags):
        if lag >= 0:
            corrs[i] = np.corrcoef(x[:n - lag], y[lag:])[0, 1] if n - lag >= 10 else np.nan
        else:
            corrs[i] = np.corrcoef(x[-lag:], y[:n + lag])[0, 1] if n + lag >= 10 else np.nan
    return lags, corrs


def find_peak(lags, corrs):
    valid = ~np.isnan(corrs)
    if not valid.any():
        return 0, 0.0
    idx = np.nanargmax(np.abs(corrs))
    return lags[idx], corrs[idx]


def run_phase_xcorr(df, phases):
    """Per-phase xcorr of differenced 7d-smoothed tweet volume vs cases/deaths."""
    targets = [(f"{CASES_COL}_7d", "Cases"), (f"{DEATHS_COL}_7d", "Deaths")]
    phase_results = {}
    for phase_name, (start, end) in phases.items():
        mask = (df[DATE_COL] >= start) & (df[DATE_COL] <= end)
        phase_df = df[mask].copy()
        n = len(phase_df)
        if n < 30:
            print(f"  {phase_name.replace(chr(10),' ')}: only {n} rows, skipping")
            continue
        print(f"  {phase_name.replace(chr(10),' ')} ({start} → {end}): n={n}")
        phase_results[phase_name] = {"n": n, "start": start, "end": end}
        for col, label in targets:
            tv_diff = phase_df[f"{TWEET_VOL_COL}_7d"].diff().dropna().values
            var_diff = phase_df[col].diff().dropna().values
            m = min(len(tv_diff), len(var_diff))
            tv_diff, var_diff = tv_diff[:m], var_diff[:m]
            lags, corrs = compute_xcorr(tv_diff, var_diff, min(MAX_LAG, n // 3))
            peak_lag, peak_corr = find_peak(lags, corrs)
            phase_results[phase_name][label] = {
                "lags": lags, "corrs": corrs,
                "peak_lag": peak_lag, "peak_corr": peak_corr,
            }
            print(f"    vs {label}: peak lag={peak_lag:+d}, r={peak_corr:+.4f}")
    return phase_results


def plot_phase_xcorr(phase_results, title, filename, output_dir):
    n_phases = len(phase_results)
    fig, axes = plt.subplots(n_phases, 2, figsize=(14, 4 * n_phases))
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)
    for i, (phase_name, pdata) in enumerate(phase_results.items()):
        for j, (label, color) in enumerate([("Cases", "steelblue"), ("Deaths", "firebrick")]):
            ax = axes[i, j] if n_phases > 1 else axes[j]
            if label not in pdata:
                continue
            r = pdata[label]
            ax.bar(r["lags"], r["corrs"], color=color, alpha=0.7, width=0.8)
            sig = 1.96 / np.sqrt(pdata["n"])
            ax.axhline(sig, color="gray", linestyle=":", alpha=0.5)
            ax.axhline(-sig, color="gray", linestyle=":", alpha=0.5)
            ax.axhline(0, color="black", linewidth=0.5)
            ax.axvline(0, color="black", linewidth=0.5, alpha=0.3)
            ax.plot(r["peak_lag"], r["peak_corr"], "v", color="black", markersize=10)
            ax.set_title(f"{phase_name}\nvs {label} (peak: lag={r['peak_lag']}, r={r['peak_corr']:.3f})",
                         fontsize=9)
            ax.set_xlim(-MAX_LAG - 1, MAX_LAG + 1)
            if i == n_phases - 1:
                ax.set_xlabel("Lag (days)")
            if j == 0:
                ax.set_ylabel("Correlation")
    plt.tight_layout()
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_summary_heatmap(phase_results, title, filename, output_dir):
    phases_list = list(phase_results.keys())
    targets_list = ["Cases", "Deaths"]
    peak_lag_matrix = np.zeros((len(phases_list), len(targets_list)))
    peak_corr_matrix = np.zeros((len(phases_list), len(targets_list)))
    for i, phase in enumerate(phases_list):
        for j, target in enumerate(targets_list):
            if target in phase_results[phase]:
                peak_lag_matrix[i, j] = phase_results[phase][target]["peak_lag"]
                peak_corr_matrix[i, j] = phase_results[phase][target]["peak_corr"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    ax = axes[0]
    im = ax.imshow(peak_lag_matrix, cmap="RdBu_r", aspect="auto", vmin=-MAX_LAG, vmax=MAX_LAG)
    ax.set_xticks(range(len(targets_list))); ax.set_xticklabels(targets_list)
    ax.set_yticks(range(len(phases_list)))
    ax.set_yticklabels([p.replace("\n", " ") for p in phases_list], fontsize=8)
    ax.set_title("Peak Lag (days)\nBlue=tweet leads, Red=tweet lags")
    for i in range(len(phases_list)):
        for j in range(len(targets_list)):
            ax.text(j, i, f"{peak_lag_matrix[i,j]:.0f}",
                    ha="center", va="center", fontsize=11, fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.8)

    ax = axes[1]
    im = ax.imshow(peak_corr_matrix, cmap="RdYlGn", aspect="auto", vmin=-0.5, vmax=0.5)
    ax.set_xticks(range(len(targets_list))); ax.set_xticklabels(targets_list)
    ax.set_yticks(range(len(phases_list)))
    ax.set_yticklabels([p.replace("\n", " ") for p in phases_list], fontsize=8)
    ax.set_title("Peak Correlation (r)")
    for i in range(len(phases_list)):
        for j in range(len(targets_list)):
            ax.text(j, i, f"{peak_corr_matrix[i,j]:.3f}",
                    ha="center", va="center", fontsize=11, fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_comparison(orig, adj, filename, output_dir):
    o_phases = list(orig.keys())
    a_phases = list(adj.keys())
    targets_list = ["Cases", "Deaths"]

    def to_matrix(results, phase_list):
        corr = np.zeros((len(phase_list), len(targets_list)))
        for i, p in enumerate(phase_list):
            for j, t in enumerate(targets_list):
                if t in results[p]:
                    corr[i, j] = results[p][t]["peak_corr"]
        return corr

    o_corr = to_matrix(orig, o_phases)
    a_corr = to_matrix(adj, a_phases)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle("Phase Peak-Correlation: Original vs Data-Validated Boundaries",
                 fontsize=14, fontweight="bold")
    for ax, mat, phase_list, subtitle in [
        (axes[0], o_corr, o_phases, "Original (hand-drawn)"),
        (axes[1], a_corr, a_phases, "Adjusted (PELT-validated)"),
    ]:
        im = ax.imshow(mat, cmap="RdYlGn", aspect="auto", vmin=-0.5, vmax=0.5)
        ax.set_xticks(range(len(targets_list))); ax.set_xticklabels(targets_list)
        ax.set_yticks(range(len(phase_list)))
        ax.set_yticklabels([p.replace("\n", " ") for p in phase_list], fontsize=8)
        ax.set_title(subtitle)
        for i in range(len(phase_list)):
            for j in range(len(targets_list)):
                ax.text(j, i, f"{mat[i,j]:.3f}",
                        ha="center", va="center", fontsize=11, fontweight="bold")
        plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def save_comparison_table(orig, adj, filename, output_dir):
    rows = []
    for i, (o_name, a_name) in enumerate(zip(orig.keys(), adj.keys())):
        for target in ["Cases", "Deaths"]:
            if target in orig[o_name] and target in adj[a_name]:
                o = orig[o_name][target]
                a = adj[a_name][target]
                rows.append({
                    "phase_num": i + 1,
                    "target": target,
                    "original_phase": o_name.replace("\n", " "),
                    "original_lag": o["peak_lag"],
                    "original_r": round(o["peak_corr"], 4),
                    "adjusted_phase": a_name.replace("\n", " "),
                    "adjusted_lag": a["peak_lag"],
                    "adjusted_r": round(a["peak_corr"], 4),
                    "delta_r": round(a["peak_corr"] - o["peak_corr"], 4),
                })
    df_out = pd.DataFrame(rows)
    path = os.path.join(output_dir, filename)
    df_out.to_csv(path, index=False)
    print(f"  Saved: {path}")
    print("\nNumeric comparison:")
    print(df_out.to_string(index=False))


def run_adjusted_phase_xcorr():
    """Stage 5: re-run phase-segmented xcorr with PELT-validated boundaries."""
    os.makedirs(PHASE_ADJUSTED_DIR, exist_ok=True)
    df = pd.read_csv(INPUT_PATH)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)
    for col in [TWEET_VOL_COL, CASES_COL, DEATHS_COL]:
        df[f"{col}_7d"] = df[col].rolling(7, center=True).mean()
    print(f"\nLoaded: {len(df)} rows")

    print("\n─ RUN 1: ORIGINAL PHASES ─")
    results_orig = run_phase_xcorr(df, PHASES_ORIGINAL)

    print("\n─ RUN 2: ADJUSTED PHASES ─")
    results_adj = run_phase_xcorr(df, PHASES_ADJUSTED)

    print("\n─ FIGURES ─")
    plot_phase_xcorr(results_adj,
                     "Phase-Segmented Cross-Correlation (Adjusted Boundaries)",
                     "06_xcorr_by_phase_ADJUSTED.png", PHASE_ADJUSTED_DIR)
    plot_summary_heatmap(results_adj,
                         "Phase-Segmented Summary (Adjusted Boundaries)",
                         "07_phase_summary_heatmap_ADJUSTED.png", PHASE_ADJUSTED_DIR)
    plot_comparison(results_orig, results_adj,
                    "08_heatmap_comparison.png", PHASE_ADJUSTED_DIR)
    save_comparison_table(results_orig, results_adj,
                          "phase_comparison_table.csv", PHASE_ADJUSTED_DIR)


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 70)
    print("UNSUPERVISED PHASE DISCOVERY")
    print("=" * 70)

    df, X, labels_researcher = load_and_prepare()

    print("\n--- Method 1: K-means, k=5 ---")
    labels_k5 = run_kmeans_k5(X)

    print("\n--- Method 2: Silhouette sweep k=2..10 ---")
    sil_scores, best_k = silhouette_sweep(X)
    for k, s in sil_scores.items():
        marker = "  <-- best" if k == best_k else ("  (researcher)" if k == K_TARGET else "")
        print(f"  k={k:2d}: silhouette = {s:+.4f}{marker}")

    print("\n--- Method 3: PELT change-point detection ---")
    pelt_result = run_pelt(X, n_bkps=4)
    if pelt_result is not None:
        print(f"  Forced 4 breakpoints (for 5 segments): indices = {pelt_result['forced']}")
        print(f"  Data-driven (BIC-style penalty):       indices = {pelt_result['data_driven']}")

    print("\n--- Method 4: HDBSCAN ---")
    labels_hdbscan = run_hdbscan(X)

    # =====================================================
    # EVALUATION
    # =====================================================
    print("\n" + "=" * 70)
    print("AGREEMENT WITH RESEARCHER PHASES (ARI, NMI)")
    print("=" * 70)
    print("  ARI ranges from -1 (disagree) to 0 (chance) to 1 (identical)")
    print("  NMI ranges from 0 (independent) to 1 (identical)\n")

    evaluate_labels(labels_k5, labels_researcher, "K-means k=5")
    if labels_hdbscan is not None:
        evaluate_labels(labels_hdbscan, labels_researcher, "HDBSCAN")

    if pelt_result is not None:
        print("\n" + "=" * 70)
        print("BOUNDARY PROXIMITY (days to nearest discovered change-point)")
        print("=" * 70)
        forced_dists = boundary_proximity(pelt_result["forced"][:-1], df[DATE_COL])  # drop last (end of series)
        dd_dists = boundary_proximity([i - 1 for i in pelt_result["data_driven"][:-1]], df[DATE_COL])
        print(f"  Researcher boundaries: {[b.strftime('%Y-%m-%d') for b in RESEARCHER_BOUNDARIES]}")
        print(f"  PELT (forced 4):  nearest-CP distance in days = {forced_dists}")
        print(f"  PELT (data-drv):  nearest-CP distance in days = {dd_dists}")
        print("  Interpretation: small numbers (<30 days) = researcher boundary aligns with real regime shift")

    # =====================================================
    # PLOTS
    # =====================================================
    print("\n" + "=" * 70)
    print("SAVING FIGURES")
    print("=" * 70)
    plot_timeline(df, labels_k5, pelt_result, labels_hdbscan, OUTPUT_DIR)
    plot_silhouette(sil_scores, best_k, OUTPUT_DIR)

    # =====================================================
    # STAGE 5: XCORR WITH PELT-VALIDATED PHASE BOUNDARIES
    # =====================================================
    print("\n" + "=" * 70)
    print("STAGE 5: PHASE-SEGMENTED XCORR (ORIGINAL vs ADJUSTED)")
    print("=" * 70)
    run_adjusted_phase_xcorr()

    print("\nDone.")


if __name__ == "__main__":
    main()
