"""
Phase-Segmented Random Forest: Non-Linearity Robustness Check
======================================================================
Re-runs the IV.A phase-segmented analysis using Random Forest instead
of OLS. The goal is to test whether OLS's three-phase R² pattern is a
genuine data feature or an artifact of the linearity assumption.

KEY QUESTIONS:
1. If RF's R² in Phase 1-2 remains near zero, the "decoupled discussion"
   finding is robust to non-linearity.
2. If RF's R² in Phase 3-4 closely matches OLS, the linear assumption
   captures most of the real signal.
3. Feature importance from RF provides a non-parametric ranking of
   cases vs deaths importance.

DESIGN CHOICES:
- TimeSeriesSplit (not random K-fold) — preserves temporal ordering
- 5 splits per phase
- n_estimators=200, max_depth=5 (conservative to prevent overfitting on small samples)
- Comparison metrics: R² (out-of-fold mean), feature importance

USAGE:
    python tweets_rf_phased.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score
import os
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# PATHS
# ============================================================
INPUT_PATH = "/home/qqq/ADS_pandas_new/.csv/new/full_analysis_data.csv"
OUTPUT_DIR = "/home/qqq/ADS_pandas_new/graphs/modelling"

# ============================================================
# CONFIG
# ============================================================
DATE_COL = "date"
TWEET_VOL_COL = "tweet_volume"
CASES_COL = "daily_new_cases"
DEATHS_COL = "daily_new_deaths"
STRINGENCY_COL = "StringencyIndex_Average"

PHASES = [
    ("Phase 1: Initial Outbreak", "2020-03-19", "2020-08-31"),
    ("Phase 2: Second Wave", "2020-09-01", "2021-02-28"),
    ("Phase 3: Vaccination Era", "2021-03-01", "2021-09-30"),
    ("Phase 4: Omicron", "2021-10-01", "2022-03-31"),
    ("Phase 5: Post-Restrictions", "2022-04-01", "2022-12-31"),
]

# RF hyperparameters — conservative to prevent overfitting on small phase samples
RF_PARAMS = {
    "n_estimators": 200,
    "max_depth": 5,
    "min_samples_split": 10,
    "min_samples_leaf": 5,
    "random_state": 42,
    "n_jobs": -1,
}

N_SPLITS = 5  # TimeSeriesSplit folds per phase

# OLS R² values from tweets_ols.py (for comparison)
OLS_R2_REFERENCE = {
    "Phase 1: Initial Outbreak": 0.0091,
    "Phase 2: Second Wave": 0.0001,
    "Phase 3: Vaccination Era": 0.1577,
    "Phase 4: Omicron": 0.2165,
    "Phase 5: Post-Restrictions": 0.0622,
}


# ============================================================
# DATA PREP
# ============================================================
def load_and_difference(filepath):
    """Load and first-difference the data."""
    print(f"\n{'='*70}")
    print("LOADING AND PREPARING DATA")
    print(f"{'='*70}")

    df = pd.read_csv(filepath)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)
    print(f"Loaded: {len(df)} rows, {df[DATE_COL].min().date()} to {df[DATE_COL].max().date()}")

    # First difference (same as OLS pipeline)
    cols = [TWEET_VOL_COL, CASES_COL, DEATHS_COL, STRINGENCY_COL]
    for col in cols:
        df[f"d_{col}"] = df[col].diff()

    df = df.dropna(subset=[f"d_{c}" for c in cols]).reset_index(drop=True)
    print(f"After differencing: {len(df)} rows")
    return df


# ============================================================
# RUN RF FOR ONE PHASE
# ============================================================
def run_rf_for_phase(df, phase_name, start, end):
    """Fit Random Forest within a single phase using TimeSeriesSplit CV."""
    mask = (df[DATE_COL] >= start) & (df[DATE_COL] <= end)
    phase_df = df[mask].copy().reset_index(drop=True)
    n = len(phase_df)

    if n < 50:
        print(f"  {phase_name}: only {n} rows, skipping")
        return None

    X = phase_df[[f"d_{CASES_COL}", f"d_{DEATHS_COL}"]].values
    y = phase_df[f"d_{TWEET_VOL_COL}"].values

    # TimeSeriesSplit
    n_splits = min(N_SPLITS, max(2, n // 30))
    tscv = TimeSeriesSplit(n_splits=n_splits)

    cv_r2 = []
    cv_train_r2 = []  # to detect overfitting
    feature_importances = []

    for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        rf = RandomForestRegressor(**RF_PARAMS)
        rf.fit(X_train, y_train)

        # Out-of-fold R²
        y_pred_test = rf.predict(X_test)
        y_pred_train = rf.predict(X_train)
        r2_test = r2_score(y_test, y_pred_test)
        r2_train = r2_score(y_train, y_pred_train)

        cv_r2.append(r2_test)
        cv_train_r2.append(r2_train)
        feature_importances.append(rf.feature_importances_)

    # Aggregate
    mean_r2_test = np.mean(cv_r2)
    std_r2_test = np.std(cv_r2)
    mean_r2_train = np.mean(cv_train_r2)
    mean_importances = np.mean(feature_importances, axis=0)

    # Also fit on full phase for in-sample comparison (matches OLS)
    rf_full = RandomForestRegressor(**RF_PARAMS)
    rf_full.fit(X, y)
    y_pred_full = rf_full.predict(X)
    r2_in_sample = r2_score(y, y_pred_full)

    return {
        "phase": phase_name,
        "n": n,
        "n_splits": n_splits,
        "r2_in_sample": r2_in_sample,
        "r2_cv_mean": mean_r2_test,
        "r2_cv_std": std_r2_test,
        "r2_train_mean": mean_r2_train,
        "ols_r2": OLS_R2_REFERENCE.get(phase_name, np.nan),
        "imp_cases": mean_importances[0],
        "imp_deaths": mean_importances[1],
        "cv_r2_per_fold": cv_r2,
    }


# ============================================================
# PRINT RESULTS
# ============================================================
def print_results(all_results):
    print(f"\n{'='*70}")
    print("PHASE-SEGMENTED RANDOM FOREST RESULTS")
    print(f"{'='*70}")

    for r in all_results:
        if r is None:
            continue
        print(f"\n{'─'*60}")
        print(f"{r['phase']}, n={r['n']}, CV folds={r['n_splits']}")
        print(f"{'─'*60}")
        print(f"  In-sample R²:        {r['r2_in_sample']:.4f}")
        print(f"  Train R² (CV mean):  {r['r2_train_mean']:.4f}")
        print(f"  Test R² (CV mean):   {r['r2_cv_mean']:.4f}  (± {r['r2_cv_std']:.4f})")
        print(f"  Per-fold test R²:    {[f'{x:.3f}' for x in r['cv_r2_per_fold']]}")
        print(f"  OLS R² (reference):  {r['ols_r2']:.4f}")
        diff = r['r2_in_sample'] - r['ols_r2']
        print(f"  RF − OLS (in-sample): {diff:+.4f}  "
              f"({'RF higher' if diff > 0.02 else 'comparable' if abs(diff) <= 0.02 else 'RF lower'})")
        print(f"  Feature importance:")
        print(f"    Δcases:  {r['imp_cases']:.3f}")
        print(f"    Δdeaths: {r['imp_deaths']:.3f}")


# ============================================================
# PLOTS
# ============================================================
def plot_rf_vs_ols(all_results, output_dir):
    """Side-by-side bar chart: RF vs OLS R² across phases."""
    valid = [r for r in all_results if r is not None]
    phases = [r["phase"].replace("Phase ", "P") for r in valid]
    x = np.arange(len(phases))
    width = 0.35

    rf_in = [r["r2_in_sample"] for r in valid]
    rf_cv = [max(0, r["r2_cv_mean"]) for r in valid]  # clip negative R² to 0 for visualization
    ols = [r["ols_r2"] for r in valid]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle("Random Forest vs OLS — R² Comparison Across Phases\n"
                 "(Tests whether OLS three-phase pattern is robust to non-linearity)",
                 fontsize=13, fontweight="bold")

    # Left: in-sample comparison (apples-to-apples with OLS)
    ax = axes[0]
    ax.bar(x - width/2, ols, width, label="OLS R²",
           color="steelblue", alpha=0.85, edgecolor="black", linewidth=0.5)
    ax.bar(x + width/2, rf_in, width, label="RF R² (in-sample)",
           color="firebrick", alpha=0.85, edgecolor="black", linewidth=0.5)
    for i, (o, rf) in enumerate(zip(ols, rf_in)):
        diff = rf - o
        ax.text(i + width/2, rf + 0.01, f"{diff:+.3f}", ha="center", fontsize=9, color="firebrick")
    ax.set_xticks(x)
    ax.set_xticklabels(phases, rotation=20, ha="right")
    ax.set_ylabel("R²")
    ax.set_title("In-sample R² (Direct OLS Comparison)")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")

    # Right: out-of-fold CV R² (more honest measure of generalization)
    ax = axes[1]
    ax.bar(x - width/2, ols, width, label="OLS R²",
           color="steelblue", alpha=0.85, edgecolor="black", linewidth=0.5)
    ax.bar(x + width/2, rf_cv, width, label="RF R² (CV out-of-fold)",
           color="darkorange", alpha=0.85, edgecolor="black", linewidth=0.5)
    # Error bars for CV
    cv_stds = [r["r2_cv_std"] for r in valid]
    ax.errorbar(x + width/2, rf_cv, yerr=cv_stds, fmt="none",
                color="black", capsize=4, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(phases, rotation=20, ha="right")
    ax.set_ylabel("R²")
    ax.set_title("Out-of-fold R² (Generalization Estimate)")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    path = os.path.join(output_dir, "rf_vs_ols_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {path}")
    plt.close()


def plot_feature_importance(all_results, output_dir):
    """Stacked bar chart of feature importance across phases."""
    valid = [r for r in all_results if r is not None]
    phases = [r["phase"].replace("Phase ", "P") for r in valid]
    x = np.arange(len(phases))
    width = 0.35

    imp_cases = [r["imp_cases"] for r in valid]
    imp_deaths = [r["imp_deaths"] for r in valid]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width/2, imp_cases, width, label="Δcases",
           color="steelblue", alpha=0.85, edgecolor="black", linewidth=0.5)
    ax.bar(x + width/2, imp_deaths, width, label="Δdeaths",
           color="firebrick", alpha=0.85, edgecolor="black", linewidth=0.5)
    for i, (c, d) in enumerate(zip(imp_cases, imp_deaths)):
        ax.text(i - width/2, c + 0.01, f"{c:.2f}", ha="center", fontsize=9)
        ax.text(i + width/2, d + 0.01, f"{d:.2f}", ha="center", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(phases, rotation=20, ha="right")
    ax.set_ylabel("Random Forest Feature Importance")
    ax.set_title("Feature Importance Across Phases\n"
                 "(Non-parametric estimate of cases vs deaths predictive power)",
                 fontsize=13, fontweight="bold")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    path = os.path.join(output_dir, "rf_feature_importance.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()


# ============================================================
# INTERPRETATION
# ============================================================
def interpret(all_results):
    print(f"\n{'='*70}")
    print("INTERPRETATION")
    print(f"{'='*70}")

    valid = [r for r in all_results if r is not None]

    # Compare RF in-sample vs OLS
    print("\n--- Q1: Is the three-phase pattern robust to non-linearity? ---")
    consistent = True
    for r in valid:
        rf_r2 = r["r2_in_sample"]
        ols_r2 = r["ols_r2"]
        diff = rf_r2 - ols_r2
        # Check: does RF dramatically change the conclusion in any phase?
        if abs(diff) > 0.10:
            consistent = False
            print(f"  ⚠ {r['phase']}: RF R² ({rf_r2:.3f}) vs OLS R² ({ols_r2:.3f}) "
                  f"differ by {diff:+.3f} — substantial difference")
        else:
            print(f"  ✓ {r['phase']}: RF R² = {rf_r2:.3f}, OLS R² = {ols_r2:.3f} "
                  f"(diff {diff:+.3f}, comparable)")
    if consistent:
        print("\n  → All phases show comparable R² between RF and OLS.")
        print("    The three-phase structure is robust to non-linearity.")
        print("    OLS's linearity assumption is not driving the IV.A findings.")

    # Phase 1-2 specifically: are they truly low-signal?
    print("\n--- Q2: Are Phase 1-2 truly low-signal? ---")
    p12 = [r for r in valid if "Phase 1" in r["phase"] or "Phase 2" in r["phase"]]
    if all(r["r2_in_sample"] < 0.1 for r in p12):
        print("  ✓ Phases 1-2 RF R² remain below 0.1 — confirms 'decoupled discussion' finding.")
        print("    Even with non-linear flexibility, severity does not predict tweet volume in these phases.")
    else:
        print("  ⚠ Some Phase 1-2 RF R² exceeds 0.1 — non-linear signal may exist.")

    # CV vs in-sample (overfitting check)
    print("\n--- Overfitting Diagnostic ---")
    for r in valid:
        gap = r["r2_train_mean"] - r["r2_cv_mean"]
        if gap > 0.20:
            print(f"  ⚠ {r['phase']}: Train R²={r['r2_train_mean']:.3f}, "
                  f"CV R²={r['r2_cv_mean']:.3f} — large gap suggests overfitting")
        else:
            print(f"  ✓ {r['phase']}: Train-CV gap = {gap:+.3f} (acceptable)")

    # Feature importance pattern
    print("\n--- Feature Importance Pattern ---")
    for r in valid:
        c, d = r["imp_cases"], r["imp_deaths"]
        dominant = "deaths" if d > c else "cases"
        ratio = max(c, d) / max(min(c, d), 0.01)
        print(f"  {r['phase']}: cases={c:.2f}, deaths={d:.2f} → {dominant} dominant "
              f"({ratio:.1f}x)")


# ============================================================
# MAIN
# ============================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 70)
    print("PHASE-SEGMENTED RANDOM FOREST ANALYSIS")
    print("=" * 70)

    df = load_and_difference(INPUT_PATH)

    all_results = []
    for phase_name, start, end in PHASES:
        print(f"\n{'─'*50}")
        print(f"Fitting RF: {phase_name}")
        result = run_rf_for_phase(df, phase_name, start, end)
        if result:
            all_results.append(result)

    # Print
    print_results(all_results)

    # Save summary CSV
    rows = []
    for r in all_results:
        if r is None:
            continue
        rows.append({
            "phase": r["phase"],
            "n": r["n"],
            "ols_r2": r["ols_r2"],
            "rf_r2_in_sample": r["r2_in_sample"],
            "rf_r2_cv_mean": r["r2_cv_mean"],
            "rf_r2_cv_std": r["r2_cv_std"],
            "rf_r2_train_mean": r["r2_train_mean"],
            "imp_cases": r["imp_cases"],
            "imp_deaths": r["imp_deaths"],
        })
    summary_df = pd.DataFrame(rows)
    csv_path = os.path.join(OUTPUT_DIR, "rf_phased_summary.csv")
    summary_df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    # Plots
    print("\nGenerating plots...")
    plot_rf_vs_ols(all_results, OUTPUT_DIR)
    plot_feature_importance(all_results, OUTPUT_DIR)

    # Interpretation
    interpret(all_results)

    print(f"\n{'='*70}")
    print(f"OUTPUTS SAVED TO: {OUTPUT_DIR}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()