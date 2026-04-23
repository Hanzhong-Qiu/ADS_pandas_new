"""
Phase-Segmented Regression: Quantifying Pandemic Fatigue
=========================================================
For each pandemic phase, regress Δtweet_volume on Δcases and Δdeaths
to measure how much public discussion responds to pandemic changes,
and whether this response diminishes over time (fatigue).

USAGE:
    python phase_regression.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
import statsmodels.api as sm
import os
import sys
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# PATHS
# ============================================================
INPUT_PATH = "/home/qqq/ADS_pandas_new/.csv/new/full_analysis_data.csv"
OUTPUT_DIR = "/home/qqq/ADS_pandas_new/graphs/exploration"

# ============================================================
# COLUMN CONFIG
# ============================================================
DATE_COL = "date"
TWEET_VOL_COL = "tweet_volume"
CASES_COL = "daily_new_cases"
DEATHS_COL = "daily_new_deaths"
STRINGENCY_COL = "StringencyIndex_Average"

# Pandemic phases
PHASES = [
    ("Phase 1: Initial Outbreak", "2020-03-19", "2020-08-31"),
    ("Phase 2: Second Wave", "2020-09-01", "2021-02-28"),
    ("Phase 3: Vaccination Era", "2021-03-01", "2021-09-30"),
    ("Phase 4: Omicron", "2021-10-01", "2022-03-31"),
    ("Phase 5: Post-Restrictions", "2022-04-01", "2022-12-31"),
]


def load_and_difference(filepath):
    """Load raw data, compute first differences."""
    print(f"Loading: {filepath}")
    df = pd.read_csv(filepath)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)
    print(f"Loaded: {len(df)} rows, {df[DATE_COL].min().date()} to {df[DATE_COL].max().date()}")

    # First difference
    diff_cols = [TWEET_VOL_COL, CASES_COL, DEATHS_COL]
    if STRINGENCY_COL in df.columns:
        diff_cols.append(STRINGENCY_COL)

    for col in diff_cols:
        df[f"d_{col}"] = df[col].diff()

    df = df.dropna(subset=[f"d_{col}" for col in diff_cols]).reset_index(drop=True)
    print(f"After differencing: {len(df)} rows")

    return df


def run_regression_for_phase(df, phase_name, start, end):
    """
    Run OLS regression within a single phase.
    Base model:     Δtweet_volume ~ Δcases + Δdeaths
    Extended model:  Δtweet_volume ~ Δcases + Δdeaths + ΔStringency
    """
    mask = (df[DATE_COL] >= start) & (df[DATE_COL] <= end)
    phase_df = df[mask].copy()
    n = len(phase_df)

    if n < 20:
        print(f"  {phase_name}: only {n} rows, skipping")
        return None

    y = phase_df[f"d_{TWEET_VOL_COL}"].values

    # ---- Base model: Δcases + Δdeaths ----
    X_base = phase_df[[f"d_{CASES_COL}", f"d_{DEATHS_COL}"]].values
    X_base = sm.add_constant(X_base)

    try:
        model_base = sm.OLS(y, X_base).fit()
    except Exception as e:
        print(f"  {phase_name}: regression failed — {e}")
        return None

    result = {
        "phase": phase_name,
        "start": start,
        "end": end,
        "n": n,
        # Base model results
        "base_r2": model_base.rsquared,
        "base_adj_r2": model_base.rsquared_adj,
        "base_f_pval": model_base.f_pvalue,
        # Cases coefficient
        "beta_cases": model_base.params[1],
        "se_cases": model_base.bse[1],
        "p_cases": model_base.pvalues[1],
        "ci_cases_lo": model_base.conf_int()[1, 0],
        "ci_cases_hi": model_base.conf_int()[1, 1],
        # Deaths coefficient
        "beta_deaths": model_base.params[2],
        "se_deaths": model_base.bse[2],
        "p_deaths": model_base.pvalues[2],
        "ci_deaths_lo": model_base.conf_int()[2, 0],
        "ci_deaths_hi": model_base.conf_int()[2, 1],
        # Full model summary for printing
        "base_summary": model_base.summary().as_text(),
    }

    # ---- Extended model: add ΔStringency ----
    str_col = f"d_{STRINGENCY_COL}"
    if str_col in phase_df.columns:
        X_ext = phase_df[[f"d_{CASES_COL}", f"d_{DEATHS_COL}", str_col]].values
        X_ext = sm.add_constant(X_ext)
        model_ext = sm.OLS(y, X_ext).fit()

        result["ext_r2"] = model_ext.rsquared
        result["ext_adj_r2"] = model_ext.rsquared_adj
        result["beta_stringency"] = model_ext.params[3]
        result["se_stringency"] = model_ext.bse[3]
        result["p_stringency"] = model_ext.pvalues[3]
        result["ci_stringency_lo"] = model_ext.conf_int()[3, 0]
        result["ci_stringency_hi"] = model_ext.conf_int()[3, 1]
        # R² improvement
        result["r2_improvement"] = model_ext.rsquared - model_base.rsquared
        result["ext_summary"] = model_ext.summary().as_text()

    # ---- Standardized coefficients (for comparability) ----
    # Standardize within phase
    y_std = (y - y.mean()) / y.std() if y.std() > 0 else y
    X_cases_std = (phase_df[f"d_{CASES_COL}"].values - phase_df[f"d_{CASES_COL}"].mean()) / phase_df[f"d_{CASES_COL}"].std() if phase_df[f"d_{CASES_COL}"].std() > 0 else phase_df[f"d_{CASES_COL}"].values
    X_deaths_std = (phase_df[f"d_{DEATHS_COL}"].values - phase_df[f"d_{DEATHS_COL}"].mean()) / phase_df[f"d_{DEATHS_COL}"].std() if phase_df[f"d_{DEATHS_COL}"].std() > 0 else phase_df[f"d_{DEATHS_COL}"].values

    X_std = sm.add_constant(np.column_stack([X_cases_std, X_deaths_std]))
    model_std = sm.OLS(y_std, X_std).fit()

    result["std_beta_cases"] = model_std.params[1]
    result["std_beta_deaths"] = model_std.params[2]
    result["std_se_cases"] = model_std.bse[1]
    result["std_se_deaths"] = model_std.bse[2]

    return result


def print_results(all_results):
    """Print formatted results."""
    print("\n" + "=" * 70)
    print("PHASE-SEGMENTED REGRESSION RESULTS")
    print("=" * 70)

    for r in all_results:
        print(f"\n{'─' * 60}")
        print(f"{r['phase']} ({r['start']} to {r['end']}), n={r['n']}")
        print(f"{'─' * 60}")
        print(f"  Base Model: Δtweet_vol ~ Δcases + Δdeaths")
        print(f"    R² = {r['base_r2']:.4f}, Adj R² = {r['base_adj_r2']:.4f}, "
              f"F p-value = {r['base_f_pval']:.4e}")
        print(f"    β_cases  = {r['beta_cases']:>12.4f} (SE={r['se_cases']:.4f}, "
              f"p={r['p_cases']:.4e}) "
              f"{'✓ sig' if r['p_cases'] < 0.05 else '✗ not sig'}")
        print(f"    β_deaths = {r['beta_deaths']:>12.4f} (SE={r['se_deaths']:.4f}, "
              f"p={r['p_deaths']:.4e}) "
              f"{'✓ sig' if r['p_deaths'] < 0.05 else '✗ not sig'}")
        print(f"    Standardized: β_cases={r['std_beta_cases']:.4f}, "
              f"β_deaths={r['std_beta_deaths']:.4f}")

        if "beta_stringency" in r:
            print(f"\n  Extended Model: + ΔStringency")
            print(f"    R² = {r['ext_r2']:.4f} (improvement: +{r['r2_improvement']:.4f})")
            print(f"    β_stringency = {r['beta_stringency']:>12.4f} "
                  f"(SE={r['se_stringency']:.4f}, p={r['p_stringency']:.4e}) "
                  f"{'✓ sig' if r['p_stringency'] < 0.05 else '✗ not sig'}")


def plot_forest(all_results, output_dir):
    """
    Forest plot: regression coefficients across phases.
    Uses STANDARDIZED coefficients so phases are comparable.
    """
    phases = [r["phase"] for r in all_results]
    n_phases = len(phases)
    y_pos = np.arange(n_phases)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle("Standardized Regression Coefficients by Pandemic Phase\n"
                 "(Δtweet_volume ~ Δcases + Δdeaths)",
                 fontsize=14, fontweight="bold")

    # ---- Cases forest plot ----
    ax = axes[0]
    betas = [r["std_beta_cases"] for r in all_results]
    ses = [r["std_se_cases"] for r in all_results]
    ci_lo = [b - 1.96 * s for b, s in zip(betas, ses)]
    ci_hi = [b + 1.96 * s for b, s in zip(betas, ses)]
    sigs = [r["p_cases"] < 0.05 for r in all_results]

    colors = ["steelblue" if s else "lightgray" for s in sigs]
    ax.barh(y_pos, betas, xerr=[np.array(betas) - np.array(ci_lo),
                                 np.array(ci_hi) - np.array(betas)],
            color=colors, edgecolor="black", linewidth=0.5, capsize=4,
            height=0.6, alpha=0.85)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(phases, fontsize=10)
    ax.set_xlabel("Standardized β (Cases)")
    ax.set_title("Effect of Δcases on Δtweet_volume")
    ax.invert_yaxis()

    # Add value labels
    for i, (b, p) in enumerate(zip(betas, [r["p_cases"] for r in all_results])):
        label = f"{b:.3f}{'*' if p < 0.05 else ''}"
        ax.text(b + 0.02 if b >= 0 else b - 0.02, i, label,
                va="center", ha="left" if b >= 0 else "right", fontsize=9)

    # ---- Deaths forest plot ----
    ax = axes[1]
    betas = [r["std_beta_deaths"] for r in all_results]
    ses = [r["std_se_deaths"] for r in all_results]
    ci_lo = [b - 1.96 * s for b, s in zip(betas, ses)]
    ci_hi = [b + 1.96 * s for b, s in zip(betas, ses)]
    sigs = [r["p_deaths"] < 0.05 for r in all_results]

    colors = ["firebrick" if s else "lightgray" for s in sigs]
    ax.barh(y_pos, betas, xerr=[np.array(betas) - np.array(ci_lo),
                                 np.array(ci_hi) - np.array(betas)],
            color=colors, edgecolor="black", linewidth=0.5, capsize=4,
            height=0.6, alpha=0.85)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(phases, fontsize=10)
    ax.set_xlabel("Standardized β (Deaths)")
    ax.set_title("Effect of Δdeaths on Δtweet_volume")
    ax.invert_yaxis()

    for i, (b, p) in enumerate(zip(betas, [r["p_deaths"] for r in all_results])):
        label = f"{b:.3f}{'*' if p < 0.05 else ''}"
        ax.text(b + 0.02 if b >= 0 else b - 0.02, i, label,
                va="center", ha="left" if b >= 0 else "right", fontsize=9)

    # Legend
    sig_patch = mpatches.Patch(color="steelblue", label="p < 0.05")
    ns_patch = mpatches.Patch(color="lightgray", label="p ≥ 0.05")
    fig.legend(handles=[sig_patch, ns_patch], loc="lower center", ncol=2,
               fontsize=10, frameon=True)

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    path = os.path.join(output_dir, "08_forest_plot_by_phase.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {path}")
    plt.close()


def plot_coefficient_trend(all_results, output_dir):
    """
    Line plot showing how coefficients change across phases — fatigue visualization.
    """
    phases = [r["phase"].replace("Phase ", "P") for r in all_results]
    x = np.arange(len(phases))

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Pandemic Fatigue: How Public Response Weakens Over Time",
                 fontsize=14, fontweight="bold")

    # ---- Standardized beta (cases) trend ----
    ax = axes[0]
    betas = [r["std_beta_cases"] for r in all_results]
    ses = [r["std_se_cases"] for r in all_results]
    sigs = [r["p_cases"] < 0.05 for r in all_results]

    ax.errorbar(x, betas, yerr=[1.96 * s for s in ses], fmt="o-",
                color="steelblue", linewidth=2, markersize=10, capsize=5)
    for i, (b, sig) in enumerate(zip(betas, sigs)):
        marker = "★" if sig else ""
        ax.annotate(f"{b:.3f}{marker}", (x[i], betas[i]),
                    textcoords="offset points", xytext=(0, 12),
                    ha="center", fontsize=9)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(phases, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Standardized β")
    ax.set_title("Δcases → Δtweet_volume")

    # ---- Standardized beta (deaths) trend ----
    ax = axes[1]
    betas = [r["std_beta_deaths"] for r in all_results]
    ses = [r["std_se_deaths"] for r in all_results]
    sigs = [r["p_deaths"] < 0.05 for r in all_results]

    ax.errorbar(x, betas, yerr=[1.96 * s for s in ses], fmt="o-",
                color="firebrick", linewidth=2, markersize=10, capsize=5)
    for i, (b, sig) in enumerate(zip(betas, sigs)):
        marker = "★" if sig else ""
        ax.annotate(f"{b:.3f}{marker}", (x[i], betas[i]),
                    textcoords="offset points", xytext=(0, 12),
                    ha="center", fontsize=9)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(phases, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Standardized β")
    ax.set_title("Δdeaths → Δtweet_volume")

    # ---- R² trend ----
    ax = axes[2]
    r2s = [r["base_r2"] for r in all_results]
    ax.bar(x, r2s, color="mediumpurple", alpha=0.8, edgecolor="black", linewidth=0.5)
    for i, val in enumerate(r2s):
        ax.text(i, val + 0.005, f"{val:.3f}", ha="center", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(phases, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("R²")
    ax.set_title("Model Explanatory Power")
    ax.set_ylim(0, max(r2s) * 1.3 if max(r2s) > 0 else 0.1)

    plt.tight_layout()
    path = os.path.join(output_dir, "09_fatigue_coefficient_trend.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()


def plot_stringency_contribution(all_results, output_dir):
    """
    Bar chart: does adding Stringency improve the model?
    """
    phases_with_str = [r for r in all_results if "beta_stringency" in r]
    if not phases_with_str:
        return

    phases = [r["phase"].replace("Phase ", "P") for r in phases_with_str]
    x = np.arange(len(phases))
    r2_base = [r["base_r2"] for r in phases_with_str]
    r2_ext = [r["ext_r2"] for r in phases_with_str]
    p_str = [r["p_stringency"] for r in phases_with_str]

    fig, ax = plt.subplots(figsize=(12, 5))
    width = 0.35
    bars1 = ax.bar(x - width / 2, r2_base, width, label="Base (Δcases + Δdeaths)",
                   color="steelblue", alpha=0.8, edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x + width / 2, r2_ext, width, label="Extended (+ ΔStringency)",
                   color="deeppink", alpha=0.8, edgecolor="black", linewidth=0.5)

    for i, (b, e, p) in enumerate(zip(r2_base, r2_ext, p_str)):
        improvement = e - b
        sig = "★" if p < 0.05 else ""
        ax.text(i + width / 2, e + 0.003, f"+{improvement:.4f}{sig}",
                ha="center", fontsize=9, color="deeppink")

    ax.set_xticks(x)
    ax.set_xticklabels(phases, rotation=30, ha="right")
    ax.set_ylabel("R²")
    ax.set_title("Does Policy (Stringency) Add Explanatory Power Beyond Pandemic Data?",
                 fontsize=13, fontweight="bold")
    ax.legend()

    plt.tight_layout()
    path = os.path.join(output_dir, "10_stringency_contribution.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()


def interpret(all_results):
    """Print interpretation and next steps."""
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    # Fatigue check
    std_cases = [r["std_beta_cases"] for r in all_results]
    std_deaths = [r["std_beta_deaths"] for r in all_results]

    print("\n--- Fatigue Effect ---")
    print(f"  Cases β across phases:  {['%.3f' % b for b in std_cases]}")
    print(f"  Deaths β across phases: {['%.3f' % b for b in std_deaths]}")

    # Check if Phase 4 breaks the trend
    if len(std_cases) >= 4:
        if abs(std_cases[3]) > abs(std_cases[2]):
            print(f"  → Phase 4 (Omicron) shows INCREASED response vs Phase 3")
            print(f"    (Omicron's scale re-engaged public attention)")
        if abs(std_cases[0]) > abs(std_cases[-1]):
            print(f"  → Overall fatigue confirmed: initial |β|={abs(std_cases[0]):.3f} "
                  f"> final |β|={abs(std_cases[-1]):.3f}")

    # Stringency contribution
    str_results = [r for r in all_results if "beta_stringency" in r]
    if str_results:
        any_sig = any(r["p_stringency"] < 0.05 for r in str_results)
        avg_improvement = np.mean([r["r2_improvement"] for r in str_results])
        print(f"\n--- Stringency Contribution ---")
        print(f"  Any phase significant: {'Yes' if any_sig else 'No'}")
        print(f"  Average R² improvement: {avg_improvement:.4f}")
        if avg_improvement < 0.01:
            print(f"  → Policy adds negligible explanatory power beyond pandemic data")

    # Overall model fit
    r2s = [r["base_r2"] for r in all_results]
    print(f"\n--- Model Fit ---")
    print(f"  R² across phases: {['%.4f' % r for r in r2s]}")
    print(f"  Average R²: {np.mean(r2s):.4f}")
    if np.mean(r2s) < 0.1:
        print(f"  → Models explain limited variance — tweet volume driven by")
        print(f"     many factors beyond pandemic data (media, platform, etc.)")

    print(f"\n{'=' * 70}")
    print("NEXT STEPS")
    print("=" * 70)
    print("Based on these results:")
    print("1. If fatigue confirmed → Write up as key finding (RQ3)")
    print("2. If Stringency not significant → Supports the narrative that")
    print("   policy doesn't independently drive public discussion")
    print("3. If R² is low → Discuss in limitations; tweet volume has")
    print("   many drivers beyond pandemic severity")
    print("4. Consider: response speed comparison (RQ2) using the")
    print("   cross-correlation lags from previous analysis")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load and difference
    df = load_and_difference(INPUT_PATH)

    # Run regression for each phase
    all_results = []
    for phase_name, start, end in PHASES:
        print(f"\n{'─' * 50}")
        print(f"Fitting: {phase_name}")
        result = run_regression_for_phase(df, phase_name, start, end)
        if result:
            all_results.append(result)

    if not all_results:
        print("ERROR: No phases produced results")
        sys.exit(1)

    # Print results
    print_results(all_results)

    # Save detailed results
    summary_cols = [
        "phase", "n", "base_r2", "base_adj_r2",
        "beta_cases", "se_cases", "p_cases",
        "beta_deaths", "se_deaths", "p_deaths",
        "std_beta_cases", "std_beta_deaths",
    ]
    if "beta_stringency" in all_results[0]:
        summary_cols += ["ext_r2", "r2_improvement",
                         "beta_stringency", "se_stringency", "p_stringency"]

    summary_df = pd.DataFrame(all_results)[summary_cols]
    csv_path = os.path.join(OUTPUT_DIR, "phase_regression_summary.csv")
    summary_df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    # Save full OLS summaries
    with open(os.path.join(OUTPUT_DIR, "phase_regression_full_output.txt"), "w") as f:
        for r in all_results:
            f.write(f"\n{'=' * 70}\n{r['phase']}\n{'=' * 70}\n")
            f.write("\nBASE MODEL:\n")
            f.write(r["base_summary"])
            if "ext_summary" in r:
                f.write("\n\nEXTENDED MODEL (+ Stringency):\n")
                f.write(r["ext_summary"])
            f.write("\n")
    print(f"Saved: {os.path.join(OUTPUT_DIR, 'phase_regression_full_output.txt')}")

    # Plots
    print("\nGenerating plots...")
    plot_forest(all_results, OUTPUT_DIR)
    plot_coefficient_trend(all_results, OUTPUT_DIR)
    plot_stringency_contribution(all_results, OUTPUT_DIR)

    # Interpretation
    interpret(all_results)


if __name__ == "__main__":
    main()