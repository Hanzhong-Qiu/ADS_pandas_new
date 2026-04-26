"""
Phase-Segmented Regression: Is Policy a Response to Pandemic Severity?
=======================================================================
Mirror analysis of tweets_ols.py, but with stringency as the RESPONSE variable
instead of tweet_volume.

For each pandemic phase, regress Δstringency on Δcases and Δdeaths
to test whether policy responses are driven by pandemic severity.

This directly answers the second half of sub-RQ (ii):
  "Is policy itself a response to pandemic severity, or is it independent?"

If β_cases and β_deaths are significant across phases, we establish
that policy and public engagement are BOTH responses to severity
(consistent with the proposed causal structure).

USAGE:
    python tweets_ols_stringency.py
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

# Pandemic phases (identical to tweets_ols.py for direct comparability)
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
    diff_cols = [STRINGENCY_COL, CASES_COL, DEATHS_COL, TWEET_VOL_COL]
    for col in diff_cols:
        df[f"d_{col}"] = df[col].diff()

    df = df.dropna(subset=[f"d_{col}" for col in diff_cols]).reset_index(drop=True)
    print(f"After differencing: {len(df)} rows")
    return df


def run_regression_for_phase(df, phase_name, start, end):
    """
    Run OLS regression within a single phase.

    Base model:     Δstringency ~ Δcases + Δdeaths
    Extended model: Δstringency ~ Δcases + Δdeaths + Δtweet_volume
                    (Does public engagement predict policy beyond severity?)
    """
    mask = (df[DATE_COL] >= start) & (df[DATE_COL] <= end)
    phase_df = df[mask].copy()
    n = len(phase_df)

    if n < 20:
        print(f"  {phase_name}: only {n} rows, skipping")
        return None

    y = phase_df[f"d_{STRINGENCY_COL}"].values

    # ---- Base model: Δstringency ~ Δcases + Δdeaths ----
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
        # Base model
        "base_r2": model_base.rsquared,
        "base_adj_r2": model_base.rsquared_adj,
        "base_f_pval": model_base.f_pvalue,
        # Cases
        "beta_cases": model_base.params[1],
        "se_cases": model_base.bse[1],
        "p_cases": model_base.pvalues[1],
        "ci_cases_lo": model_base.conf_int()[1, 0],
        "ci_cases_hi": model_base.conf_int()[1, 1],
        # Deaths
        "beta_deaths": model_base.params[2],
        "se_deaths": model_base.bse[2],
        "p_deaths": model_base.pvalues[2],
        "ci_deaths_lo": model_base.conf_int()[2, 0],
        "ci_deaths_hi": model_base.conf_int()[2, 1],
        "base_summary": model_base.summary().as_text(),
    }

    # ---- Extended model: add Δtweet_volume ----
    tv_col = f"d_{TWEET_VOL_COL}"
    if tv_col in phase_df.columns:
        X_ext = phase_df[[f"d_{CASES_COL}", f"d_{DEATHS_COL}", tv_col]].values
        X_ext = sm.add_constant(X_ext)
        model_ext = sm.OLS(y, X_ext).fit()

        result["ext_r2"] = model_ext.rsquared
        result["ext_adj_r2"] = model_ext.rsquared_adj
        result["beta_tweet_vol"] = model_ext.params[3]
        result["se_tweet_vol"] = model_ext.bse[3]
        result["p_tweet_vol"] = model_ext.pvalues[3]
        result["ci_tweet_vol_lo"] = model_ext.conf_int()[3, 0]
        result["ci_tweet_vol_hi"] = model_ext.conf_int()[3, 1]
        result["r2_improvement"] = model_ext.rsquared - model_base.rsquared
        result["ext_summary"] = model_ext.summary().as_text()

    # ---- Standardized coefficients ----
    y_std = (y - y.mean()) / y.std() if y.std() > 0 else y

    def std_col(col):
        vals = phase_df[col].values
        return (vals - vals.mean()) / vals.std() if vals.std() > 0 else vals

    X_cases_std = std_col(f"d_{CASES_COL}")
    X_deaths_std = std_col(f"d_{DEATHS_COL}")

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
    print("RESPONSE VARIABLE: Δstringency (policy response)")
    print("=" * 70)

    for r in all_results:
        print(f"\n{'─' * 60}")
        print(f"{r['phase']} ({r['start']} to {r['end']}), n={r['n']}")
        print(f"{'─' * 60}")
        print(f"  Base Model: Δstringency ~ Δcases + Δdeaths")
        print(f"    R² = {r['base_r2']:.4f}, Adj R² = {r['base_adj_r2']:.4f}, "
              f"F p-value = {r['base_f_pval']:.4e}")
        print(f"    β_cases  = {r['beta_cases']:>14.6e} (SE={r['se_cases']:.4e}, "
              f"p={r['p_cases']:.4e}) "
              f"{'✓ sig' if r['p_cases'] < 0.05 else '✗ not sig'}")
        print(f"    β_deaths = {r['beta_deaths']:>14.6e} (SE={r['se_deaths']:.4e}, "
              f"p={r['p_deaths']:.4e}) "
              f"{'✓ sig' if r['p_deaths'] < 0.05 else '✗ not sig'}")
        print(f"    Standardized: β_cases={r['std_beta_cases']:.4f}, "
              f"β_deaths={r['std_beta_deaths']:.4f}")

        if "beta_tweet_vol" in r:
            print(f"\n  Extended Model: + Δtweet_volume")
            print(f"    R² = {r['ext_r2']:.4f} (improvement: +{r['r2_improvement']:.4f})")
            print(f"    β_tweet_vol = {r['beta_tweet_vol']:>14.6e} "
                  f"(SE={r['se_tweet_vol']:.4e}, p={r['p_tweet_vol']:.4e}) "
                  f"{'✓ sig' if r['p_tweet_vol'] < 0.05 else '✗ not sig'}")


def plot_forest(all_results, output_dir):
    """Forest plot: standardized coefficients across phases."""
    phases = [r["phase"] for r in all_results]
    n_phases = len(phases)
    y_pos = np.arange(n_phases)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle("Standardized Regression Coefficients by Pandemic Phase\n"
                 "(Δstringency ~ Δcases + Δdeaths)  —  Is policy a response to severity?",
                 fontsize=14, fontweight="bold")

    # ---- Cases ----
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
    ax.set_title("Effect of Δcases on Δstringency")
    ax.invert_yaxis()

    for i, (b, p) in enumerate(zip(betas, [r["p_cases"] for r in all_results])):
        label = f"{b:.3f}{'*' if p < 0.05 else ''}"
        ax.text(b + 0.02 if b >= 0 else b - 0.02, i, label,
                va="center", ha="left" if b >= 0 else "right", fontsize=9)

    # ---- Deaths ----
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
    ax.set_title("Effect of Δdeaths on Δstringency")
    ax.invert_yaxis()

    for i, (b, p) in enumerate(zip(betas, [r["p_deaths"] for r in all_results])):
        label = f"{b:.3f}{'*' if p < 0.05 else ''}"
        ax.text(b + 0.02 if b >= 0 else b - 0.02, i, label,
                va="center", ha="left" if b >= 0 else "right", fontsize=9)

    sig_patch = mpatches.Patch(color="steelblue", label="p < 0.05")
    ns_patch = mpatches.Patch(color="lightgray", label="p ≥ 0.05")
    fig.legend(handles=[sig_patch, ns_patch], loc="lower center", ncol=2,
               fontsize=10, frameon=True)

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    path = os.path.join(output_dir, "11_forest_plot_stringency_response.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {path}")
    plt.close()


def plot_coefficient_trend(all_results, output_dir):
    """Trajectory of standardized coefficients and R² across phases."""
    phases = [r["phase"].replace("Phase ", "P") for r in all_results]
    x = np.arange(len(phases))

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("How Does Policy Respond to Pandemic Severity Across Phases?",
                 fontsize=14, fontweight="bold")

    # Cases
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
    ax.set_title("Δcases → Δstringency")

    # Deaths
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
    ax.set_title("Δdeaths → Δstringency")

    # R²
    ax = axes[2]
    r2s = [r["base_r2"] for r in all_results]
    ax.bar(x, r2s, color="mediumseagreen", alpha=0.8, edgecolor="black", linewidth=0.5)
    for i, val in enumerate(r2s):
        ax.text(i, val + 0.005, f"{val:.3f}", ha="center", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(phases, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("R²")
    ax.set_title("Model Explanatory Power")
    ax.set_ylim(0, max(r2s) * 1.3 if max(r2s) > 0 else 0.1)

    plt.tight_layout()
    path = os.path.join(output_dir, "12_stringency_coefficient_trend.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()


def plot_tweet_vol_contribution(all_results, output_dir):
    """Does adding Δtweet_volume improve the model?
    If yes → public engagement influences policy beyond severity.
    If no  → policy responds to severity only, not directly to engagement.
    """
    phases_with_tv = [r for r in all_results if "beta_tweet_vol" in r]
    if not phases_with_tv:
        return

    phases = [r["phase"].replace("Phase ", "P") for r in phases_with_tv]
    x = np.arange(len(phases))
    r2_base = [r["base_r2"] for r in phases_with_tv]
    r2_ext = [r["ext_r2"] for r in phases_with_tv]
    p_tv = [r["p_tweet_vol"] for r in phases_with_tv]

    fig, ax = plt.subplots(figsize=(12, 5))
    width = 0.35
    ax.bar(x - width / 2, r2_base, width, label="Base (Δcases + Δdeaths)",
           color="steelblue", alpha=0.8, edgecolor="black", linewidth=0.5)
    ax.bar(x + width / 2, r2_ext, width, label="Extended (+ Δtweet_volume)",
           color="orange", alpha=0.8, edgecolor="black", linewidth=0.5)

    for i, (b, e, p) in enumerate(zip(r2_base, r2_ext, p_tv)):
        improvement = e - b
        sig = "★" if p < 0.05 else ""
        ax.text(i + width / 2, e + 0.003, f"+{improvement:.4f}{sig}",
                ha="center", fontsize=9, color="orange")

    ax.set_xticks(x)
    ax.set_xticklabels(phases, rotation=30, ha="right")
    ax.set_ylabel("R²")
    ax.set_title("Does Public Engagement Add Explanatory Power for Policy Beyond Severity?",
                 fontsize=13, fontweight="bold")
    ax.legend()

    plt.tight_layout()
    path = os.path.join(output_dir, "13_tweet_vol_contribution_to_policy.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()


def interpret(all_results):
    """Interpret the results relative to the theoretical framework."""
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    # Severity → policy
    std_cases = [r["std_beta_cases"] for r in all_results]
    std_deaths = [r["std_beta_deaths"] for r in all_results]
    sig_cases = [r["p_cases"] < 0.05 for r in all_results]
    sig_deaths = [r["p_deaths"] < 0.05 for r in all_results]

    print("\n--- Severity → Policy Pattern ---")
    print(f"  Cases β across phases:  {['%.3f' % b for b in std_cases]}")
    print(f"    Significant in:       {sum(sig_cases)}/{len(sig_cases)} phases")
    print(f"  Deaths β across phases: {['%.3f' % b for b in std_deaths]}")
    print(f"    Significant in:       {sum(sig_deaths)}/{len(sig_deaths)} phases")

    # R²
    r2s = [r["base_r2"] for r in all_results]
    print(f"\n--- Model Fit ---")
    print(f"  R² across phases: {['%.4f' % r for r in r2s]}")
    print(f"  Average R²: {np.mean(r2s):.4f}")

    # Compare to tweet_volume model (from tweets_ols.py)
    print(f"\n--- Comparison with Engagement Model ---")
    print(f"  This model (stringency as Y): Avg R² = {np.mean(r2s):.4f}")
    print(f"  tweets_ols.py (tweet_vol Y):  Avg R² = 0.0891 (from prior run)")
    if np.mean(r2s) > 0.0891:
        print(f"  → Policy is MORE predictable from severity than engagement is.")
    else:
        print(f"  → Policy is LESS predictable from severity than engagement is.")

    # Extended model: does tweet_volume add to policy prediction?
    ext_results = [r for r in all_results if "beta_tweet_vol" in r]
    if ext_results:
        any_sig = any(r["p_tweet_vol"] < 0.05 for r in ext_results)
        avg_improvement = np.mean([r["r2_improvement"] for r in ext_results])
        print(f"\n--- Does Public Engagement Influence Policy Beyond Severity? ---")
        print(f"  Any phase significant: {'Yes' if any_sig else 'No'}")
        print(f"  Average R² improvement: {avg_improvement:.4f}")
        if avg_improvement < 0.01 and not any_sig:
            print(f"  → Public engagement does NOT add predictive power for policy.")
            print(f"    Both policy and engagement appear to be parallel responses to severity,")
            print(f"    with no direct influence between them (static view).")
        elif any_sig:
            print(f"  → Public engagement adds explanatory power in some phases.")
            print(f"    This suggests public discourse may influence policy beyond")
            print(f"    what severity data alone predicts.")

    # Answer sub-RQ (ii) second half
    print(f"\n{'=' * 70}")
    print("ANSWERING sub-RQ (ii): Is policy a response to pandemic severity?")
    print(f"{'=' * 70}")

    any_sig_severity = any(sig_cases[i] or sig_deaths[i] for i in range(len(all_results)))
    if any_sig_severity:
        print("  ✓ YES: Pandemic severity significantly predicts policy changes in")
        print(f"    at least one phase (cases or deaths significant).")
        print(f"  → Supports the hypothesis: BOTH policy and engagement are severity responses.")
    else:
        print("  ✗ Severity does not significantly predict policy — unexpected.")
        print("  → Framework may need revision.")

    print(f"\n{'=' * 70}")
    print("NEXT STEPS")
    print(f"{'=' * 70}")
    print("1. Compare β magnitudes: tweets_ols.py vs tweets_ols_stringency.py")
    print("   If β(severity→tweets) > β(severity→stringency), engagement is")
    print("   more sensitive to severity than policy is.")
    print("2. Static OLS cannot answer 'who responds faster?'")
    print("   → Proceed to VAR + IRF for temporal priority (sub-RQ iii).")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load and difference
    df = load_and_difference(INPUT_PATH)

    # Run regressions
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

    # Save summary CSV
    summary_cols = [
        "phase", "n", "base_r2", "base_adj_r2",
        "beta_cases", "se_cases", "p_cases",
        "beta_deaths", "se_deaths", "p_deaths",
        "std_beta_cases", "std_beta_deaths",
    ]
    if "beta_tweet_vol" in all_results[0]:
        summary_cols += ["ext_r2", "r2_improvement",
                         "beta_tweet_vol", "se_tweet_vol", "p_tweet_vol"]

    summary_df = pd.DataFrame(all_results)[summary_cols]
    csv_path = os.path.join(OUTPUT_DIR, "phase_regression_stringency_summary.csv")
    summary_df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    # Save full OLS summaries
    txt_path = os.path.join(OUTPUT_DIR, "phase_regression_stringency_full_output.txt")
    with open(txt_path, "w") as f:
        for r in all_results:
            f.write(f"\n{'=' * 70}\n{r['phase']}\n{'=' * 70}\n")
            f.write("\nBASE MODEL (Δstringency ~ Δcases + Δdeaths):\n")
            f.write(r["base_summary"])
            if "ext_summary" in r:
                f.write("\n\nEXTENDED MODEL (+ Δtweet_volume):\n")
                f.write(r["ext_summary"])
            f.write("\n")
    print(f"Saved: {txt_path}")

    # Plots
    print("\nGenerating plots...")
    plot_forest(all_results, OUTPUT_DIR)
    plot_coefficient_trend(all_results, OUTPUT_DIR)
    plot_tweet_vol_contribution(all_results, OUTPUT_DIR)

    # Interpretation
    interpret(all_results)


if __name__ == "__main__":
    main()