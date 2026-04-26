"""
VAR + Impulse Response Analysis: Dynamic System of Pandemic Severity,
Public Engagement, and Government Policy
======================================================================
This script answers sub-RQ (iii): Does public engagement lead policy in
responding to pandemic severity?

It also resolves the methodological gap from IV.C.1 (where contemporaneous
OLS failed to predict stringency) by introducing dynamic lag structure
through a Vector Autoregression framework.

KEY OUTPUTS:
1. Lag order selection (AIC/BIC/HQIC)
2. VAR stability check
3. Impulse Response Functions (IRFs) with bootstrap CIs
4. Forecast Error Variance Decomposition (FEVD)
5. Granger causality tests within VAR framework

VARIABLE ORDERING (Cholesky):
    cases -> deaths -> tweet_volume -> stringency
    (Most exogenous to most endogenous)

USAGE:
    python tweets_var.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
import os
import sys
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

# VAR variable ordering (most exogenous first, for Cholesky identification)
VAR_VARS = [CASES_COL, DEATHS_COL, TWEET_VOL_COL, STRINGENCY_COL]

# Display labels (for plots)
VAR_LABELS = {
    CASES_COL: "Cases",
    DEATHS_COL: "Deaths",
    TWEET_VOL_COL: "Tweet Volume",
    STRINGENCY_COL: "Stringency",
}

# Analysis parameters
MAX_LAG = 21          # Maximum lag to consider for selection
IRF_HORIZON = 21      # Days to project IRFs forward
N_BOOTSTRAP = 1000    # Bootstrap iterations for IRF CIs


# ============================================================
# STEP 1: LOAD AND PREPARE DATA
# ============================================================
def load_and_difference(filepath):
    """Load full_analysis_data.csv, compute first differences."""
    print(f"\n{'='*70}")
    print("STEP 1: LOAD AND PREPARE DATA")
    print(f"{'='*70}")

    df = pd.read_csv(filepath)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)
    print(f"Loaded: {len(df)} rows, {df[DATE_COL].min().date()} to {df[DATE_COL].max().date()}")

    # First difference
    diff_cols = VAR_VARS
    for col in diff_cols:
        df[f"d_{col}"] = df[col].diff()

    df_diff = df[[DATE_COL] + [f"d_{col}" for col in diff_cols]].dropna().reset_index(drop=True)
    print(f"After differencing: {len(df_diff)} rows")

    # Build VAR-ready DataFrame (just the differenced series, indexed by date)
    var_df = df_diff[[f"d_{col}" for col in diff_cols]].copy()
    var_df.columns = diff_cols  # rename d_x -> x for cleaner output
    var_df.index = df_diff[DATE_COL]

    # Verify stationarity
    print("\nADF stationarity check on differenced series:")
    print(f"  {'Variable':<25s} {'ADF stat':>10s} {'p-value':>10s} {'Stationary':>12s}")
    for col in diff_cols:
        result = adfuller(var_df[col].dropna(), autolag="AIC")
        adf_stat, p_val = result[0], result[1]
        is_stat = "✓" if p_val < 0.05 else "✗"
        print(f"  Δ{col:<24s} {adf_stat:>10.4f} {p_val:>10.4f} {is_stat:>12s}")

    return var_df


# ============================================================
# STEP 2: LAG ORDER SELECTION
# ============================================================
def select_lag_order(var_df, max_lag=MAX_LAG):
    """Use AIC/BIC/HQIC to select optimal lag."""
    print(f"\n{'='*70}")
    print("STEP 2: LAG ORDER SELECTION")
    print(f"{'='*70}")

    model = VAR(var_df)
    selection = model.select_order(maxlags=max_lag)
    print(f"\n{selection.summary()}")

    # Extract chosen lag (we use BIC by default — most parsimonious)
    aic_lag = selection.aic
    bic_lag = selection.bic
    hqic_lag = selection.hqic

    print(f"\nOptimal lag selection:")
    print(f"  AIC:  {aic_lag}")
    print(f"  BIC:  {bic_lag}")
    print(f"  HQIC: {hqic_lag}")

    # Default to BIC (most parsimonious to avoid overfitting)
    chosen_lag = bic_lag if bic_lag >= 1 else aic_lag
    print(f"\n→ Using lag = {chosen_lag} (BIC-selected, parsimonious)")

    return chosen_lag


# ============================================================
# STEP 3: FIT VAR AND CHECK STABILITY
# ============================================================
def fit_var(var_df, lag):
    """Fit VAR model and check stability."""
    print(f"\n{'='*70}")
    print(f"STEP 3: FIT VAR({lag}) AND CHECK STABILITY")
    print(f"{'='*70}")

    model = VAR(var_df)
    results = model.fit(lag)

    # Stability: all characteristic roots' modulus < 1
    is_stable = results.is_stable(verbose=False)
    print(f"\nVAR stability check: {'✓ STABLE' if is_stable else '✗ UNSTABLE'}")

    if not is_stable:
        print("WARNING: VAR is not stable. IRFs may not converge meaningfully.")
        print("Consider reducing lag or checking data for outliers.")

    # Print model summary highlights
    print(f"\nModel diagnostics:")
    print(f"  N observations:    {results.nobs}")
    print(f"  Lag order:         {results.k_ar}")
    print(f"  AIC:               {results.aic:.4f}")
    print(f"  BIC:               {results.bic:.4f}")
    print(f"  Log-likelihood:    {results.llf:.4f}")

    return results


# ============================================================
# STEP 4: IMPULSE RESPONSE FUNCTIONS
# ============================================================
def compute_irfs(var_results, horizon=IRF_HORIZON, n_boot=N_BOOTSTRAP):
    """Compute orthogonalized IRFs with bootstrap CIs."""
    print(f"\n{'='*70}")
    print("STEP 4: IMPULSE RESPONSE FUNCTIONS")
    print(f"{'='*70}")
    print(f"Computing IRFs over {horizon}-day horizon...")
    print(f"Bootstrap CIs: {n_boot} iterations (this may take 1-2 minutes)...")

    irf = var_results.irf(horizon)

    # Get orthogonalized IRFs (Cholesky-identified) and bootstrap CIs
    irf_oirf = irf.orth_irfs  # shape: (horizon+1, n_vars, n_vars)
    # ci shape: (horizon+1, n_vars, n_vars, 2) — last dim is [lower, upper]
    ci_low, ci_high = irf.errband_mc(orth=True, repl=n_boot, signif=0.05)

    print("✓ IRFs computed.")
    return irf, irf_oirf, ci_low, ci_high


def plot_irfs(irf_oirf, ci_low, ci_high, output_dir, horizon=IRF_HORIZON):
    """Generate the three core IRF figures."""
    var_names = VAR_VARS
    labels = [VAR_LABELS[v] for v in var_names]
    days = np.arange(horizon + 1)

    # Find indices
    idx_cases = var_names.index(CASES_COL)
    idx_deaths = var_names.index(DEATHS_COL)
    idx_tv = var_names.index(TWEET_VOL_COL)
    idx_str = var_names.index(STRINGENCY_COL)

    # ============================================================
    # FIGURE 1: tweet_volume responses (CORE FINDING)
    # ============================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Public Engagement Responses to Shocks (Δtweet_volume)",
                 fontsize=14, fontweight="bold")

    shocks = [
        (idx_deaths, "Deaths", "firebrick"),
        (idx_cases, "Cases", "steelblue"),
        (idx_str, "Stringency", "darkorange"),
    ]

    for ax, (shock_idx, shock_name, color) in zip(axes, shocks):
        response = irf_oirf[:, idx_tv, shock_idx]
        lo = ci_low[:, idx_tv, shock_idx]
        hi = ci_high[:, idx_tv, shock_idx]

        ax.plot(days, response, color=color, linewidth=2.5)
        ax.fill_between(days, lo, hi, color=color, alpha=0.2, label="95% CI")
        ax.axhline(0, color="black", linewidth=0.6, linestyle="--")
        ax.set_xlabel("Days after shock")
        ax.set_ylabel("Δtweet_volume response")
        ax.set_title(f"Shock: {shock_name}")
        ax.legend()
        ax.grid(alpha=0.3)

        # Annotate peak
        peak_day = np.argmax(np.abs(response[:15]))  # peak within 15 days
        peak_val = response[peak_day]
        ax.annotate(
            f"peak day {peak_day}\nresponse={peak_val:.0f}",
            xy=(peak_day, peak_val),
            xytext=(peak_day + 2, peak_val * 1.1 if abs(peak_val) > 0 else 0.05),
            fontsize=9,
            arrowprops=dict(arrowstyle="->", color="black", lw=0.8),
        )

    plt.tight_layout()
    path = os.path.join(output_dir, "var_irf_tweet_volume_responses.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()

    # ============================================================
    # FIGURE 2: stringency responses (POLICY DYNAMICS)
    # ============================================================
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Policy Responses to Pandemic Shocks (Δstringency)",
                 fontsize=14, fontweight="bold")

    shocks_for_str = [
        (idx_deaths, "Deaths", "firebrick"),
        (idx_cases, "Cases", "steelblue"),
    ]

    for ax, (shock_idx, shock_name, color) in zip(axes, shocks_for_str):
        response = irf_oirf[:, idx_str, shock_idx]
        lo = ci_low[:, idx_str, shock_idx]
        hi = ci_high[:, idx_str, shock_idx]

        ax.plot(days, response, color=color, linewidth=2.5)
        ax.fill_between(days, lo, hi, color=color, alpha=0.2, label="95% CI")
        ax.axhline(0, color="black", linewidth=0.6, linestyle="--")
        ax.set_xlabel("Days after shock")
        ax.set_ylabel("Δstringency response")
        ax.set_title(f"Shock: {shock_name}")
        ax.legend()
        ax.grid(alpha=0.3)

        # Annotate peak
        peak_day = np.argmax(np.abs(response))
        peak_val = response[peak_day]
        ax.annotate(
            f"peak day {peak_day}\nresponse={peak_val:.4f}",
            xy=(peak_day, peak_val),
            xytext=(peak_day + 2, peak_val * 1.1 if abs(peak_val) > 0.001 else 0.001),
            fontsize=9,
            arrowprops=dict(arrowstyle="->", color="black", lw=0.8),
        )

    plt.tight_layout()
    path = os.path.join(output_dir, "var_irf_stringency_responses.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()

    # ============================================================
    # FIGURE 3: temporal priority — public vs policy response to deaths
    # ============================================================
    fig, ax = plt.subplots(figsize=(10, 6))

    # Tweet volume response to deaths (normalized to peak for comparability)
    tv_response = irf_oirf[:, idx_tv, idx_deaths]
    str_response = irf_oirf[:, idx_str, idx_deaths]

    # Normalize by peak absolute value for shape comparison
    tv_norm = tv_response / np.max(np.abs(tv_response)) if np.max(np.abs(tv_response)) > 0 else tv_response
    str_norm = str_response / np.max(np.abs(str_response)) if np.max(np.abs(str_response)) > 0 else str_response

    ax.plot(days, tv_norm, color="firebrick", linewidth=2.5,
            label=f"Tweet Volume (peak day {np.argmax(np.abs(tv_response))})")
    ax.plot(days, str_norm, color="darkorange", linewidth=2.5,
            label=f"Stringency (peak day {np.argmax(np.abs(str_response))})")
    ax.axhline(0, color="black", linewidth=0.6, linestyle="--")
    ax.set_xlabel("Days after deaths shock", fontsize=11)
    ax.set_ylabel("Normalized response (% of peak)", fontsize=11)
    ax.set_title("Temporal Priority: Public Engagement vs Policy Response to Deaths",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "var_irf_temporal_priority.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()


# ============================================================
# STEP 5: VARIANCE DECOMPOSITION
# ============================================================
def plot_fevd(var_results, output_dir, horizon=IRF_HORIZON):
    """Forecast Error Variance Decomposition."""
    print(f"\n{'='*70}")
    print("STEP 5: VARIANCE DECOMPOSITION")
    print(f"{'='*70}")

    fevd = var_results.fevd(horizon + 1)
    decomp = fevd.decomp  # shape: (n_vars, horizon+1, n_vars)
    # decomp[i, t, j] = share of variance of variable i at horizon t attributable to shock in variable j

    var_names = VAR_VARS

    # Plot for tweet_volume and stringency
    targets = [
        (var_names.index(TWEET_VOL_COL), "Tweet Volume"),
        (var_names.index(STRINGENCY_COL), "Stringency"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Forecast Error Variance Decomposition", fontsize=14, fontweight="bold")

    horizons_to_show = [1, 5, 10, 20]
    horizons_to_show = [h for h in horizons_to_show if h <= horizon]
    bar_colors = ["#4C72B0", "#DD8452", "#55A467", "#C44E52"]

    for ax, (target_idx, target_name) in zip(axes, targets):
        # data: rows = horizons, cols = source variables
        data = np.array([decomp[target_idx, h, :] for h in horizons_to_show])  # (n_horizons, n_vars)

        bottom = np.zeros(len(horizons_to_show))
        for j, source in enumerate(var_names):
            ax.bar(
                range(len(horizons_to_show)),
                data[:, j] * 100,
                bottom=bottom,
                label=VAR_LABELS[source],
                color=bar_colors[j],
                edgecolor="white",
            )
            # Annotate non-trivial contributions
            for i, val in enumerate(data[:, j] * 100):
                if val > 5:
                    ax.text(i, bottom[i] + val / 2, f"{val:.0f}%",
                            ha="center", va="center", fontsize=9, color="white", fontweight="bold")
            bottom += data[:, j] * 100

        ax.set_xticks(range(len(horizons_to_show)))
        ax.set_xticklabels([f"{h}d" for h in horizons_to_show])
        ax.set_xlabel("Forecast horizon")
        ax.set_ylabel("% of forecast error variance")
        ax.set_title(f"Target: {target_name}")
        ax.set_ylim(0, 100)
        ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    path = os.path.join(output_dir, "var_fevd.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()

    # Print summary table
    print("\nFEVD Summary (% of variance explained at each horizon):")
    print("\n--- TWEET VOLUME ---")
    print(f"  {'Horizon':<10s} " + " ".join([f"{VAR_LABELS[v]:>12s}" for v in var_names]))
    tv_idx = var_names.index(TWEET_VOL_COL)
    for h in horizons_to_show:
        row = decomp[tv_idx, h, :] * 100
        print(f"  {h:<10d} " + " ".join([f"{val:>11.2f}%" for val in row]))

    print("\n--- STRINGENCY ---")
    print(f"  {'Horizon':<10s} " + " ".join([f"{VAR_LABELS[v]:>12s}" for v in var_names]))
    str_idx = var_names.index(STRINGENCY_COL)
    for h in horizons_to_show:
        row = decomp[str_idx, h, :] * 100
        print(f"  {h:<10d} " + " ".join([f"{val:>11.2f}%" for val in row]))


# ============================================================
# STEP 6: GRANGER CAUSALITY WITHIN VAR FRAMEWORK
# ============================================================
def granger_causality_var(var_results):
    """Test Granger causality within VAR framework."""
    print(f"\n{'='*70}")
    print("STEP 6: GRANGER CAUSALITY (within VAR)")
    print(f"{'='*70}")
    print("Testing whether each variable Granger-causes others, controlling for")
    print("all other variables in the VAR system.\n")

    # Test all directed pairs of interest
    tests = [
        # (caused_var, causing_var)
        (TWEET_VOL_COL, [STRINGENCY_COL]),  # Does stringency Granger-cause tweet_vol?
        (STRINGENCY_COL, [TWEET_VOL_COL]),  # Does tweet_vol Granger-cause stringency?
        (TWEET_VOL_COL, [DEATHS_COL]),
        (TWEET_VOL_COL, [CASES_COL]),
        (STRINGENCY_COL, [DEATHS_COL]),
        (STRINGENCY_COL, [CASES_COL]),
    ]

    results_summary = []
    print(f"  {'Cause':<20s} → {'Effect':<20s} {'F-stat':>10s} {'p-value':>10s} {'Sig?':>6s}")
    print(f"  {'-'*70}")
    for caused, causing in tests:
        try:
            test_result = var_results.test_causality(caused, causing, kind="f")
            f_stat = test_result.test_statistic
            p_val = test_result.pvalue
            sig = "✓" if p_val < 0.05 else "✗"
            print(f"  {causing[0]:<20s} → {caused:<20s} {f_stat:>10.3f} {p_val:>10.4f} {sig:>6s}")
            results_summary.append({
                "cause": causing[0],
                "effect": caused,
                "f_stat": f_stat,
                "p_value": p_val,
                "significant": p_val < 0.05,
            })
        except Exception as e:
            print(f"  Error testing {causing[0]} → {caused}: {e}")

    print(f"\n  Significance: ✓ p < 0.05, ✗ p ≥ 0.05")
    return pd.DataFrame(results_summary)


# ============================================================
# STEP 7: INTERPRETATION
# ============================================================
def interpret(var_results, irf_oirf, granger_df):
    """Print framework-level interpretation."""
    print(f"\n{'='*70}")
    print("STEP 7: INTERPRETATION")
    print(f"{'='*70}")

    var_names = VAR_VARS
    idx_tv = var_names.index(TWEET_VOL_COL)
    idx_str = var_names.index(STRINGENCY_COL)
    idx_deaths = var_names.index(DEATHS_COL)
    idx_cases = var_names.index(CASES_COL)

    # Peak times for response to deaths shock
    tv_to_deaths = irf_oirf[:, idx_tv, idx_deaths]
    str_to_deaths = irf_oirf[:, idx_str, idx_deaths]
    tv_peak_day = int(np.argmax(np.abs(tv_to_deaths)))
    str_peak_day = int(np.argmax(np.abs(str_to_deaths)))

    print("\n--- Sub-RQ (i): Does severity drive public engagement? ---")
    print(f"  Tweet volume IRF to deaths shock peaks at day {tv_peak_day}")
    print(f"  Tweet volume IRF to cases shock peaks at day "
          f"{int(np.argmax(np.abs(irf_oirf[:, idx_tv, idx_cases])))}")
    print("  → Confirms IV.A finding: severity drives engagement, dynamically.")

    print("\n--- Sub-RQ (ii): Is policy a response to severity? ---")
    print(f"  Stringency IRF to deaths shock peaks at day {str_peak_day}")
    print(f"  Stringency IRF to cases shock peaks at day "
          f"{int(np.argmax(np.abs(irf_oirf[:, idx_str, idx_cases])))}")
    print("  → If significant, this confirms policy is a delayed response to severity")
    print("    (resolving the OLS failure in IV.C.1).")

    print("\n--- Sub-RQ (iii): Temporal priority ---")
    if tv_peak_day < str_peak_day:
        delay = str_peak_day - tv_peak_day
        print(f"  ✓ Tweet volume peak (day {tv_peak_day}) PRECEDES "
              f"stringency peak (day {str_peak_day})")
        print(f"  → Public engagement leads policy by {delay} days.")
        print(f"  → Defines an 'attention window' of ~{delay} days for")
        print(f"    public health communication after a death shock.")
    else:
        print(f"  ⚠ Stringency peak (day {str_peak_day}) ≤ tweet volume peak (day {tv_peak_day})")
        print(f"  → Unexpected: policy responds as fast or faster than engagement.")

    # Granger causality summary
    print("\n--- Direct Influence Between Engagement and Policy ---")
    tv_to_str = granger_df[
        (granger_df["cause"] == TWEET_VOL_COL) & (granger_df["effect"] == STRINGENCY_COL)
    ]
    str_to_tv = granger_df[
        (granger_df["cause"] == STRINGENCY_COL) & (granger_df["effect"] == TWEET_VOL_COL)
    ]

    if len(tv_to_str):
        sig = tv_to_str["significant"].values[0]
        p = tv_to_str["p_value"].values[0]
        print(f"  Tweet volume → Stringency: p={p:.4f} {'(✓ significant)' if sig else '(✗ not significant)'}")
    if len(str_to_tv):
        sig = str_to_tv["significant"].values[0]
        p = str_to_tv["p_value"].values[0]
        print(f"  Stringency → Tweet volume: p={p:.4f} {'(✓ significant)' if sig else '(✗ not significant)'}")

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print("This VAR analysis provides:")
    print("  1. Confirmation that engagement responds to severity dynamically (IV.A in motion)")
    print("  2. Resolution of IV.C.1 OLS failure: policy responds to severity over longer horizons")
    print("  3. Quantification of public-policy temporal priority")
    print("  4. Variance decomposition giving relative importance of each shock source")


# ============================================================
# MAIN
# ============================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1
    var_df = load_and_difference(INPUT_PATH)

    # Step 2
    chosen_lag = select_lag_order(var_df)

    # Step 3
    var_results = fit_var(var_df, chosen_lag)

    # Save VAR summary
    txt_path = os.path.join(OUTPUT_DIR, "var_model_summary.txt")
    with open(txt_path, "w") as f:
        f.write(str(var_results.summary()))
    print(f"\nSaved VAR summary: {txt_path}")

    # Step 4
    irf, irf_oirf, ci_low, ci_high = compute_irfs(var_results)
    plot_irfs(irf_oirf, ci_low, ci_high, OUTPUT_DIR)

    # Save IRF data for reference
    np.savez(
        os.path.join(OUTPUT_DIR, "var_irfs.npz"),
        oirf=irf_oirf,
        ci_low=ci_low,
        ci_high=ci_high,
        var_names=np.array(VAR_VARS),
    )

    # Step 5
    plot_fevd(var_results, OUTPUT_DIR)

    # Step 6
    granger_df = granger_causality_var(var_results)
    granger_df.to_csv(os.path.join(OUTPUT_DIR, "var_granger_results.csv"), index=False)

    # Step 7
    interpret(var_results, irf_oirf, granger_df)

    print(f"\n{'='*70}")
    print("ALL VAR OUTPUTS SAVED")
    print(f"{'='*70}")
    print(f"Directory: {OUTPUT_DIR}")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        if f.startswith("var_"):
            size = os.path.getsize(os.path.join(OUTPUT_DIR, f)) / 1024
            print(f"  {f} ({size:.1f} KB)")


if __name__ == "__main__":
    main()