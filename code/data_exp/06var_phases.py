"""
Phase-Segmented VAR Analysis: Robustness Check
======================================================================
Re-runs the 4-variable VAR (cases, deaths, tweet_volume, stringency)
within each pandemic phase to test whether the global findings hold
across all phases or are masked by phase-level heterogeneity.

KEY QUESTIONS:
1. Does tweet_volume's response to deaths/cases differ across phases?
   (Should mirror IV.A's three-phase structure)
2. Does stringency show significant dynamic response to severity in
   any single phase, even if it doesn't globally?
3. Do Granger causality directions/significance flip across phases?

DESIGN CHOICES:
- Each phase gets its own BIC-selected lag (sample size varies)
- IRF horizon: 14 days (shorter than global 21d due to smaller samples)
- Bootstrap CIs: 500 iterations (faster than global 1000 for runtime)
- Output: per-phase IRFs + cross-phase summary heatmap

USAGE:
    python tweets_var_phased.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
import os
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# PATHS
# ============================================================
INPUT_PATH = "/home/qqq/ADS_pandas_new/.csv/new/full_analysis_data.csv"
OUTPUT_DIR = "/home/qqq/ADS_pandas_new/graphs/modelling/phased_var"

# ============================================================
# CONFIG
# ============================================================
DATE_COL = "date"
TWEET_VOL_COL = "tweet_volume"
CASES_COL = "daily_new_cases"
DEATHS_COL = "daily_new_deaths"
STRINGENCY_COL = "StringencyIndex_Average"

VAR_VARS = [CASES_COL, DEATHS_COL, TWEET_VOL_COL, STRINGENCY_COL]

VAR_LABELS = {
    CASES_COL: "Cases",
    DEATHS_COL: "Deaths",
    TWEET_VOL_COL: "Tweet Volume",
    STRINGENCY_COL: "Stringency",
}

PHASES = [
    ("Phase 1: Initial Outbreak", "2020-03-19", "2020-08-31"),
    ("Phase 2: Second Wave", "2020-09-01", "2021-02-28"),
    ("Phase 3: Vaccination Era", "2021-03-01", "2021-09-30"),
    ("Phase 4: Omicron", "2021-10-01", "2022-03-31"),
    ("Phase 5: Post-Restrictions", "2022-04-01", "2022-12-31"),
]

# Smaller samples → smaller max lag candidates
PHASE_MAX_LAG = 7    # Try lag 1-7 per phase (BIC will pick best)
IRF_HORIZON = 14     # 14 days (shorter than global 21d for smaller samples)
N_BOOTSTRAP = 500    # 500 iterations (faster than global 1000)


# ============================================================
# DATA PREP
# ============================================================
def load_and_difference(filepath):
    """Load and difference data. Returns var_df indexed by date."""
    df = pd.read_csv(filepath)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)

    for col in VAR_VARS:
        df[f"d_{col}"] = df[col].diff()

    df_diff = df[[DATE_COL] + [f"d_{col}" for col in VAR_VARS]].dropna().reset_index(drop=True)
    var_df = df_diff[[f"d_{col}" for col in VAR_VARS]].copy()
    var_df.columns = VAR_VARS
    var_df.index = df_diff[DATE_COL]
    return var_df


def slice_phase(var_df, start, end):
    """Extract a phase window."""
    mask = (var_df.index >= start) & (var_df.index <= end)
    return var_df.loc[mask].copy()


# ============================================================
# PER-PHASE VAR ANALYSIS
# ============================================================
def analyze_phase(phase_name, start, end, var_df, output_dir):
    """Run full VAR pipeline for one phase. Returns dict of results."""
    print(f"\n{'─' * 70}")
    print(f"ANALYZING: {phase_name} ({start} to {end})")
    print(f"{'─' * 70}")

    phase_df = slice_phase(var_df, start, end)
    n = len(phase_df)
    print(f"Sample size: {n}")

    if n < 50:
        print(f"  ⚠ Sample too small (n<50), skipping.")
        return None

    # ADF check (quick)
    adf_ok = True
    for col in VAR_VARS:
        result = adfuller(phase_df[col].dropna(), autolag="AIC")
        if result[1] >= 0.05:
            adf_ok = False
            print(f"  ⚠ Δ{col} not stationary in this phase (p={result[1]:.4f})")
    if adf_ok:
        print("  ✓ All variables stationary within phase")

    # Lag selection
    try:
        model = VAR(phase_df)
        # Cap max lag based on sample size to avoid singularities
        safe_max_lag = min(PHASE_MAX_LAG, n // 20)
        if safe_max_lag < 1:
            safe_max_lag = 1
        selection = model.select_order(maxlags=safe_max_lag)
        chosen_lag = selection.bic if selection.bic >= 1 else 1
        print(f"  Lag selection (max tested={safe_max_lag}): BIC={selection.bic}, "
              f"AIC={selection.aic} → using lag={chosen_lag}")
    except Exception as e:
        print(f"  Lag selection failed: {e}")
        return None

    # Fit VAR
    try:
        var_results = model.fit(chosen_lag)
        is_stable = var_results.is_stable(verbose=False)
        print(f"  VAR({chosen_lag}) fit. Stability: {'✓' if is_stable else '✗'}")
    except Exception as e:
        print(f"  VAR fitting failed: {e}")
        return None

    # IRF
    horizon = min(IRF_HORIZON, n // 10)
    print(f"  Computing IRFs (horizon={horizon}, bootstrap={N_BOOTSTRAP})...")
    try:
        irf = var_results.irf(horizon)
        irf_oirf = irf.orth_irfs
        ci_low, ci_high = irf.errband_mc(orth=True, repl=N_BOOTSTRAP, signif=0.05)
    except Exception as e:
        print(f"  IRF failed: {e}")
        return None

    # Granger causality (key directions)
    granger_results = {}
    granger_tests = [
        (TWEET_VOL_COL, [DEATHS_COL]),
        (TWEET_VOL_COL, [CASES_COL]),
        (TWEET_VOL_COL, [STRINGENCY_COL]),
        (STRINGENCY_COL, [DEATHS_COL]),
        (STRINGENCY_COL, [CASES_COL]),
        (STRINGENCY_COL, [TWEET_VOL_COL]),
    ]
    for caused, causing in granger_tests:
        try:
            test = var_results.test_causality(caused, causing, kind="f")
            granger_results[(causing[0], caused)] = (test.test_statistic, test.pvalue)
        except Exception:
            granger_results[(causing[0], caused)] = (np.nan, np.nan)

    # Extract peak responses (IRF) for cross-phase summary
    idx = {v: VAR_VARS.index(v) for v in VAR_VARS}
    peaks = {}
    for response_var in [TWEET_VOL_COL, STRINGENCY_COL]:
        for shock_var in [DEATHS_COL, CASES_COL]:
            r = irf_oirf[:, idx[response_var], idx[shock_var]]
            lo = ci_low[:, idx[response_var], idx[shock_var]]
            hi = ci_high[:, idx[response_var], idx[shock_var]]
            # Find peak (largest absolute response)
            peak_day = int(np.argmax(np.abs(r)))
            peak_val = r[peak_day]
            # Check if peak is significant (CI excludes 0)
            peak_sig = (lo[peak_day] > 0) or (hi[peak_day] < 0)
            peaks[(shock_var, response_var)] = {
                "day": peak_day,
                "value": peak_val,
                "sig": peak_sig,
                "ci_lo": lo[peak_day],
                "ci_hi": hi[peak_day],
            }

    # Save individual phase IRF figure (compact 2x2 layout)
    plot_phase_irfs(phase_name, irf_oirf, ci_low, ci_high, horizon, output_dir)

    return {
        "phase": phase_name,
        "start": start,
        "end": end,
        "n": n,
        "lag": chosen_lag,
        "horizon": horizon,
        "stable": is_stable,
        "irf_oirf": irf_oirf,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "granger": granger_results,
        "peaks": peaks,
    }


def plot_phase_irfs(phase_name, irf_oirf, ci_low, ci_high, horizon, output_dir):
    """Compact 2x2 IRF figure: tweet_vol responses + stringency responses."""
    days = np.arange(horizon + 1)
    idx = {v: VAR_VARS.index(v) for v in VAR_VARS}

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle(f"VAR IRFs — {phase_name}", fontsize=13, fontweight="bold")

    # Top row: tweet_vol responses
    for j, (shock_var, color) in enumerate([(DEATHS_COL, "firebrick"),
                                             (CASES_COL, "steelblue")]):
        ax = axes[0, j]
        i_resp = idx[TWEET_VOL_COL]
        i_shock = idx[shock_var]
        resp = irf_oirf[:, i_resp, i_shock]
        lo = ci_low[:, i_resp, i_shock]
        hi = ci_high[:, i_resp, i_shock]
        ax.plot(days, resp, color=color, linewidth=2)
        ax.fill_between(days, lo, hi, color=color, alpha=0.2)
        ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
        ax.set_xlabel("Days after shock")
        ax.set_ylabel("Δtweet_volume response")
        ax.set_title(f"Shock: {VAR_LABELS[shock_var]} → Tweet Volume")
        ax.grid(alpha=0.3)

    # Bottom row: stringency responses
    for j, (shock_var, color) in enumerate([(DEATHS_COL, "firebrick"),
                                             (CASES_COL, "steelblue")]):
        ax = axes[1, j]
        i_resp = idx[STRINGENCY_COL]
        i_shock = idx[shock_var]
        resp = irf_oirf[:, i_resp, i_shock]
        lo = ci_low[:, i_resp, i_shock]
        hi = ci_high[:, i_resp, i_shock]
        ax.plot(days, resp, color=color, linewidth=2)
        ax.fill_between(days, lo, hi, color=color, alpha=0.2)
        ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
        ax.set_xlabel("Days after shock")
        ax.set_ylabel("Δstringency response")
        ax.set_title(f"Shock: {VAR_LABELS[shock_var]} → Stringency")
        ax.grid(alpha=0.3)

    plt.tight_layout()
    safe_name = phase_name.replace(":", "").replace(" ", "_").lower()
    path = os.path.join(output_dir, f"phase_irf_{safe_name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


# ============================================================
# CROSS-PHASE SUMMARY
# ============================================================
def plot_cross_phase_summary(all_results, output_dir):
    """Heatmap-style summary: peak response value across phases."""
    valid = [r for r in all_results if r is not None]
    if not valid:
        print("No valid results for summary.")
        return

    phase_names = [r["phase"].replace("Phase ", "P") for r in valid]

    # Build 2x2 matrix of figures: rows = response vars, cols = shock vars
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Phase-Segmented VAR: Peak IRF Responses Across Phases\n"
                 "(★ = peak significant at 95% CI)",
                 fontsize=14, fontweight="bold")

    response_shock_pairs = [
        (TWEET_VOL_COL, DEATHS_COL, "firebrick"),
        (TWEET_VOL_COL, CASES_COL, "steelblue"),
        (STRINGENCY_COL, DEATHS_COL, "darkorange"),
        (STRINGENCY_COL, CASES_COL, "purple"),
    ]

    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]

    for (resp_var, shock_var, color), (row, col) in zip(response_shock_pairs, positions):
        ax = axes[row, col]
        peak_vals = []
        peak_days = []
        sigs = []
        ci_los = []
        ci_his = []
        for r in valid:
            p = r["peaks"][(shock_var, resp_var)]
            peak_vals.append(p["value"])
            peak_days.append(p["day"])
            sigs.append(p["sig"])
            ci_los.append(p["ci_lo"])
            ci_his.append(p["ci_hi"])

        x_pos = np.arange(len(phase_names))
        ci_lower_err = np.array(peak_vals) - np.array(ci_los)
        ci_upper_err = np.array(ci_his) - np.array(peak_vals)
        bar_colors = [color if s else "lightgray" for s in sigs]
        ax.bar(x_pos, peak_vals, color=bar_colors, edgecolor="black",
               linewidth=0.5, alpha=0.85)
        ax.errorbar(x_pos, peak_vals, yerr=[ci_lower_err, ci_upper_err],
                    fmt="none", color="black", capsize=4, linewidth=1)
        ax.axhline(0, color="black", linewidth=0.7)

        # Annotate peak day and significance
        for i, (val, day, sig) in enumerate(zip(peak_vals, peak_days, sigs)):
            marker = "★" if sig else ""
            ax.text(i, val, f"d{day}{marker}",
                    ha="center", va="bottom" if val >= 0 else "top",
                    fontsize=9, fontweight="bold")

        ax.set_xticks(x_pos)
        ax.set_xticklabels(phase_names, rotation=20, ha="right", fontsize=9)
        ax.set_ylabel("Peak IRF response")
        ax.set_title(f"{VAR_LABELS[shock_var]} shock → {VAR_LABELS[resp_var]}")
        ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    path = os.path.join(output_dir, "cross_phase_irf_summary.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {path}")
    plt.close()


def plot_granger_summary(all_results, output_dir):
    """Heatmap of Granger p-values across phases."""
    valid = [r for r in all_results if r is not None]
    if not valid:
        return

    phase_names = [r["phase"].replace("Phase ", "P") for r in valid]
    granger_directions = [
        (DEATHS_COL, TWEET_VOL_COL, "Deaths → TweetVol"),
        (CASES_COL, TWEET_VOL_COL, "Cases → TweetVol"),
        (STRINGENCY_COL, TWEET_VOL_COL, "Stringency → TweetVol"),
        (DEATHS_COL, STRINGENCY_COL, "Deaths → Stringency"),
        (CASES_COL, STRINGENCY_COL, "Cases → Stringency"),
        (TWEET_VOL_COL, STRINGENCY_COL, "TweetVol → Stringency"),
    ]

    matrix = np.full((len(granger_directions), len(valid)), np.nan)
    for i, (cause, effect, _) in enumerate(granger_directions):
        for j, r in enumerate(valid):
            if (cause, effect) in r["granger"]:
                _, p = r["granger"][(cause, effect)]
                matrix[i, j] = p

    fig, ax = plt.subplots(figsize=(10, 6))
    # Use truncated colormap (red = significant, green = not)
    masked = np.ma.masked_invalid(matrix)
    im = ax.imshow(masked, cmap="RdYlGn_r", vmin=0, vmax=0.2, aspect="auto")

    ax.set_xticks(range(len(phase_names)))
    ax.set_xticklabels(phase_names, rotation=20, ha="right")
    ax.set_yticks(range(len(granger_directions)))
    ax.set_yticklabels([g[2] for g in granger_directions])

    # Annotate p-values
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            if not np.isnan(val):
                marker = " ✓" if val < 0.05 else ""
                ax.text(j, i, f"{val:.3f}{marker}",
                        ha="center", va="center", fontsize=10,
                        color="white" if val < 0.05 else "black",
                        fontweight="bold" if val < 0.05 else "normal")

    plt.colorbar(im, ax=ax, label="Granger p-value (capped at 0.2)")
    ax.set_title("Phase-Segmented VAR: Granger Causality p-values\n"
                 "(✓ = significant at p<0.05)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(output_dir, "cross_phase_granger_summary.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()


# ============================================================
# MAIN
# ============================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("PHASE-SEGMENTED VAR ANALYSIS")
    print("=" * 70)

    var_df = load_and_difference(INPUT_PATH)
    print(f"\nLoaded data: {len(var_df)} rows total")

    # Run VAR for each phase
    all_results = []
    for phase_name, start, end in PHASES:
        result = analyze_phase(phase_name, start, end, var_df, OUTPUT_DIR)
        all_results.append(result)

    # Cross-phase summaries
    print(f"\n{'=' * 70}")
    print("GENERATING CROSS-PHASE SUMMARIES")
    print(f"{'=' * 70}")
    plot_cross_phase_summary(all_results, OUTPUT_DIR)
    plot_granger_summary(all_results, OUTPUT_DIR)

    # Print final summary table
    print(f"\n{'=' * 70}")
    print("CROSS-PHASE SUMMARY TABLE")
    print(f"{'=' * 70}")

    print(f"\n{'Phase':<30s} {'n':>5s} {'lag':>4s} {'Stable':>7s}")
    print("─" * 50)
    for r in all_results:
        if r is None:
            continue
        print(f"{r['phase']:<30s} {r['n']:>5d} {r['lag']:>4d} {'✓' if r['stable'] else '✗':>7s}")

    print(f"\n--- KEY PEAK RESPONSES (★ = significant) ---")
    print(f"\n{'Phase':<30s} {'D→TV':>20s} {'C→TV':>20s} {'D→Str':>20s} {'C→Str':>20s}")
    print("─" * 110)
    for r in all_results:
        if r is None:
            continue
        peaks = r["peaks"]
        d_tv = peaks[(DEATHS_COL, TWEET_VOL_COL)]
        c_tv = peaks[(CASES_COL, TWEET_VOL_COL)]
        d_str = peaks[(DEATHS_COL, STRINGENCY_COL)]
        c_str = peaks[(CASES_COL, STRINGENCY_COL)]
        def fmt(p):
            return f"d{p['day']} {p['value']:>+8.3f}{'★' if p['sig'] else ''}"
        print(f"{r['phase']:<30s} {fmt(d_tv):>20s} {fmt(c_tv):>20s} {fmt(d_str):>20s} {fmt(c_str):>20s}")

    print(f"\n--- KEY GRANGER CAUSALITY (p-values, ✓ = sig) ---")
    print(f"\n{'Phase':<30s} {'Deaths→TV':>14s} {'Stringency→TV':>17s} {'TV→Stringency':>17s}")
    print("─" * 90)
    for r in all_results:
        if r is None:
            continue
        g = r["granger"]
        def fmt_p(key):
            if key not in g:
                return "N/A"
            _, p = g[key]
            return f"{p:.3f}{'✓' if p < 0.05 else ''}"
        print(f"{r['phase']:<30s} "
              f"{fmt_p((DEATHS_COL, TWEET_VOL_COL)):>14s} "
              f"{fmt_p((STRINGENCY_COL, TWEET_VOL_COL)):>17s} "
              f"{fmt_p((TWEET_VOL_COL, STRINGENCY_COL)):>17s}")

    # Save data summary
    rows = []
    for r in all_results:
        if r is None:
            continue
        for (shock, response), peak in r["peaks"].items():
            rows.append({
                "phase": r["phase"],
                "shock": VAR_LABELS[shock],
                "response": VAR_LABELS[response],
                "peak_day": peak["day"],
                "peak_value": peak["value"],
                "peak_sig": peak["sig"],
                "ci_lo": peak["ci_lo"],
                "ci_hi": peak["ci_hi"],
            })
    summary_df = pd.DataFrame(rows)
    csv_path = os.path.join(OUTPUT_DIR, "cross_phase_irf_summary.csv")
    summary_df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    print(f"\n{'=' * 70}")
    print(f"ALL OUTPUTS SAVED TO: {OUTPUT_DIR}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()