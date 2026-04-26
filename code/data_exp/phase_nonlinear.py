"""
Phase-Segmented Non-Linear Validation
======================================
Random Forest + Gradient Boosting extension of the phase-segmented OLS.

Goal
----
1. Validate the three-phase structural story with a model-agnostic approach.
2. Identify WHICH policy sub-component drives sentiment in WHICH phase
   (can't be done with composite Stringency alone).

Design
------
- Target:   Δsentiment_mean  (stationary, matches prior analyses)
- Features: Δcases, Δdeaths + OxCGRT sub-indicators
            C1 school, C2 workplace, C4 gatherings, C6 stay-home,
            C7 internal movement, C8 international travel, H6 masks
            (composite StringencyIndex dropped → avoids multicollinearity)
- Phases:   3-phase Granger-aligned window
- Models:   OLS (baseline) | RandomForest | GradientBoosting
- Eval:     TimeSeriesSplit 5-fold CV on first 80% (R² mean±std)
            + held-out last 20% test block (R², MSE, MAE)
            + permutation importance on held-out block (model-agnostic)

Outputs
-------
phase_nonlinear_metrics.csv       model performance per phase
phase_nonlinear_importance.csv    permutation importance per phase × model × feature
phase_nonlinear_top_features.csv  leading feature per phase × model (agreement check)
phase_nonlinear_r2.png            grouped bar: R² by phase × model
phase_nonlinear_importance_heatmap.png  importance heatmaps, one per model
phase_nonlinear_importance_bars.png     importance bars, one panel per phase

Usage
-----
    python phase_nonlinear.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
INPUT  = "/home/mohsin/ADS/ADS_pandas_new/.csv/new/full_analysis_data.csv"
OUTDIR = "/home/mohsin/ADS/ADS_pandas_new/graphs/modelling"
os.makedirs(OUTDIR, exist_ok=True)

TARGET   = "sentiment_mean"
FEATURES = [
    "daily_new_cases", "daily_new_deaths",
    "C1_School_closing", "C2_Workplace_closing",
    "C4_Restrictions_gatherings", "C6_Stay_at_home",
    "C7_Internal_movement", "C8_International_travel",
    "H6_Facial_coverings",
]
FEAT_LABELS = [
    "Δcases", "Δdeaths",
    "ΔC1 school", "ΔC2 work",
    "ΔC4 gather", "ΔC6 stay-home",
    "ΔC7 internal", "ΔC8 intl", "ΔH6 masks",
]

PHASES = [
    ("Phase 1: Initial outbreak",      "2020-03-01", "2020-09-30"),
    ("Phase 2: Second wave & vaccine", "2020-10-01", "2021-06-30"),
    ("Phase 3: Reopening & late",      "2021-07-01", "2022-12-31"),
]

SEED       = 42
TEST_FRAC  = 0.2     # final chronological block held out for test + permutation
N_SPLITS   = 5       # TimeSeriesSplit folds on the training block
N_REPEATS  = 50      # permutation importance repeats


# ----------------------------------------------------------------------
# LOAD + DIFFERENCE
# ----------------------------------------------------------------------
df = (pd.read_csv(INPUT, parse_dates=["date"])
        .sort_values("date")
        .reset_index(drop=True))

for c in [TARGET] + FEATURES:
    df[f"d_{c}"] = df[c].diff()

df = (df.dropna(subset=[f"d_{c}" for c in [TARGET] + FEATURES])
        .reset_index(drop=True))

print(f"Data: {len(df)} rows after differencing  "
      f"({df['date'].min().date()} → {df['date'].max().date()})")


# ----------------------------------------------------------------------
# MODELS
# ----------------------------------------------------------------------
def make_models():
    """Fresh instances each call (needed inside CV loop)."""
    return {
        "OLS": LinearRegression(),
        "RandomForest": RandomForestRegressor(
            n_estimators=400, max_depth=6, min_samples_leaf=5,
            random_state=SEED, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=400, max_depth=3, learning_rate=0.05,
            subsample=0.8, random_state=SEED),
    }


# ----------------------------------------------------------------------
# PER-PHASE EVALUATION
# ----------------------------------------------------------------------
def evaluate_phase(X, y, phase_name):
    split = int(len(X) * (1 - TEST_FRAC))
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]

    rows_metrics, rows_imp = [], []

    for name, mdl in make_models().items():
        # --- Cross-validation on training block ---
        tscv = TimeSeriesSplit(n_splits=N_SPLITS)
        cv_r2 = []
        for tr_idx, va_idx in tscv.split(X_tr):
            mdl_cv = make_models()[name]
            mdl_cv.fit(X_tr[tr_idx], y_tr[tr_idx])
            cv_r2.append(r2_score(y_tr[va_idx], mdl_cv.predict(X_tr[va_idx])))

        # --- Fit on full training block, evaluate on held-out test ---
        mdl.fit(X_tr, y_tr)
        p_te = mdl.predict(X_te)

        rows_metrics.append({
            "phase": phase_name, "model": name,
            "n_train": len(X_tr), "n_test": len(X_te),
            "r2_cv_mean":  float(np.mean(cv_r2)),
            "r2_cv_std":   float(np.std(cv_r2)),
            "r2_test":     r2_score(y_te, p_te),
            "mse_test":    mean_squared_error(y_te, p_te),
            "mae_test":    mean_absolute_error(y_te, p_te),
        })

        # --- Permutation importance on held-out test (model-agnostic) ---
        perm = permutation_importance(
            mdl, X_te, y_te,
            n_repeats=N_REPEATS, random_state=SEED,
            scoring="r2", n_jobs=-1,
        )
        for i, lab in enumerate(FEAT_LABELS):
            rows_imp.append({
                "phase": phase_name, "model": name, "feature": lab,
                "perm_mean": float(perm.importances_mean[i]),
                "perm_std":  float(perm.importances_std[i]),
            })

    return rows_metrics, rows_imp


# ----------------------------------------------------------------------
# RUN
# ----------------------------------------------------------------------
all_metrics, all_imp = [], []
x_cols = [f"d_{c}" for c in FEATURES]
y_col  = f"d_{TARGET}"

for phase, start, end in PHASES:
    sub = df[(df["date"] >= start) & (df["date"] <= end)].copy()
    if len(sub) < 60:
        print(f"Skip {phase}: only {len(sub)} rows")
        continue

    # Standardize features (OLS coefs comparable; trees invariant)
    X  = sub[x_cols].values
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    Xs = (X - mu) / sd
    y  = sub[y_col].values

    print(f"\n{phase}  (n={len(sub)})")
    m, i = evaluate_phase(Xs, y, phase)
    all_metrics.extend(m); all_imp.extend(i)

    for r in m:
        print(f"  {r['model']:18s}  "
              f"R²_cv={r['r2_cv_mean']:+.3f}±{r['r2_cv_std']:.3f}   "
              f"R²_test={r['r2_test']:+.3f}   "
              f"MSE={r['mse_test']:.2e}   "
              f"MAE={r['mae_test']:.2e}")

met_df = pd.DataFrame(all_metrics)
imp_df = pd.DataFrame(all_imp)
met_df.to_csv(f"{OUTDIR}/phase_nonlinear_metrics.csv",    index=False)
imp_df.to_csv(f"{OUTDIR}/phase_nonlinear_importance.csv", index=False)


# ----------------------------------------------------------------------
# PLOTS
# ----------------------------------------------------------------------
MODELS  = ["OLS", "RandomForest", "GradientBoosting"]
PALETTE = {"OLS": "#7f7f7f", "RandomForest": "#2E86AB",
           "GradientBoosting": "#E63946"}

# (1) R² across phases × models — held-out test
fig, ax = plt.subplots(figsize=(9, 4.5))
pivot = met_df.pivot(index="phase", columns="model",
                     values="r2_test")[MODELS]
pivot.plot(kind="bar", ax=ax,
           color=[PALETTE[m] for m in MODELS],
           edgecolor="black", width=0.75)
ax.axhline(0, color="black", linewidth=0.6)
ax.set_ylabel("R² on held-out test block")
ax.set_title("Predictive power of pandemic drivers on Δsentiment")
ax.set_xticklabels([p.split(":")[0] for p in pivot.index], rotation=0)
ax.legend(title="", fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/phase_nonlinear_r2.png", dpi=150)
plt.close()

# (2) Permutation importance heatmaps — one subplot per model
fig, axes = plt.subplots(len(MODELS), 1, figsize=(10, 10), sharex=True)
for ax, m in zip(axes, MODELS):
    sub  = imp_df[imp_df["model"] == m]
    grid = sub.pivot(index="feature", columns="phase",
                     values="perm_mean").loc[FEAT_LABELS]
    vmax = max(abs(grid.values.min()), abs(grid.values.max()))
    sns.heatmap(grid, annot=True, fmt=".3f", center=0,
                cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                cbar_kws={"label": "ΔR² if feature permuted"},
                linewidths=0.4, linecolor="white", ax=ax)
    ax.set_title(m, fontweight="bold")
    ax.set_xlabel(""); ax.set_ylabel("")
fig.suptitle("Permutation importance by phase — model-agnostic validation",
             fontweight="bold", fontsize=12)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/phase_nonlinear_importance_heatmap.png", dpi=150)
plt.close()

# (3) Grouped horizontal bars — one panel per phase, all models overlaid
fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
for ax, (phase, _, _) in zip(axes, PHASES):
    sub  = imp_df[imp_df["phase"] == phase]
    grid = sub.pivot(index="feature", columns="model",
                     values="perm_mean")[MODELS].loc[FEAT_LABELS]
    grid.plot(kind="barh", ax=ax,
              color=[PALETTE[m] for m in MODELS], edgecolor="black")
    ax.axvline(0, color="black", linewidth=0.6)
    ax.set_title(phase.split(":")[1].strip(), fontsize=10)
    ax.set_xlabel("Permutation ΔR²")
    ax.invert_yaxis()
    ax.legend(fontsize=8)
axes[0].set_ylabel("")
fig.suptitle("Model-agnostic feature importance by phase", fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTDIR}/phase_nonlinear_importance_bars.png", dpi=150)
plt.close()


# ----------------------------------------------------------------------
# AGREEMENT SUMMARY (leading feature per phase × model)
# ----------------------------------------------------------------------
top = (imp_df.sort_values("perm_mean", ascending=False)
             .groupby(["phase", "model"]).head(1)
             [["phase", "model", "feature", "perm_mean"]]
             .sort_values(["phase", "model"])
             .reset_index(drop=True))
top.to_csv(f"{OUTDIR}/phase_nonlinear_top_features.csv", index=False)

print("\n" + "=" * 70)
print("TOP FEATURE PER PHASE × MODEL  (agreement check)")
print("=" * 70)
print(top.to_string(index=False))

print(f"\nOutputs saved to: {OUTDIR}/")
for f in ["phase_nonlinear_metrics.csv",
          "phase_nonlinear_importance.csv",
          "phase_nonlinear_top_features.csv",
          "phase_nonlinear_r2.png",
          "phase_nonlinear_importance_heatmap.png",
          "phase_nonlinear_importance_bars.png"]:
    print(f"  {f}")
