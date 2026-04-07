import matplotlib.pyplot as plt
import pandas as pd

from analysis_common import DATA_OUTPUT_DIR, PLOTS_OUTPUT_DIR, ensure_output_dirs, load_inputs
from analysis_config import FINAL_COUNTRIES, PRE2022_CUTOFF

try:
    from sklearn.ensemble import RandomForestRegressor
except ImportError as exc:
    raise SystemExit(
        "scikit-learn is required for policyDecoderRF.py. "
        "Install it in the project virtualenv with: .venv/bin/pip install scikit-learn"
    ) from exc


FEATURE_COLUMNS = [
    "C1M_School closing",
    "C2M_Workplace closing",
    "C6M_Stay at home requirements",
    "C8EV_International travel controls",
]
TARGET_SHIFT_DAYS = 7


def build_policy_decoder_dataset():
    df_tweets, df_oxford, _ = load_inputs()

    df_policy = (
        df_oxford[df_oxford["CountryName"].isin(FINAL_COUNTRIES)]
        .groupby("date")[FEATURE_COLUMNS]
        .mean()
        .reset_index()
    )

    df_merged = pd.merge(df_tweets, df_policy, on="date", how="inner").sort_values("date")
    df_merged = df_merged[df_merged["date"] <= PRE2022_CUTOFF].copy()

    # Shift the target backward so today's policy mix predicts volatility seven days later.
    df_merged["volatility_plus_7d"] = df_merged["sentiment_volatility"].shift(-TARGET_SHIFT_DAYS)
    df_model = df_merged.dropna(subset=FEATURE_COLUMNS + ["volatility_plus_7d"]).copy()
    return df_model


def plot_feature_importance(importances):
    importance_series = pd.Series(importances, index=FEATURE_COLUMNS).sort_values(ascending=True)

    plt.figure(figsize=(10, 5))
    plt.barh(importance_series.index, importance_series.values, color="#2c7fb8")
    plt.xlabel("Random Forest Feature Importance")
    plt.title("Policy Decoder: Which Lockdown Rules Best Predicted Volatility?")
    plt.tight_layout()
    plt.savefig(PLOTS_OUTPUT_DIR / "policy_decoder_feature_importance.png", dpi=300)
    plt.close()


def main():
    ensure_output_dirs()
    df_model = build_policy_decoder_dataset()

    X = df_model[FEATURE_COLUMNS]
    y = df_model["volatility_plus_7d"]

    model = RandomForestRegressor(
        n_estimators=500,
        random_state=42,
        min_samples_leaf=5,
    )
    model.fit(X, y)

    importance_df = (
        pd.DataFrame(
            {
                "feature": FEATURE_COLUMNS,
                "importance": model.feature_importances_,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    importance_df.to_csv(DATA_OUTPUT_DIR / "policy_decoder_feature_importance.csv", index=False)
    plot_feature_importance(model.feature_importances_)

    print("🌲 Policy Decoder complete.")
    print(f"Rows used: {len(df_model)}")
    print("Feature importances:")
    for row in importance_df.itertuples(index=False):
        print(f"- {row.feature}: {row.importance:.4f}")


if __name__ == "__main__":
    main()
