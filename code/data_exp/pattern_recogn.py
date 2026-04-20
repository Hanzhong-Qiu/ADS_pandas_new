import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — must be before pyplot import
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (classification_report, confusion_matrix,
                              ConfusionMatrixDisplay, roc_auc_score)

# ── 1. LOAD DATA ──────────────────────────────────────────────────────────────
df = pd.read_csv(Path(__file__).resolve().parents[1] / '.csv' / 'cleaned_sentiment_data.csv')
print("Shape:", df.shape)
print(df.head())

# ── 2. FEATURE SELECTION ──────────────────────────────────────────────────────
feature_cols = ["sentiment_mean", "sentiment_volatility", "tweet_volume"]  # <-- EDIT
X = df[feature_cols].dropna()

# ── 3. SCALE FEATURES ─────────────────────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── 4. ELBOW METHOD — find optimal k ─────────────────────────────────────────
inertias = []
sil_scores = []
K_range = range(2, 11)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, labels))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(K_range, inertias, marker="o", color="steelblue")
axes[0].set_title("Elbow Method")
axes[0].set_xlabel("Number of Clusters (k)")
axes[0].set_ylabel("Inertia")
axes[0].grid(True)
axes[1].plot(K_range, sil_scores, marker="o", color="darkorange")
axes[1].set_title("Silhouette Score vs k")
axes[1].set_xlabel("Number of Clusters (k)")
axes[1].set_ylabel("Silhouette Score")
axes[1].grid(True)
plt.tight_layout()
plt.savefig("elbow_silhouette.png", dpi=150)
plt.close()
print("Saved: elbow_silhouette.png")

# ── 5. FIT K-MEANS WITH CHOSEN k ─────────────────────────────────────────────
OPTIMAL_K = 4   # <-- EDIT based on elbow/silhouette plots

kmeans = KMeans(n_clusters=OPTIMAL_K, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_scaled)
df_clustered = X.copy()
df_clustered["cluster"] = cluster_labels

# ── 6. SILHOUETTE PLOT FOR CHOSEN k ──────────────────────────────────────────
sil_vals = silhouette_samples(X_scaled, cluster_labels)
avg_sil  = silhouette_score(X_scaled, cluster_labels)

fig, ax = plt.subplots(figsize=(10, 6))
y_lower = 10
colors = cm.nipy_spectral(np.linspace(0, 1, OPTIMAL_K))

for i in range(OPTIMAL_K):
    ith_sil = np.sort(sil_vals[cluster_labels == i])
    size_i  = ith_sil.shape[0]
    y_upper = y_lower + size_i
    ax.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_sil,
                     facecolor=colors[i], edgecolor=colors[i], alpha=0.7)
    ax.text(-0.05, y_lower + 0.5 * size_i, str(i))
    y_lower = y_upper + 10

ax.axvline(x=avg_sil, color="red", linestyle="--", label=f"Avg score: {avg_sil:.3f}")
ax.set_title(f"Silhouette Plot (k={OPTIMAL_K})")
ax.set_xlabel("Silhouette Coefficient")
ax.set_ylabel("Cluster")
ax.legend()
plt.tight_layout()
plt.savefig("silhouette_plot.png", dpi=150)
plt.close()
print("Saved: silhouette_plot.png")

# ── 7. PCA → 2D (pre-reduction before t-SNE) ─────────────────────────────────
pca = PCA(n_components=min(50, X_scaled.shape[1]), random_state=42)
X_pca = pca.fit_transform(X_scaled)
print(f"Variance explained by PCA: {pca.explained_variance_ratio_.sum():.2%}")

# ── 8. t-SNE ─────────────────────────────────────────────────────────────────
tsne = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate=200,
    max_iter=1000,      # <-- fixed: was n_iter (renamed in sklearn 1.5+)
    random_state=42
)
X_tsne = tsne.fit_transform(X_pca)

# ── 9. t-SNE PLOT COLOURED BY K-MEANS CLUSTER ────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 7))
scatter = ax.scatter(
    X_tsne[:, 0], X_tsne[:, 1],
    c=cluster_labels,
    cmap="nipy_spectral",
    alpha=0.6,
    s=15
)
plt.colorbar(scatter, ax=ax, label="Cluster")
ax.set_title(f"t-SNE coloured by K-Means clusters (k={OPTIMAL_K})")
ax.set_xlabel("t-SNE Dim 1")
ax.set_ylabel("t-SNE Dim 2")
plt.tight_layout()
plt.savefig("tsne_clusters.png", dpi=150)
plt.close()
print("Saved: tsne_clusters.png")

# ── 10. CLUSTER PROFILE SUMMARY ───────────────────────────────────────────────
print("\n── Cluster Means ──")
print(df_clustered.groupby("cluster").mean().round(3))
print("\n── Cluster Sizes ──")
print(df_clustered["cluster"].value_counts().sort_index())

# ── 11. FURTHER ANALYSIS: per-cluster distributions ──────────────────────────
fig, axes = plt.subplots(1, len(feature_cols), figsize=(5 * len(feature_cols), 5))
for idx, col in enumerate(feature_cols):
    for c in range(OPTIMAL_K):
        axes[idx].hist(df_clustered[df_clustered["cluster"] == c][col],
                       bins=30, alpha=0.5, label=f"Cluster {c}")
    axes[idx].set_title(f"Distribution: {col}")
    axes[idx].set_xlabel(col)
    axes[idx].set_ylabel("Count")
    axes[idx].legend()
plt.tight_layout()
plt.savefig("cluster_distributions.png", dpi=150)
plt.close()
print("Saved: cluster_distributions.png")

# ── 12. PAIRPLOT-STYLE SCATTER MATRIX ────────────────────────────────────────
n_cols = len(feature_cols)
fig, axes = plt.subplots(n_cols, n_cols, figsize=(4 * n_cols, 4 * n_cols))
cmap = plt.cm.get_cmap("nipy_spectral", OPTIMAL_K)
for i, col_i in enumerate(feature_cols):
    for j, col_j in enumerate(feature_cols):
        ax = axes[i][j]
        if i == j:
            for c in range(OPTIMAL_K):
                mask = df_clustered["cluster"] == c
                ax.hist(df_clustered[mask][col_i], bins=20, alpha=0.5,
                        color=cmap(c), label=f"C{c}")
        else:
            sc = ax.scatter(df_clustered[col_j], df_clustered[col_i],
                            c=df_clustered["cluster"], cmap="nipy_spectral",
                            alpha=0.4, s=10, vmin=0, vmax=OPTIMAL_K - 1)
        if i == n_cols - 1:
            ax.set_xlabel(col_j)
        if j == 0:
            ax.set_ylabel(col_i)
plt.suptitle("Scatter Matrix coloured by Cluster", y=1.01, fontsize=14)
plt.tight_layout()
plt.savefig("scatter_matrix.png", dpi=150)
plt.close()
print("Saved: scatter_matrix.png")

# ── 13. GRADIENT BOOSTING — predict cluster label ────────────────────────────
# Use K-Means cluster labels as the supervised target
y_gb = cluster_labels

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_gb, test_size=0.2, random_state=42, stratify=y_gb
)

gb = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    random_state=42
)
gb.fit(X_train, y_train)
y_pred = gb.predict(X_test)

print("\n── Gradient Boosting — Classification Report ──")
print(classification_report(y_test, y_pred, target_names=[f"Cluster {i}" for i in range(OPTIMAL_K)]))

# Cross-validated accuracy
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(gb, X_scaled, y_gb, cv=cv, scoring="accuracy")
print(f"5-Fold CV Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# ── 14. CONFUSION MATRIX ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6))
ConfusionMatrixDisplay(
    confusion_matrix(y_test, y_pred),
    display_labels=[f"C{i}" for i in range(OPTIMAL_K)]
).plot(ax=ax, colorbar=True, cmap="Blues")
ax.set_title("Gradient Boosting — Confusion Matrix")
plt.tight_layout()
plt.savefig("gb_confusion_matrix.png", dpi=150)
plt.close()
print("Saved: gb_confusion_matrix.png")

# ── 15. FEATURE IMPORTANCES ──────────────────────────────────────────────────
importances = gb.feature_importances_
sorted_idx = np.argsort(importances)[::-1]

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(range(len(feature_cols)),
       importances[sorted_idx],
       color="steelblue", edgecolor="white")
ax.set_xticks(range(len(feature_cols)))
ax.set_xticklabels([feature_cols[i] for i in sorted_idx], rotation=30, ha="right")
ax.set_title("Gradient Boosting — Feature Importances")
ax.set_ylabel("Importance")
ax.grid(axis="y", alpha=0.4)
plt.tight_layout()
plt.savefig("gb_feature_importances.png", dpi=150)
plt.close()
print("Saved: gb_feature_importances.png")

# ── 16. LEARNING CURVE (train vs test loss) ──────────────────────────────────
train_scores = gb.train_score_
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(train_scores, label="Train deviance", color="steelblue")
ax.set_title("Gradient Boosting — Training Deviance")
ax.set_xlabel("Boosting Iterations")
ax.set_ylabel("Deviance")
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.savefig("gb_learning_curve.png", dpi=150)
plt.close()
print("Saved: gb_learning_curve.png")

print("\n── All done. Saved charts: cluster_distributions, scatter_matrix,")
print("   gb_confusion_matrix, gb_feature_importances, gb_learning_curve ──")