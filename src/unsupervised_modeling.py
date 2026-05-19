"""Clusterização de aeroportos com KMeans e visualização por PCA."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.config import FIGURES_DIR, RANDOM_STATE

CLUSTER_FEATURES = [
    "total_flights",
    "avg_arrival_delay",
    "delayed_rate",
    "avg_distance",
    "cancelled_rate",
    "destinations_served",
    "morning_delay_rate",
    "afternoon_delay_rate",
    "night_delay_rate",
]


def aggregate_airports_for_clustering(df: pd.DataFrame) -> pd.DataFrame:
    base = df.copy()
    if "IS_DELAYED" not in base and "ARRIVAL_DELAY" in base:
        base["IS_DELAYED"] = (base["ARRIVAL_DELAY"] > 15).astype(int)
    grouped = base.groupby("ORIGIN_AIRPORT").agg(
        total_flights=("ORIGIN_AIRPORT", "size"),
        avg_arrival_delay=("ARRIVAL_DELAY", "mean"),
        delayed_rate=("IS_DELAYED", "mean"),
        avg_distance=("DISTANCE", "mean"),
        cancelled_rate=("CANCELLED", "mean"),
        destinations_served=("DESTINATION_AIRPORT", "nunique"),
    )
    period_rates = base.pivot_table(index="ORIGIN_AIRPORT", columns="DEPARTURE_PERIOD", values="IS_DELAYED", aggfunc="mean", observed=False)
    period_rates = period_rates.rename(columns={"manha": "morning_delay_rate", "tarde": "afternoon_delay_rate", "noite": "night_delay_rate"})
    airport_features = grouped.join(period_rates, how="left").reset_index()
    for col in CLUSTER_FEATURES:
        if col not in airport_features:
            airport_features[col] = 0.0
    return airport_features.fillna(0)


def run_kmeans(airport_features: pd.DataFrame, k: int = 4) -> tuple[pd.DataFrame, KMeans, StandardScaler, PCA, float]:
    X = airport_features[CLUSTER_FEATURES]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20)
    labels = model.fit_predict(X_scaled)
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(X_scaled)
    output = airport_features.copy()
    output["cluster"] = labels
    output["pca_1"] = coords[:, 0]
    output["pca_2"] = coords[:, 1]
    score = silhouette_score(X_scaled, labels) if len(set(labels)) > 1 else float("nan")
    return output, model, scaler, pca, score


def elbow_silhouette(airport_features: pd.DataFrame, k_values: range = range(2, 9)) -> pd.DataFrame:
    X_scaled = StandardScaler().fit_transform(airport_features[CLUSTER_FEATURES])
    rows = []
    for k in k_values:
        if k >= len(airport_features):
            continue
        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20)
        labels = model.fit_predict(X_scaled)
        rows.append({"k": k, "inertia": model.inertia_, "silhouette": silhouette_score(X_scaled, labels)})
    return pd.DataFrame(rows)


def plot_elbow(metrics: pd.DataFrame) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(9, 5))
    sns.lineplot(data=metrics, x="k", y="inertia", marker="o", ax=ax1, color="#4E79A7")
    ax1.set_ylabel("Inércia")
    ax2 = ax1.twinx()
    sns.lineplot(data=metrics, x="k", y="silhouette", marker="s", ax=ax2, color="#E15759")
    ax2.set_ylabel("Silhouette")
    ax1.set_title("Método do cotovelo e silhouette para KMeans")
    path = FIGURES_DIR / "kmeans_cotovelo_silhouette.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_clusters(clustered: pd.DataFrame) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=clustered, x="pca_1", y="pca_2", hue="cluster", size="total_flights", palette="tab10", sizes=(40, 350))
    for _, row in clustered.sort_values("total_flights", ascending=False).head(15).iterrows():
        plt.text(row["pca_1"], row["pca_2"], row["ORIGIN_AIRPORT"], fontsize=8)
    plt.title("Clusters de aeroportos visualizados com PCA")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    path = FIGURES_DIR / "clusters_aeroportos_pca.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def describe_clusters(clustered: pd.DataFrame) -> pd.DataFrame:
    return clustered.groupby("cluster")[CLUSTER_FEATURES].mean().round(3).reset_index()
