"""Executa o pipeline completo do projeto.

Uso:
    python -m src.run_pipeline --sample-size 250000

O parâmetro de amostragem deixa a execução viável em computadores pessoais. Para rodar
com a base completa, use --sample-size 0.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.config import DEFAULT_SAMPLE_SIZE, FIGURES_DIR, REPORTS_DIR
from src.eda import (
    airport_criticality,
    cancellation_summary,
    delay_cause_summary,
    plot_airline_delays,
    plot_airport_criticality,
    plot_delay_distribution,
    plot_time_patterns,
    route_summary,
)
from src.load_data import enrich_with_reference_tables, load_all, validate_tables
from src.preprocessing import missing_value_strategy, prepare_for_modeling
from src.supervised_modeling import (
    plot_confusion_and_roc,
    plot_feature_importance,
    random_forest_feature_importance,
    train_and_evaluate,
)
from src.unsupervised_modeling import (
    aggregate_airports_for_clustering,
    describe_clusters,
    elbow_silhouette,
    plot_clusters,
    plot_elbow,
    run_kmeans,
)
from src.utils import ensure_directories, save_dataframe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pipeline Tech Challenge Fase 3 - voos")
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE, help="Número de linhas de flights.csv; use 0 para base completa.")
    parser.add_argument("--no-model-save", action="store_true", help="Não salva modelos joblib.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_directories()
    nrows = None if args.sample_size == 0 else args.sample_size

    airlines, airports, flights = load_all(nrows=nrows)
    validation = validate_tables(airlines, airports, flights)
    for key in ["airlines_missing", "airports_missing", "flights_missing"]:
        save_dataframe(validation[key], REPORTS_DIR / f"{key}.csv")
    if "arrival_delay_description" in validation:
        validation["arrival_delay_description"].to_csv(REPORTS_DIR / "arrival_delay_distribution.csv")

    enriched = enrich_with_reference_tables(flights, airlines, airports)
    modeling = prepare_for_modeling(enriched)
    save_dataframe(missing_value_strategy(enriched), REPORTS_DIR / "missing_value_strategy.csv")

    plot_delay_distribution(modeling)
    plot_airline_delays(modeling)
    plot_time_patterns(modeling)
    plot_airport_criticality(modeling)
    save_dataframe(airport_criticality(modeling), REPORTS_DIR / "airport_criticality.csv")
    save_dataframe(route_summary(modeling), REPORTS_DIR / "route_summary.csv")
    save_dataframe(cancellation_summary(enriched), REPORTS_DIR / "cancellation_summary.csv")
    save_dataframe(delay_cause_summary(enriched), REPORTS_DIR / "delay_cause_summary.csv")

    metrics, models, split = train_and_evaluate(modeling, save_models=not args.no_model_save)
    save_dataframe(metrics, REPORTS_DIR / "supervised_metrics.csv")
    _, X_test, _, y_test = split
    best_model_name = metrics.iloc[0]["model"]
    if best_model_name == "baseline_majority":
        best_model_name = "random_forest"
    plot_confusion_and_roc(models[best_model_name], X_test, y_test, str(best_model_name))
    if "random_forest" in models:
        importance = random_forest_feature_importance(models["random_forest"])
        save_dataframe(importance, REPORTS_DIR / "random_forest_feature_importance.csv")
        plot_feature_importance(importance)

    airport_features = aggregate_airports_for_clustering(modeling)
    cluster_metrics = elbow_silhouette(airport_features)
    save_dataframe(cluster_metrics, REPORTS_DIR / "cluster_k_metrics.csv")
    plot_elbow(cluster_metrics)
    chosen_k = int(cluster_metrics.sort_values("silhouette", ascending=False).iloc[0]["k"]) if not cluster_metrics.empty else 4
    clustered, _, _, _, score = run_kmeans(airport_features, k=chosen_k)
    save_dataframe(clustered, REPORTS_DIR / "airport_clusters.csv")
    save_dataframe(describe_clusters(clustered), REPORTS_DIR / "cluster_profiles.csv")
    plot_clusters(clustered)

    report = REPORTS_DIR / "execucao_pipeline.md"
    report.write_text(
        "# Execução do pipeline\n\n"
        f"- Linhas carregadas em flights.csv: {flights.shape[0]:,}\n"
        f"- Linhas usadas na modelagem após filtros: {modeling.shape[0]:,}\n"
        f"- Figuras geradas em: `{FIGURES_DIR}`\n"
        f"- Melhor silhouette usado nos clusters: {score:.3f}\n"
        f"- Métricas supervisionadas salvas em: `{REPORTS_DIR / 'supervised_metrics.csv'}`\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
