"""Funções de EDA e geração de gráficos."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config import FIGURES_DIR, MIN_GROUP_FLIGHTS

sns.set_theme(style="whitegrid")


def save_current_figure(filename: str) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / filename
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_delay_distribution(df: pd.DataFrame) -> Path:
    data = df["ARRIVAL_DELAY"].dropna().clip(lower=-60, upper=240)
    plt.figure(figsize=(10, 5))
    sns.histplot(data, bins=80, kde=True, color="#376795")
    plt.axvline(15, color="red", linestyle="--", label="limiar 15 min")
    plt.title("Distribuição do atraso de chegada (limitada entre -60 e 240 min)")
    plt.xlabel("Atraso de chegada em minutos")
    plt.ylabel("Quantidade de voos")
    plt.legend()
    return save_current_figure("distribuicao_atraso_chegada.png")


def airline_delay_summary(df: pd.DataFrame, min_flights: int = MIN_GROUP_FLIGHTS) -> pd.DataFrame:
    summary = df.groupby("AIRLINE").agg(
        flights=("AIRLINE", "size"),
        avg_arrival_delay=("ARRIVAL_DELAY", "mean"),
        delayed_rate=("IS_DELAYED", "mean"),
    ).reset_index()
    return summary.query("flights >= @min_flights").sort_values("avg_arrival_delay", ascending=False)


def plot_airline_delays(df: pd.DataFrame) -> Path:
    summary = airline_delay_summary(df).head(15)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=summary, y="AIRLINE", x="avg_arrival_delay", color="#59A14F")
    plt.title("Companhias com maior atraso médio de chegada")
    plt.xlabel("Atraso médio (min)")
    plt.ylabel("Companhia")
    return save_current_figure("atraso_medio_companhia.png")


def plot_time_patterns(df: pd.DataFrame) -> list[Path]:
    paths: list[Path] = []
    for col, filename, title in [
        ("MONTH", "atraso_por_mes.png", "Taxa de voos atrasados por mês"),
        ("DAY_OF_WEEK", "atraso_por_dia_semana.png", "Taxa de voos atrasados por dia da semana"),
        ("DEPARTURE_PERIOD", "atraso_por_periodo_dia.png", "Taxa de voos atrasados por período do dia"),
    ]:
        summary = df.groupby(col, observed=False)["IS_DELAYED"].mean().reset_index()
        plt.figure(figsize=(9, 5))
        sns.barplot(data=summary, x=col, y="IS_DELAYED", color="#F28E2B")
        plt.title(title)
        plt.xlabel(col)
        plt.ylabel("Proporção de atrasos > 15 min")
        paths.append(save_current_figure(filename))
    return paths


def airport_criticality(df: pd.DataFrame, min_flights: int = MIN_GROUP_FLIGHTS) -> pd.DataFrame:
    summary = df.groupby("ORIGIN_AIRPORT").agg(
        flights=("ORIGIN_AIRPORT", "size"),
        avg_arrival_delay=("ARRIVAL_DELAY", "mean"),
        delayed_rate=("IS_DELAYED", "mean"),
        destinations=("DESTINATION_AIRPORT", "nunique"),
    ).reset_index()
    filtered = summary.query("flights >= @min_flights").copy()
    filtered["criticality_score"] = filtered["flights"].rank(pct=True) * filtered["delayed_rate"].rank(pct=True)
    return filtered.sort_values("criticality_score", ascending=False)


def plot_airport_criticality(df: pd.DataFrame) -> Path:
    summary = airport_criticality(df).head(20)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=summary, x="flights", y="delayed_rate", size="avg_arrival_delay", hue="avg_arrival_delay", palette="Reds", sizes=(50, 350))
    for _, row in summary.head(10).iterrows():
        plt.text(row["flights"], row["delayed_rate"], row["ORIGIN_AIRPORT"], fontsize=8)
    plt.title("Aeroportos críticos: volume versus taxa de atraso")
    plt.xlabel("Quantidade de voos")
    plt.ylabel("Taxa de atraso")
    return save_current_figure("aeroportos_criticos.png")


def route_summary(df: pd.DataFrame, min_flights: int = MIN_GROUP_FLIGHTS) -> pd.DataFrame:
    summary = df.groupby("ROUTE").agg(
        flights=("ROUTE", "size"),
        avg_arrival_delay=("ARRIVAL_DELAY", "mean"),
        delayed_rate=("IS_DELAYED", "mean"),
        avg_distance=("DISTANCE", "mean"),
    ).reset_index()
    return summary.query("flights >= @min_flights").sort_values("delayed_rate", ascending=False)


def cancellation_summary(df: pd.DataFrame) -> pd.DataFrame:
    if "CANCELLATION_REASON" not in df or "CANCELLED" not in df:
        return pd.DataFrame()
    cancelled = df[df["CANCELLED"].fillna(0).eq(1)]
    return cancelled["CANCELLATION_REASON"].value_counts(dropna=False).rename_axis("reason").reset_index(name="flights")


def delay_cause_summary(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["AIR_SYSTEM_DELAY", "SECURITY_DELAY", "AIRLINE_DELAY", "LATE_AIRCRAFT_DELAY", "WEATHER_DELAY"]
    existing = [col for col in cols if col in df]
    if not existing:
        return pd.DataFrame()
    return df[existing].mean().sort_values(ascending=False).rename("avg_minutes").reset_index().rename(columns={"index": "cause"})
