"""Feature engineering com cuidado para evitar vazamento de dados."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import DELAY_THRESHOLD_MINUTES


def hhmm_to_minutes(value: object) -> float:
    """Converte horário HHMM em minutos desde meia-noite; retorna NaN para valores inválidos."""
    if pd.isna(value):
        return np.nan
    try:
        hhmm = int(float(value))
    except (TypeError, ValueError):
        return np.nan
    if hhmm == 2400:
        hhmm = 0
    hour, minute = divmod(hhmm, 100)
    if hour < 0 or hour > 23 or minute < 0 or minute > 59:
        return np.nan
    return float(hour * 60 + minute)


def period_from_minutes(minutes: object) -> str:
    """Classifica o horário em madrugada, manhã, tarde ou noite."""
    if pd.isna(minutes):
        return "desconhecido"
    minutes = int(minutes)
    if 0 <= minutes < 360:
        return "madrugada"
    if 360 <= minutes < 720:
        return "manha"
    if 720 <= minutes < 1080:
        return "tarde"
    return "noite"


def add_target(df: pd.DataFrame, threshold: int = DELAY_THRESHOLD_MINUTES) -> pd.DataFrame:
    """Cria IS_DELAYED usando ARRIVAL_DELAY > threshold minutos."""
    output = df.copy()
    output["IS_DELAYED"] = (output["ARRIVAL_DELAY"] > threshold).astype(int)
    return output


def add_route_and_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cria rota, horários em minutos, período do dia e categoria de distância."""
    output = df.copy()
    output["ROUTE"] = output["ORIGIN_AIRPORT"].astype(str) + "-" + output["DESTINATION_AIRPORT"].astype(str)
    output["SCHEDULED_DEPARTURE_MINUTES"] = output["SCHEDULED_DEPARTURE"].map(hhmm_to_minutes)
    output["SCHEDULED_ARRIVAL_MINUTES"] = output["SCHEDULED_ARRIVAL"].map(hhmm_to_minutes)
    output["DEPARTURE_PERIOD"] = output["SCHEDULED_DEPARTURE_MINUTES"].map(period_from_minutes)
    bins = [-np.inf, 500, 1500, np.inf]
    labels = ["curta", "media", "longa"]
    output["DISTANCE_CATEGORY"] = pd.cut(output["DISTANCE"], bins=bins, labels=labels).astype(str)
    return output


def add_busy_route_flag(df: pd.DataFrame, quantile: float = 0.75) -> pd.DataFrame:
    """Marca rotas com volume acima do quantil informado, calculado sem usar o alvo."""
    output = df.copy()
    route_counts = output["ROUTE"].value_counts()
    cutoff = route_counts.quantile(quantile) if not route_counts.empty else 0
    busy_routes = set(route_counts[route_counts >= cutoff].index)
    output["IS_BUSY_ROUTE"] = np.where(output["ROUTE"].isin(busy_routes), "sim", "nao")
    return output


def create_modeling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica as transformações necessárias para EDA e modelagem."""
    output = add_route_and_time_features(df)
    output = add_target(output)
    output = add_busy_route_flag(output)
    return output
