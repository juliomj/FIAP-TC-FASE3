"""Preparação e limpeza da base para análise e modelagem."""
from __future__ import annotations

import pandas as pd

from src.config import MODEL_FEATURES
from src.features import create_modeling_features


def remove_cancelled_diverted(df: pd.DataFrame) -> pd.DataFrame:
    """Remove voos cancelados/desviados para prever atraso de chegada observável."""
    mask = pd.Series(True, index=df.index)
    if "CANCELLED" in df:
        mask &= df["CANCELLED"].fillna(0).eq(0)
    if "DIVERTED" in df:
        mask &= df["DIVERTED"].fillna(0).eq(0)
    output = df.loc[mask].copy()
    if "ARRIVAL_DELAY" in output:
        output = output.dropna(subset=["ARRIVAL_DELAY"])
    return output


def prepare_for_modeling(df: pd.DataFrame) -> pd.DataFrame:
    """Limpa voos inválidos, cria features e remove registros sem dados essenciais."""
    output = remove_cancelled_diverted(df)
    output = create_modeling_features(output)
    essential = [col for col in MODEL_FEATURES + ["IS_DELAYED"] if col in output.columns]
    output = output.dropna(subset=essential)
    return output


def missing_value_strategy(df: pd.DataFrame) -> pd.DataFrame:
    """Resume decisão de tratamento de nulos por coluna."""
    missing = df.isna().mean().sort_values(ascending=False).rename("missing_pct").reset_index()
    missing = missing.rename(columns={"index": "column"})
    missing["strategy"] = "manter para EDA; imputar no pipeline ou remover se alvo/feature essencial"
    post_event = {
        "AIR_SYSTEM_DELAY",
        "SECURITY_DELAY",
        "AIRLINE_DELAY",
        "LATE_AIRCRAFT_DELAY",
        "WEATHER_DELAY",
        "CANCELLATION_REASON",
    }
    missing.loc[missing["column"].isin(post_event), "strategy"] = "usar apenas em EDA; não usar na previsão pré-voo"
    missing.loc[missing["column"].isin(["ARRIVAL_DELAY"]), "strategy"] = "alvo/base da variável IS_DELAYED; remover nulos na modelagem"
    return missing
