"""Funções utilitárias para diretórios, persistência e relatórios."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import FIGURES_DIR, MODELS_DIR, REPORTS_DIR


def ensure_directories() -> None:
    """Cria diretórios de saída usados pelo projeto."""
    for path in (FIGURES_DIR, MODELS_DIR, REPORTS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def save_dataframe(df: pd.DataFrame, path: Path, index: bool = False) -> None:
    """Salva um DataFrame em CSV criando o diretório de destino quando necessário."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)


def save_json(payload: dict[str, Any], path: Path) -> None:
    """Salva dicionários de métricas em JSON legível."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def summarize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Retorna resumo com tipo, nulos e percentual de nulos por coluna."""
    return pd.DataFrame(
        {
            "column": df.columns,
            "dtype": [str(dtype) for dtype in df.dtypes],
            "missing": df.isna().sum().to_numpy(),
            "missing_pct": (df.isna().mean() * 100).round(3).to_numpy(),
            "unique_values": df.nunique(dropna=True).to_numpy(),
        }
    )
