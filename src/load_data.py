"""Carregamento e validação inicial das bases de voos."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import AIRLINES_PATH, AIRPORTS_PATH, FLIGHTS_PATH, DATA_DIR, DATA_DICTIONARY_GLOB
from src.utils import summarize_dataframe


def require_file(path: Path) -> None:
    """Garante que o arquivo existe e explica como corrigir quando não existir."""
    if not path.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {path}. Baixe os CSVs do desafio e coloque-os em data/ "
            "com os nomes airlines.csv, airports.csv e flights.csv."
        )


def load_airlines(path: Path = AIRLINES_PATH) -> pd.DataFrame:
    require_file(path)
    return pd.read_csv(path)


def load_airports(path: Path = AIRPORTS_PATH) -> pd.DataFrame:
    require_file(path)
    return pd.read_csv(path)


def load_flights(path: Path = FLIGHTS_PATH, nrows: int | None = None) -> pd.DataFrame:
    require_file(path)
    return pd.read_csv(path, low_memory=False, nrows=nrows)


def find_data_dictionary(data_dir: Path = DATA_DIR) -> list[Path]:
    """Localiza arquivos de dicionário de dados do flights, quando disponíveis."""
    return sorted(data_dir.glob(DATA_DICTIONARY_GLOB))


def load_all(nrows: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Carrega airlines, airports e flights."""
    return load_airlines(), load_airports(), load_flights(nrows=nrows)


def validate_tables(
    airlines: pd.DataFrame, airports: pd.DataFrame, flights: pd.DataFrame
) -> dict[str, object]:
    """Calcula validações solicitadas na etapa inicial do desafio."""
    validation = {
        "airlines_shape": airlines.shape,
        "airports_shape": airports.shape,
        "flights_shape": flights.shape,
        "airlines_columns": list(airlines.columns),
        "airports_columns": list(airports.columns),
        "flights_columns": list(flights.columns),
        "airlines_duplicates": int(airlines.duplicated().sum()),
        "airports_duplicates": int(airports.duplicated().sum()),
        "flights_duplicates": int(flights.duplicated().sum()),
        "airlines_missing": summarize_dataframe(airlines),
        "airports_missing": summarize_dataframe(airports),
        "flights_missing": summarize_dataframe(flights),
    }
    if "CANCELLED" in flights:
        validation["cancelled_rate"] = float(flights["CANCELLED"].mean())
    if "DIVERTED" in flights:
        validation["diverted_rate"] = float(flights["DIVERTED"].mean())
    if "ARRIVAL_DELAY" in flights:
        validation["arrival_delay_description"] = flights["ARRIVAL_DELAY"].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    return validation


def enrich_with_reference_tables(
    flights: pd.DataFrame, airlines: pd.DataFrame, airports: pd.DataFrame
) -> pd.DataFrame:
    """Faz merges sem remover linhas, preservando a cardinalidade de flights."""

    # Companhias aéreas
    airlines_ref = airlines.rename(
        columns={
            "IATA_CODE": "AIRLINE",
            "AIRLINE": "AIRLINE_NAME",
        }
    )

    enriched = flights.merge(
        airlines_ref,
        on="AIRLINE",
        how="left",
        validate="many_to_one",
    )

    # Aeroportos de origem
    origin_cols = {
        "IATA_CODE": "ORIGIN_AIRPORT",
        "AIRPORT": "ORIGIN_AIRPORT_NAME",
        "CITY": "ORIGIN_CITY",
        "STATE": "ORIGIN_STATE",
        "COUNTRY": "ORIGIN_COUNTRY",
        "LATITUDE": "ORIGIN_LATITUDE",
        "LONGITUDE": "ORIGIN_LONGITUDE",
    }

    origin = airports.rename(columns=origin_cols)
    origin = origin[list(origin_cols.values())]

    enriched = enriched.merge(
        origin,
        on="ORIGIN_AIRPORT",
        how="left",
        validate="many_to_one",
    )

    # Aeroportos de destino
    destination_cols = {
        "IATA_CODE": "DESTINATION_AIRPORT",
        "AIRPORT": "DESTINATION_AIRPORT_NAME",
        "CITY": "DESTINATION_CITY",
        "STATE": "DESTINATION_STATE",
        "COUNTRY": "DESTINATION_COUNTRY",
        "LATITUDE": "DESTINATION_LATITUDE",
        "LONGITUDE": "DESTINATION_LONGITUDE",
    }

    destination = airports.rename(columns=destination_cols)
    destination = destination[list(destination_cols.values())]

    enriched = enriched.merge(
        destination,
        on="DESTINATION_AIRPORT",
        how="left",
        validate="many_to_one",
    )

    return enriched
