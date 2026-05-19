"""Modelagem supervisionada para prever atraso relevante na chegada."""
from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    CATEGORICAL_FEATURES,
    FIGURES_DIR,
    MODEL_FEATURES,
    MODELS_DIR,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    TEST_SIZE,
)


def split_features_target(df: pd.DataFrame):
    features = [col for col in MODEL_FEATURES if col in df.columns]
    X = df[features].copy()
    y = df["IS_DELAYED"].astype(int)
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)


def make_preprocessor() -> ColumnTransformer:
    categorical = [col for col in CATEGORICAL_FEATURES if col in MODEL_FEATURES]
    numeric = [col for col in NUMERIC_FEATURES if col in MODEL_FEATURES]
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=20))]), categorical),
        ]
    )


def build_models() -> dict[str, Pipeline]:
    preprocessor = make_preprocessor()
    return {
        "baseline_majority": Pipeline([("preprocessor", preprocessor), ("model", DummyClassifier(strategy="most_frequent"))]),
        "logistic_regression": Pipeline([("preprocessor", make_preprocessor()), ("model", LogisticRegression(max_iter=500, class_weight="balanced", n_jobs=-1))]),
        "random_forest": Pipeline([("preprocessor", make_preprocessor()), ("model", RandomForestClassifier(n_estimators=120, max_depth=18, min_samples_leaf=20, class_weight="balanced_subsample", n_jobs=-1, random_state=RANDOM_STATE))]),
    }


def _predict_proba_or_score(model: Pipeline, X: pd.DataFrame):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def evaluate_model(name: str, model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float | str]:
    y_pred = model.predict(X_test)
    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }
    try:
        y_score = _predict_proba_or_score(model, X_test)
        metrics["roc_auc"] = roc_auc_score(y_test, y_score)
    except Exception:
        metrics["roc_auc"] = float("nan")
    return metrics


def train_and_evaluate(df: pd.DataFrame, save_models: bool = True) -> tuple[pd.DataFrame, dict[str, Pipeline], tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]]:
    X_train, X_test, y_train, y_test = split_features_target(df)
    trained: dict[str, Pipeline] = {}
    rows: list[dict[str, float | str]] = []
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for name, model in build_models().items():
        model.fit(X_train, y_train)
        trained[name] = model
        rows.append(evaluate_model(name, model, X_test, y_test))
        if save_models:
            joblib.dump(model, MODELS_DIR / f"{name}.joblib")
    metrics = pd.DataFrame(rows).sort_values("f1", ascending=False)
    return metrics, trained, (X_train, X_test, y_train, y_test)


def plot_confusion_and_roc(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, model_name: str) -> list[Path]:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["no prazo", "atrasado"])
    disp.plot(cmap="Blues")
    plt.title(f"Matriz de confusão - {model_name}")
    path = FIGURES_DIR / f"matriz_confusao_{model_name}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    paths.append(path)
    try:
        RocCurveDisplay.from_estimator(model, X_test, y_test)
        plt.title(f"Curva ROC - {model_name}")
        path = FIGURES_DIR / f"curva_roc_{model_name}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        paths.append(path)
    except Exception:
        pass
    return paths


def random_forest_feature_importance(model: Pipeline, top_n: int = 25) -> pd.DataFrame:
    preprocessor = model.named_steps["preprocessor"]
    estimator = model.named_steps["model"]
    names = preprocessor.get_feature_names_out()
    importances = estimator.feature_importances_
    return pd.DataFrame({"feature": names, "importance": importances}).sort_values("importance", ascending=False).head(top_n)


def plot_feature_importance(importance: pd.DataFrame) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 7))
    plt.barh(importance["feature"][::-1], importance["importance"][::-1], color="#4E79A7")
    plt.title("Importância de variáveis - Random Forest")
    plt.xlabel("Importância")
    plt.ylabel("Variável")
    path = FIGURES_DIR / "importancia_variaveis_random_forest.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path
