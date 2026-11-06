"""Chargement des données du challenge, avec cache binaire, et métrique d'évaluation.

Les CSV bruts (``data/raw/``) sont lus une seule fois, en float32, puis mis en cache dans
``data/processed/`` sous forme de trois fichiers par jeu :

- ``<nom>_values.npy`` : matrice (temps × courbes) en float32 ;
- ``<nom>_index.npy`` : horodates en ``datetime64[ns]`` ;
- ``<nom>_columns.json`` : noms des colonnes.

Ce format est choisi pour la vitesse : sur ``X_train`` (1057 × 21 000), ``np.load`` prend
~0,05 s contre ~4,4 s pour parquet, dont le coût croît avec le nombre de colonnes.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
RAW_DIR: Path = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR: Path = PROJECT_ROOT / "data" / "processed"

INDEX_COL: str = "Horodate"
HOLED_PREFIX: str = "holed_"


def _read_csv_float32(path: Path) -> pd.DataFrame:
    """Lit un CSV du challenge en float32, avec un index datetime.

    Le moteur C de pandas avec un ``dtype`` float32 par défaut est ~5 fois plus rapide
    que le moteur pyarrow sur ces fichiers très larges, et n'alloue jamais de float64.

    Args:
        path: Chemin du CSV.

    Returns:
        DataFrame (temps × courbes) en float32, indexé par ``Horodate``.
    """
    dtypes: defaultdict[str, type] = defaultdict(lambda: np.float32)
    dtypes[INDEX_COL] = str
    df = pd.read_csv(path, dtype=dtypes, index_col=INDEX_COL)
    df.index = pd.DatetimeIndex(pd.to_datetime(df.index), name=INDEX_COL)
    return df


def _cache_paths(name: str) -> tuple[Path, Path, Path]:
    """Renvoie les chemins (valeurs, index, colonnes) du cache d'un jeu de données."""
    return (
        PROCESSED_DIR / f"{name}_values.npy",
        PROCESSED_DIR / f"{name}_index.npy",
        PROCESSED_DIR / f"{name}_columns.json",
    )


def load_frame(name: str, *, refresh: bool = False) -> pd.DataFrame:
    """Charge un fichier du challenge, depuis le cache s'il existe, sinon depuis le CSV.

    Args:
        name: Nom du fichier sans extension (``"X_train"``, ``"y_train"``, ``"X_test"``…).
        refresh: Si vrai, relit le CSV et réécrit le cache.

    Returns:
        DataFrame (temps × courbes) en float32, indexé par ``Horodate``.
    """
    values_path, index_path, columns_path = _cache_paths(name)

    if refresh or not values_path.exists():
        df = _read_csv_float32(RAW_DIR / f"{name}.csv")
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        np.save(values_path, df.to_numpy(dtype=np.float32))
        np.save(index_path, df.index.to_numpy(dtype="datetime64[ns]"))
        columns_path.write_text(json.dumps(df.columns.tolist()))
        return df

    values = np.load(values_path)
    index = pd.DatetimeIndex(np.load(index_path), name=INDEX_COL)
    columns = json.loads(columns_path.read_text())
    return pd.DataFrame(values, index=index, columns=columns, copy=False)


def split_holed(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sépare les courbes complètes des courbes trouées (colonnes ``holed_*``).

    Args:
        df: DataFrame contenant les deux types de courbes.

    Returns:
        ``(X_full, X_holed)``.
    """
    is_holed = df.columns.str.startswith(HOLED_PREFIX)
    return df.loc[:, ~is_holed], df.loc[:, is_holed]


def load_train(*, refresh: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Charge le jeu d'entraînement.

    Args:
        refresh: Si vrai, relit les CSV et réécrit le cache.

    Returns:
        ``(X_full, X_holed, y)`` : les 20 000 courbes complètes, les 1 000 courbes trouées,
        et leurs vraies valeurs, avec les colonnes de ``y`` alignées sur celles de ``X_holed``.
    """
    X_full, X_holed = split_holed(load_frame("X_train", refresh=refresh))
    y = load_frame("y_train", refresh=refresh).loc[X_holed.index, X_holed.columns]
    return X_full, X_holed, y


def load_test(*, refresh: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Charge le jeu de test.

    Args:
        refresh: Si vrai, relit le CSV et réécrit le cache.

    Returns:
        ``(X_full, X_holed)`` : les 37 140 courbes complètes et les 1 000 courbes trouées.
    """
    return split_holed(load_frame("X_test", refresh=refresh))


def mae_on_holes(
    pred: pd.DataFrame | npt.ArrayLike,
    y: pd.DataFrame | npt.ArrayLike,
    mask: pd.DataFrame | npt.ArrayLike,
) -> float:
    """MAE calculée uniquement sur les valeurs manquantes, en excluant les NaN de ``y``.

    Args:
        pred: Prédictions, de même forme que ``y``.
        y: Vraies valeurs ; ses NaN sont exclus du calcul.
        mask: Booléen, vrai là où la valeur était manquante (typiquement ``X_holed.isna()``).

    Returns:
        L'erreur absolue moyenne sur les positions évaluées.

    Raises:
        ValueError: Si les formes diffèrent, si les colonnes de DataFrames ne sont pas
            alignées, si aucune position n'est évaluable, ou si ``pred`` contient des NaN
            sur des positions évaluées.
    """
    frames = [a for a in (pred, y, mask) if isinstance(a, pd.DataFrame)]
    if any(not f.columns.equals(frames[0].columns) for f in frames[1:]):
        raise ValueError("Les colonnes de pred, y et mask ne sont pas alignées.")

    pred_arr = np.asarray(pred, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    mask_arr = np.asarray(mask, dtype=bool)
    if not pred_arr.shape == y_arr.shape == mask_arr.shape:
        raise ValueError(
            f"Formes incompatibles : pred {pred_arr.shape}, y {y_arr.shape}, mask {mask_arr.shape}."
        )

    evaluated = mask_arr & ~np.isnan(y_arr)
    if not evaluated.any():
        raise ValueError("Aucune position à évaluer (masque vide ou y entièrement NaN).")
    errors = np.abs(pred_arr[evaluated] - y_arr[evaluated])
    if np.isnan(errors).any():
        raise ValueError("pred contient des NaN sur des positions évaluées.")
    return float(errors.mean())
