"""Fichier de soumission au format du challenge.

Le fichier attendu reprend exactement ``sample_submission.csv`` : colonne ``Horodate``
(index, mêmes horodates dans le même ordre, même écriture), puis ``holed_1`` …
``holed_1000`` dans le même ordre. Seuls les trous sont remplacés : les valeurs observées
restent identiques.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from enedis_imputation.data import PROCESSED_DIR, RAW_DIR

SAMPLE_PATH = RAW_DIR / "sample_submission.csv"
SUBMISSION_PATH = PROCESSED_DIR / "submission.csv"


def read_sample(path: Path = SAMPLE_PATH) -> pd.DataFrame:
    """Lit le fichier exemple en gardant l'écriture exacte des horodates (texte)."""
    return pd.read_csv(path, index_col=0, dtype={0: str})


def to_submission(filled: pd.DataFrame, sample: pd.DataFrame) -> pd.DataFrame:
    """Met des courbes complétées au format exact du fichier exemple.

    Args:
        filled: Courbes complétées (temps × ``holed_*``), index datetime.
        sample: Fichier exemple (``read_sample``).

    Returns:
        Un DataFrame avec l'index texte et l'ordre des colonnes du fichier exemple.
    """
    out = filled[sample.columns].copy()
    if not pd.to_datetime(sample.index).equals(pd.DatetimeIndex(out.index)):
        raise ValueError("Les horodates ne correspondent pas au fichier exemple.")
    out.index = sample.index
    return out


def check_submission(sub: pd.DataFrame, sample: pd.DataFrame, X_holed: pd.DataFrame) -> None:
    """Vérifie une soumission avant dépôt.

    Args:
        sub: Soumission (format ``to_submission``).
        sample: Fichier exemple.
        X_holed: Courbes trouées d'origine (test), pour vérifier les points observés.

    Raises:
        ValueError: Au premier écart trouvé (format, NaN, valeur négative, valeur
            observée modifiée).
    """
    if not sub.columns.equals(sample.columns):
        raise ValueError("Colonnes différentes du fichier exemple (noms ou ordre).")
    if not sub.index.equals(sample.index):
        raise ValueError("Index Horodate différent du fichier exemple.")
    values = sub.to_numpy(dtype=np.float64)
    if np.isnan(values).any():
        raise ValueError(f"{int(np.isnan(values).sum())} valeur(s) manquante(s).")
    if (values < 0).any():
        raise ValueError(f"{int((values < 0).sum())} valeur(s) négative(s).")
    original = X_holed[sample.columns].to_numpy(dtype=np.float64)
    observed = ~np.isnan(original)
    if not np.allclose(values[observed], original[observed], rtol=0, atol=1e-3):
        raise ValueError("Des valeurs observées ont été modifiées.")


def write_submission(sub: pd.DataFrame, path: Path = SUBMISSION_PATH) -> Path:
    """Écrit la soumission en CSV, comme le fichier exemple.

    Horodate en première colonne, 3 décimales, fins de ligne Windows (CRLF) comme
    ``sample_submission.csv``.
    """
    sub.to_csv(
        path,
        float_format="%.3f",
        index_label=sub.index.name or "Horodate",
        lineterminator="\r\n",
    )
    return path
