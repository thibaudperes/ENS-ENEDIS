"""Description des trous (suites consécutives de valeurs manquantes) dans les courbes."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd

LENGTH_BINS: list[float] = [0, 1, 2, 6, 48, np.inf]
"""Bornes des classes de longueur de trou, en nombre de pas de 30 min."""

LENGTH_LABELS: list[str] = ["1", "2", "3-6", "7-48", ">48"]
"""Libellés des classes de longueur de trou."""


def hole_runs(mask: npt.ArrayLike) -> pd.DataFrame:
    """Liste tous les trous d'une matrice de masques (temps × courbes).

    Calcul vectorisé : on détecte les fronts montants et descendants du masque le long
    de l'axe temporel. Les trous sont triés par courbe, puis par début.

    Args:
        mask: Booléen (temps × courbes), vrai là où la valeur est manquante.

    Returns:
        DataFrame avec une ligne par trou et les colonnes ``curve`` (indice de colonne),
        ``start`` (indice du premier pas manquant) et ``length`` (nombre de pas).
    """
    m = np.asarray(mask, dtype=bool)
    if m.ndim == 1:
        m = m[:, None]
    padded = np.zeros((m.shape[0] + 2, m.shape[1]), dtype=np.int8)
    padded[1:-1] = m
    edges = np.diff(padded, axis=0).T  # (courbes × temps+1), parcouru courbe par courbe
    curve, start = np.nonzero(edges == 1)
    _, end = np.nonzero(edges == -1)
    return pd.DataFrame({"curve": curve, "start": start, "length": end - start})


def hole_length_per_point(mask: npt.ArrayLike) -> npt.NDArray[np.int64]:
    """Associe à chaque valeur manquante la longueur du trou auquel elle appartient.

    Args:
        mask: Booléen (temps × courbes), vrai là où la valeur est manquante.

    Returns:
        Entiers de même forme que ``mask`` : longueur du trou sur les positions manquantes,
        0 ailleurs.
    """
    m = np.asarray(mask, dtype=bool)
    lengths = hole_runs(m)["length"].to_numpy()
    out = np.zeros(m.shape, dtype=np.int64)
    # Parcourir m.T dans l'ordre (courbe, temps) énumère les points manquants trou par trou,
    # dans le même ordre que hole_runs : chaque longueur est donc répétée autant de fois.
    out.T[m.T] = np.repeat(lengths, lengths)
    return out


def length_bucket(lengths: npt.ArrayLike) -> pd.Categorical:
    """Range des longueurs de trou dans les classes ``LENGTH_LABELS``.

    Args:
        lengths: Longueurs de trou (en pas de 30 min), strictement positives.

    Returns:
        Catégories ordonnées ``"1"``, ``"2"``, ``"3-6"``, ``"7-48"``, ``">48"``.
    """
    return pd.cut(np.asarray(lengths), bins=LENGTH_BINS, labels=LENGTH_LABELS)
