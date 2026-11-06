"""Méthodes d'imputation de référence.

Toutes les méthodes ont la même signature, attendue par ``validation.evaluate`` :
``method(X_holed, reservoir) -> X_filled``, où ``X_holed`` contient les courbes à compléter
(NaN dans les trous), ``reservoir`` des courbes complètes utilisables comme voisines, et
``X_filled`` a les mêmes colonnes et le même index que ``X_holed``, sans NaN dans les trous.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import pandas as pd

Method = Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame]


def linear_interpolation(X_holed: pd.DataFrame, reservoir: pd.DataFrame) -> pd.DataFrame:
    """Benchmark officiel : interpolation linéaire, prolongée par constante aux bords.

    Args:
        X_holed: Courbes à compléter (temps × courbes), NaN dans les trous.
        reservoir: Ignoré (présent pour respecter la signature commune).

    Returns:
        Courbes complétées.
    """
    return X_holed.interpolate(limit_direction="both")


def nearest_neighbors(
    X_holed: pd.DataFrame, reservoir: pd.DataFrame, k: int = 5
) -> npt.NDArray[np.int64]:
    """Indices des k courbes du réservoir les plus proches de chaque courbe trouée.

    La distance est la MAE calculée sur les seuls points observés de la courbe trouée.
    Le calcul se fait courbe par courbe : la plus grosse matrice temporaire a la taille du
    réservoir (~70 Mo pour 16 000 courbes), quel que soit le nombre de courbes à remplir.

    Args:
        X_holed: Courbes à compléter (temps × n), NaN dans les trous.
        reservoir: Courbes complètes (temps × m), sans NaN.
        k: Nombre de voisins.

    Returns:
        Entiers (n × k) : indices de colonnes de ``reservoir``, du plus proche au moins proche.
        Une courbe sans aucun point observé reçoit des voisins arbitraires (à traiter par
        ``with_fallback``).
    """
    R = reservoir.to_numpy(dtype=np.float32)
    X = X_holed.to_numpy(dtype=np.float32)
    observed = ~np.isnan(X)
    out = np.empty((X.shape[1], k), dtype=np.int64)
    for i in range(X.shape[1]):
        w = observed[:, i].astype(np.float32)
        x = np.where(observed[:, i], X[:, i], 0.0).astype(np.float32)
        dist = w @ np.abs(R - x[:, None]) / max(w.sum(), 1.0)  # MAE sur les points observés
        nn = np.argpartition(dist, k)[:k]
        out[i] = nn[np.argsort(dist[nn])]
    return out


def knn_median(X_holed: pd.DataFrame, reservoir: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    """KNN naïf : chaque trou est rempli par la médiane des k courbes les plus proches.

    Args:
        X_holed: Courbes à compléter (temps × n), NaN dans les trous.
        reservoir: Courbes complètes (temps × m), sans NaN.
        k: Nombre de voisins.

    Returns:
        Courbes complétées (les points observés sont conservés tels quels).
    """
    nn = nearest_neighbors(X_holed, reservoir, k)
    R = reservoir.to_numpy(dtype=np.float32)
    pred = np.stack([np.median(R[:, row], axis=1) for row in nn], axis=1)
    return X_holed.fillna(pd.DataFrame(pred, index=X_holed.index, columns=X_holed.columns))


def scaled_median_fill(X_holed: pd.DataFrame, reservoir: pd.DataFrame) -> pd.DataFrame:
    """Secours pour courbes quasi vides : profil médian du réservoir, mis à l'échelle.

    Le profil est la médiane du réservoir à chaque pas de temps. Il est multiplié par le
    rapport (somme des points observés / somme du profil sur ces mêmes points), pour
    retrouver le niveau de la courbe. Sans aucun point observé, le profil est utilisé tel
    quel ; si tous les points observés sont nuls, la courbe est complétée par des zéros.

    Args:
        X_holed: Courbes à compléter (temps × n), NaN dans les trous.
        reservoir: Courbes complètes (temps × m), sans NaN.

    Returns:
        Courbes complétées (les points observés sont conservés tels quels).
    """
    profile = np.median(reservoir.to_numpy(dtype=np.float32), axis=1)
    X = X_holed.to_numpy(dtype=np.float64)
    observed = ~np.isnan(X)
    obs_sum = np.where(observed, X, 0.0).sum(axis=0)
    prof_sum = (observed * profile[:, None]).sum(axis=0)
    scale = np.where(observed.any(axis=0), obs_sum / np.maximum(prof_sum, 1e-9), 1.0)
    pred = profile[:, None] * scale[None, :]
    return X_holed.fillna(pd.DataFrame(pred, index=X_holed.index, columns=X_holed.columns))


def with_fallback(method: Method, min_observed: int = 1) -> Method:
    """Enveloppe une méthode : les courbes trop vides passent par ``scaled_median_fill``.

    Par défaut, seules les courbes sans aucun point observé sont concernées : la validation
    (notebook 02) montre que le KNN se dégrade bien et domine dès 12 points observés, et
    qu'avec un seul point toutes les méthodes se valent. Sans aucun point, ni l'interpolation
    ni le KNN ne peuvent rien faire, et le profil médian du réservoir est le choix naturel.

    Args:
        method: Méthode principale.
        min_observed: En dessous de ce nombre de points observés, la courbe est complétée
            par le secours.

    Returns:
        Une méthode de même signature.
    """

    def wrapped(X_holed: pd.DataFrame, reservoir: pd.DataFrame) -> pd.DataFrame:
        sparse = X_holed.notna().sum() < min_observed
        filled = method(X_holed.loc[:, ~sparse], reservoir)
        if sparse.any():
            rescue = scaled_median_fill(X_holed.loc[:, sparse], reservoir)
            filled = pd.concat([filled, rescue], axis=1)
        return filled[X_holed.columns]

    wrapped.__name__ = f"{getattr(method, '__name__', 'method')}+secours"
    return wrapped
