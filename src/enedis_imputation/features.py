"""Variables décrivant chaque point manquant, pour le modèle de correction (LightGBM).

Une ligne = un point manquant (une courbe, un instant). Toutes les variables ne dépendent
que des points observés de la courbe (``X_holed``) et du réservoir de voisines : jamais
des vraies valeurs. Elles se calculent par blocs de courbes pour tenir en mémoire.

Familles de variables :

- **prédictions de base** : KNN ancré, KNN non ancré (voisines mises à l'échelle),
  interpolation linéaire ;
- **désaccord des voisines** : écart-type et écart interquartile des K voisines mises à
  l'échelle à cet instant, distance moyenne aux voisines ;
- **position** : longueur du trou, distances aux bords, position relative dans le trou ;
- **calendrier** : heure (sinus, cosinus), jour de semaine, week-end ;
- **mémoire de la courbe** : valeurs observées à t ± 48 (veille, lendemain) et t ± 336
  (semaine d'avant, d'après), médiane de la courbe à ce créneau horaire ;
- **contexte** : valeurs observées aux bords du trou, niveau, écart-type et part de zéros
  de la courbe.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd

from enedis_imputation.holes import hole_length_per_point
from enedis_imputation.knn import (
    BEST_PREDICTOR,
    Neighbors,
    Predictor,
    aggregate_neighbors,
    anchor_residual,
    scale_neighbors,
)

FEATURES: list[str] = [
    # prédictions de base
    "knn_anchored",
    "knn_scaled",
    "interp",
    # désaccord des voisines
    "nb_std",
    "nb_iqr",
    "nb_dist",
    # position dans le trou
    "hole_len",
    "dist_left",
    "dist_right",
    "rel_pos",
    # calendrier
    "hour_sin",
    "hour_cos",
    "dow",
    "weekend",
    # mémoire de la courbe
    "lag_m48",
    "lag_p48",
    "lag_m336",
    "lag_p336",
    "slot_median",
    # contexte
    "left_val",
    "right_val",
    "curve_mean",
    "curve_std",
    "curve_zero_share",
]
"""Noms des variables, dans l'ordre des colonnes."""

_BLOCK = 256


def _shift(values: np.ndarray, lag: int) -> np.ndarray:
    """Valeur à l'instant t − lag (NaN hors de la période), pour chaque courbe."""
    out = np.full_like(values, np.nan)
    if lag > 0:
        out[lag:] = values[:-lag]
    else:
        out[:lag] = values[-lag:]
    return out


def _edges(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Indice du dernier point observé avant t et du premier après t (NaN si aucun)."""
    t = np.arange(mask.shape[0], dtype=np.float64)[:, None] * np.ones(mask.shape)
    observed_t = np.where(mask, np.nan, t)
    last = pd.DataFrame(observed_t).ffill().to_numpy()
    nxt = pd.DataFrame(observed_t).bfill().to_numpy()
    return last, nxt


def _block_features(
    X: np.ndarray,
    R: np.ndarray,
    index: pd.DatetimeIndex,
    nb_index: np.ndarray,
    nb_dist: np.ndarray,
    weights: np.ndarray,
    predictor: Predictor,
) -> dict[str, np.ndarray]:
    """Toutes les variables d'un bloc de courbes, sous forme de matrices (temps × courbes)."""
    T, B = X.shape
    mask = np.isnan(X)
    Xz = np.where(mask, 0.0, X)
    k = predictor.k
    feats: dict[str, np.ndarray] = {}

    # Prédictions de base et désaccord des voisines
    scaled = scale_neighbors(
        R[:, nb_index[:, :k]].astype(np.float64), Xz, weights, predictor.scaling
    )
    agg = aggregate_neighbors(scaled, nb_dist[:, :k], predictor.aggregate)
    anchored = agg + anchor_residual(
        Xz - agg, mask, predictor.anchor_points, predictor.anchor_decay
    )
    feats["knn_anchored"] = np.clip(anchored, 0.0, None)
    feats["knn_scaled"] = np.clip(agg, 0.0, None)
    feats["interp"] = pd.DataFrame(X).interpolate(limit_direction="both").to_numpy()
    feats["nb_std"] = scaled.std(axis=2)
    q25, q75 = np.percentile(scaled, [25, 75], axis=2)
    feats["nb_iqr"] = q75 - q25
    feats["nb_dist"] = np.broadcast_to(nb_dist[:, :k].mean(axis=1), (T, B))

    # Position dans le trou
    t = np.arange(T, dtype=np.float64)[:, None]
    last, nxt = _edges(mask)
    length = hole_length_per_point(mask).astype(np.float64)
    feats["hole_len"] = length
    feats["dist_left"] = t - last
    feats["dist_right"] = nxt - t
    start = np.where(np.isnan(last), 0.0, last + 1)
    feats["rel_pos"] = (t - start + 1) / (length + 1)

    # Calendrier
    hour = (index.hour + index.minute / 60).to_numpy()[:, None]
    feats["hour_sin"] = np.broadcast_to(np.sin(2 * np.pi * hour / 24), (T, B))
    feats["hour_cos"] = np.broadcast_to(np.cos(2 * np.pi * hour / 24), (T, B))
    dow = index.day_of_week.to_numpy()[:, None].astype(np.float64)
    feats["dow"] = np.broadcast_to(dow, (T, B))
    feats["weekend"] = np.broadcast_to((dow >= 5).astype(np.float64), (T, B))

    # Mémoire de la courbe (NaN si l'instant de référence est lui-même manquant)
    for name, lag in [("lag_m48", 48), ("lag_p48", -48), ("lag_m336", 336), ("lag_p336", -336)]:
        feats[name] = _shift(X, lag)
    slot = (index.hour * 2 + index.minute // 30).to_numpy()
    slot_med = np.full((48, B), np.nan)
    for s in range(48):
        rows = X[slot == s]
        has = ~np.isnan(rows).all(axis=0)
        slot_med[s, has] = np.nanmedian(rows[:, has], axis=0)
    feats["slot_median"] = slot_med[slot]

    # Contexte
    cols = np.arange(B)[None, :] * np.ones((T, 1), dtype=np.int64)
    feats["left_val"] = np.where(
        np.isnan(last), np.nan, X[np.nan_to_num(last).astype(np.int64), cols]
    )
    feats["right_val"] = np.where(
        np.isnan(nxt), np.nan, X[np.nan_to_num(nxt).astype(np.int64), cols]
    )
    observed = ~mask
    n_obs = np.maximum(observed.sum(axis=0), 1)
    mean = Xz.sum(axis=0) / n_obs
    var = (np.where(observed, (X - mean) ** 2, 0.0)).sum(axis=0) / n_obs
    zeros = (observed & (Xz == 0)).sum(axis=0) / n_obs
    feats["curve_mean"] = np.broadcast_to(mean, (T, B))
    feats["curve_std"] = np.broadcast_to(np.sqrt(var), (T, B))
    feats["curve_zero_share"] = np.broadcast_to(zeros, (T, B))
    return feats


def build_features(
    X_holed: pd.DataFrame,
    reservoir: pd.DataFrame,
    neighbors: Neighbors,
    predictor: Predictor = BEST_PREDICTOR,
) -> tuple[pd.DataFrame, npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Calcule les variables de tous les points manquants.

    Args:
        X_holed: Courbes à compléter (temps × courbes), NaN dans les trous.
        reservoir: Le réservoir utilisé par ``knn.search`` (vérifié).
        neighbors: Voisines trouvées par ``knn.search`` pour ces courbes.
        predictor: Configuration du KNN ancré dont on tire les prédictions de base.

    Returns:
        ``(variables, t, c)`` : un DataFrame float32 avec une ligne par point manquant
        (colonnes ``FEATURES``), et les indices de temps et de courbe de chaque ligne.
        Les lignes sont rangées courbe par courbe, puis dans l'ordre du temps.

    Raises:
        ValueError: Si le réservoir n'est pas celui de la recherche des voisines.
    """
    if not reservoir.columns.equals(neighbors.reservoir_columns):
        raise ValueError("Le réservoir doit être celui utilisé par search().")
    R = reservoir.to_numpy(dtype=np.float32)
    X_all = X_holed.to_numpy(dtype=np.float64)
    parts, ts, cs = [], [], []
    for b0 in range(0, X_all.shape[1], _BLOCK):
        b1 = min(b0 + _BLOCK, X_all.shape[1])
        X = X_all[:, b0:b1]
        feats = _block_features(
            X,
            R,
            X_holed.index,
            neighbors.index[b0:b1],
            neighbors.distance[b0:b1],
            neighbors.weights[:, b0:b1].astype(np.float64),
            predictor,
        )
        mask_T = np.isnan(X).T  # parcours courbe par courbe, puis dans le temps
        c_loc, t_loc = np.nonzero(mask_T)
        parts.append(np.column_stack([feats[f].T[mask_T] for f in FEATURES]).astype(np.float32))
        ts.append(t_loc)
        cs.append(c_loc + b0)
    table = pd.DataFrame(np.vstack(parts), columns=FEATURES)
    return table, np.concatenate(ts), np.concatenate(cs)


def curve_features(
    X_holed: pd.DataFrame,
    reservoir: pd.DataFrame,
    neighbors: Neighbors,
    names: list[str],
    predictor: Predictor = BEST_PREDICTOR,
) -> dict[str, npt.NDArray[np.float32]]:
    """Calcule des variables à chaque pas de temps (et pas seulement aux points manquants).

    Sert aux modèles qui voient la courbe entière (réseaux de neurones).

    Args:
        X_holed: Courbes à compléter (temps × courbes), NaN dans les trous.
        reservoir: Le réservoir utilisé par ``knn.search`` (vérifié).
        neighbors: Voisines trouvées par ``knn.search`` pour ces courbes.
        names: Variables voulues, parmi ``FEATURES``.
        predictor: Configuration du KNN ancré.

    Returns:
        ``{nom: matrice float32 (temps × courbes)}``.

    Raises:
        ValueError: Si le réservoir n'est pas celui de la recherche des voisines.
    """
    if not reservoir.columns.equals(neighbors.reservoir_columns):
        raise ValueError("Le réservoir doit être celui utilisé par search().")
    R = reservoir.to_numpy(dtype=np.float32)
    X_all = X_holed.to_numpy(dtype=np.float64)
    parts: dict[str, list[np.ndarray]] = {name: [] for name in names}
    for b0 in range(0, X_all.shape[1], _BLOCK):
        b1 = min(b0 + _BLOCK, X_all.shape[1])
        feats = _block_features(
            X_all[:, b0:b1],
            R,
            X_holed.index,
            neighbors.index[b0:b1],
            neighbors.distance[b0:b1],
            neighbors.weights[:, b0:b1].astype(np.float64),
            predictor,
        )
        for name in names:
            parts[name].append(np.asarray(feats[name], dtype=np.float32))
    return {name: np.hstack(blocks) for name, blocks in parts.items()}
