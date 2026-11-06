"""KNN configurable pour l'imputation : recherche de voisins, mise à l'échelle, ancrage.

Le calcul se fait en deux temps, pour pouvoir tester beaucoup de variantes à moindre coût :

1. ``search`` : trouve les ``k_max`` plus proches voisines de chaque courbe dans le
   réservoir, pour une distance donnée (``Distance``). C'est l'étape coûteuse.
2. ``predict`` : à partir de ces voisines, construit la prédiction (``Predictor``) :
   mise à l'échelle des voisines, agrégation des K premières, puis ancrage éventuel sur
   les points observés au bord des trous.

``make_knn(distance, predictor)`` assemble les deux en une méthode standard
``(X_holed, reservoir) -> X_filled``.

**Ancrage.** La prédiction ancrée vaut ``KNN + résidu``, où le résidu (observé − KNN) est
mesuré au bord de chaque trou puis interpolé linéairement le long du trou (prolongé par
constante en début ou fin de courbe). C'est équivalent à : interpolation de la courbe
+ (forme des voisines − interpolation de cette forme). On garde les niveaux aux bords et
la forme des voisines à l'intérieur.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

from enedis_imputation.methods import Method

Metric = Literal["l1", "l2"]
Scaling = Literal["none", "mult", "affine_ls", "affine_med"]
Aggregate = Literal["median", "wmean"]

_BLOCK = 256  # cibles traitées par bloc (taille des matrices temporaires : bloc × réservoir)


@dataclass(frozen=True)
class Distance:
    """Définition de la distance entre une courbe trouée et une courbe du réservoir.

    Attributes:
        metric: ``"l1"`` (MAE) ou ``"l2"`` (erreur quadratique moyenne).
        shape: Si vrai, compare la forme seule (courbes centrées-réduites). En L2, c'est
            exactement ``1 − corrélation`` pondérée sur les points observés.
        window: ``None`` pour toute la courbe, sinon nombre de pas de part et d'autre des
            trous : seuls les points observés à moins de ``window`` pas d'un trou comptent.
    """

    metric: Metric = "l1"
    shape: bool = False
    window: int | None = None

    @property
    def label(self) -> str:
        """Libellé court, par ex. « L2 forme ±48 »."""
        space = "forme" if self.shape else "brut"
        zone = f"±{self.window}" if self.window else "global"
        return f"{self.metric.upper()} {space} {zone}"


@dataclass(frozen=True)
class Predictor:
    """Définition de la prédiction à partir des voisines.

    Attributes:
        k: Nombre de voisines agrégées.
        aggregate: ``"median"`` (optimale pour la MAE) ou ``"wmean"`` (moyenne pondérée par
            l'inverse de la distance).
        scaling: Mise à l'échelle de chaque voisine sur les points observés de la cible :
            ``"none"``, ``"mult"`` (a·r, moindres carrés), ``"affine_ls"`` (a·r + b,
            moindres carrés), ``"affine_med"`` (pente des moindres carrés, ordonnée médiane).
        anchor: Si vrai, ancre la prédiction sur les points observés au bord des trous.
        anchor_points: Nombre de points observés moyennés de chaque côté pour l'ancre
            (1 = le seul point voisin du trou).
        anchor_decay: ``None`` pour une correction interpolée sur tout le trou ; sinon
            constante τ (en pas) d'atténuation : la correction est multipliée par
            ``exp(−(d − 1) / τ)``, où d est la distance au bord le plus proche. Au milieu
            d'un long trou, on fait alors confiance au KNN.
    """

    k: int = 5
    aggregate: Aggregate = "median"
    scaling: Scaling = "none"
    anchor: bool = False
    anchor_points: int = 1
    anchor_decay: float | None = None

    @property
    def label(self) -> str:
        """Libellé court, par ex. « K=10 médiane, affine_ls, ancré h=2 »."""
        agg = "médiane" if self.aggregate == "median" else "moy. pondérée"
        decay = f" τ={self.anchor_decay:g}" if self.anchor_decay else ""
        anchor = f"ancré h={self.anchor_points}{decay}" if self.anchor else "non ancré"
        return f"K={self.k} {agg}, échelle {self.scaling}, {anchor}"


@dataclass(frozen=True)
class Neighbors:
    """Résultat d'une recherche de voisins.

    Attributes:
        index: Entiers (cibles × k_max) : colonnes du réservoir, de la plus proche à la moins
            proche.
        distance: Distances correspondantes.
        reservoir_columns: Colonnes du réservoir utilisé ; ``predict`` vérifie qu'on lui
            fournit le même, pour que les indices désignent bien les mêmes courbes.
        weights: Poids (temps × cibles) des points pris en compte (observés, et dans la
            fenêtre éventuelle) ; réutilisés pour la mise à l'échelle.
        seconds: Durée de la recherche.
    """

    index: npt.NDArray[np.int64]
    distance: npt.NDArray[np.float32]
    reservoir_columns: pd.Index
    weights: npt.NDArray[np.float32] = field(repr=False)
    seconds: float = 0.0


# --------------------------------------------------------------------------- distances


def point_weights(mask: npt.NDArray[np.bool_], window: int | None) -> npt.NDArray[np.float32]:
    """Poids des points pour la distance : 1 sur les points observés retenus, 0 ailleurs.

    Args:
        mask: Booléens (temps × cibles), vrai dans les trous.
        window: ``None`` pour tous les points observés ; sinon seuls ceux à moins de
            ``window`` pas d'un trou. Une courbe sans aucun point dans la fenêtre garde
            tous ses points observés.

    Returns:
        Poids float32 (temps × cibles).
    """
    observed = ~mask
    if window is None:
        return observed.astype(np.float32)
    # Distance au trou le plus proche via une dilatation du masque (somme glissante)
    padded = np.pad(mask.astype(np.int32), ((window, window), (0, 0)))
    csum = np.cumsum(padded, axis=0)
    csum = np.vstack([np.zeros((1, mask.shape[1]), np.int64), csum])
    near = (csum[2 * window + 1 :] - csum[: -2 * window - 1]) > 0
    w = observed & near
    empty = ~w.any(axis=0)
    w[:, empty] = observed[:, empty]
    return w.astype(np.float32)


def _l2_block(
    X: np.ndarray, W: np.ndarray, R: np.ndarray, R2: np.ndarray, shape: bool
) -> np.ndarray:
    """Distances L2 masquées d'un bloc de cibles à tout le réservoir, par produits matriciels.

    ``Σ w (x − r)² = Σ w x² − 2 (w x)·r + w·r²`` : trois produits matriciels au lieu d'une
    boucle. En mode forme, la corrélation pondérée s'obtient avec les mêmes sommes.
    """
    n = W.sum(axis=0)[:, None]  # (B, 1)
    WX = W * X
    s_x = WX.sum(axis=0)[:, None]
    s_xx = (WX * X).sum(axis=0)[:, None]
    s_r = W.T @ R  # (B, N)
    s_rr = W.T @ R2
    s_xr = WX.T @ R
    mse = (s_xx - 2 * s_xr + s_rr) / np.maximum(n, 1)
    if not shape:
        return mse
    var_x = n * s_xx - s_x**2
    var_r = n * s_rr - s_r**2
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = (n * s_xr - s_x * s_r) / np.sqrt(np.maximum(var_x, 0) * np.maximum(var_r, 0))
    dist = 1.0 - corr
    dist = np.where(np.isfinite(dist), dist, 2.0)  # voisine constante : forme indéfinie
    # Cible constante sur ses points (ex. courbe nulle) : la forme n'a pas de sens, on
    # revient à la distance brute (le classement ne se fait qu'au sein d'une ligne).
    flat = (var_x <= 1e-9 * np.maximum(n * s_xx, 1)).ravel()
    dist[flat] = mse[flat]
    return dist


def _l1_target(
    x: np.ndarray, w: np.ndarray, R: np.ndarray, Rz: np.ndarray | None, shape: bool
) -> np.ndarray:
    """Distances L1 masquées d'une cible à tout le réservoir (boucle : pas d'identité)."""
    rows = w > 0
    if not rows.any():
        return np.zeros(R.shape[1], dtype=np.float32)
    xs = x[rows]
    if shape:
        sd = xs.std()
        if sd > 0:
            return np.abs(Rz[rows] - ((xs - xs.mean()) / sd)[:, None]).mean(axis=0)
    return np.abs(R[rows] - xs[:, None]).mean(axis=0)


def search(
    X_holed: pd.DataFrame, reservoir: pd.DataFrame, distance: Distance, k_max: int = 50
) -> Neighbors:
    """Cherche les ``k_max`` plus proches voisines de chaque courbe trouée.

    Args:
        X_holed: Courbes à compléter (temps × cibles), NaN dans les trous.
        reservoir: Courbes complètes (temps × N), sans NaN.
        distance: Définition de la distance.
        k_max: Nombre de voisines conservées (les variantes de K les réutilisent).

    Returns:
        Les voisines, triées de la plus proche à la moins proche.
    """
    start = time.perf_counter()
    # L2 : float64, car l'identité développée soustrait des sommes de l'ordre de 1e10
    dtype = np.float64 if distance.metric == "l2" else np.float32
    R = reservoir.to_numpy(dtype=dtype)
    mask = X_holed.isna().to_numpy()
    X = np.where(mask, 0.0, X_holed.to_numpy(dtype=dtype)).astype(dtype)
    W = point_weights(mask, distance.window).astype(dtype)
    k_max = min(k_max, R.shape[1])

    R2 = R * R if distance.metric == "l2" else None
    Rz = None
    if distance.metric == "l1" and distance.shape:
        sd = R.std(axis=0)
        Rz = ((R - R.mean(axis=0)) / np.where(sd > 0, sd, 1.0)).astype(np.float32)

    n = X.shape[1]
    index = np.empty((n, k_max), dtype=np.int64)
    dist_out = np.empty((n, k_max), dtype=np.float32)
    for b0 in range(0, n, _BLOCK):
        b1 = min(b0 + _BLOCK, n)
        if distance.metric == "l2":
            D = _l2_block(X[:, b0:b1], W[:, b0:b1], R, R2, distance.shape)
        else:
            D = np.stack(
                [_l1_target(X[:, j], W[:, j], R, Rz, distance.shape) for j in range(b0, b1)]
            )
        part = np.argpartition(D, k_max - 1, axis=1)[:, :k_max]
        d_part = np.take_along_axis(D, part, axis=1)
        order = np.argsort(d_part, axis=1)
        index[b0:b1] = np.take_along_axis(part, order, axis=1)
        dist_out[b0:b1] = np.take_along_axis(d_part, order, axis=1)
    return Neighbors(
        index, dist_out, reservoir.columns, W.astype(np.float32), time.perf_counter() - start
    )


# --------------------------------------------------------------------------- prédiction


def _scale(Rn: np.ndarray, x: np.ndarray, w: np.ndarray, scaling: Scaling) -> np.ndarray:
    """Met chaque voisine à l'échelle de la cible sur les points pondérés.

    Args:
        Rn: Voisines (temps × cibles × K).
        x: Cibles (temps × cibles), 0 dans les trous.
        w: Poids (temps × cibles).
        scaling: Type de mise à l'échelle.

    Returns:
        Voisines mises à l'échelle, même forme que ``Rn``.
    """
    if scaling == "none":
        return Rn
    w3, x3 = w[:, :, None], x[:, :, None]
    n = np.maximum(w3.sum(axis=0), 1.0)
    if scaling == "mult":
        num, den = (w3 * x3 * Rn).sum(axis=0), (w3 * Rn * Rn).sum(axis=0)
        a = np.where(den > 0, num / np.maximum(den, 1e-12), 1.0)
        return Rn * np.clip(a, 0.0, None)[None]
    mean_r, mean_x = (w3 * Rn).sum(axis=0) / n, (w3 * x3).sum(axis=0) / n
    cov = (w3 * (Rn - mean_r) * (x3 - mean_x)).sum(axis=0)
    var = (w3 * (Rn - mean_r) ** 2).sum(axis=0)
    a = np.clip(np.where(var > 0, cov / np.maximum(var, 1e-12), 0.0), 0.0, None)
    if scaling == "affine_ls":
        b = mean_x - a * mean_r
    else:  # ordonnée à l'origine médiane des résidus, sur les points pondérés
        resid = np.where(w3 > 0, x3 - a[None] * Rn, np.nan)
        b = np.nanmedian(resid, axis=0)
        b = np.where(np.isnan(b), 0.0, b)
    return a[None] * Rn + b[None]


def _aggregate(Rn: np.ndarray, dist: np.ndarray, aggregate: Aggregate) -> np.ndarray:
    """Combine les K voisines (temps × cibles × K) en une prédiction (temps × cibles)."""
    if aggregate == "median":
        return np.median(Rn, axis=2)
    weights = 1.0 / np.maximum(dist, 1e-6)
    weights = weights / weights.sum(axis=1, keepdims=True)
    return (Rn * weights[None]).sum(axis=2)


def anchor_residual(
    residual: npt.NDArray[np.float64],
    mask: npt.NDArray[np.bool_],
    points: int = 1,
    decay: float | None = None,
) -> npt.NDArray[np.float64]:
    """Interpole dans les trous le résidu mesuré à leurs bords.

    Le résidu de bord est la moyenne des ``points`` derniers points observés avant le trou
    (ancre gauche) et des ``points`` premiers après (ancre droite). Dans le trou, on passe
    linéairement de l'ancre gauche à l'ancre droite ; sans ancre d'un côté (trou en début
    ou en fin de courbe), on prolonge l'autre par constante.

    Args:
        residual: Résidus (temps × cibles), valeurs ignorées dans les trous.
        mask: Booléens (temps × cibles), vrai dans les trous.
        points: Nombre de points observés moyennés de chaque côté.
        decay: Si fourni, constante τ (en pas) : la correction est multipliée par
            ``exp(−(d − 1) / τ)``, d étant la distance au bord observé le plus proche.

    Returns:
        Correction (temps × cibles), nulle hors des trous.
    """
    T = mask.shape[0]
    e = pd.DataFrame(np.where(mask, np.nan, residual))
    # Ancre gauche : moyenne glissante « passée » des points observés, propagée dans le trou
    left = e.rolling(points, min_periods=1).mean().where(~mask).ffill().to_numpy()
    right = e[::-1].rolling(points, min_periods=1).mean()[::-1].where(~mask).bfill().to_numpy()
    t = np.arange(T, dtype=np.float64)[:, None]
    last = pd.DataFrame(np.where(mask, np.nan, t * np.ones(mask.shape))).ffill().to_numpy()
    nxt = pd.DataFrame(np.where(mask, np.nan, t * np.ones(mask.shape))).bfill().to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = (t - last) / (nxt - last)
    corr = np.where(
        np.isnan(left), right, np.where(np.isnan(right), left, left + frac * (right - left))
    )
    if decay is not None:
        dist = np.fmin(t - last, nxt - t)  # distance au bord observé le plus proche
        corr = corr * np.exp(-(np.nan_to_num(dist, nan=1.0) - 1.0) / decay)
    corr = np.where(mask, corr, 0.0)
    return np.nan_to_num(corr, nan=0.0)  # courbe sans aucun point observé : pas de correction


def predict(
    X_holed: pd.DataFrame, reservoir: pd.DataFrame, neighbors: Neighbors, predictor: Predictor
) -> pd.DataFrame:
    """Complète les trous à partir de voisines déjà trouvées.

    Args:
        X_holed: Courbes à compléter (temps × cibles), NaN dans les trous.
        reservoir: Le réservoir utilisé par ``search`` (vérifié).
        neighbors: Résultat de ``search`` sur ces mêmes courbes.
        predictor: Définition de la prédiction.

    Returns:
        Courbes complétées (points observés inchangés, prédictions ≥ 0).

    Raises:
        ValueError: Si le réservoir diffère de celui de la recherche, ou si ``k`` dépasse
            le nombre de voisines disponibles.
    """
    if not reservoir.columns.equals(neighbors.reservoir_columns):
        raise ValueError("Le réservoir doit être celui utilisé par search().")
    if predictor.k > neighbors.index.shape[1]:
        raise ValueError(f"k={predictor.k} > {neighbors.index.shape[1]} voisines cherchées.")
    R = reservoir.to_numpy(dtype=np.float32)
    mask = X_holed.isna().to_numpy()
    X = np.where(mask, 0.0, X_holed.to_numpy(dtype=np.float32)).astype(np.float32)
    k = predictor.k
    pred = np.empty(X.shape, dtype=np.float64)
    for b0 in range(0, X.shape[1], _BLOCK):
        b1 = min(b0 + _BLOCK, X.shape[1])
        Rn = R[:, neighbors.index[b0:b1, :k]].astype(np.float64)  # (temps, bloc, K)
        Rn = _scale(Rn, X[:, b0:b1], neighbors.weights[:, b0:b1], predictor.scaling)
        pred[:, b0:b1] = _aggregate(Rn, neighbors.distance[b0:b1, :k], predictor.aggregate)
    if predictor.anchor:
        pred = pred + anchor_residual(
            X - pred, mask, predictor.anchor_points, predictor.anchor_decay
        )
    pred = np.clip(pred, 0.0, None)  # une consommation n'est jamais négative
    filled = np.where(mask, pred, X_holed.to_numpy(dtype=np.float64))
    return pd.DataFrame(filled, index=X_holed.index, columns=X_holed.columns)


def make_knn(distance: Distance, predictor: Predictor) -> Method:
    """Assemble recherche et prédiction en une méthode ``(X_holed, reservoir) -> X_filled``.

    Args:
        distance: Définition de la distance.
        predictor: Définition de la prédiction.

    Returns:
        La méthode, nommée d'après sa configuration.
    """

    def method(X_holed: pd.DataFrame, reservoir: pd.DataFrame) -> pd.DataFrame:
        nb = search(X_holed, reservoir, distance, k_max=predictor.k)
        return predict(X_holed, reservoir, nb, predictor)

    method.__name__ = f"KNN {distance.label} | {predictor.label}"
    return method


BEST_DISTANCE = Distance(metric="l2", shape=True, window=None)
"""Distance retenue sur dev (notebook 03) : forme seule (1 − corrélation), toute la courbe."""

BEST_PREDICTOR = Predictor(
    k=10, aggregate="wmean", scaling="affine_med", anchor=True, anchor_points=1, anchor_decay=3.0
)
"""Prédiction retenue sur dev (notebook 03) : 10 voisines mises à l'échelle (pente des
moindres carrés, ordonnée médiane), moyenne pondérée, ancrage sur le point voisin du trou
atténué avec τ = 3 pas."""


def anchored_knn(X_holed: pd.DataFrame, reservoir: pd.DataFrame) -> pd.DataFrame:
    """KNN ancré avec la configuration retenue (``BEST_DISTANCE``, ``BEST_PREDICTOR``).

    Args:
        X_holed: Courbes à compléter (temps × cibles), NaN dans les trous.
        reservoir: Courbes complètes (temps × N), sans NaN.

    Returns:
        Courbes complétées.
    """
    return make_knn(BEST_DISTANCE, BEST_PREDICTOR)(X_holed, reservoir)
