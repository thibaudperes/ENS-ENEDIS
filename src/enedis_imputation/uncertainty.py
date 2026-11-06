"""Intervalles de confiance par prédiction conforme, et analyse du biais.

**Prédiction conforme.** Sur des courbes de calibration jamais vues par les modèles, on
mesure un score de non-conformité pour chaque point manquant. Le quantile conforme de ces
scores, au niveau 1 − α, donne un intervalle qui contient la vraie valeur avec une
probabilité d'au moins 1 − α, sous hypothèse d'échangeabilité. Les points d'une même courbe
étant corrélés, la garantie n'est qu'approximative : la couverture se contrôle donc sur dev
et real.

Trois variantes :

- ``"classe"`` : score = |erreur|, un quantile par classe de longueur de trou ;
- ``"normalisé"`` : score = |erreur| / σ̂, un seul quantile ; σ̂ est la difficulté prédite
  pour chaque point par un petit modèle (``fit_difficulty``) ;
- ``"normalisé par classe"`` : idem, avec un quantile par classe.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import lightgbm as lgb
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch

from enedis_imputation import SEED
from enedis_imputation.deep import lightgbm_prediction, make_sequences, network_prediction
from enedis_imputation.features import build_features
from enedis_imputation.gbm import GBMConfig, TrainingSet
from enedis_imputation.holes import LENGTH_LABELS, length_bucket
from enedis_imputation.knn import BEST_DISTANCE, BEST_PREDICTOR, search

Kind = Literal["classe", "normalisé", "normalisé par classe"]
ALPHAS: tuple[float, ...] = (0.2, 0.1)
"""Niveaux d'erreur : intervalles à 80 % et 90 %."""

DIFFICULTY_FEATURES: list[str] = [
    "nb_std",
    "nb_iqr",
    "nb_dist",
    "hole_len",
    "dist_left",
    "dist_right",
    "rel_pos",
    "hour_sin",
    "hour_cos",
    "curve_mean",
    "curve_std",
    "curve_zero_share",
    "knn_anchored",
    "interp",
]
"""Variables du modèle de difficulté (aucune ne dépend des vraies valeurs)."""

MIN_SIGMA: float = 1.0
"""Plancher de la difficulté prédite (W), pour éviter une division par ~0."""


# --------------------------------------------------------------------------- prédictions


@dataclass(frozen=True)
class Components:
    """Prédictions du mélange et de ses composantes, une ligne par point manquant.

    Attributes:
        feats: Variables des points manquants (``features.FEATURES``).
        t: Indice de temps de chaque ligne.
        c: Indice de courbe de chaque ligne.
        lightgbm: Prédiction du LightGBM (W).
        network: Prédiction du réseau (W).
        blend: Prédiction du mélange (W), ≥ 0.
        classes: Classe de longueur de trou de chaque ligne.
    """

    feats: pd.DataFrame
    t: npt.NDArray[np.int64]
    c: npt.NDArray[np.int64]
    lightgbm: npt.NDArray[np.float64]
    network: npt.NDArray[np.float64]
    blend: npt.NDArray[np.float64]
    classes: pd.Categorical = field(repr=False)


def predict_components(
    X_holed: pd.DataFrame,
    reservoir: pd.DataFrame,
    booster: lgb.Booster,
    cfg: GBMConfig,
    model: torch.nn.Module,
    weight: float,
) -> Components:
    """Calcule, en une passe, les variables et les prédictions du mélange et de ses parts.

    Args:
        X_holed: Courbes à compléter (temps × courbes).
        reservoir: Réservoir de voisines (sans aucune des courbes à compléter).
        booster: LightGBM entraîné.
        cfg: Sa configuration.
        model: Réseau entraîné.
        weight: Poids w du LightGBM dans le mélange.

    Returns:
        Les prédictions aux points manquants.
    """
    nb = search(X_holed, reservoir, BEST_DISTANCE, k_max=BEST_PREDICTOR.k)
    feats, t, c = build_features(X_holed, reservoir, nb)
    gbm = lightgbm_prediction(booster, cfg, X_holed, reservoir, nb)[c, t]
    net = network_prediction(model, make_sequences(X_holed, reservoir, nb))[c, t]
    blend = np.clip(weight * gbm + (1 - weight) * net, 0.0, None)
    classes = length_bucket(feats["hole_len"].to_numpy())
    return Components(feats, t, c, gbm, net, blend, classes)


# --------------------------------------------------------------------------- difficulté


def fit_difficulty(ts: TrainingSet, n_rows: int = 500_000, seed: int = SEED) -> lgb.Booster:
    """Entraîne le modèle de difficulté sur les données d'entraînement du réservoir.

    La cible est |y − KNN ancré|, l'erreur du KNN ancré **hors pli** : contrairement à
    l'erreur du mélange (entraîné sur ces mêmes courbes, donc trop optimiste ici), elle est
    honnête, et elle classe bien les points du plus facile au plus difficile. Seul l'ordre
    compte : la calibration conforme fixe ensuite l'échelle.

    Args:
        ts: Données d'entraînement (étape LightGBM).
        n_rows: Nombre de lignes tirées pour l'entraînement.
        seed: Graine aléatoire.

    Returns:
        Le modèle (objectif L1 : il prédit la médiane de l'erreur absolue).
    """
    rng = np.random.default_rng(seed)
    rows = rng.choice(len(ts.y), size=min(n_rows, len(ts.y)), replace=False)
    X = ts.X.iloc[rows][DIFFICULTY_FEATURES]
    target = np.abs(ts.y[rows] - ts.X["knn_anchored"].to_numpy()[rows])
    params = {
        "objective": "l1",
        "num_leaves": 31,
        "learning_rate": 0.1,
        "min_child_samples": 500,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.5,
        "bagging_freq": 1,
        "seed": seed,
        "verbosity": -1,
    }
    return lgb.train(params, lgb.Dataset(X, target), num_boost_round=300)


def difficulty(booster: lgb.Booster, feats: pd.DataFrame) -> npt.NDArray[np.float64]:
    """Difficulté prédite σ̂ de chaque point (W), au moins ``MIN_SIGMA``."""
    return np.maximum(booster.predict(feats[DIFFICULTY_FEATURES]), MIN_SIGMA)


# --------------------------------------------------------------------------- conformal


def conformal_quantile(scores: npt.ArrayLike, alpha: float) -> float:
    """Quantile conforme : la ⌈(n + 1)(1 − α)⌉-ième plus petite valeur des scores.

    Args:
        scores: Scores de non-conformité de calibration.
        alpha: Niveau d'erreur visé (0,1 pour un intervalle à 90 %).

    Returns:
        Le quantile (``inf`` si l'échantillon est trop petit pour ce niveau).
    """
    s = np.sort(np.asarray(scores, dtype=np.float64))
    k = int(np.ceil((len(s) + 1) * (1 - alpha)))
    return float(s[k - 1]) if k <= len(s) else float("inf")


@dataclass(frozen=True)
class ConformalModel:
    """Intervalles conformes calibrés.

    Attributes:
        kind: Variante (``"classe"``, ``"normalisé"``, ``"normalisé par classe"``).
        quantiles: ``{(alpha, classe): quantile}`` ; classe ``"toutes"`` si un seul quantile.
    """

    kind: Kind
    quantiles: dict[tuple[float, str], float]

    @property
    def normalized(self) -> bool:
        """Vrai si le score est divisé par la difficulté σ̂."""
        return self.kind.startswith("normalisé")

    @property
    def by_class(self) -> bool:
        """Vrai si un quantile est calibré par classe de longueur de trou."""
        return self.kind.endswith("classe")


def fit_conformal(
    abs_errors: npt.ArrayLike,
    classes: pd.Categorical,
    kind: Kind,
    sigma: npt.ArrayLike | None = None,
    alphas: tuple[float, ...] = ALPHAS,
) -> ConformalModel:
    """Calibre des intervalles conformes sur des erreurs de calibration.

    Args:
        abs_errors: Erreurs absolues du modèle sur la calibration.
        classes: Classe de longueur de trou de chaque point.
        kind: Variante.
        sigma: Difficulté prédite de chaque point (variantes normalisées).
        alphas: Niveaux d'erreur.

    Returns:
        Le modèle conforme.
    """
    err = np.asarray(abs_errors, dtype=np.float64)
    scores = err / np.asarray(sigma) if kind.startswith("normalisé") else err
    cls = np.asarray(classes)
    quantiles = {}
    for alpha in alphas:
        if kind.endswith("classe"):
            for c in LENGTH_LABELS:
                quantiles[(alpha, c)] = conformal_quantile(scores[cls == c], alpha)
        else:
            quantiles[(alpha, "toutes")] = conformal_quantile(scores, alpha)
    return ConformalModel(kind, quantiles)


def intervals(
    model: ConformalModel,
    pred: npt.ArrayLike,
    classes: pd.Categorical,
    sigma: npt.ArrayLike | None = None,
    alpha: float = 0.1,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Bornes de l'intervalle ``prédiction ± quantile (× σ̂)``, borne basse ≥ 0.

    Args:
        model: Modèle conforme calibré.
        pred: Prédictions.
        classes: Classe de longueur de trou de chaque point.
        sigma: Difficulté prédite (variantes normalisées).
        alpha: Niveau d'erreur.

    Returns:
        ``(borne basse, borne haute)``.
    """
    p = np.asarray(pred, dtype=np.float64)
    if model.by_class:
        cls = np.asarray(classes)
        q = np.array([model.quantiles[(alpha, c)] for c in cls])
    else:
        q = np.full(len(p), model.quantiles[(alpha, "toutes")])
    half = q * np.asarray(sigma) if model.normalized else q
    return np.clip(p - half, 0.0, None), p + half


def coverage_table(
    y: npt.ArrayLike,
    low: npt.ArrayLike,
    high: npt.ArrayLike,
    classes: pd.Categorical,
) -> pd.DataFrame:
    """Couverture réelle et largeur moyenne, par classe de longueur de trou et globalement.

    Args:
        y: Vraies valeurs (NaN ignorés).
        low: Bornes basses.
        high: Bornes hautes.
        classes: Classe de chaque point.

    Returns:
        Colonnes ``couverture`` et ``largeur (W)``, index : classes puis ``toutes``.
    """
    y, low, high = (np.asarray(a, dtype=np.float64) for a in (y, low, high))
    ok = ~np.isnan(y)
    df = pd.DataFrame(
        {
            "classe": np.asarray(classes)[ok],
            "couvert": ((y >= low) & (y <= high))[ok],
            "largeur": (high - low)[ok],
        }
    )
    table = df.groupby("classe", observed=False).agg(
        couverture=("couvert", "mean"), largeur=("largeur", "mean")
    )
    table = table.reindex(LENGTH_LABELS)
    table.loc["toutes"] = [df["couvert"].mean(), df["largeur"].mean()]
    return table.rename(columns={"largeur": "largeur (W)"})


# --------------------------------------------------------------------------- biais


def bias_table(pred: npt.ArrayLike, y: npt.ArrayLike, classes: pd.Categorical) -> pd.DataFrame:
    """Erreur signée moyenne (prédiction − vérité) et MAE, par classe et globalement.

    Un modèle entraîné en L1 vise la médiane conditionnelle : sur des consommations à
    distribution asymétrique (pics rares), il sous-estime en moyenne.

    Returns:
        Colonnes ``biais (W)``, ``biais (%)`` (relatif à la moyenne vraie) et ``MAE (W)``.
    """
    p, t = np.asarray(pred, dtype=np.float64), np.asarray(y, dtype=np.float64)
    ok = ~np.isnan(t)
    df = pd.DataFrame({"classe": np.asarray(classes)[ok], "e": (p - t)[ok], "y": t[ok]})
    g = df.groupby("classe", observed=False)
    table = pd.DataFrame(
        {
            "biais (W)": g["e"].mean(),
            "vraie moyenne (W)": g["y"].mean(),
            "MAE (W)": g["e"].apply(lambda e: e.abs().mean()),
        }
    ).reindex(LENGTH_LABELS)
    table.loc["toutes"] = [df["e"].mean(), df["y"].mean(), df["e"].abs().mean()]
    table["biais (%)"] = table["biais (W)"] / table["vraie moyenne (W)"]
    return table[["biais (W)", "biais (%)", "MAE (W)"]]


def save_conformal(model: ConformalModel, path: Path) -> None:
    """Enregistre un modèle conforme calibré (JSON)."""
    data = {"kind": model.kind, "quantiles": [[a, c, q] for (a, c), q in model.quantiles.items()]}
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1))


def load_conformal(path: Path) -> ConformalModel:
    """Relit un modèle conforme enregistré par ``save_conformal``."""
    data = json.loads(path.read_text())
    return ConformalModel(data["kind"], {(a, c): q for a, c, q in data["quantiles"]})
