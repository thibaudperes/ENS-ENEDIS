"""LightGBM qui corrige le KNN ancré (stacking), entraîné sans fuite.

**Données d'entraînement.** Les cibles sont des courbes du réservoir (jamais dev, holdout
ni real), trouées par greffe de plusieurs motifs réels. Pour que les prédictions KNN
données au modèle ressemblent à celles qu'il verra sur dev, elles sont calculées *hors
pli* : les courbes sont réparties en plis, et celles du pli k sont remplies avec un
réservoir qui exclut tout le pli k (donc la courbe elle-même et ses autres copies).

**Modèle.** Objectif L1 (on est jugé sur la MAE), deux formulations :

- ``"residual"`` : le modèle prédit y − KNN ancré, et l'on ajoute sa sortie au KNN ;
- ``"direct"`` : le modèle prédit y.

Le nombre d'arbres est choisi par validation croisée groupée par courbe sur les données
d'entraînement, avec arrêt anticipé : dev ne sert jamais à l'apprentissage.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from typing import Literal

import lightgbm as lgb
import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.model_selection import GroupKFold

from enedis_imputation import SEED
from enedis_imputation.data import PROCESSED_DIR
from enedis_imputation.features import FEATURES, build_features
from enedis_imputation.knn import (
    BEST_DISTANCE,
    BEST_PREDICTOR,
    Distance,
    Neighbors,
    Predictor,
    search,
)
from enedis_imputation.methods import Method
from enedis_imputation.validation import exact_duplicates, graft

Formulation = Literal["residual", "direct"]
TRAIN_CACHE = PROCESSED_DIR / "gbm_train"


# --------------------------------------------------------------------------- données


@dataclass(frozen=True)
class FoldData:
    """Un pli de l'entraînement hors pli.

    Attributes:
        fold: Numéro du pli.
        sources: Pour chaque courbe trouée, le nom de la courbe d'origine du réservoir.
        X_holed: Courbes trouées par greffe (colonnes ``"<courbe>|<j>"``).
        y: Vraies valeurs, mêmes colonnes.
        pool: Réservoir de voisines du pli : le réservoir moins toutes les courbes du pli.
        neighbors: Voisines trouvées dans ``pool``.
    """

    fold: int
    sources: list[str]
    X_holed: pd.DataFrame
    y: pd.DataFrame
    pool: pd.DataFrame
    neighbors: Neighbors


@dataclass(frozen=True)
class TrainingSet:
    """Données d'entraînement : une ligne par point manquant.

    Attributes:
        X: Variables (colonnes ``FEATURES``), float32.
        y: Vraies valeurs.
        groups: Identifiant de la courbe d'origine (pour la validation croisée groupée).
    """

    X: pd.DataFrame
    y: npt.NDArray[np.float32]
    groups: npt.NDArray[np.int64]


def iter_folds(
    reservoir: pd.DataFrame,
    motifs: npt.NDArray[np.bool_],
    *,
    n_curves: int,
    motifs_per_curve: int = 2,
    n_folds: int = 5,
    seed: int = SEED,
    distance: Distance = BEST_DISTANCE,
    k: int = BEST_PREDICTOR.k,
) -> Iterator[FoldData]:
    """Construit les plis : greffe des motifs et recherche des voisines hors pli.

    Args:
        reservoir: Courbes complètes disponibles (le réservoir de validation).
        motifs: Motifs de NaN (temps × motifs).
        n_curves: Nombre de courbes du réservoir utilisées comme cibles d'entraînement.
        motifs_per_curve: Nombre de motifs différents greffés sur chaque courbe.
        n_folds: Nombre de plis.
        seed: Graine aléatoire.
        distance: Distance de recherche des voisines.
        k: Nombre de voisines.

    Yields:
        Les plis, un par un (pour limiter la mémoire).

    Raises:
        ValueError: Si une courbe du pli se retrouve dans son propre réservoir.
    """
    rng = np.random.default_rng(seed)
    eligible = reservoir.columns.difference(exact_duplicates(reservoir), sort=False)
    curves = rng.choice(eligible.to_numpy(), size=n_curves, replace=False)
    fold_of = rng.permutation(np.arange(n_curves) % n_folds)
    n_motifs = motifs.shape[1]

    for fold in range(n_folds):
        fold_curves = curves[fold_of == fold]
        pool = reservoir.drop(columns=fold_curves)
        if pool.columns.isin(fold_curves).any():
            raise ValueError("Fuite : une courbe du pli est dans son propre réservoir.")
        # motifs distincts pour chaque courbe : les premiers d'une permutation aléatoire
        picks = np.argsort(rng.random((len(fold_curves), n_motifs)), axis=1)[:, :motifs_per_curve]
        sources = np.repeat(fold_curves, motifs_per_curve)
        names = [f"{c}|{j}" for c in fold_curves for j in range(motifs_per_curve)]
        y = pd.DataFrame(reservoir[sources].to_numpy(), index=reservoir.index, columns=names)
        X_holed = graft(y, motifs[:, picks.ravel()])
        nb = search(X_holed, pool, distance, k_max=k)
        yield FoldData(fold, [str(s) for s in sources], X_holed, y, pool, nb)


def build_training_set(
    reservoir: pd.DataFrame,
    motifs: npt.NDArray[np.bool_],
    *,
    n_curves: int = 8000,
    motifs_per_curve: int = 2,
    n_folds: int = 5,
    seed: int = SEED,
    predictor: Predictor = BEST_PREDICTOR,
    refresh: bool = False,
) -> TrainingSet:
    """Construit (ou relit depuis ``data/processed/``) les données d'entraînement.

    Args:
        reservoir: Courbes complètes disponibles.
        motifs: Motifs de NaN (temps × motifs).
        n_curves: Nombre de courbes du réservoir utilisées comme cibles.
        motifs_per_curve: Motifs greffés par courbe.
        n_folds: Nombre de plis pour les prédictions KNN hors pli.
        seed: Graine aléatoire.
        predictor: Configuration du KNN ancré.
        refresh: Si vrai, reconstruit même si le cache existe.

    Returns:
        Les données d'entraînement.
    """
    params = {
        "n_curves": n_curves,
        "motifs_per_curve": motifs_per_curve,
        "n_folds": n_folds,
        "seed": seed,
        "predictor": repr(predictor),
        "reservoir": reservoir.shape[1],
    }
    meta_path = TRAIN_CACHE.with_suffix(".json")
    if not refresh and meta_path.exists() and json.loads(meta_path.read_text()) == params:
        arrays = np.load(TRAIN_CACHE.with_suffix(".npz"))
        return TrainingSet(
            pd.DataFrame(arrays["X"], columns=FEATURES), arrays["y"], arrays["groups"]
        )

    curve_ids = {name: i for i, name in enumerate(reservoir.columns)}
    X_parts, y_parts, g_parts = [], [], []
    for fd in iter_folds(
        reservoir,
        motifs,
        n_curves=n_curves,
        motifs_per_curve=motifs_per_curve,
        n_folds=n_folds,
        seed=seed,
        k=predictor.k,
    ):
        feats, t, c = build_features(fd.X_holed, fd.pool, fd.neighbors, predictor)
        X_parts.append(feats.to_numpy())
        y_parts.append(fd.y.to_numpy(dtype=np.float32)[t, c])
        g_parts.append(np.array([curve_ids[fd.sources[j]] for j in c], dtype=np.int64))
    ts = TrainingSet(
        pd.DataFrame(np.vstack(X_parts), columns=FEATURES),
        np.concatenate(y_parts),
        np.concatenate(g_parts),
    )
    np.savez(TRAIN_CACHE.with_suffix(".npz"), X=ts.X.to_numpy(), y=ts.y, groups=ts.groups)
    meta_path.write_text(json.dumps(params))
    return ts


# --------------------------------------------------------------------------- modèle


@dataclass(frozen=True)
class GBMConfig:
    """Configuration d'un LightGBM.

    Attributes:
        formulation: ``"residual"`` (prédit y − KNN ancré) ou ``"direct"`` (prédit y).
        num_leaves: Nombre maximal de feuilles par arbre.
        learning_rate: Taux d'apprentissage (0,1 : 2,5 fois plus rapide que 0,05 pour
            0,4 W de perte, mesuré sur un pli d'entraînement).
        min_child_samples: Effectif minimal d'une feuille.
        feature_fraction: Part des variables tirées pour chaque arbre.
        bagging_fraction: Part des lignes tirées pour chaque arbre (0,5 divise le temps
            par arbre par ~2 ; l'objectif L1 recalcule des médianes, coûteuses).
        lambda_l2: Régularisation L2 des feuilles.
        max_rounds: Nombre maximal d'arbres.
        early_stopping: Arrêt si la validation ne s'améliore plus pendant ce nombre d'arbres.
    """

    formulation: Formulation = "residual"
    num_leaves: int = 63
    learning_rate: float = 0.1
    min_child_samples: int = 200
    feature_fraction: float = 0.8
    bagging_fraction: float = 0.5
    lambda_l2: float = 1.0
    max_rounds: int = 3000
    early_stopping: int = 100

    @property
    def label(self) -> str:
        """Libellé court, par ex. « LightGBM résidu, 63 feuilles »."""
        kind = "résidu" if self.formulation == "residual" else "direct"
        return f"LightGBM {kind}, {self.num_leaves} feuilles"

    def params(self) -> dict[str, object]:
        """Paramètres au format LightGBM."""
        return {
            "objective": "l1",
            "metric": "l1",
            "num_leaves": self.num_leaves,
            "learning_rate": self.learning_rate,
            "min_child_samples": self.min_child_samples,
            "feature_fraction": self.feature_fraction,
            "bagging_fraction": self.bagging_fraction,
            "bagging_freq": 1,
            "lambda_l2": self.lambda_l2,
            "seed": SEED,
            "verbosity": -1,
        }


def _target(X: pd.DataFrame, y: np.ndarray, cfg: GBMConfig) -> np.ndarray:
    """Cible d'apprentissage selon la formulation."""
    return y - X["knn_anchored"].to_numpy() if cfg.formulation == "residual" else y


def to_prediction(
    X: pd.DataFrame, raw: np.ndarray, cfg: GBMConfig, clip: bool = True
) -> npt.NDArray[np.float64]:
    """Convertit la sortie du modèle en prédiction de consommation.

    Args:
        X: Variables des lignes prédites.
        raw: Sortie brute du modèle.
        cfg: Configuration (formulation).
        clip: Si vrai, force les prédictions à être positives ou nulles.

    Returns:
        Prédictions (W).
    """
    pred = raw + X["knn_anchored"].to_numpy() if cfg.formulation == "residual" else raw
    return np.clip(pred, 0.0, None) if clip else pred


@dataclass(frozen=True)
class CVResult:
    """Résultat de la validation croisée groupée sur les données d'entraînement.

    Attributes:
        mae: MAE hors pli, toutes lignes confondues.
        best_rounds: Nombre d'arbres retenu par l'arrêt anticipé, pour chaque pli.
        seconds: Durée totale.
    """

    mae: float
    best_rounds: list[int]
    seconds: float


def cross_validate(ts: TrainingSet, cfg: GBMConfig, n_splits: int = 3) -> CVResult:
    """Validation croisée groupée par courbe, avec arrêt anticipé sur le pli de validation.

    Args:
        ts: Données d'entraînement.
        cfg: Configuration.
        n_splits: Nombre de plis.

    Returns:
        MAE hors pli et nombre d'arbres retenu par pli.
    """
    start = time.perf_counter()
    target = _target(ts.X, ts.y, cfg)
    abs_err = np.empty(len(ts.y))
    rounds = []
    for train_idx, valid_idx in GroupKFold(n_splits=n_splits).split(ts.X, groups=ts.groups):
        train = lgb.Dataset(ts.X.iloc[train_idx], target[train_idx], free_raw_data=True)
        valid = lgb.Dataset(ts.X.iloc[valid_idx], target[valid_idx], reference=train)
        booster = lgb.train(
            cfg.params(),
            train,
            num_boost_round=cfg.max_rounds,
            valid_sets=[valid],
            callbacks=[lgb.early_stopping(cfg.early_stopping, verbose=False)],
        )
        rounds.append(booster.best_iteration)
        Xv = ts.X.iloc[valid_idx]
        pred = to_prediction(Xv, booster.predict(Xv, num_iteration=booster.best_iteration), cfg)
        abs_err[valid_idx] = np.abs(pred - ts.y[valid_idx])
    return CVResult(float(abs_err.mean()), rounds, time.perf_counter() - start)


def fit(ts: TrainingSet, cfg: GBMConfig, n_rounds: int) -> lgb.Booster:
    """Entraîne le modèle final sur toutes les données d'entraînement.

    Args:
        ts: Données d'entraînement.
        cfg: Configuration.
        n_rounds: Nombre d'arbres (issu de la validation croisée).

    Returns:
        Le modèle entraîné.
    """
    data = lgb.Dataset(ts.X, _target(ts.X, ts.y, cfg), free_raw_data=True)
    return lgb.train(cfg.params(), data, num_boost_round=n_rounds)


MODEL_DIR = PROCESSED_DIR / "gbm_models"


def train_cached(
    ts: TrainingSet, cfg: GBMConfig, ts_key: str, n_splits: int = 3
) -> tuple[CVResult, lgb.Booster, float]:
    """Validation croisée puis entraînement final, mis en cache sur disque.

    Le premier appel calcule tout (plusieurs minutes) et enregistre le modèle, le résultat
    de la validation croisée et les temps mesurés ; les appels suivants les relisent. Les
    temps renvoyés sont toujours ceux du calcul d'origine.

    Args:
        ts: Données d'entraînement.
        cfg: Configuration.
        ts_key: Identifiant des données d'entraînement (invalide le cache s'il change),
            par ex. le contenu de ``TRAIN_CACHE.with_suffix(".json")``.
        n_splits: Nombre de plis de la validation croisée.

    Returns:
        ``(résultat de la validation croisée, modèle final, durée de l'entraînement final)``.
    """
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    key = hashlib.md5(json.dumps([config_dict(cfg), ts_key, n_splits]).encode()).hexdigest()[:12]
    model_path, info_path = MODEL_DIR / f"{key}.txt", MODEL_DIR / f"{key}.json"
    if model_path.exists() and info_path.exists():
        info = json.loads(info_path.read_text())
        cv = CVResult(info["cv_mae"], info["best_rounds"], info["cv_seconds"])
        return cv, lgb.Booster(model_file=str(model_path)), info["fit_seconds"]

    cv = cross_validate(ts, cfg, n_splits)
    start = time.perf_counter()
    booster = fit(ts, cfg, n_rounds=int(np.mean(cv.best_rounds)))
    fit_seconds = time.perf_counter() - start
    booster.save_model(str(model_path))
    info_path.write_text(
        json.dumps(
            {
                "cv_mae": cv.mae,
                "best_rounds": cv.best_rounds,
                "cv_seconds": cv.seconds,
                "fit_seconds": fit_seconds,
            }
        )
    )
    return cv, booster, fit_seconds


def make_method(
    booster: lgb.Booster,
    cfg: GBMConfig,
    *,
    clip: bool = True,
    distance: Distance = BEST_DISTANCE,
    predictor: Predictor = BEST_PREDICTOR,
) -> Method:
    """Méthode d'imputation ``(X_holed, reservoir) -> X_filled`` fondée sur le modèle.

    Args:
        booster: Modèle entraîné.
        cfg: Sa configuration (formulation).
        clip: Si vrai, force les prédictions à être positives ou nulles.
        distance: Distance de recherche des voisines (celle de l'entraînement).
        predictor: Configuration du KNN ancré (celle de l'entraînement).

    Returns:
        La méthode.
    """

    def method(X_holed: pd.DataFrame, reservoir: pd.DataFrame) -> pd.DataFrame:
        nb = search(X_holed, reservoir, distance, k_max=predictor.k)
        feats, t, c = build_features(X_holed, reservoir, nb, predictor)
        pred = to_prediction(feats, booster.predict(feats), cfg, clip=clip)
        filled = X_holed.to_numpy(dtype=np.float64).copy()
        filled[t, c] = pred
        return pd.DataFrame(filled, index=X_holed.index, columns=X_holed.columns)

    method.__name__ = cfg.label + ("" if clip else " (sans plancher à 0)")
    return method


def config_dict(cfg: GBMConfig) -> dict[str, object]:
    """Configuration sous forme de dictionnaire (pour MLflow)."""
    return asdict(cfg)
