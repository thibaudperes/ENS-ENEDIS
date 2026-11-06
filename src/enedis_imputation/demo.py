"""Imputation d'une seule courbe, pour la démo interactive et les visuels du README.

Tout part des caches et modèles déjà calculés : jeu dev et son réservoir, LightGBM, TCN,
quantiles conformes. Rien n'est réentraîné. Le seul calcul un peu long (~40 s), la
calibration des intervalles, est fait une fois puis enregistré.
"""

from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch

from enedis_imputation.data import PROCESSED_DIR
from enedis_imputation.deep import TCN, calibration_set
from enedis_imputation.gbm import GBMConfig, load_cached_booster
from enedis_imputation.holes import hole_length_per_point, hole_runs, length_bucket
from enedis_imputation.knn import anchored_knn
from enedis_imputation.methods import linear_interpolation
from enedis_imputation.uncertainty import (
    ConformalModel,
    fit_conformal,
    intervals,
    load_conformal,
    predict_components,
    save_conformal,
)
from enedis_imputation.validation import Split, load_split, motif_masks

BLEND_WEIGHT: float = 0.20
"""Poids du LightGBM dans le mélange (choisi sur la calibration, notebook 05)."""

CONFORMAL_PATH = PROCESSED_DIR / "conformal_classe.json"
GBM_CONFIG = GBMConfig("residual", 63)


@dataclass
class Demo:
    """Tout ce qu'il faut pour reconstruire une courbe de dev.

    Attributes:
        dev: Jeu dev (courbes et réservoir de voisines).
        booster: LightGBM.
        tcn: Réseau TCN.
        conformal: Quantiles conformes par classe de trou.
        motifs: Les 1 999 motifs de trous réels (temps × motifs).
    """

    dev: Split
    booster: lgb.Booster
    tcn: torch.nn.Module
    conformal: ConformalModel
    motifs: pd.DataFrame


def calibrate_conformal(booster: lgb.Booster, tcn: torch.nn.Module) -> ConformalModel:
    """Calibre les intervalles par classe sur les 1 000 courbes de calibration (~40 s)."""
    dev = load_split("dev")
    calib = calibration_set(dev.reservoir, motif_masks().to_numpy())
    comp = predict_components(calib.X_holed, calib.pool, booster, GBM_CONFIG, tcn, BLEND_WEIGHT)
    y = calib.y.to_numpy()[comp.t, comp.c]
    ok = ~np.isnan(y)
    return fit_conformal(np.abs(comp.blend - y)[ok], comp.classes[ok], "classe")


def load_demo() -> Demo:
    """Charge le jeu dev et les modèles ; calibre les intervalles au premier appel."""
    tcn = TCN(repeats=1)
    tcn.load_state_dict(torch.load(PROCESSED_DIR / "deep_tcn.pt", map_location="cpu"))
    tcn.eval()
    booster = load_cached_booster(GBM_CONFIG)
    if CONFORMAL_PATH.exists():
        conformal = load_conformal(CONFORMAL_PATH)
    else:
        conformal = calibrate_conformal(booster, tcn)
        save_conformal(conformal, CONFORMAL_PATH)
    return Demo(load_split("dev"), booster, tcn, conformal, motif_masks())


def custom_mask(length: int, start: int, n_steps: int) -> npt.NDArray[np.bool_]:
    """Masque d'un seul trou de ``n_steps`` pas à partir du pas ``start``."""
    mask = np.zeros(length, dtype=bool)
    mask[start : start + n_steps] = True
    return mask


@dataclass(frozen=True)
class Reconstruction:
    """Reconstructions d'une courbe et leurs erreurs.

    Attributes:
        truth: Vraie courbe (index horaire).
        mask: Points masqués (vrai = trou).
        preds: ``{méthode: courbe complétée}``.
        low: Borne basse de l'intervalle à 90 % (NaN hors des trous).
        high: Borne haute de l'intervalle à 90 % (NaN hors des trous).
        mae: ``{méthode: MAE sur les trous}`` (W).
        coverage: Part des vraies valeurs dans l'intervalle à 90 %.
    """

    truth: pd.Series
    mask: npt.NDArray[np.bool_]
    preds: dict[str, npt.NDArray[np.float64]]
    low: npt.NDArray[np.float64]
    high: npt.NDArray[np.float64]
    mae: dict[str, float]
    coverage: float


def reconstruct(demo: Demo, truth: pd.Series, mask: npt.ArrayLike) -> Reconstruction:
    """Masque une courbe, la reconstruit avec chaque méthode et calcule ses erreurs.

    La courbe ne fait pas partie du réservoir (c'est une courbe de dev) : aucune méthode
    ne voit ses vraies valeurs.

    Args:
        demo: Données et modèles.
        truth: Vraie courbe complète.
        mask: Points à masquer.

    Returns:
        Les reconstructions.
    """
    m = np.asarray(mask, dtype=bool)
    X = truth.to_frame().mask(m[:, None])
    reservoir = demo.dev.reservoir
    comp = predict_components(X, reservoir, demo.booster, GBM_CONFIG, demo.tcn, BLEND_WEIGHT)
    blend = X.iloc[:, 0].to_numpy(dtype=np.float64).copy()
    blend[comp.t] = comp.blend
    low, high = np.full(len(m), np.nan), np.full(len(m), np.nan)
    low[comp.t], high[comp.t] = intervals(demo.conformal, comp.blend, comp.classes, alpha=0.1)
    preds = {
        "Interpolation linéaire": linear_interpolation(X, reservoir).iloc[:, 0].to_numpy(),
        "KNN ancré": anchored_knn(X, reservoir).iloc[:, 0].to_numpy(),
        "Modèle final (mélange)": blend,
    }
    y = truth.to_numpy(dtype=np.float64)
    mae = {name: float(np.abs(p[m] - y[m]).mean()) for name, p in preds.items()}
    coverage = float(((y[m] >= low[m]) & (y[m] <= high[m])).mean())
    return Reconstruction(truth, m, preds, low, high, mae, coverage)


def zoom(mask: npt.ArrayLike, context: int = 96) -> slice:
    """Fenêtre centrée sur le plus long trou, avec ``context`` pas de chaque côté."""
    m = np.asarray(mask, dtype=bool)
    runs = hole_runs(m)
    longest = runs.loc[runs["length"].idxmax()]
    start, end = int(longest["start"]), int(longest["start"] + longest["length"])
    return slice(max(0, start - context), min(len(m), end + context))


def hole_summary(mask: npt.ArrayLike) -> pd.Series:
    """Nombre de points masqués par classe de longueur de trou."""
    m = np.asarray(mask, dtype=bool)
    return pd.Series(length_bucket(hole_length_per_point(m)[m])).value_counts().sort_index()
