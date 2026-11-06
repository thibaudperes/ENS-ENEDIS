"""Tests du stacking LightGBM : entraînement hors pli sans fuite, variables, prédiction."""

import numpy as np
import pandas as pd
import pytest

from enedis_imputation.features import FEATURES, build_features
from enedis_imputation.gbm import (
    GBMConfig,
    TrainingSet,
    fit,
    iter_folds,
    make_method,
    to_prediction,
)
from enedis_imputation.knn import BEST_DISTANCE, BEST_PREDICTOR, predict, search

T = 400
INDEX = pd.date_range("2023-01-09", periods=T, freq="30min")


def _reservoir(n: int = 60, seed: int = 0) -> pd.DataFrame:
    """Courbes synthétiques : un cycle journalier commun, des niveaux et bruits différents."""
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    base = 1 + 0.5 * np.sin(2 * np.pi * t / 48)
    values = base[:, None] * rng.gamma(2, 100, n) + rng.normal(0, 20, (T, n))
    return pd.DataFrame(np.clip(values, 0, None), INDEX, [f"r{j}" for j in range(n)])


def _motifs(m: int = 10, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    motifs = rng.random((T, m)) < 0.1
    for j in range(m):  # un long trou par motif, comme dans les vraies données
        s = rng.integers(0, T - 48)
        motifs[s : s + 48, j] = True
    return motifs


def test_out_of_fold_neighbors_never_include_the_curve_or_its_fold() -> None:
    """Aucune courbe n'est sa propre voisine, ni voisine d'une autre courbe de son pli."""
    reservoir = _reservoir()
    seen = set()
    for fd in iter_folds(reservoir, _motifs(), n_curves=30, motifs_per_curve=2, n_folds=3, k=10):
        fold_sources = set(fd.sources)
        assert not fold_sources & set(fd.pool.columns)
        for j, src in enumerate(fd.sources):
            neighbor_names = set(fd.pool.columns[fd.neighbors.index[j]])
            assert src not in neighbor_names
            assert not neighbor_names & fold_sources
        seen |= fold_sources
    assert len(seen) == 30  # chaque courbe cible appartient à exactement un pli


def test_copies_of_a_curve_get_distinct_motifs() -> None:
    """Les deux copies d'une même courbe ne portent pas le même motif."""
    fd = next(iter_folds(_reservoir(), _motifs(), n_curves=10, motifs_per_curve=2, n_folds=2))
    masks = fd.X_holed.isna().to_numpy()
    for j in range(0, masks.shape[1], 2):
        assert not np.array_equal(masks[:, j], masks[:, j + 1])


def test_features_match_the_anchored_knn_and_lags() -> None:
    """La variable knn_anchored égale la prédiction du KNN ancré ; les décalages sont exacts."""
    reservoir = _reservoir()
    X = reservoir.iloc[:, :3].copy()
    pool = reservoir.iloc[:, 3:]
    X = X.mask(_motifs(3))
    nb = search(X, pool, BEST_DISTANCE, k_max=BEST_PREDICTOR.k)
    feats, t, c = build_features(X, pool, nb)
    assert list(feats.columns) == FEATURES
    assert len(feats) == X.isna().to_numpy().sum()
    expected = predict(X, pool, nb, BEST_PREDICTOR).to_numpy()[t, c]
    np.testing.assert_allclose(feats["knn_anchored"], expected, rtol=1e-4, atol=1e-3)
    lag = np.where(t >= 48, X.to_numpy()[np.maximum(t - 48, 0), c], np.nan)
    np.testing.assert_allclose(feats["lag_m48"], lag, equal_nan=True, rtol=1e-6)
    assert (feats["dist_left"].fillna(1) >= 1).all() and (feats["rel_pos"].between(0, 1)).all()


def test_to_prediction_residual_and_clip() -> None:
    """Formulation résidu : KNN + sortie ; le plancher à 0 s'applique."""
    X = pd.DataFrame({"knn_anchored": [10.0, 5.0]})
    cfg = GBMConfig(formulation="residual")
    np.testing.assert_allclose(to_prediction(X, np.array([1.0, -8.0]), cfg), [11.0, 0.0])
    np.testing.assert_allclose(to_prediction(X, np.array([1.0, -8.0]), cfg, clip=False), [11, -3])


def test_method_fills_only_holes() -> None:
    """La méthode LightGBM remplit les trous sans toucher aux points observés."""
    reservoir = _reservoir()
    fd = next(iter_folds(reservoir, _motifs(), n_curves=20, motifs_per_curve=1, n_folds=2))
    feats, t, c = build_features(fd.X_holed, fd.pool, fd.neighbors)
    ts = TrainingSet(feats, fd.y.to_numpy(np.float32)[t, c], c.astype(np.int64))
    cfg = GBMConfig(num_leaves=7, min_child_samples=5)
    method = make_method(fit(ts, cfg, n_rounds=20), cfg)
    out = method(fd.X_holed, fd.pool)
    observed = fd.X_holed.notna().to_numpy()
    np.testing.assert_allclose(out.to_numpy()[observed], fd.X_holed.to_numpy()[observed])
    assert not out.isna().any().any() and (out.to_numpy() >= 0).all()


@pytest.mark.parametrize("formulation", ["residual", "direct"])
def test_config_params_use_l1(formulation: str) -> None:
    """L'objectif est la MAE (L1), quelle que soit la formulation."""
    assert GBMConfig(formulation=formulation).params()["objective"] == "l1"
