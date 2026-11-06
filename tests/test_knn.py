"""Tests du KNN configurable : distances vectorisées, ancrage, garde-fous."""

import numpy as np
import pandas as pd
import pytest

from enedis_imputation.knn import (
    Distance,
    Predictor,
    anchor_residual,
    make_knn,
    point_weights,
    predict,
    search,
)

T = 40
INDEX = pd.date_range("2023-01-09", periods=T, freq="30min")


def _random_problem(seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    reservoir = pd.DataFrame(
        rng.gamma(2.0, 100.0, size=(T, 30)), INDEX, [f"r{j}" for j in range(30)]
    )
    X = pd.DataFrame(rng.gamma(2.0, 100.0, size=(T, 4)), INDEX, [f"x{j}" for j in range(4)])
    X.iloc[5:9, 0] = np.nan
    X.iloc[[0, 20, 21], 1] = np.nan
    X.iloc[30:, 2] = np.nan
    X.iloc[12, 3] = np.nan
    return X, reservoir


def _brute_force(X: pd.DataFrame, R: pd.DataFrame, distance: Distance) -> np.ndarray:
    """Distances calculées naïvement, cible par cible et voisine par voisine."""
    W = point_weights(X.isna().to_numpy(), distance.window).astype(bool)
    out = np.empty((X.shape[1], R.shape[1]))
    for i in range(X.shape[1]):
        x = X.to_numpy()[W[:, i], i]
        for j in range(R.shape[1]):
            r = R.to_numpy()[W[:, i], j]
            if distance.shape and distance.metric == "l2":
                out[i, j] = 1 - np.corrcoef(x, r)[0, 1]
            elif distance.metric == "l2":
                out[i, j] = np.mean((x - r) ** 2)
            else:
                out[i, j] = np.mean(np.abs(x - r))
    return out


@pytest.mark.parametrize(
    "distance",
    [
        Distance("l2"),
        Distance("l2", window=3),
        Distance("l2", shape=True),
        Distance("l1"),
        Distance("l1", window=3),
    ],
)
def test_search_matches_brute_force(distance: Distance) -> None:
    """Les distances vectorisées (produits matriciels) égalent le calcul naïf."""
    X, R = _random_problem()
    nb = search(X, R, distance, k_max=5)
    brute = _brute_force(X, R, distance)
    np.testing.assert_array_equal(nb.index, np.argsort(brute, axis=1)[:, :5])
    np.testing.assert_allclose(nb.distance, np.sort(brute, axis=1)[:, :5], rtol=1e-4)


def test_point_weights_window() -> None:
    """Seuls les points observés à moins de `window` pas d'un trou sont retenus."""
    mask = np.zeros((10, 1), dtype=bool)
    mask[5] = True
    w = point_weights(mask, window=2)[:, 0]
    np.testing.assert_array_equal(np.flatnonzero(w), [3, 4, 6, 7])


def test_anchor_residual_interpolates_between_edges() -> None:
    """Le résidu passe linéairement de l'ancre gauche à l'ancre droite, constant aux bords."""
    residual = np.array([[0.0, 10, 99, 99, 99, 30, 0, 99, 99]]).T
    mask = np.array([[False, False, True, True, True, False, False, True, True]]).T
    corr = anchor_residual(residual, mask, points=1)[:, 0]
    np.testing.assert_allclose(corr, [0, 0, 15, 20, 25, 0, 0, 0, 0])
    corr2 = anchor_residual(residual, mask, points=2)[:, 0]
    np.testing.assert_allclose(corr2[2:5], [5 + 10 * 1 / 4, 5 + 10 * 2 / 4, 5 + 10 * 3 / 4])


def test_anchor_decay_fades_in_the_middle() -> None:
    """Avec atténuation, la correction est pleine au bord et décroît vers le centre."""
    residual = np.array([[10.0] + [0.0] * 9 + [10.0]]).T
    mask = np.zeros_like(residual, dtype=bool)
    mask[1:10] = True
    corr = anchor_residual(residual, mask, decay=2.0)[:, 0]
    assert corr[1] == pytest.approx(10.0) and corr[9] == pytest.approx(10.0)
    assert corr[5] == pytest.approx(10.0 * np.exp(-4 / 2))


def test_anchored_knn_recovers_shifted_shape() -> None:
    """Si la cible est une voisine décalée d'une constante, le KNN ancré la retrouve."""
    t = np.arange(T, dtype=float)
    shape = 100 + 50 * np.sin(t / 3)
    R = pd.DataFrame({"r0": shape, "r1": shape[::-1], "r2": np.full(T, 100.0)}, INDEX)
    truth = shape + 200
    X = pd.DataFrame({"x": truth}, INDEX)
    X.iloc[10:20, 0] = np.nan
    method = make_knn(Distance("l2", shape=True), Predictor(k=1, anchor=True))
    np.testing.assert_allclose(method(X, R)["x"], truth, atol=1e-6)


def test_predict_refuses_another_reservoir() -> None:
    """Les indices de voisines n'ont de sens que pour le réservoir de la recherche."""
    X, R = _random_problem()
    nb = search(X, R, Distance("l2"), k_max=3)
    with pytest.raises(ValueError, match="réservoir"):
        predict(X, R.iloc[:, :-1], nb, Predictor(k=3))


def test_predictions_are_non_negative_and_keep_observed() -> None:
    """Les points observés sont conservés, les prédictions jamais négatives."""
    X, R = _random_problem()
    out = make_knn(Distance("l2"), Predictor(k=3, scaling="affine_ls", anchor=True))(X, R)
    observed = X.notna().to_numpy()
    np.testing.assert_allclose(out.to_numpy()[observed], X.to_numpy()[observed])
    assert (out.to_numpy() >= 0).all() and not out.isna().any().any()
