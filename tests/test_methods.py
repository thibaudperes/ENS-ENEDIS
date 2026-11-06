"""Tests des méthodes de référence."""

import numpy as np
import pandas as pd
import pytest

from enedis_imputation.methods import (
    knn_median,
    linear_interpolation,
    nearest_neighbors,
    scaled_median_fill,
    with_fallback,
)

INDEX = pd.date_range("2023-01-09", periods=8, freq="30min")


def test_linear_interpolation_fills_edges() -> None:
    """L'interpolation relie les points et prolonge les bords par constante."""
    X = pd.DataFrame({"a": [np.nan, 2.0, np.nan, 4.0, np.nan, np.nan, np.nan, np.nan]}, INDEX)
    out = linear_interpolation(X, pd.DataFrame(index=INDEX))
    np.testing.assert_allclose(out["a"], [2, 2, 3, 4, 4, 4, 4, 4])


def test_knn_finds_the_matching_curve() -> None:
    """Le plus proche voisin est la courbe identique sur les points observés."""
    t = np.arange(8, dtype=float)
    reservoir = pd.DataFrame({"r0": t, "r1": t + 100, "r2": -t}, INDEX)
    X = pd.DataFrame({"x": np.where(t % 3 == 0, np.nan, t + 100)}, INDEX)
    assert nearest_neighbors(X, reservoir, k=1)[0, 0] == 1
    filled = knn_median(X, reservoir, k=1)
    np.testing.assert_allclose(filled["x"], t + 100)


def test_knn_keeps_observed_values() -> None:
    """Les points observés ne sont jamais modifiés."""
    reservoir = pd.DataFrame({"r0": np.zeros(8), "r1": np.ones(8)}, INDEX)
    X = pd.DataFrame({"x": [5.0, np.nan, 7.0, np.nan, 1, 1, 1, 1]}, INDEX)
    filled = knn_median(X, reservoir, k=1)
    np.testing.assert_allclose(filled["x"].iloc[[0, 2, 4]], [5.0, 7.0, 1.0])


def test_scaled_median_fill_matches_level() -> None:
    """Le profil médian est remis au niveau des points observés."""
    profile = np.array([1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0])
    reservoir = pd.DataFrame({"r0": profile, "r1": profile, "r2": profile}, INDEX)
    X = pd.DataFrame({"x": [10.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]}, INDEX)
    filled = scaled_median_fill(X, reservoir)
    np.testing.assert_allclose(filled["x"], profile * 10)


def test_with_fallback_routes_sparse_curves() -> None:
    """Les courbes trop vides passent par le secours, les autres par la méthode."""
    reservoir = pd.DataFrame({"r0": np.full(8, 3.0)}, INDEX)
    dense = [1.0, np.nan, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    sparse = [6.0] + [np.nan] * 7
    X = pd.DataFrame({"dense": dense, "sparse": sparse}, INDEX)
    filled = with_fallback(lambda X, R: X.fillna(-1.0), min_observed=4)(X, reservoir)
    assert list(filled.columns) == ["dense", "sparse"]
    assert filled.loc[INDEX[1], "dense"] == -1.0
    assert filled.loc[INDEX[1], "sparse"] == pytest.approx(6.0)
