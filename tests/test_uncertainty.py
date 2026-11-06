"""Tests des intervalles conformes et de l'analyse du biais."""

import numpy as np
import pandas as pd
import pytest

from enedis_imputation.holes import length_bucket
from enedis_imputation.uncertainty import (
    bias_table,
    conformal_quantile,
    coverage_table,
    fit_conformal,
    intervals,
)


def test_conformal_quantile_rank() -> None:
    """Le quantile conforme est la ⌈(n + 1)(1 − α)⌉-ième plus petite valeur."""
    scores = np.arange(1, 10)  # n = 9 ; α = 0,2 → rang ⌈10 × 0,8⌉ = 8
    assert conformal_quantile(scores, 0.2) == 8
    assert conformal_quantile(scores, 0.01) == float("inf")  # rang 10 > n


@pytest.mark.parametrize("kind", ["classe", "normalisé", "normalisé par classe"])
def test_coverage_on_exchangeable_data(kind: str) -> None:
    """Sur des données échangeables, la couverture atteint le niveau visé."""
    rng = np.random.default_rng(0)
    n = 20_000
    lengths = rng.choice([1, 2, 5, 20, 60], size=2 * n)
    sigma = 5.0 + lengths  # les trous longs sont plus difficiles
    errors = rng.normal(0, sigma)
    classes = length_bucket(lengths)
    cal, test = slice(0, n), slice(n, 2 * n)
    model = fit_conformal(np.abs(errors[cal]), classes[cal], kind, sigma=sigma[cal])
    y = 100 + errors[test]
    pred = np.full(n, 100.0)
    for alpha in (0.2, 0.1):
        low, high = intervals(model, pred, classes[test], sigma=sigma[test], alpha=alpha)
        coverage = ((y >= low) & (y <= high)).mean()
        assert abs(coverage - (1 - alpha)) < 0.01


def test_intervals_never_below_zero() -> None:
    """Une consommation n'est jamais négative : la borne basse est ramenée à 0."""
    classes = length_bucket(np.array([1, 1, 1]))
    model = fit_conformal(np.array([50.0, 60.0, 70.0]), classes, "classe", alphas=(0.5,))
    low, high = intervals(model, np.array([10.0, 0.0, 200.0]), classes, alpha=0.5)
    assert (low >= 0).all()
    # n = 3, α = 0,5 → rang ⌈4 × 0,5⌉ = 2 → quantile 60
    np.testing.assert_allclose(high - np.array([10.0, 0.0, 200.0]), 60.0)


def test_coverage_table_counts() -> None:
    """Couverture et largeur calculées par classe, NaN de la vérité ignorés."""
    classes = length_bucket(np.array([1, 1, 2, 2]))
    table = coverage_table(
        y=np.array([5.0, 50.0, 1.0, np.nan]),
        low=np.zeros(4),
        high=np.array([10.0, 10.0, 2.0, 2.0]),
        classes=classes,
    )
    assert table.loc["1", "couverture"] == 0.5
    assert table.loc["2", "couverture"] == 1.0
    assert table.loc["toutes", "largeur (W)"] == pytest.approx((10 + 10 + 2) / 3)


def test_bias_sign() -> None:
    """Le biais est la moyenne de (prédiction − vérité) : négatif si l'on sous-estime."""
    classes = length_bucket(np.array([1, 1]))
    table = bias_table(np.array([90.0, 100.0]), np.array([100.0, 110.0]), classes)
    assert table.loc["toutes", "biais (W)"] == pytest.approx(-10.0)
    assert table.loc["toutes", "biais (%)"] == pytest.approx(-10.0 / 105.0)
    assert isinstance(table, pd.DataFrame)
