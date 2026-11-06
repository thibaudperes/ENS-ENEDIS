"""Tests de la métrique ``mae_on_holes``."""

import numpy as np
import pandas as pd
import pytest

from enedis_imputation.data import mae_on_holes


def test_only_masked_positions_count() -> None:
    """Les erreurs hors du masque sont ignorées."""
    y = np.array([[10.0, 20.0], [30.0, 40.0]])
    pred = np.array([[11.0, 1000.0], [27.0, 1000.0]])
    mask = np.array([[True, False], [True, False]])
    assert mae_on_holes(pred, y, mask) == pytest.approx((1 + 3) / 2)


def test_nan_in_y_are_excluded() -> None:
    """Une position masquée dont la vérité est NaN n'entre pas dans la moyenne."""
    y = np.array([[10.0, np.nan], [30.0, 40.0]])
    pred = np.array([[12.0, 999.0], [30.0, 44.0]])
    mask = np.ones_like(y, dtype=bool)
    assert mae_on_holes(pred, y, mask) == pytest.approx((2 + 0 + 4) / 3)


def test_accepts_dataframes() -> None:
    """La métrique accepte des DataFrames alignés."""
    y = pd.DataFrame({"holed_1": [1.0, 2.0], "holed_2": [3.0, np.nan]})
    pred = y.fillna(0.0) + 1.0
    mask = pd.DataFrame(True, index=y.index, columns=y.columns)
    assert mae_on_holes(pred, y, mask) == pytest.approx(1.0)


def test_misaligned_dataframes_raise() -> None:
    """Des colonnes dans un ordre différent lèvent une erreur au lieu d'un score faux."""
    y = pd.DataFrame({"a": [1.0], "b": [2.0]})
    with pytest.raises(ValueError, match="alignées"):
        mae_on_holes(y[["b", "a"]], y, y.notna())


def test_nan_in_pred_raises() -> None:
    """Une prédiction manquante sur un trou évalué est une erreur, pas un score silencieux."""
    y = np.array([[1.0, 2.0]])
    pred = np.array([[1.0, np.nan]])
    with pytest.raises(ValueError, match="NaN"):
        mae_on_holes(pred, y, np.ones_like(y, dtype=bool))


def test_shape_mismatch_raises() -> None:
    """Des formes différentes lèvent une erreur."""
    with pytest.raises(ValueError, match="Formes"):
        mae_on_holes(np.zeros((2, 2)), np.zeros((2, 3)), np.ones((2, 3), dtype=bool))
