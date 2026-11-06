"""Tests de l'identité visuelle."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from enedis_imputation.viz import PALETTE, method_color, plot_imputation, set_style


def test_method_color_by_keyword() -> None:
    """La couleur d'une méthode est reconnue dans un libellé libre."""
    assert method_color("Interpolation linéaire") == PALETTE["interpolation"]
    assert method_color("KNN (médiane de 5)") == PALETTE["knn"]
    assert method_color("KNN naïf (médiane de 5)") == PALETTE["knn_naive"]
    assert method_color("Modèle final") == PALETTE["model"]
    assert method_color("Vraie courbe") == PALETTE["true"]
    with pytest.raises(KeyError):
        method_color("Méthode inconnue")


def test_plot_imputation_draws_predictions_only_around_holes() -> None:
    """Les reconstructions n'apparaissent que dans les trous et à leurs bords."""
    set_style()
    index = pd.date_range("2023-01-09", periods=10, freq="30min")
    true = pd.Series(np.arange(10, dtype=float), index=index)
    mask = np.zeros(10, dtype=bool)
    mask[4:6] = True
    fig = plot_imputation(true, {"Interpolation": np.full(10, 99.0)}, mask, "Conclusion")

    line = next(ln for ln in fig.axes[0].get_lines() if ln.get_label() == "Interpolation")
    drawn = ~np.isnan(np.asarray(line.get_ydata(), dtype=float))
    np.testing.assert_array_equal(np.flatnonzero(drawn), [3, 4, 5, 6])
    assert fig.get_suptitle() == "Conclusion"
    plt.close(fig)
