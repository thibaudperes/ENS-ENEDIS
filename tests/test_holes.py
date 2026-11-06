"""Tests de la description des trous."""

import numpy as np

from enedis_imputation.holes import hole_length_per_point, hole_runs, length_bucket


def test_hole_runs_detects_edges() -> None:
    """Les trous en début, milieu et fin de courbe sont détectés, courbe par courbe."""
    mask = np.array(
        [
            [True, False],
            [True, True],
            [False, True],
            [True, True],
        ]
    )
    runs = hole_runs(mask)
    assert runs.to_dict("list") == {
        "curve": [0, 0, 1],
        "start": [0, 3, 1],
        "length": [2, 1, 3],
    }


def test_hole_length_per_point() -> None:
    """Chaque point manquant reçoit la longueur de son trou, les autres 0."""
    mask = np.array([[True, False], [True, True], [False, True], [True, True]])
    expected = np.array([[2, 0], [2, 3], [0, 3], [1, 3]])
    np.testing.assert_array_equal(hole_length_per_point(mask), expected)


def test_length_bucket() -> None:
    """Les bornes des classes sont incluses à droite."""
    assert list(length_bucket([1, 2, 3, 6, 7, 48, 49])) == [
        "1",
        "2",
        "3-6",
        "3-6",
        "7-48",
        "7-48",
        ">48",
    ]
