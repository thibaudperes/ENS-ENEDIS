"""Tests du protocole de validation : greffe, découpage, anti-fuite, évaluation."""

import numpy as np
import pandas as pd
import pytest

from enedis_imputation.data import RAW_DIR
from enedis_imputation.validation import (
    Split,
    assign_curves,
    check_no_leak,
    evaluate,
    exact_duplicates,
    graft,
    load_split,
)

HAS_DATA = (RAW_DIR / "X_train.csv").exists()
INDEX = pd.date_range("2023-01-09", periods=6, freq="30min")


def _frame(values: np.ndarray, prefix: str) -> pd.DataFrame:
    return pd.DataFrame(
        values, index=INDEX, columns=[f"{prefix}{j}" for j in range(values.shape[1])]
    )


def test_graft_puts_nan_exactly_on_motif() -> None:
    """La greffe met des NaN aux positions du motif, et nulle part ailleurs."""
    full = _frame(np.arange(12, dtype=float).reshape(6, 2), "c")
    motif = np.zeros((6, 2), dtype=bool)
    motif[1:3, 0] = True
    motif[5, 1] = True
    holed = graft(full, motif)
    np.testing.assert_array_equal(holed.isna().to_numpy(), motif)
    np.testing.assert_array_equal(holed.to_numpy()[~motif], full.to_numpy()[~motif])


def test_assign_curves_is_disjoint_and_respects_forbidden() -> None:
    """Cibles dev, cibles holdout et réservoir sont disjoints et couvrent tout."""
    candidates = pd.Index([f"c{i}" for i in range(100)])
    forbidden = pd.Index(["c0", "c1", "c2"])
    groups = assign_curves(candidates, forbidden, n_dev=30, n_holdout=20, seed=0)
    dev, holdout, reservoir = (set(groups[k]) for k in ("dev", "holdout", "reservoir"))
    assert (len(dev), len(holdout), len(reservoir)) == (30, 20, 50)
    assert not (dev & holdout or dev & reservoir or holdout & reservoir)
    assert dev | holdout | reservoir == set(candidates)
    assert set(forbidden) <= reservoir


def test_assign_curves_is_reproducible() -> None:
    """Même graine, même découpage."""
    candidates = pd.Index([f"c{i}" for i in range(50)])
    a = assign_curves(candidates, pd.Index([]), 10, 10, seed=42)
    b = assign_curves(candidates, pd.Index([]), 10, 10, seed=42)
    assert a == b


def test_exact_duplicates_ignores_zero_curves() -> None:
    """Les doubles non nuls sont détectés ; les courbes nulles identiques sont tolérées."""
    values = np.column_stack([np.ones(6), np.ones(6), np.zeros(6), np.zeros(6), np.arange(6)])
    assert list(exact_duplicates(_frame(values, "c"))) == ["c0", "c1"]


def test_check_no_leak_raises() -> None:
    """Une cible présente dans le réservoir est une fuite."""
    targets = _frame(np.ones((6, 2)), "c")
    with pytest.raises(ValueError, match="Fuite"):
        check_no_leak(targets, _frame(np.ones((6, 3)), "c"))
    check_no_leak(targets, _frame(np.ones((6, 3)), "r"))  # pas d'erreur


def _toy_split(name: str) -> Split:
    y = _frame(np.arange(12, dtype=float).reshape(6, 2), "c")
    motif = np.zeros((6, 2), dtype=bool)
    motif[2:4, 0] = True
    return Split(name, graft(y, motif), y, _frame(np.zeros((6, 3)), "r"))


def test_evaluate_refuses_holdout_without_flag() -> None:
    """Le holdout ne s'évalue qu'avec une autorisation explicite."""
    with pytest.raises(PermissionError):
        evaluate(lambda X, R: X.fillna(0.0), _toy_split("holdout"), register=False)
    evaluate(lambda X, R: X.fillna(0.0), _toy_split("holdout"), allow_holdout=True, register=False)


def test_evaluate_computes_mae_on_holes_only() -> None:
    """Une erreur constante de 5 W dans les trous donne une MAE de 5, classe '2'."""
    split = _toy_split("dev")
    result = evaluate(lambda X, R: X.fillna(split.y + 5.0), split, register=False)
    assert result.mae == pytest.approx(5.0)
    assert result.by_class.loc["2", "mae"] == pytest.approx(5.0)
    assert result.by_class.loc["2", "points"] == 2
    assert result.by_class["points"].sum() == 2


def test_evaluate_detects_leak() -> None:
    """La fonction evaluate refuse un jeu dont une cible est dans le réservoir."""
    split = _toy_split("dev")
    leaky = Split("dev", split.X_holed, split.y, split.y)
    with pytest.raises(ValueError, match="Fuite"):
        evaluate(lambda X, R: X.fillna(0.0), leaky, register=False)


@pytest.mark.skipif(not HAS_DATA, reason="données du challenge absentes")
def test_real_splits_never_leak() -> None:
    """Sur les vrais jeux : aucune cible dans le réservoir, jeux disjoints, greffe exacte."""
    splits = {name: load_split(name) for name in ("dev", "holdout", "real")}
    reservoir = set(splits["dev"].reservoir.columns)
    for split in splits.values():
        assert not set(split.X_holed.columns) & reservoir
        assert set(split.reservoir.columns) == reservoir  # même réservoir pour les 3 jeux
    assert not set(splits["dev"].X_holed.columns) & set(splits["holdout"].X_holed.columns)

    # Les cibles n'ont pas de double exact (non nul) dans le réservoir
    dev = splits["dev"]
    assert not set(dev.X_holed.columns) & set(exact_duplicates(dev.y.join(dev.reservoir)))

    # Chaque cible porte exactement le motif de NaN qui lui a été attribué
    real_train = splits["real"].X_holed
    for target, motif in list(zip(dev.X_holed.columns, dev.motifs, strict=True))[:50]:
        if motif.startswith("train:"):
            expected = real_train[motif.removeprefix("train:")].isna().to_numpy()
            np.testing.assert_array_equal(dev.X_holed[target].isna().to_numpy(), expected)
