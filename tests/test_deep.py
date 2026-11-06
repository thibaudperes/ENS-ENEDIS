"""Tests des réseaux à masquage : formes, perte masquée, séquences, anti-fuite."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from enedis_imputation.deep import (
    CHANNELS,
    TCN,
    PatchTransformer,
    SequenceSet,
    TrainConfig,
    calibration_set,
    make_sequences,
    masked_l1_watts,
    split_by_group,
    train_model,
)
from enedis_imputation.gbm import training_curves
from enedis_imputation.knn import BEST_DISTANCE, BEST_PREDICTOR, search

T = 400
INDEX = pd.date_range("2023-01-09", periods=T, freq="30min")


def _reservoir(n: int = 60, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = 1 + 0.5 * np.sin(2 * np.pi * np.arange(T) / 48)
    values = base[:, None] * rng.gamma(2, 100, n) + rng.normal(0, 20, (T, n))
    return pd.DataFrame(np.clip(values, 0, None), INDEX, [f"r{j}" for j in range(n)])


def test_networks_output_one_value_per_time_step() -> None:
    """Les deux réseaux renvoient (lot × temps), y compris pour une longueur impaire."""
    x = torch.randn(3, len(CHANNELS), 1057)
    assert TCN(repeats=1)(x).shape == (3, 1057)
    assert PatchTransformer()(x).shape == (3, 1057)


def _without_norms(module: torch.nn.Module) -> torch.nn.Module:
    """Remplace les GroupNorm par l'identité (leurs statistiques portent sur toute la courbe)."""
    for name, child in module.named_children():
        if isinstance(child, torch.nn.GroupNorm):
            setattr(module, name, torch.nn.Identity())
        else:
            _without_norms(child)
    return module


def test_tcn_receptive_field() -> None:
    """Le chemin convolutif d'une sortie ne voit que les entrées à moins de 127 pas.

    (Les GroupNorm ajoutent en plus des statistiques globales de la courbe : on les
    neutralise pour isoler le champ réceptif des convolutions.)
    """
    model = _without_norms(TCN(repeats=1)).eval()
    x = torch.randn(1, len(CHANNELS), 1057, requires_grad=True)
    model(x)[0, 528].backward()
    influenced = np.flatnonzero(x.grad.abs().sum(dim=1)[0].numpy() > 0)
    half = (model.receptive_field - 1) // 2
    assert model.receptive_field == 255
    assert influenced.min() == 528 - half and influenced.max() == 528 + half


def test_loss_ignores_points_outside_the_mask() -> None:
    """Modifier la prédiction hors du masque ne change pas la perte ; elle est en watts."""
    pred = torch.zeros(2, 5)
    target = torch.tensor([[10.0, 0, 0, 0, 0], [0, 0, -20.0, 0, 0]])
    mask = torch.tensor([[1.0, 0, 0, 0, 0], [0, 0, 1.0, 0, 0]])
    scale = torch.tensor([100.0, 50.0])
    assert masked_l1_watts(pred, target, mask, scale).item() == 15.0
    pred_outside = pred.clone()
    pred_outside[mask == 0] = 1e6
    assert masked_l1_watts(pred_outside, target, mask, scale).item() == 15.0
    pred_inside = torch.tensor([[0.1, 0, 0, 0, 0], [0, 0, -0.4, 0, 0]])  # × échelle = cible
    assert masked_l1_watts(pred_inside, target, mask, scale).item() == 0.0


def test_make_sequences_is_consistent() -> None:
    """Canaux normalisés, cible nulle hors masque, KNN cohérent avec son canal."""
    reservoir = _reservoir()
    y = reservoir.iloc[:, :4]
    mask = np.random.default_rng(3).random(y.shape) < 0.15
    X = y.mask(mask)
    pool = reservoir.iloc[:, 4:]
    nb = search(X, pool, BEST_DISTANCE, k_max=BEST_PREDICTOR.k)
    seqs = make_sequences(X, pool, nb, y)
    assert seqs.inputs.shape == (4, len(CHANNELS), T)
    np.testing.assert_array_equal(seqs.mask, mask.T.astype(np.float32))
    assert (seqs.target[seqs.mask == 0] == 0).all()
    knn_channel = seqs.inputs[:, CHANNELS.index("knn_anchored")] * seqs.scale[:, None]
    np.testing.assert_allclose(knn_channel, seqs.knn, rtol=1e-5)
    np.testing.assert_allclose(
        seqs.target, np.where(mask.T, y.to_numpy().T - seqs.knn, 0), atol=1e-3
    )


def test_split_by_group_keeps_copies_together() -> None:
    """Les copies d'une même courbe sont toutes du même côté."""
    groups = np.repeat(np.arange(50), 2)
    train, valid = split_by_group(groups, 0.2, seed=0)
    assert not set(groups[train]) & set(groups[valid])
    assert len(train) + len(valid) == 100


def test_calibration_curves_never_leak() -> None:
    """Courbes de calibration : jamais cibles d'entraînement, jamais dans leur propre pool."""
    reservoir = _reservoir(80)
    motifs = np.random.default_rng(1).random((T, 6)) < 0.1
    calib = calibration_set(reservoir, motifs, n_calib=10, n_train_curves=40, n_folds=4)
    train_curves, _, _ = training_curves(reservoir, 40, 4)
    assert not set(calib.curves) & set(train_curves)
    assert not set(calib.curves) & set(calib.pool.columns)
    assert calib.X_holed.isna().any().all()


def test_train_model_saves_best_state(tmp_path: Path) -> None:
    """Un mini-entraînement tourne, enregistre le meilleur état et son historique."""
    rng = np.random.default_rng(0)
    n = 8
    seqs = SequenceSet(
        inputs=rng.normal(size=(n, len(CHANNELS), 64)).astype(np.float32),
        target=rng.normal(size=(n, 64)).astype(np.float32),
        mask=(rng.random((n, 64)) < 0.2).astype(np.float32),
        holes=np.zeros((n, 64), np.float32),
        scale=np.ones(n, np.float32),
        knn=np.zeros((n, 64), np.float32),
        groups=np.arange(n),
    )
    path = tmp_path / "model.pt"
    hist = train_model(TCN(repeats=1), seqs, seqs, TrainConfig(max_epochs=2, batch_size=4), path)
    assert path.exists() and path.with_suffix(".csv").exists()
    assert list(hist["epoch"]) == [1, 2]
