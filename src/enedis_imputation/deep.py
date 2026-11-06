"""Réseaux de neurones à masquage : une séquence = une courbe entière.

Le réseau reçoit, pour chaque courbe, des canaux alignés sur les 1 057 pas de temps
(valeurs observées, masque, KNN ancré, interpolation, désaccord des voisines, calendrier),
normalisés par le niveau moyen observé de la courbe. Il prédit le **résidu** (y − KNN
ancré) en unités normalisées ; la perte est une L1 calculée **uniquement sur les points
masqués**, remise en watts (on remultiplie par l'échelle de la courbe) pour coller à la
métrique du challenge.

Deux architectures :

- ``TCN`` : convolutions 1D dilatées non causales, à connexions résiduelles ;
- ``PatchTransformer`` : transformer sur des patchs de 8 pas.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, fields
from pathlib import Path

import lightgbm as lgb
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from torch import nn

from enedis_imputation import SEED
from enedis_imputation.data import PROCESSED_DIR
from enedis_imputation.features import build_features, curve_features
from enedis_imputation.gbm import GBMConfig, iter_folds, to_prediction, training_curves
from enedis_imputation.knn import BEST_DISTANCE, BEST_PREDICTOR, Neighbors, search
from enedis_imputation.methods import Method
from enedis_imputation.validation import exact_duplicates, graft

CHANNELS: list[str] = [
    "observed",
    "mask",
    "knn_anchored",
    "interp",
    "nb_std",
    "hour_sin",
    "hour_cos",
    "weekend",
]
"""Canaux d'entrée, dans l'ordre."""


class ResidualBlock(nn.Module):
    """Bloc résiduel : normalisation, convolution dilatée, activation, convolution 1×1."""

    def __init__(self, width: int, dilation: int, kernel: int = 3, dropout: float = 0.0) -> None:
        """Construit le bloc.

        Args:
            width: Nombre de canaux internes.
            dilation: Dilatation de la convolution (espacement entre les points vus).
            kernel: Taille du noyau.
            dropout: Taux de dropout.
        """
        super().__init__()
        self.norm = nn.GroupNorm(8, width)
        self.conv = nn.Conv1d(
            width, width, kernel, dilation=dilation, padding=dilation * (kernel - 1) // 2
        )
        self.act = nn.GELU()
        self.mix = nn.Conv1d(width, width, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applique le bloc : x + f(x), même longueur en sortie."""
        return x + self.drop(self.mix(self.act(self.conv(self.norm(x)))))


class TCN(nn.Module):
    """Réseau convolutif temporel à dilatations croissantes (non causal).

    Avec un noyau de 3 et les dilatations 1, 2, 4, …, 64, le chemin convolutif de chaque
    sortie voit 1 + 2 × 127 = 255 pas (plus de 5 jours) ; 509 pas si la pile est répétée
    deux fois. Les ``GroupNorm`` y ajoutent des statistiques globales de la courbe (moyenne
    et variance sur toute sa longueur), soit un contexte global.
    """

    def __init__(
        self,
        in_channels: int = len(CHANNELS),
        width: int = 64,
        dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64),
        repeats: int = 2,
        kernel: int = 3,
        dropout: float = 0.0,
    ) -> None:
        """Construit le réseau.

        Args:
            in_channels: Nombre de canaux d'entrée.
            width: Nombre de canaux internes.
            dilations: Dilatations d'une pile de blocs.
            repeats: Nombre de répétitions de la pile.
            kernel: Taille des noyaux.
            dropout: Taux de dropout.
        """
        super().__init__()
        self.kernel, self.dilations, self.repeats = kernel, dilations, repeats
        self.inp = nn.Conv1d(in_channels, width, 1)
        self.blocks = nn.Sequential(
            *[ResidualBlock(width, d, kernel, dropout) for _ in range(repeats) for d in dilations]
        )
        self.out = nn.Sequential(nn.GroupNorm(8, width), nn.GELU(), nn.Conv1d(width, 1, 1))

    @property
    def receptive_field(self) -> int:
        """Nombre de pas de temps vus par chaque sortie."""
        return 1 + (self.kernel - 1) * sum(self.dilations) * self.repeats

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Prédit le résidu normalisé.

        Args:
            x: Entrées (lot × canaux × temps).

        Returns:
            Sorties (lot × temps).
        """
        return self.out(self.blocks(self.inp(x))).squeeze(1)


class PatchTransformer(nn.Module):
    """Transformer sur des patchs de quelques pas (la courbe devient ~130 jetons)."""

    def __init__(
        self,
        in_channels: int = len(CHANNELS),
        length: int = 1057,
        patch: int = 8,
        dim: int = 96,
        heads: int = 4,
        layers: int = 3,
        dropout: float = 0.0,
    ) -> None:
        """Construit le réseau.

        Args:
            in_channels: Nombre de canaux d'entrée.
            length: Longueur des séquences.
            patch: Nombre de pas par jeton.
            dim: Dimension des jetons.
            heads: Nombre de têtes d'attention.
            layers: Nombre de couches.
            dropout: Taux de dropout.
        """
        super().__init__()
        self.patch, self.length = patch, length
        self.n_tokens = -(-length // patch)
        self.embed = nn.Conv1d(in_channels, dim, kernel_size=patch, stride=patch)
        self.pos = nn.Parameter(torch.zeros(1, self.n_tokens, dim))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            dim,
            heads,
            dim_feedforward=2 * dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, patch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Prédit le résidu normalisé.

        Args:
            x: Entrées (lot × canaux × temps).

        Returns:
            Sorties (lot × temps).
        """
        pad = self.n_tokens * self.patch - x.shape[-1]
        tokens = self.embed(nn.functional.pad(x, (0, pad))).transpose(1, 2) + self.pos
        out = self.head(self.encoder(tokens))  # (lot, jetons, patch)
        return out.reshape(x.shape[0], -1)[:, : x.shape[-1]]


def count_parameters(model: nn.Module) -> int:
    """Nombre de paramètres entraînables."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def masked_l1_watts(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """L1 en watts, calculée uniquement sur les points masqués.

    Args:
        pred: Résidu prédit, en unités normalisées (lot × temps).
        target: Résidu vrai (y − KNN ancré), en watts (lot × temps).
        mask: 1 sur les points à prédire, 0 ailleurs (lot × temps).
        scale: Échelle de chaque courbe, en watts (lot,).

    Returns:
        Erreur absolue moyenne (W) sur l'ensemble des points masqués du lot.
    """
    err = (pred * scale[:, None] - target).abs() * mask
    return err.sum() / mask.sum().clamp(min=1.0)


# --------------------------------------------------------------------------- séquences

_FEATURE_CHANNELS = ["knn_anchored", "interp", "nb_std", "hour_sin", "hour_cos", "weekend"]
SCALE_OFFSET: float = 1.0
"""Échelle d'une courbe = niveau moyen observé + 1 W (évite la division par zéro)."""


@dataclass
class SequenceSet:
    """Courbes entières prêtes pour un réseau.

    Attributes:
        inputs: Canaux normalisés (courbes × ``CHANNELS`` × temps), float32.
        target: Résidu vrai y − KNN ancré en watts, 0 hors des points évalués.
        mask: 1 sur les points à prédire dont la vérité est connue, 0 ailleurs.
        holes: 1 sur tous les points manquants (à remplir), 0 ailleurs.
        scale: Échelle de chaque courbe (W).
        knn: KNN ancré (W), courbes × temps.
        groups: Identifiant de la courbe d'origine (pour séparer entraînement et validation).
    """

    inputs: npt.NDArray[np.float32]
    target: npt.NDArray[np.float32]
    mask: npt.NDArray[np.float32]
    holes: npt.NDArray[np.float32]
    scale: npt.NDArray[np.float32]
    knn: npt.NDArray[np.float32]
    groups: npt.NDArray[np.int64]

    def subset(self, idx: npt.NDArray[np.int64]) -> SequenceSet:
        """Sous-ensemble de courbes."""
        return SequenceSet(*(getattr(self, f.name)[idx] for f in fields(self)))

    @staticmethod
    def concat(parts: list[SequenceSet]) -> SequenceSet:
        """Concatène plusieurs ensembles."""
        return SequenceSet(
            *(np.concatenate([getattr(p, f.name) for p in parts]) for f in fields(SequenceSet))
        )


def make_sequences(
    X_holed: pd.DataFrame,
    reservoir: pd.DataFrame,
    neighbors: Neighbors,
    y: pd.DataFrame | None = None,
    groups: npt.NDArray[np.int64] | None = None,
) -> SequenceSet:
    """Construit les séquences d'entrée d'un réseau à partir de courbes trouées.

    Args:
        X_holed: Courbes à compléter (temps × courbes), NaN dans les trous.
        reservoir: Réservoir utilisé par ``knn.search``.
        neighbors: Voisines de ces courbes.
        y: Vraies valeurs (pour l'entraînement) ; ``None`` en prédiction.
        groups: Identifiant de la courbe d'origine de chaque colonne.

    Returns:
        Les séquences.
    """
    feats = curve_features(X_holed, reservoir, neighbors, _FEATURE_CHANNELS)
    X = X_holed.to_numpy(dtype=np.float32)
    holes = np.isnan(X)
    observed = np.where(holes, 0.0, X)
    n_obs = np.maximum((~holes).sum(axis=0), 1)
    scale = (observed.sum(axis=0) / n_obs + SCALE_OFFSET).astype(np.float32)
    channels = {
        "observed": observed / scale,
        "mask": holes.astype(np.float32),
        "knn_anchored": feats["knn_anchored"] / scale,
        "interp": feats["interp"] / scale,
        "nb_std": feats["nb_std"] / scale,
        "hour_sin": feats["hour_sin"],
        "hour_cos": feats["hour_cos"],
        "weekend": feats["weekend"],
    }
    inputs = np.stack([channels[c].T for c in CHANNELS], axis=1).astype(np.float32)
    knn = feats["knn_anchored"].T.astype(np.float32)
    if y is None:
        target = np.zeros_like(knn)
        mask = np.zeros_like(knn)
    else:
        y_arr = y.to_numpy(dtype=np.float32).T
        mask = (holes.T & ~np.isnan(y_arr)).astype(np.float32)
        target = np.where(mask > 0, y_arr - knn, 0.0).astype(np.float32)
    n = X.shape[1]
    return SequenceSet(
        inputs,
        target,
        mask,
        holes.T.astype(np.float32),
        scale,
        knn,
        np.arange(n, dtype=np.int64) if groups is None else groups,
    )


def build_sequence_training_set(
    reservoir: pd.DataFrame,
    motifs: npt.NDArray[np.bool_],
    *,
    n_curves: int = 8000,
    motifs_per_curve: int = 2,
    n_folds: int = 5,
    seed: int = SEED,
    refresh: bool = False,
) -> SequenceSet:
    """Séquences d'entraînement : mêmes courbes, motifs et KNN hors pli que le LightGBM.

    Args:
        reservoir: Courbes complètes disponibles.
        motifs: Motifs de NaN (temps × motifs).
        n_curves: Nombre de courbes du réservoir utilisées comme cibles.
        motifs_per_curve: Motifs greffés par courbe.
        n_folds: Plis de l'entraînement hors pli du KNN.
        seed: Graine (identique à l'étape LightGBM : mêmes courbes, mêmes motifs).
        refresh: Si vrai, reconstruit même si le cache existe.

    Returns:
        Les séquences (entrées stockées en float16 sur disque, float32 en mémoire).
    """
    path = PROCESSED_DIR / "deep_train.npz"
    params = f"{n_curves}-{motifs_per_curve}-{n_folds}-{seed}-{reservoir.shape[1]}"
    if path.exists() and not refresh:
        arrays = np.load(path)
        if str(arrays["params"]) == params:
            return SequenceSet(
                arrays["inputs"].astype(np.float32),
                *(arrays[f.name] for f in fields(SequenceSet)[1:]),
            )
    curve_ids = {name: i for i, name in enumerate(reservoir.columns)}
    parts = []
    for fd in iter_folds(
        reservoir,
        motifs,
        n_curves=n_curves,
        motifs_per_curve=motifs_per_curve,
        n_folds=n_folds,
        seed=seed,
    ):
        groups = np.array([curve_ids[s] for s in fd.sources], dtype=np.int64)
        parts.append(make_sequences(fd.X_holed, fd.pool, fd.neighbors, fd.y, groups))
    seqs = SequenceSet.concat(parts)
    np.savez(
        path,
        params=params,
        inputs=seqs.inputs.astype(np.float16),
        **{f.name: getattr(seqs, f.name) for f in fields(SequenceSet)[1:]},
    )
    return seqs


# --------------------------------------------------------------------------- entraînement


@dataclass(frozen=True)
class TrainConfig:
    """Hyperparamètres d'entraînement.

    Attributes:
        lr: Taux d'apprentissage initial (AdamW).
        weight_decay: Décroissance des poids.
        batch_size: Taille des lots.
        max_epochs: Nombre maximal d'epochs.
        patience: Arrêt si la validation ne s'améliore pas pendant ce nombre d'epochs.
        valid_fraction: Part des courbes d'origine réservée à la validation.
    """

    lr: float = 2e-3
    weight_decay: float = 1e-4
    batch_size: int = 32
    max_epochs: int = 20
    patience: int = 3
    valid_fraction: float = 0.1


def device() -> torch.device:
    """Le GPU Apple (mps) s'il est disponible, sinon le CPU."""
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def split_by_group(
    groups: npt.NDArray[np.int64], fraction: float, seed: int = SEED
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Sépare les séquences en entraînement / validation, par courbe d'origine.

    Les copies d'une même courbe (motifs différents) restent du même côté.

    Returns:
        ``(indices d'entraînement, indices de validation)``.
    """
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    valid_groups = rng.choice(unique, size=max(1, int(len(unique) * fraction)), replace=False)
    is_valid = np.isin(groups, valid_groups)
    return np.flatnonzero(~is_valid), np.flatnonzero(is_valid)


def _batches(n: int, size: int, rng: np.random.Generator | None) -> list[np.ndarray]:
    order = rng.permutation(n) if rng is not None else np.arange(n)
    return [order[i : i + size] for i in range(0, n, size)]


def _tensors(seqs: SequenceSet, idx: np.ndarray, dev: torch.device) -> tuple[torch.Tensor, ...]:
    return tuple(
        torch.from_numpy(np.ascontiguousarray(a[idx])).to(dev)
        for a in (seqs.inputs, seqs.target, seqs.mask, seqs.scale)
    )


@torch.no_grad()
def masked_mae(model: nn.Module, seqs: SequenceSet, dev: torch.device, batch: int = 64) -> float:
    """MAE en watts du réseau sur les points masqués d'un ensemble de séquences."""
    model.eval()
    err = torch.zeros((), device=dev)
    count = torch.zeros((), device=dev)
    for idx in _batches(len(seqs.scale), batch, None):
        x, target, mask, scale = _tensors(seqs, idx, dev)
        err += ((model(x) * scale[:, None] - target).abs() * mask).sum()
        count += mask.sum()
    return float(err / count.clamp(min=1.0))


def train_model(
    model: nn.Module,
    train: SequenceSet,
    valid: SequenceSet,
    cfg: TrainConfig,
    path: Path,
    seed: int = SEED,
    verbose: bool = True,
) -> pd.DataFrame:
    """Entraîne un réseau avec arrêt anticipé et sauvegarde le meilleur état.

    Args:
        model: Réseau à entraîner.
        train: Séquences d'entraînement.
        valid: Séquences de validation (courbes du réservoir, jamais dev).
        cfg: Hyperparamètres.
        path: Fichier où sauvegarder le meilleur état (``.pt``).
        seed: Graine (initialisation, ordre des lots).
        verbose: Si vrai, affiche la perte et la durée de chaque epoch.

    Returns:
        Historique : une ligne par epoch (perte d'entraînement, MAE de validation, durée).
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = device()
    model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=1)
    history, best, stale = [], np.inf, 0
    for epoch in range(1, cfg.max_epochs + 1):
        start = time.perf_counter()
        model.train()
        # Perte accumulée sur le GPU : une seule synchronisation CPU–GPU par epoch (lire une
        # valeur à chaque lot forcerait le CPU à attendre le GPU 450 fois par epoch).
        total = torch.zeros((), device=dev)
        points = torch.zeros((), device=dev)
        for idx in _batches(len(train.scale), cfg.batch_size, rng):
            x, target, mask, scale = _tensors(train, idx, dev)
            loss = masked_l1_watts(model(x), target, mask, scale)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            n = mask.sum()
            total += loss.detach() * n
            points += n
        train_mae = float(total / points)  # synchronisation unique
        valid_mae = masked_mae(model, valid, dev)
        sched.step(valid_mae)
        history.append(
            {
                "epoch": epoch,
                "train_mae": train_mae,
                "valid_mae": valid_mae,
                "lr": opt.param_groups[0]["lr"],
                "seconds": time.perf_counter() - start,
            }
        )
        if verbose:
            h = history[-1]
            print(
                f"epoch {epoch:2d}/{cfg.max_epochs}  entraînement {h['train_mae']:7.2f} W  "
                f"validation {h['valid_mae']:7.2f} W  {h['seconds']:5.0f} s",
                flush=True,
            )
        if valid_mae < best - 1e-3:
            best, stale = valid_mae, 0
            torch.save(model.state_dict(), path)
        else:
            stale += 1
            if stale >= cfg.patience:
                break
    model.load_state_dict(torch.load(path, map_location=dev))
    hist = pd.DataFrame(history)
    hist.to_csv(path.with_suffix(".csv"), index=False)
    return hist


@torch.no_grad()
def predict_residual(model: nn.Module, seqs: SequenceSet, batch: int = 64) -> np.ndarray:
    """Résidu prédit en watts (courbes × temps)."""
    dev = device()
    model.to(dev).eval()
    out = np.empty_like(seqs.knn)
    for idx in _batches(len(seqs.scale), batch, None):
        x = torch.from_numpy(np.ascontiguousarray(seqs.inputs[idx])).to(dev)
        out[idx] = (model(x).float().cpu().numpy()) * seqs.scale[idx, None]
    return out


def network_prediction(model: nn.Module, seqs: SequenceSet) -> np.ndarray:
    """Prédiction du réseau en watts (courbes × temps) : KNN ancré + résidu, ≥ 0."""
    return np.clip(seqs.knn + predict_residual(model, seqs), 0.0, None)


# --------------------------------------------------------------------------- méthodes


def _fill(X_holed: pd.DataFrame, pred_ct: np.ndarray) -> pd.DataFrame:
    """Remplit les trous de X_holed avec une prédiction (courbes × temps)."""
    X = X_holed.to_numpy(dtype=np.float64)
    filled = np.where(np.isnan(X), pred_ct.T, X)
    return pd.DataFrame(filled, index=X_holed.index, columns=X_holed.columns)


def make_network_method(model: nn.Module) -> Method:
    """Méthode ``(X_holed, reservoir) -> X_filled`` fondée sur un réseau entraîné."""

    def method(X_holed: pd.DataFrame, reservoir: pd.DataFrame) -> pd.DataFrame:
        nb = search(X_holed, reservoir, BEST_DISTANCE, k_max=BEST_PREDICTOR.k)
        seqs = make_sequences(X_holed, reservoir, nb)
        return _fill(X_holed, network_prediction(model, seqs))

    return method


def lightgbm_prediction(
    booster: lgb.Booster,
    cfg: GBMConfig,
    X_holed: pd.DataFrame,
    reservoir: pd.DataFrame,
    neighbors: Neighbors,
) -> np.ndarray:
    """Prédiction du LightGBM en watts (courbes × temps ; NaN hors des trous)."""
    feats, t, c = build_features(X_holed, reservoir, neighbors)
    out = np.full((X_holed.shape[1], X_holed.shape[0]), np.nan)
    out[c, t] = to_prediction(feats, booster.predict(feats), cfg)
    return out


def make_blend_method(
    booster: lgb.Booster, cfg: GBMConfig, model: nn.Module, weight: float
) -> Method:
    """Mélange ``w · LightGBM + (1 − w) · réseau`` (prédictions ≥ 0).

    Args:
        booster: LightGBM entraîné.
        cfg: Sa configuration.
        model: Réseau entraîné.
        weight: Poids w du LightGBM.

    Returns:
        La méthode.
    """

    def method(X_holed: pd.DataFrame, reservoir: pd.DataFrame) -> pd.DataFrame:
        nb = search(X_holed, reservoir, BEST_DISTANCE, k_max=BEST_PREDICTOR.k)
        gbm_pred = lightgbm_prediction(booster, cfg, X_holed, reservoir, nb)
        net_pred = network_prediction(model, make_sequences(X_holed, reservoir, nb))
        return _fill(X_holed, np.clip(weight * gbm_pred + (1 - weight) * net_pred, 0.0, None))

    return method


# --------------------------------------------------------------------------- calibration


@dataclass(frozen=True)
class CalibrationSet:
    """Courbes du réservoir jamais utilisées comme cibles, pour choisir le poids du mélange.

    Attributes:
        curves: Courbes du réservoir choisies.
        X_holed: Ces courbes, trouées par greffe d'un motif chacune.
        y: Leurs vraies valeurs.
        pool: Réservoir de voisines : le réservoir moins ces courbes.
    """

    curves: list[str]
    X_holed: pd.DataFrame
    y: pd.DataFrame
    pool: pd.DataFrame


def calibration_set(
    reservoir: pd.DataFrame,
    motifs: npt.NDArray[np.bool_],
    *,
    n_calib: int = 1000,
    n_train_curves: int = 8000,
    n_folds: int = 5,
    seed: int = SEED,
) -> CalibrationSet:
    """Courbes de calibration : hors cibles d'entraînement, hors doubles, hors leur pool.

    Args:
        reservoir: Courbes complètes disponibles.
        motifs: Motifs de NaN (temps × motifs).
        n_calib: Nombre de courbes de calibration.
        n_train_curves: Nombre de courbes d'entraînement (à exclure).
        n_folds: Plis de l'entraînement (pour retrouver les mêmes courbes).
        seed: Graine de l'entraînement.

    Returns:
        L'ensemble de calibration.
    """
    train_curves, _, _ = training_curves(reservoir, n_train_curves, n_folds, seed)
    excluded = set(train_curves) | set(exact_duplicates(reservoir))
    rng = np.random.default_rng(seed + 2)
    candidates = [c for c in reservoir.columns if c not in excluded]
    curves = list(rng.choice(candidates, size=n_calib, replace=False))
    picks = rng.integers(0, motifs.shape[1], size=n_calib)
    y = reservoir[curves]
    pool = reservoir.drop(columns=curves)
    return CalibrationSet(curves, graft(y, motifs[:, picks]), y, pool)


def train_or_load(
    model: nn.Module, train: SequenceSet, valid: SequenceSet, cfg: TrainConfig, name: str
) -> tuple[nn.Module, pd.DataFrame]:
    """Entraîne un réseau, ou recharge le meilleur état et l'historique s'ils existent.

    Args:
        model: Réseau (architecture) à entraîner ou à recharger.
        train: Séquences d'entraînement.
        valid: Séquences de validation.
        cfg: Hyperparamètres.
        name: Nom du fichier dans ``data/processed/`` (sans extension).

    Returns:
        ``(réseau au meilleur état, historique d'entraînement)``.
    """
    path = PROCESSED_DIR / f"{name}.pt"
    if path.exists() and path.with_suffix(".csv").exists():
        model.load_state_dict(torch.load(path, map_location="cpu"))
        return model, pd.read_csv(path.with_suffix(".csv"))
    history = train_model(model, train, valid, cfg, path)
    return model, history
