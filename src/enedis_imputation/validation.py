"""Protocole de validation par faux trous simulés (« greffe de masques »).

Principe : on prend le motif exact de NaN d'une vraie courbe trouée (train ou test) et on
l'applique, aux mêmes horodates, à une courbe complète du train dont on connaît toutes
les valeurs. Les longueurs de trous, leur calendrier (blocs de 4 h, journées à minuit…)
et le taux de trous par courbe sont ainsi reproduits à l'identique.

Trois jeux d'évaluation :

- ``dev`` : ~2 000 courbes complètes du train trouées par greffe, pour régler les méthodes ;
- ``holdout`` : ~2 000 autres courbes trouées par greffe, réservées au score final
  (``evaluate`` refuse de l'utiliser sans ``allow_holdout=True``) ;
- ``real`` : les 1 000 vraies courbes trouées du train, jeu de contrôle.

Les trois jeux partagent le même réservoir de voisins : les courbes complètes du train qui
ne sont cibles ni de ``dev`` ni de ``holdout``. Règle anti-fuite : une courbe cible n'est
jamais dans le réservoir utilisé pour la remplir (vérifié à chaque évaluation).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
import warnings
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import numpy.typing as npt
import pandas as pd

from enedis_imputation import SEED
from enedis_imputation.data import PROCESSED_DIR, PROJECT_ROOT, load_test, load_train
from enedis_imputation.holes import LENGTH_LABELS, hole_length_per_point, length_bucket
from enedis_imputation.methods import Method

SPLITS: tuple[str, ...] = ("dev", "holdout", "real")
EXCLUDED_MOTIFS: tuple[str, ...] = ("train:holed_192",)
"""Motifs exclus : holed_192 n'a qu'un point observé, sa vérité est entièrement NaN, et
aucune courbe du test n'est comparable (la plus trouée l'est à 33 %)."""

REGISTRY_PATH = PROJECT_ROOT / "reports" / "results.csv"
META_PATH = PROCESSED_DIR / "splits_meta.json"
N_BOOT: int = 1000


@dataclass(frozen=True)
class Split:
    """Un jeu d'évaluation : courbes trouées, vérité et réservoir de voisins.

    Attributes:
        name: ``"dev"``, ``"holdout"`` ou ``"real"``.
        X_holed: Courbes à compléter (temps × cibles), NaN dans les trous.
        y: Vraies valeurs, mêmes colonnes que ``X_holed``.
        reservoir: Courbes complètes utilisables comme voisines (aucune cible).
        motifs: Pour chaque cible, le motif de NaN greffé (``None`` pour ``real``).
    """

    name: str
    X_holed: pd.DataFrame
    y: pd.DataFrame
    reservoir: pd.DataFrame
    motifs: list[str] | None = None

    @property
    def mask(self) -> npt.NDArray[np.bool_]:
        """Positions à prédire (vrai = manquant dans ``X_holed``)."""
        return self.X_holed.isna().to_numpy()


@dataclass(frozen=True)
class EvalResult:
    """Résultat d'une évaluation.

    Attributes:
        method: Nom de la méthode.
        split: Jeu évalué.
        mae: MAE globale (W).
        ci: Intervalle de confiance à 95 % de la MAE globale (bootstrap sur les courbes).
        by_class: MAE, bornes de l'intervalle et nombre de points, par classe de longueur.
        seconds: Temps de calcul de la méthode.
        n_curves: Nombre de courbes évaluées.
    """

    method: str
    split: str
    mae: float
    ci: tuple[float, float]
    by_class: pd.DataFrame
    seconds: float
    n_curves: int

    def to_row(self) -> dict[str, object]:
        """Ligne du registre des résultats."""
        row: dict[str, object] = {
            "date": datetime.now().isoformat(timespec="seconds"),
            "commit": _git_commit(),
            "method": self.method,
            "split": self.split,
            "mae": round(self.mae, 3),
            "mae_ci_low": round(self.ci[0], 3),
            "mae_ci_high": round(self.ci[1], 3),
        }
        row |= {f"mae_{c}": round(v, 3) for c, v in self.by_class["mae"].items()}
        row |= {
            "n_points": int(self.by_class["points"].sum()),
            "n_curves": self.n_curves,
            "seconds": round(self.seconds, 2),
        }
        return row


# --------------------------------------------------------------------------- construction


def motif_masks() -> pd.DataFrame:
    """Motifs de NaN des vraies courbes trouées (train et test), hors motifs exclus.

    Returns:
        Booléens (temps × motifs), colonnes nommées ``"train:holed_k"`` / ``"test:holed_k"``.
    """
    _, X_holed_train, _ = load_train()
    _, X_holed_test = load_test()
    masks = pd.concat(
        [X_holed_train.isna().add_prefix("train:"), X_holed_test.isna().add_prefix("test:")],
        axis=1,
    )
    return masks.drop(columns=list(EXCLUDED_MOTIFS))


def graft(full: pd.DataFrame, masks: npt.ArrayLike) -> pd.DataFrame:
    """Applique des motifs de NaN à des courbes complètes.

    Args:
        full: Courbes complètes (temps × n).
        masks: Booléens (temps × n), vrai là où la valeur doit devenir manquante.

    Returns:
        Copie de ``full`` avec des NaN aux positions des motifs.
    """
    m = np.asarray(masks, dtype=bool)
    if m.shape != full.shape:
        raise ValueError(f"Formes incompatibles : courbes {full.shape}, motifs {m.shape}.")
    return full.mask(m)


def exact_duplicates(F: pd.DataFrame) -> pd.Index:
    """Courbes non nulles qui ont un double exact dans ``F``.

    Une telle courbe ne doit pas être cible : son double, dans le réservoir, la remplirait
    parfaitement. (Les courbes identiquement nulles, nombreuses, ne posent pas ce problème :
    prédire 0 pour une courbe nulle est légitime.)

    Args:
        F: Courbes complètes (temps × n).

    Returns:
        Noms des courbes concernées.
    """
    values = F.to_numpy()
    digests = pd.Series(
        [hashlib.md5(values[:, j].tobytes()).hexdigest() for j in range(values.shape[1])],
        index=F.columns,
    )
    nonzero = np.abs(values).max(axis=0) > 0
    return F.columns[digests.duplicated(keep=False).to_numpy() & nonzero]


def assign_curves(
    candidates: pd.Index, forbidden_targets: pd.Index, n_dev: int, n_holdout: int, seed: int
) -> dict[str, list[str]]:
    """Répartit les courbes complètes entre cibles ``dev``, cibles ``holdout`` et réservoir.

    Args:
        candidates: Courbes complètes disponibles.
        forbidden_targets: Courbes qui ne peuvent pas être cibles (elles vont au réservoir).
        n_dev: Nombre de cibles ``dev``.
        n_holdout: Nombre de cibles ``holdout``.
        seed: Graine aléatoire.

    Returns:
        ``{"dev": [...], "holdout": [...], "reservoir": [...]}``, trois listes disjointes
        dont l'union est ``candidates``.
    """
    rng = np.random.default_rng(seed)
    eligible = candidates.difference(forbidden_targets, sort=False)
    drawn = rng.permutation(eligible.to_numpy())
    dev, holdout = drawn[:n_dev], drawn[n_dev : n_dev + n_holdout]
    targets = set(dev) | set(holdout)
    return {
        "dev": [str(c) for c in dev],
        "holdout": [str(c) for c in holdout],
        "reservoir": [str(c) for c in candidates if c not in targets],
    }


def make_splits(*, seed: int = SEED, refresh: bool = False) -> dict[str, object]:
    """Construit les jeux ``dev`` et ``holdout`` et les sauvegarde dans ``data/processed/``.

    Chaque jeu reçoit une permutation aléatoire de tous les motifs : il compte donc autant
    de cibles que de motifs (1 999), et ses trous ont exactement la distribution des vrais.

    Args:
        seed: Graine aléatoire (découpage et attribution des motifs).
        refresh: Si vrai, reconstruit même si les fichiers existent.

    Returns:
        Les métadonnées (graine, listes de courbes, motifs attribués).
    """
    if META_PATH.exists() and not refresh:
        return json.loads(META_PATH.read_text())

    X_full, _, _ = load_train()
    candidates = X_full.columns[X_full.notna().all().to_numpy()]
    masks = motif_masks()
    n = masks.shape[1]
    groups = assign_curves(candidates, exact_duplicates(X_full[candidates]), n, n, seed)

    rng = np.random.default_rng(seed + 1)
    meta: dict[str, object] = {"seed": seed, "reservoir": groups["reservoir"]}
    for name in ("dev", "holdout"):
        motifs = list(rng.permutation(masks.columns.to_numpy()))
        meta[name] = {"targets": groups[name], "motifs": [str(m) for m in motifs]}
        y = X_full[groups[name]]
        X_holed = graft(y, masks[motifs].to_numpy())
        np.save(PROCESSED_DIR / f"split_{name}_X.npy", X_holed.to_numpy(dtype=np.float32))
    META_PATH.write_text(json.dumps(meta))
    return meta


def load_split(name: str) -> Split:
    """Charge un jeu d'évaluation (le construit au premier appel).

    Args:
        name: ``"dev"``, ``"holdout"`` ou ``"real"``.

    Returns:
        Le jeu, avec son réservoir de voisins.
    """
    if name not in SPLITS:
        raise ValueError(f"Jeu inconnu : {name!r}. Choix : {SPLITS}.")
    meta = make_splits()
    X_full, X_holed_real, y_real = load_train()
    reservoir = X_full[meta["reservoir"]]

    if name == "real":
        return Split(name, X_holed_real, y_real, reservoir)

    targets = meta[name]["targets"]
    y = X_full[targets]
    values = np.load(PROCESSED_DIR / f"split_{name}_X.npy")
    X_holed = pd.DataFrame(values, index=y.index, columns=y.columns, copy=False)
    return Split(name, X_holed, y, reservoir, motifs=meta[name]["motifs"])


# --------------------------------------------------------------------------- évaluation


def check_no_leak(X_holed: pd.DataFrame, reservoir: pd.DataFrame) -> None:
    """Vérifie qu'aucune courbe à remplir n'est dans le réservoir de voisins.

    Raises:
        ValueError: Si une courbe cible figure dans le réservoir.
    """
    leaked = X_holed.columns.intersection(reservoir.columns)
    if len(leaked):
        raise ValueError(f"Fuite : {len(leaked)} cible(s) dans le réservoir, ex. {leaked[0]!r}.")


def per_curve_errors(
    pred: pd.DataFrame, y: pd.DataFrame, mask: npt.ArrayLike
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    """Somme des erreurs absolues et nombre de points évalués, par courbe et par classe.

    Les NaN de ``y`` sont exclus. Ces agrégats suffisent pour la MAE (globale ou par
    classe) et pour un bootstrap par courbe.

    Args:
        pred: Courbes complétées.
        y: Vraies valeurs.
        mask: Positions à évaluer (vrai = trou).

    Returns:
        ``(sommes, effectifs)``, chacun de forme (courbes × classes), classes dans l'ordre
        de ``LENGTH_LABELS``.

    Raises:
        ValueError: Si ``pred`` contient des NaN sur des positions évaluées.
    """
    m = np.asarray(mask, dtype=bool)
    y_arr = y.to_numpy(dtype=np.float64)
    evaluated = m & ~np.isnan(y_arr)
    err = np.abs(pred.to_numpy(dtype=np.float64) - y_arr)
    if np.isnan(err[evaluated]).any():
        raise ValueError("pred contient des NaN sur des positions évaluées.")
    cls = np.full(m.shape, -1, dtype=np.int64)
    cls[m] = length_bucket(hole_length_per_point(m)[m]).codes
    sums = np.zeros((m.shape[1], len(LENGTH_LABELS)))
    counts = np.zeros((m.shape[1], len(LENGTH_LABELS)), dtype=np.int64)
    for c in range(len(LENGTH_LABELS)):
        sel = evaluated & (cls == c)
        sums[:, c] = np.where(sel, err, 0.0).sum(axis=0)
        counts[:, c] = sel.sum(axis=0)
    return sums, counts


def bootstrap_mae(
    sums: npt.NDArray[np.float64],
    counts: npt.NDArray[np.int64],
    n_boot: int = N_BOOT,
    seed: int = SEED,
) -> npt.NDArray[np.float64]:
    """Distribution bootstrap de la MAE, en rééchantillonnant les courbes.

    On rééchantillonne des courbes entières (et non des points) : les erreurs d'une même
    courbe sont corrélées, et c'est la variabilité entre clients qui compte.

    Args:
        sums: Sommes d'erreurs (courbes × classes).
        counts: Effectifs (courbes × classes).
        n_boot: Nombre de tirages.
        seed: Graine aléatoire.

    Returns:
        Tableau (n_boot × (classes + 1)) : MAE par classe, puis MAE globale.
    """
    rng = np.random.default_rng(seed)
    S = np.column_stack([sums, sums.sum(axis=1)])
    N = np.column_stack([counts, counts.sum(axis=1)])
    draws = rng.integers(0, len(S), size=(n_boot, len(S)))
    with np.errstate(invalid="ignore", divide="ignore"):
        return S[draws].sum(axis=1) / N[draws].sum(axis=1)


def evaluate(
    method: Method,
    split: str | Split,
    *,
    name: str | None = None,
    allow_holdout: bool = False,
    register: bool = True,
) -> EvalResult:
    """Évalue une méthode sur un jeu : MAE globale, MAE par classe de trou, temps de calcul.

    Args:
        method: Fonction ``(X_holed, reservoir) -> X_filled``.
        split: Nom du jeu (``"dev"``, ``"holdout"``, ``"real"``) ou jeu déjà chargé.
        name: Nom de la méthode dans le registre (par défaut, ``method.__name__``).
        allow_holdout: Doit être vrai pour évaluer sur ``holdout``, réservé au score final.
        register: Si vrai, ajoute une ligne au registre ``reports/results.csv``.

    Returns:
        Le résultat de l'évaluation.

    Raises:
        PermissionError: Si ``holdout`` est demandé sans ``allow_holdout=True``.
        ValueError: En cas de fuite, ou si la sortie de la méthode est mal formée.
    """
    split = load_split(split) if isinstance(split, str) else split
    if split.name == "holdout" and not allow_holdout:
        raise PermissionError("Le holdout est réservé au score final : allow_holdout=True.")
    check_no_leak(split.X_holed, split.reservoir)

    start = time.perf_counter()
    pred = method(split.X_holed.copy(), split.reservoir)
    seconds = time.perf_counter() - start
    if not (pred.columns.equals(split.X_holed.columns) and pred.index.equals(split.X_holed.index)):
        raise ValueError("La méthode doit renvoyer les mêmes colonnes et le même index.")

    sums, counts = per_curve_errors(pred, split.y, split.mask)
    boot = bootstrap_mae(sums, counts)
    with np.errstate(invalid="ignore", divide="ignore"):
        point = np.append(sums.sum(axis=0) / counts.sum(axis=0), sums.sum() / counts.sum())
    with warnings.catch_warnings():  # classe sans aucun point : intervalle NaN, attendu
        warnings.simplefilter("ignore", RuntimeWarning)
        low, high = np.nanpercentile(boot, [2.5, 97.5], axis=0)
    by_class = pd.DataFrame(
        {"mae": point[:-1], "low": low[:-1], "high": high[:-1], "points": counts.sum(axis=0)},
        index=pd.Index(LENGTH_LABELS, name="classe"),
    )
    result = EvalResult(
        method=name or getattr(method, "__name__", "method"),
        split=split.name,
        mae=float(point[-1]),
        ci=(float(low[-1]), float(high[-1])),
        by_class=by_class,
        seconds=seconds,
        n_curves=int((counts.sum(axis=1) > 0).sum()),
    )
    if register:
        append_to_registry(result)
    return result


# --------------------------------------------------------------------------- registre


def append_to_registry(result: EvalResult) -> None:
    """Ajoute une ligne au registre des résultats (``reports/results.csv``)."""
    row = pd.DataFrame([result.to_row()])
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    row.to_csv(REGISTRY_PATH, mode="a", header=not REGISTRY_PATH.exists(), index=False)


def load_registry(*, latest_only: bool = True) -> pd.DataFrame:
    """Lit le registre des résultats.

    Args:
        latest_only: Si vrai, ne garde que la dernière évaluation de chaque couple
            (méthode, jeu).

    Returns:
        Le registre (vide s'il n'existe pas encore).
    """
    if not REGISTRY_PATH.exists():
        return pd.DataFrame()
    reg = pd.read_csv(REGISTRY_PATH)
    if latest_only:
        reg = reg.drop_duplicates(["method", "split"], keep="last").reset_index(drop=True)
    return reg


def _git_commit() -> str:
    """Hash court du commit courant (chaîne vide hors dépôt git)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return ""
    return out.stdout.strip()
