"""Identité visuelle du projet : style matplotlib, palette par méthode, figures types.

Toutes les figures du projet (EDA, résultats, README, animation) passent par ce module :

- ``set_style()`` à appeler une fois en début de notebook ou de script ;
- ``PALETTE`` : une couleur par rôle, toujours la même (vraie courbe, interpolation, KNN,
  modèle final, intervalle de confiance, trous, données descriptives) ;
- ``set_conclusion(fig, texte)`` : titre de figure qui énonce la conclusion ;
- ``plot_imputation(...)`` : une courbe, ses trous grisés et les reconstructions ;
- ``save_fig(fig, nom)`` : export PNG dans ``reports/figures/``.

La palette a été vérifiée : toutes les paires de couleurs de méthodes restent distinctes
(ΔE OKLab ≥ 12) en vision normale et sous simulation de protanopie, deutéranopie et
tritanopie.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping
from pathlib import Path

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.units as munits
import numpy as np
import numpy.typing as npt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from enedis_imputation.data import PROJECT_ROOT
from enedis_imputation.holes import hole_runs

FIG_DIR: Path = PROJECT_ROOT / "reports" / "figures"

PALETTE: dict[str, str] = {
    # Méthodes : une couleur par méthode, réutilisée dans toutes les figures
    "true": "#2f2f2c",  # vraie courbe : gris foncé
    "interpolation": "#d9480f",  # orange
    "knn": "#2a78d6",  # bleu
    "model": "#12b886",  # vert (modèle final)
    "interval": "#12b886",  # vert, utilisé transparent (intervalle de confiance)
    # Contexte
    "hole": "#e6e5df",  # zones grisées des trous
    "neutral": "#a3a29b",  # éléments secondaires (voisines, références)
    # Graphiques descriptifs (sans méthode) : violet, pour ne jamais évoquer une méthode
    "data": "#5b4bc4",
    "data_light": "#b3aaf0",
    # Encre
    "ink": "#1a1a19",
    "ink_secondary": "#5f5e5a",
    "muted": "#8a8983",
    "grid": "#ebeae5",
    "axis": "#c3c2b7",
    "background": "#ffffff",
}
"""Couleurs du projet, par rôle."""

INTERVAL_ALPHA: float = 0.22
"""Opacité de la bande d'intervalle de confiance."""

_METHOD_KEYS: list[tuple[tuple[str, ...], str]] = [
    (("vrai", "true", "réel"), "true"),
    (("interp",), "interpolation"),
    (("knn", "voisin"), "knn"),
    (("modèle", "model", "final"), "model"),
]
_EXTRA_COLORS: list[str] = ["#e87ba4", "#eda100", "#6250d6"]

_LINE_WIDTH: dict[str, float] = {"true": 1.3, "interpolation": 1.8, "knn": 1.8, "model": 2.4}


def method_color(name: str) -> str:
    """Couleur associée à une méthode, d'après son libellé.

    Le libellé est libre (« Interpolation linéaire », « KNN (médiane de 5) »…) : on
    reconnaît la méthode par mot-clé, sans tenir compte de la casse.

    Args:
        name: Libellé de la méthode.

    Returns:
        Code hexadécimal de la couleur.

    Raises:
        KeyError: Si le libellé ne correspond à aucune méthode connue.
    """
    key = _method_key(name)
    if key is None:
        raise KeyError(f"Méthode inconnue : {name!r}. Ajouter un mot-clé dans viz._METHOD_KEYS.")
    return PALETTE[key]


def _method_key(name: str) -> str | None:
    """Clé de palette d'une méthode, ou None si elle est inconnue."""
    lowered = name.lower()
    return next((key for kws, key in _METHOD_KEYS if any(k in lowered for k in kws)), None)


def set_style() -> None:
    """Applique le style matplotlib du projet.

    Fond blanc (lisible aussi en mode sombre sur GitHub), police sans empattement,
    grille légère, sans bordures haute et droite, mise en page automatique, export
    PNG à 200 dpi, dates au format numérique français (jj/mm, hh:mm).
    """
    p = PALETTE
    mpl.rcParams.update(
        {
            "figure.facecolor": p["background"],
            "axes.facecolor": p["background"],
            "savefig.facecolor": p["background"],
            "figure.dpi": 100,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.15,
            "figure.constrained_layout.use": True,
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 10.5,
            "axes.titlesize": 11,
            "axes.titleweight": "normal",
            "axes.titlelocation": "left",
            "axes.titlecolor": p["ink_secondary"],
            "axes.titlepad": 8,
            "figure.titlesize": 14,
            "figure.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.labelcolor": p["ink_secondary"],
            "axes.edgecolor": p["axis"],
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": p["grid"],
            "grid.linewidth": 0.7,
            "xtick.color": p["muted"],
            "ytick.color": p["muted"],
            "xtick.labelcolor": p["ink_secondary"],
            "ytick.labelcolor": p["ink_secondary"],
            "xtick.major.size": 0,
            "ytick.major.size": 0,
            "text.color": p["ink"],
            "lines.linewidth": 1.8,
            "lines.solid_capstyle": "round",
            "legend.frameon": False,
            "legend.fontsize": 9.5,
            "axes.prop_cycle": mpl.cycler(color=[p["data"], p["data_light"], *_EXTRA_COLORS]),
            "patch.linewidth": 0,
        }
    )
    converter = mdates.ConciseDateConverter(
        formats=["%Y", "%d/%m", "%d/%m", "%Hh", "%H:%M", "%S"],
        zero_formats=["", "%Y", "%m/%Y", "%d/%m", "%H:%M", "%H:%M"],
        offset_formats=["", "%Y", "%m/%Y", "%m/%Y", "%d/%m/%Y", "%d/%m/%Y %H:%M"],
    )
    for t in (np.datetime64, dt.date, dt.datetime, pd.Timestamp):
        munits.registry[t] = converter


def set_conclusion(fig: Figure, text: str) -> None:
    """Donne à une figure un titre qui énonce sa conclusion (aligné à gauche, en gras).

    Le sujet de chaque graphique va dans le titre de ses axes (``ax.set_title``), en gris.

    Args:
        fig: Figure à titrer.
        text: Conclusion, par ex. « L'interpolation échoue sur les trous de plus de 3 h ».
    """
    fig.suptitle(text, x=0.0, ha="left")


def shade_holes(
    ax: Axes, index: pd.DatetimeIndex, mask: npt.ArrayLike, *, label: str | None = "Trou"
) -> None:
    """Grise les trous d'une courbe.

    Chaque zone couvre le trou à une demi-période près de chaque côté, pour qu'un trou
    d'un seul point reste visible.

    Args:
        ax: Axes sur lesquels dessiner.
        index: Horodates de la courbe.
        mask: Booléen, vrai là où la valeur est manquante.
        label: Libellé de légende (une seule entrée pour toutes les zones).
    """
    m = np.asarray(mask, dtype=bool)
    half = (index[1] - index[0]) / 2
    for i, (start, length) in enumerate(hole_runs(m)[["start", "length"]].itertuples(index=False)):
        ax.axvspan(
            index[start] - half,
            index[start + length - 1] + half,
            color=PALETTE["hole"],
            lw=0,
            zorder=0,
            label=label if i == 0 else None,
        )


def plot_imputation(
    true: pd.Series,
    preds: Mapping[str, pd.Series | npt.ArrayLike],
    mask: pd.Series | npt.ArrayLike,
    title: str,
    *,
    interval: tuple[npt.ArrayLike, npt.ArrayLike] | None = None,
    window: slice | None = None,
    ax: Axes | None = None,
    legend: bool = True,
) -> Figure:
    """Trace une courbe, ses trous grisés et les reconstructions de chaque méthode.

    Les reconstructions ne sont tracées que dans les trous, raccordées aux points observés
    voisins : un trou d'un point apparaît comme un « V » partant de la vraie courbe.

    Args:
        true: Vraie courbe (indexée par horodate), valeurs des trous comprises.
        preds: Reconstructions par méthode, ``{libellé: valeurs}``, alignées sur ``true``.
            La couleur est déduite du libellé (voir ``method_color``).
        mask: Booléen aligné sur ``true``, vrai là où la valeur était manquante.
        title: Titre. Sans ``ax``, c'est le titre de la figure et il doit énoncer la
            conclusion ; avec ``ax``, c'est le titre du graphique.
        interval: Bornes ``(basse, haute)`` d'un intervalle de confiance, alignées sur
            ``true`` ; tracé (vert transparent) uniquement dans les trous.
        window: Tranche d'indices entiers à afficher (par défaut, toute la courbe).
        ax: Axes existants (pour une grille de graphiques) ; sinon une figure est créée.
        legend: Afficher la légende : au-dessus du graphique pour une figure créée ici,
            dans le coin haut gauche sinon. Pour une grille, préférer ``legend=False`` et
            une légende commune (``fig.legend``).

    Returns:
        La figure contenant le graphique.
    """
    window = window or slice(None)
    index = pd.DatetimeIndex(true.index[window])
    y = np.asarray(true, dtype=np.float64)[window]
    m = np.asarray(mask, dtype=bool)[window]

    created = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))
        set_conclusion(fig, title)
    else:
        fig = ax.get_figure()
        ax.set_title(title)

    shade_holes(ax, index, m)
    # Points à afficher pour une reconstruction : le trou et ses deux voisins observés.
    shown = m.copy()
    shown[1:] |= m[:-1]
    shown[:-1] |= m[1:]

    if interval is not None:
        low, high = (np.asarray(b, dtype=np.float64)[window] for b in interval)
        ax.fill_between(
            index,
            np.where(m, low, np.nan),
            np.where(m, high, np.nan),
            color=PALETTE["interval"],
            alpha=INTERVAL_ALPHA,
            lw=0,
            label="Intervalle de confiance",
            zorder=1,
        )

    ax.plot(index, y, color=PALETTE["true"], lw=_LINE_WIDTH["true"], label="Vraie courbe", zorder=2)

    extra = iter(_EXTRA_COLORS)
    for name, values in preds.items():
        key = _method_key(name)
        color = PALETTE[key] if key else next(extra)
        v = np.asarray(values, dtype=np.float64)[window]
        ax.plot(
            index,
            np.where(shown, np.where(m, v, y), np.nan),
            color=color,
            lw=_LINE_WIDTH.get(key or "", 1.8),
            label=name,
            zorder=3,
        )

    if np.nanmin(y) >= 0:
        ax.set_ylim(bottom=0)
    ax.set_xlim(index[0], index[-1])
    ax.set_ylabel("Consommation (W)")
    if legend and created:
        ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=len(preds) + 3, borderaxespad=0.6)
    elif legend:
        ax.legend(loc="upper left")
    return fig


def save_fig(fig: Figure, name: str) -> Path:
    """Enregistre une figure en PNG (200 dpi) dans ``reports/figures/``.

    Args:
        fig: Figure à enregistrer.
        name: Nom du fichier, sans extension.

    Returns:
        Chemin du fichier écrit.
    """
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path)
    return path
