"""Visuels du README : animation phare (GIF) et image d'aperçu 1280 × 640.

Usage, depuis la racine du projet :

    uv run python scripts/make_visuals.py
"""

from __future__ import annotations

import os

# torch et LightGBM dans le même processus : un seul thread OpenMP (voir notebook 05).
os.environ["OMP_NUM_THREADS"] = "1"

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from PIL import Image  # noqa: E402

from enedis_imputation import SEED  # noqa: E402
from enedis_imputation.demo import custom_mask, load_demo, reconstruct  # noqa: E402
from enedis_imputation.viz import FIG_DIR, PALETTE, method_color, set_style  # noqa: E402

FINAL = "Modèle final (mélange)"
INTERP = "Interpolation linéaire"


def pick_example(demo, start: int, n_steps: int):
    """Courbe dont le gain du modèle final sur l'interpolation est proche du gain médian.

    On masque la même journée (à partir de minuit) sur 200 courbes de dev tirées au hasard,
    de niveau compris entre les 40e et 90e centiles (lisibles), et l'on garde la courbe au
    gain relatif médian : un exemple représentatif, pas le plus flatteur.
    """
    y = demo.dev.y
    level = y.mean()
    lo, hi = level.quantile([0.4, 0.9])
    pool = level[(level > lo) & (level < hi)].index
    rng = np.random.default_rng(SEED)
    mask = custom_mask(len(y), start, n_steps)
    cands = []
    for col in rng.choice(pool, size=200, replace=False):
        rec = reconstruct(demo, y[col], mask)
        cands.append((rec.mae[FINAL] / rec.mae[INTERP], col, rec))
    ratios = np.array([c[0] for c in cands])
    best = int(np.argmin(np.abs(ratios - np.median(ratios))))
    print(f"gain médian du modèle final sur 200 courbes : {1 - np.median(ratios):.0%}")
    return cands[best][2]


def hero_gif(rec, start: int, n_steps: int) -> None:
    """Animation : le trou, la droite du benchmark, la reconstruction et son intervalle."""
    idx = rec.truth.index
    w0, w1 = max(0, start - 72), min(len(idx), start + n_steps + 72)
    x = idx[w0:w1]
    y = rec.truth.to_numpy()[w0:w1]
    m = rec.mask[w0:w1]
    hole = np.flatnonzero(m)
    edge = np.r_[hole[0] - 1, hole, hole[-1] + 1]  # trou + ses deux voisins observés
    interp = rec.preds[INTERP][w0:w1]
    final = rec.preds[FINAL][w0:w1]
    low, high = rec.low[w0:w1], rec.high[w0:w1]
    y_obs = np.where(m, np.nan, y)
    ymax = np.nanmax(np.r_[y, high]) * 1.08

    fig, ax = plt.subplots(figsize=(10, 4.2), dpi=100, layout="none")
    ax.set_position([0.075, 0.11, 0.905, 0.70])  # place réservée au titre en haut
    title = fig.text(0.07, 0.92, "", fontsize=14, fontweight="bold", color=PALETTE["ink"])
    sub = fig.text(0.07, 0.865, "", fontsize=10.5, color=PALETTE["ink_secondary"])
    ax.axvspan(x[hole[0]], x[hole[-1]], color=PALETTE["hole"], lw=0, zorder=0)
    ax.plot(x, y_obs, color=PALETTE["true"], lw=1.3)
    (l_interp,) = ax.plot([], [], color=method_color(INTERP), lw=2.2)
    (l_final,) = ax.plot([], [], color=method_color(FINAL), lw=2.6)
    (l_true,) = ax.plot([], [], color=PALETTE["true"], lw=1.3, alpha=0.9)
    band = [None]
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(0, ymax)
    ax.set_ylabel("Consommation (W)")

    n = len(edge)
    steps = {"trou": 12, "interp": 22, "pause1": 12, "final": 30, "pause2": 14, "verite": 1}
    plan = []
    for phase, k in steps.items():
        plan += [(phase, i, k) for i in range(k)]
    plan += [("fin", 0, 1)] * 30  # dernière image tenue ~3 s

    def update(frame: int):
        phase, i, k = plan[frame]
        if phase == "trou":
            title.set_text("Une journée entière de mesures manque")
            sub.set_text("Courbe de consommation au pas de 30 min, jamais vue par les modèles")
        elif phase in ("interp", "pause1"):
            j = n if phase == "pause1" else max(2, int(n * (i + 1) / k))
            l_interp.set_data(x[edge[:j]], interp[edge[:j]])
            title.set_text("Le benchmark relie les deux bords par une droite")
            sub.set_text("Interpolation linéaire : elle ignore le rythme de la journée")
        elif phase in ("final", "pause2"):
            j = n if phase == "pause2" else max(2, int(n * (i + 1) / k))
            l_final.set_data(x[edge[:j]], final[edge[:j]])
            if band[0] is not None:
                band[0].remove()
            band[0] = ax.fill_between(
                x[hole[: max(1, j - 1)]],
                low[hole[: max(1, j - 1)]],
                high[hole[: max(1, j - 1)]],
                color=PALETTE["interval"],
                alpha=0.22,
                lw=0,
                zorder=1,
            )
            title.set_text("Le modèle final reconstruit la forme, avec son intervalle à 90 %")
            sub.set_text("Mélange LightGBM + réseau convolutif, qui corrigent un KNN ancré")
        else:
            l_true.set_data(x[edge], y[edge])
            title.set_text(
                f"Vraie courbe : erreur moyenne {rec.mae[FINAL]:.0f} W pour le modèle final, "
                f"{rec.mae[INTERP]:.0f} W pour l'interpolation"
            )
            sub.set_text(
                f"Intervalle à 90 % : {rec.coverage:.0%} des vraies valeurs couvertes ; "
                "sur tout le test, −35 % d'erreur par rapport au benchmark"
            )
        return l_interp, l_final, l_true

    anim = FuncAnimation(fig, update, frames=len(plan), blit=False)
    path = FIG_DIR / "animation_phare.gif"
    with plt.rc_context({"savefig.bbox": "standard"}):  # cadre fixe, pas de recadrage
        anim.save(path, writer=PillowWriter(fps=10), dpi=100)
    plt.close(fig)
    # Recompression : palette réduite, images identiques fusionnées par l'optimiseur
    gif = Image.open(path)
    frames = []
    for i in range(gif.n_frames):
        gif.seek(i)
        frames.append(gif.convert("RGB").quantize(colors=64, method=Image.Quantize.MEDIANCUT))
    frames[0].save(
        path, save_all=True, append_images=frames[1:], duration=100, loop=0, optimize=True
    )
    print(f"{path} : {len(frames)} images, {path.stat().st_size / 1e6:.2f} Mo")


def social_preview(rec, start: int, n_steps: int) -> None:
    """Image d'aperçu GitHub (1280 × 640) : titre, deux chiffres clés, une courbe."""
    fig = plt.figure(figsize=(12.8, 6.4), dpi=100, layout="none")
    fig.patch.set_facecolor("white")
    fig.text(
        0.06,
        0.84,
        "Reconstruire les trous des courbes",
        fontsize=30,
        fontweight="bold",
        color=PALETTE["ink"],
    )
    fig.text(
        0.06,
        0.765,
        "de consommation électrique",
        fontsize=30,
        fontweight="bold",
        color=PALETTE["ink"],
    )
    fig.text(
        0.06,
        0.70,
        "Challenge Data ENS n°160 — imputation de séries temporelles, pas de 30 min",
        fontsize=14,
        color=PALETTE["ink_secondary"],
    )
    for xpos, big, small in [
        (0.06, "−35 %", "d'erreur vs benchmark"),
        (0.36, "Top 7 %", "12ᵉ sur 172 au classement public"),
    ]:
        fig.text(xpos, 0.57, big, fontsize=38, fontweight="bold", color=method_color(FINAL))
        fig.text(xpos, 0.515, small, fontsize=13, color=PALETTE["ink_secondary"])
    ax = fig.add_axes([0.06, 0.08, 0.88, 0.36])
    idx = rec.truth.index
    w0, w1 = max(0, start - 60), min(len(idx), start + n_steps + 60)
    x, y, m = idx[w0:w1], rec.truth.to_numpy()[w0:w1], rec.mask[w0:w1]
    hole = np.flatnonzero(m)
    edge = np.r_[hole[0] - 1, hole, hole[-1] + 1]
    ax.axvspan(x[hole[0]], x[hole[-1]], color=PALETTE["hole"], lw=0)
    ax.fill_between(
        x[hole],
        rec.low[w0:w1][hole],
        rec.high[w0:w1][hole],
        color=PALETTE["interval"],
        alpha=0.22,
        lw=0,
    )
    ax.plot(x, y, color=PALETTE["true"], lw=1.2)
    ax.plot(x[edge], rec.preds[INTERP][w0:w1][edge], color=method_color(INTERP), lw=2)
    ax.plot(x[edge], rec.preds[FINAL][w0:w1][edge], color=method_color(FINAL), lw=2.6)
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(bottom=0)
    ax.set_axis_off()
    for label, color, xpos in [
        ("vraie courbe", PALETTE["true"], 0.06),
        ("interpolation (benchmark)", method_color(INTERP), 0.20),
        ("modèle final + intervalle à 90 %", method_color(FINAL), 0.42),
    ]:
        fig.text(xpos, 0.455, "—", color=color, fontsize=16, fontweight="bold")
        fig.text(xpos + 0.022, 0.458, label, color=PALETTE["ink_secondary"], fontsize=12)
    path = FIG_DIR / "social_preview.png"
    with plt.rc_context({"savefig.bbox": "standard"}):  # exactement 1280 × 640
        fig.savefig(path, dpi=100, facecolor="white")
    plt.close(fig)
    print(f"{path} : {Image.open(path).size}")


def main() -> None:
    """Choisit un exemple représentatif et produit les deux visuels."""
    set_style()
    demo = load_demo()
    start, n_steps = (
        48 * 7,
        48,
    )  # 16 janvier, de 00:00 à 23:30 (une journée, comme les vrais motifs)
    rec = pick_example(demo, start, n_steps)
    print(f"exemple : {rec.truth.name} ; MAE {rec.mae}")
    hero_gif(rec, start, n_steps)
    social_preview(rec, start, n_steps)


if __name__ == "__main__":
    main()
