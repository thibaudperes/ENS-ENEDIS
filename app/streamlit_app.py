"""Démo interactive : reconstruire les trous d'une courbe de consommation.

Lancement, depuis la racine du projet :

    uv run streamlit run app/streamlit_app.py

L'appli part des caches et modèles déjà calculés (aucun réentraînement). Les réglages
peuvent aussi être passés dans l'URL, par exemple ``?courbe=12&mode=perso&debut=480&duree=96``.
"""

import os

# torch et LightGBM embarquent chacun leur bibliothèque OpenMP : un seul thread évite le
# conflit quand les deux tournent dans le même processus. À fixer avant les imports.
os.environ["OMP_NUM_THREADS"] = "1"

import matplotlib.pyplot as plt  # noqa: E402
import streamlit as st  # noqa: E402

from enedis_imputation.demo import custom_mask, load_demo, reconstruct, zoom  # noqa: E402
from enedis_imputation.viz import method_color, plot_imputation, set_style  # noqa: E402

st.set_page_config(page_title="Imputation de courbes de consommation", layout="wide")
set_style()


@st.cache_resource(show_spinner="Chargement des modèles et des courbes…")
def demo():
    """Données et modèles, chargés une seule fois par session du serveur."""
    return load_demo()


def param(name: str, default: int) -> int:
    """Valeur entière lue dans l'URL, ou valeur par défaut."""
    try:
        return int(st.query_params.get(name, default))
    except ValueError:
        return default


d = demo()
n_curves, n_steps = d.dev.y.shape[1], len(d.dev.y)
index = d.dev.y.index

st.markdown(
    "<style>.block-container{padding-top:2rem;padding-bottom:1rem}</style>",
    unsafe_allow_html=True,
)
st.subheader("Reconstruire les trous d'une courbe de consommation")
st.caption(
    "Courbes synthétiques au pas de 30 min (Challenge Data ENS n°160). Les courbes proposées "
    "n'ont jamais servi à entraîner les modèles."
)

with st.sidebar:
    st.header("Réglages")
    curve = st.number_input("Courbe (jeu dev)", 1, n_curves, param("courbe", 12), step=1)
    modes = ["Motif réel de trous", "Trou personnalisé"]
    mode = st.radio("Trous", modes, index=1 if st.query_params.get("mode") == "perso" else 0)
    if mode == modes[0]:
        motif = st.number_input(
            "Motif réel (parmi 1 999)",
            1,
            d.motifs.shape[1],
            param("motif", 18),
            step=1,
            help="Motif exact de NaN d'une vraie courbe trouée du train ou du test, greffé sur "
            "la courbe choisie.",
        )
        mask = d.motifs.iloc[:, motif - 1].to_numpy()
    else:
        start = st.select_slider(
            "Début du trou",
            options=list(range(n_steps - 1)),
            value=param("debut", 480),
            format_func=lambda i: index[i].strftime("%d/%m %H:%M"),
        )
        duration = st.slider(
            "Durée (pas de 30 min)",
            1,
            144,
            param("duree", 96),
            help="Jusqu'à 3 jours (144 pas).",
        )
        mask = custom_mask(n_steps, start, min(duration, n_steps - start))
    whole = st.toggle("Toute la période", value=False)

truth = d.dev.y.iloc[:, curve - 1]
rec = reconstruct(d, truth, mask)
window = slice(None) if whole else zoom(mask)
n_missing = int(rec.mask.sum())

fig = plot_imputation(
    truth,
    rec.preds,
    rec.mask,
    f"{n_missing} points masqués ({n_missing / 2:.0f} h) — modèle final : "
    f"{rec.mae['Modèle final (mélange)']:.0f} W d'erreur moyenne, contre "
    f"{rec.mae['Interpolation linéaire']:.0f} W pour l'interpolation",
    interval=(rec.low, rec.high),
    window=window,
)
fig.set_size_inches(13, 4.6)
st.pyplot(fig, clear_figure=True)
plt.close(fig)

cols = st.columns(len(rec.mae) + 1)
for col, (name, mae) in zip(cols, rec.mae.items(), strict=False):
    delta = None
    if name != "Interpolation linéaire":
        delta = f"{mae / rec.mae['Interpolation linéaire'] - 1:+.0%} vs interpolation"
    col.metric(f"MAE — {name}", f"{mae:.0f} W", delta=delta, delta_color="inverse")
    col.markdown(
        f"<div style='height:4px;background:{method_color(name)};border-radius:2px'></div>",
        unsafe_allow_html=True,
    )
cols[-1].metric("Couverture de l'intervalle à 90 %", f"{rec.coverage:.0%}")

with st.expander("Comment lire ce graphique"):
    st.markdown(
        """
- **Zones grises** : les points masqués, que chaque méthode doit reconstruire sans les voir.
- **Interpolation linéaire** : le benchmark du challenge, une droite entre les bords du trou.
- **KNN ancré** : forme des courbes les plus ressemblantes, recalée sur les bords du trou.
- **Modèle final** : mélange d'un LightGBM et d'un réseau convolutif qui corrigent le KNN
  ancré ; bande verte : intervalle de confiance à 90 % (prédiction conforme).
- Sur le jeu de test du challenge, le modèle final réduit l'erreur de 35 % par rapport à
  l'interpolation. Sur une seule courbe, l'écart peut être bien plus grand ou plus petit.
"""
    )
