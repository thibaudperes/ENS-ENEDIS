"""Configuration commune des tests."""

import os

# torch et LightGBM embarquent chacun leur propre bibliothèque OpenMP. Si les deux lancent
# des calculs parallèles sur CPU dans le même processus, l'un plante (erreur de
# segmentation). Un seul thread OpenMP évite le conflit ; à fixer avant tout import.
os.environ["OMP_NUM_THREADS"] = "1"
