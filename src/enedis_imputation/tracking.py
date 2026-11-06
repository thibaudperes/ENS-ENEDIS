"""Suivi des expériences avec MLflow (stockage local, dans ``mlruns/``, ignoré par git).

Pour consulter les runs, depuis la racine du projet :

    uv run mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db

puis ouvrir http://127.0.0.1:5000 dans un navigateur.
"""

from __future__ import annotations

import mlflow

from enedis_imputation.data import PROJECT_ROOT
from enedis_imputation.validation import EvalResult

MLRUNS_DIR = PROJECT_ROOT / "mlruns"
TRACKING_URI = f"sqlite:///{MLRUNS_DIR / 'mlflow.db'}"


def setup_mlflow(experiment: str) -> str:
    """Pointe MLflow vers le stockage local du projet et sélectionne une expérience.

    Args:
        experiment: Nom de l'expérience (créée si besoin).

    Returns:
        Identifiant de l'expérience.
    """
    MLRUNS_DIR.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(TRACKING_URI)
    existing = mlflow.get_experiment_by_name(experiment)
    if existing is not None:
        exp_id = existing.experiment_id
    else:
        exp_id = mlflow.create_experiment(
            experiment, artifact_location=(MLRUNS_DIR / "artifacts" / experiment).as_uri()
        )
    mlflow.set_experiment(experiment_id=exp_id)
    return exp_id


def metric_key(label: str) -> str:
    """Nom de métrique compatible MLflow (« >48 » devient « gt48 »)."""
    return label.replace(">", "gt").replace(" ", "_")


def log_evaluation(result: EvalResult) -> None:
    """Enregistre dans le run MLflow actif la MAE globale et par classe d'un résultat.

    Args:
        result: Évaluation (``validation.evaluate``).
    """
    prefix = result.split
    mlflow.log_metric(f"{prefix}_mae", result.mae)
    mlflow.log_metric(f"{prefix}_mae_ci_low", result.ci[0])
    mlflow.log_metric(f"{prefix}_mae_ci_high", result.ci[1])
    mlflow.log_metric(f"{prefix}_seconds", result.seconds)
    for cls, mae in result.by_class["mae"].items():
        mlflow.log_metric(f"{prefix}_mae_{metric_key(str(cls))}", float(mae))
