
"""
Semaine 6 — Détection basée sur les modèles.

Fonctionnalités :
1. Détection des erreurs d'étiquetage avec Cleanlab.
2. Calcul du score de suspicion label/prédiction.

Classes :
0 = no
1 = yes
"""

import numpy as np
import pandas as pd

from cleanlab.filter import find_label_issues


def validate_inputs(y_train, pred_probs):
    """Vérifie les étiquettes et les probabilités."""

    labels = pd.Series(y_train).reset_index(drop=True)
    labels = labels.map({"no": 0, "yes": 1})

    if labels.isna().any():
        raise ValueError(
            "Les étiquettes doivent être 'no' ou 'yes'."
        )

    labels = labels.to_numpy(dtype=int)
    probabilities = np.asarray(pred_probs, dtype=float)

    if probabilities.shape != (len(labels), 2):
        raise ValueError(
            "pred_probs doit avoir deux colonnes "
            "et autant de lignes que y_train."
        )

    if not np.isfinite(probabilities).all():
        raise ValueError(
            "Les probabilités contiennent des valeurs invalides."
        )

    if np.any(probabilities < 0) or np.any(probabilities > 1):
        raise ValueError(
            "Les probabilités doivent être entre 0 et 1."
        )

    if not np.allclose(
        probabilities.sum(axis=1), 1.0, atol=1e-6
    ):
        raise ValueError(
            "Chaque ligne de probabilités doit totaliser 1."
        )

    return labels, probabilities


def detect_label_issues(y_train, pred_probs):
    """
    Détecte les étiquettes suspectes avec Cleanlab.

    Retourne un masque booléen.
    True = observation suspecte.
    """
    labels, probabilities = validate_inputs(
        y_train, pred_probs
    )

    issues = find_label_issues(
        labels=labels,
        pred_probs=probabilities,
        return_indices_ranked_by=None
    )

    return np.asarray(issues, dtype=bool)


def compute_inconsistency_scores(y_train, pred_probs):
    """
    Calcule le score de suspicion d'une observation.

    Score = 1 - probabilité de l'étiquette observée.

    Un score élevé signifie que le modèle est
    peu confiant dans l'étiquette présente.
    """
    labels, probabilities = validate_inputs(
        y_train, pred_probs
    )

    given_label_probabilities = probabilities[
        np.arange(len(labels)),
        labels
    ]

    scores = 1.0 - given_label_probabilities

    return scores
