
"""
Semaine 5 — Détection statistique du Data Poisoning.

Méthodes :
1. Isolation Forest
2. Valeurs élevées répétées (P3)
3. Détection des doublons (P4)
"""

import numpy as np

from detection_features import detect_feature_poisoning
from detection_repeated_values import detect_repeated_balance
from detection_duplicates import detect_duplicates


def detect_statistical(
    X,
    y=None,
    method="isolation_forest",
    clean_reference=None,
    contamination=0.05,
    min_frequency=20,
):
    """
    Détecte les observations suspectes.

    Retourne un masque booléen :
    True  = observation suspecte
    False = observation non signalée
    """

    if method == "isolation_forest":
        return detect_feature_poisoning(
            X,
            contamination=contamination,
        )

    if method == "repeated_balance":
        if clean_reference is None:
            raise ValueError(
                "clean_reference est obligatoire "
                "pour repeated_balance."
            )

        return detect_repeated_balance(
            X,
            clean_reference=clean_reference,
            min_frequency=min_frequency,
        )

    if method == "duplicates":
        if y is None:
            raise ValueError(
                "y est obligatoire pour duplicates."
            )

        return detect_duplicates(X, y)

    raise ValueError(
        f"Méthode inconnue : {method}. "
        "Choisir isolation_forest, "
        "repeated_balance ou duplicates."
    )
