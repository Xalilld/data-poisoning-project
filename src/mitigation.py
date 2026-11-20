
"""
Semaine 7 — Nettoyage et mitigation du Data Poisoning.

Ce module permet de supprimer les observations suspectes
avant de réentraîner un modèle de classification.
"""

import numpy as np
import pandas as pd


def clean_suspicious_data(
    X,
    y,
    suspicion_scores,
    threshold=0.50
):
    """
    Supprime les observations dont le score de suspicion
    est strictement supérieur au seuil.

    Paramètres
    ----------
    X : pandas.DataFrame
        Variables explicatives.

    y : pandas.Series
        Étiquettes observées.

    suspicion_scores : array-like
        Score de suspicion pour chaque observation.

    threshold : float
        Seuil de détection entre 0 et 1.

    Retour
    ------
    X_cleaned, y_cleaned, removed_mask
    """

    if not isinstance(X, pd.DataFrame):
        raise TypeError("X doit être un DataFrame.")

    if not isinstance(y, pd.Series):
        raise TypeError("y doit être une Series.")

    if len(X) != len(y):
        raise ValueError(
            "X et y doivent avoir la même taille."
        )

    if not X.index.equals(y.index):
        raise ValueError(
            "Les index de X et y doivent correspondre."
        )

    scores = np.asarray(
        suspicion_scores,
        dtype=float
    )

    if scores.ndim != 1 or len(scores) != len(X):
        raise ValueError(
            "Un score est nécessaire pour chaque observation."
        )

    if not np.isfinite(scores).all():
        raise ValueError(
            "Les scores doivent être des nombres finis."
        )

    if np.any(scores < 0) or np.any(scores > 1):
        raise ValueError(
            "Les scores doivent être entre 0 et 1."
        )

    if not 0 <= threshold <= 1:
        raise ValueError(
            "Le seuil doit être entre 0 et 1."
        )

    removed_mask = pd.Series(
        scores > threshold,
        index=X.index,
        name="is_removed"
    )

    X_cleaned = X.loc[
        ~removed_mask
    ].copy()

    y_cleaned = y.loc[
        ~removed_mask
    ].copy()

    if X_cleaned.empty:
        raise ValueError(
            "Le nettoyage a supprimé toutes les observations."
        )

    if y_cleaned.nunique() < 2:
        raise ValueError(
            "Le nettoyage doit conserver les deux classes."
        )

    return X_cleaned, y_cleaned, removed_mask
