
import numpy as np


def detect_repeated_balance(
    X,
    clean_reference,
    min_frequency=20
):
    """
    Détecte les valeurs élevées et anormalement répétées.

    Le seuil de valeur est calculé sur un Train propre
    de référence, jamais sur Validation ou Test.
    """

    if "balance" not in X.columns:
        raise ValueError("Colonne balance absente")

    if "balance" not in clean_reference.columns:
        raise ValueError("Référence sans balance")

    threshold = clean_reference["balance"].quantile(0.99)

    frequencies = X["balance"].value_counts()

    suspicious_values = frequencies[
        (frequencies >= min_frequency)
        & (frequencies.index >= threshold)
    ].index

    detected = X["balance"].isin(suspicious_values)

    return np.asarray(detected, dtype=bool)
