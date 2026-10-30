
import numpy as np
import pandas as pd


def detect_duplicates(X, y):
    """
    Détecte les doublons exacts dans les données.

    Compare les caractéristiques et le label.
    La première occurrence est conservée.
    """

    data = X.reset_index(drop=True).copy()
    data["target_label"] = pd.Series(y).reset_index(drop=True)

    detected = data.duplicated(keep="first")

    return np.asarray(detected, dtype=bool)
