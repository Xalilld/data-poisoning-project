
import numpy as np
from sklearn.ensemble import IsolationForest


def detect_feature_poisoning(X, contamination=0.10):
    """
    Détecte les valeurs suspectes de balance
    avec Isolation Forest.

    Retourne un masque booléen :
    True = observation suspecte.
    """

    if not 0 < contamination <= 0.5:
        raise ValueError(
            "contamination doit être entre 0 et 0.5"
        )

    if "balance" not in X.columns:
        raise ValueError("Colonne balance introuvable")

    values = X[["balance"]].to_numpy()

    detector = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )

    predictions = detector.fit_predict(values)

    return np.asarray(predictions == -1, dtype=bool)
