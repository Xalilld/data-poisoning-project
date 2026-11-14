
"""
Test du score de cohérence et de Cleanlab.
Semaine 6 — Data Poisoning.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from detection import (
    compute_inconsistency_scores,
    detect_label_issues
)


def main():
    y = pd.Series(["yes", "yes", "no", "no"])

    pred_probs = np.array([
        [0.05, 0.95],
        [0.90, 0.10],
        [0.80, 0.20],
        [0.05, 0.95]
    ])

    scores = compute_inconsistency_scores(
        y, pred_probs
    )

    print("\n=== SCORES DE COHÉRENCE ===")

    for label, score in zip(y, scores):
        print(
            f"Étiquette : {label} "
            f"| Suspicion : {score:.2f}"
        )

    expected = np.array([
        0.05, 0.90, 0.20, 0.95
    ])

    assert np.allclose(scores, expected), (
        "Erreur dans le calcul des scores."
    )

    print("\nTEST SCORE : RÉUSSI")

    issues = detect_label_issues(
        y, pred_probs
    )

    assert len(issues) == len(y)
    assert issues.dtype == np.dtype(bool)

    print("TEST CLEANLAB : RÉUSSI")
    print("Observations signalées :", int(issues.sum()))


if __name__ == "__main__":
    main()
