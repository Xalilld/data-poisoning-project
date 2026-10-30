
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from poison import injection_duplication
from detection_duplicates import detect_duplicates

SEED = 42
RATES = [0.01, 0.05, 0.10, 0.20]


def main():
    df = pd.read_csv(
        ROOT / "data/raw/bank-full.csv",
        sep=";"
    )

    X = df.drop(columns=["y"])
    y = df["y"]

    X_train, _, y_train, _ = train_test_split(
        X,
        y,
        test_size=0.30,
        stratify=y,
        random_state=SEED,
    )

    results = []

    for rate in RATES:
        print(f"\n=== P4 : {rate * 100:.0f} % ===")

        X_augmented, y_augmented, mask = injection_duplication(
            X_train,
            y_train,
            poison_rate=rate,
            target_class="yes",
            random_state=SEED,
        )

        detected = detect_duplicates(
            X_augmented,
            y_augmented,
        )

        truth = np.asarray(mask, dtype=int)
        predicted = detected.astype(int)

        result = {
            "attack": "P4",
            "poison_rate": rate,
            "n_injected": int(truth.sum()),
            "n_detected": int(predicted.sum()),
            "precision": precision_score(
                truth, predicted, zero_division=0
            ),
            "recall": recall_score(
                truth, predicted, zero_division=0
            ),
            "f1": f1_score(
                truth, predicted, zero_division=0
            ),
        }

        results.append(result)

        print("Lignes injectées :", result["n_injected"])
        print("Doublons détectés :", result["n_detected"])
        print(f"Precision : {result['precision']:.4f}")
        print(f"Recall    : {result['recall']:.4f}")
        print(f"F1-score  : {result['f1']:.4f}")
        print("Matrice de confusion :")
        print(confusion_matrix(truth, predicted))

    result_df = pd.DataFrame(results)

    output = ROOT / "results/csv/detection_p4.csv"
    output.parent.mkdir(parents=True, exist_ok=True)

    result_df.to_csv(output, index=False)

    print("\n=== COMPARAISON P4 ===")
    print(result_df.to_string(index=False))
    print("\nRésultats enregistrés :", output)


if __name__ == "__main__":
    main()
