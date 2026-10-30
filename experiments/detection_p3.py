
import sys
from pathlib import Path

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

from poison import feature_poisoning
from detection_features import detect_feature_poisoning

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
        print(f"\n=== P3 : {rate * 100:.0f} % ===")

        X_poisoned, mask = feature_poisoning(
            X_train,
            poison_rate=rate,
            feature="balance",
            random_state=SEED,
        )

        detected = detect_feature_poisoning(
            X_poisoned,
            contamination=rate,
        )

        truth = pd.Series(mask).to_numpy().astype(int)
        predicted = detected.astype(int)

        result = {
            "attack": "P3",
            "poison_rate": rate,
            "n_poisoned": int(truth.sum()),
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

        print("Empoisonnées :", result["n_poisoned"])
        print("Détectées :", result["n_detected"])
        print(f"Precision : {result['precision']:.4f}")
        print(f"Recall    : {result['recall']:.4f}")
        print(f"F1-score  : {result['f1']:.4f}")
        print("Matrice de confusion :")
        print(confusion_matrix(truth, predicted))

    output = ROOT / "results/csv/detection_p3.csv"
    output.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(results).to_csv(output, index=False)

    print("\n=== COMPARAISON P3 ===")
    print(pd.DataFrame(results).to_string(index=False))
    print("\nRésultats enregistrés :", output)


if __name__ == "__main__":
    main()
