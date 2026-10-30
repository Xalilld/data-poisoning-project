
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from poison import feature_poisoning
from detection_features import detect_feature_poisoning
from detection_repeated_values import detect_repeated_balance

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
        random_state=SEED
    )

    results = []

    for rate in RATES:
        X_poisoned, mask = feature_poisoning(
            X_train,
            poison_rate=rate,
            feature="balance",
            random_state=SEED
        )

        truth = pd.Series(mask).to_numpy().astype(int)

        methods = {
            "Isolation Forest": detect_feature_poisoning(
                X_poisoned,
                contamination=rate
            ),
            "Repeated High Balance": detect_repeated_balance(
                X_poisoned,
                clean_reference=X_train,
                min_frequency=20
            )
        }

        for method, detected in methods.items():
            predicted = detected.astype(int)

            results.append({
                "attack": "P3",
                "poison_rate": rate,
                "method": method,
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
                )
            })

    result_df = pd.DataFrame(results)

    output = ROOT / "results/csv/detection_p3_comparison.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output, index=False)

    print("\n=== COMPARAISON DES DETECTEURS P3 ===")
    print(result_df.to_string(index=False))
    print("\nResultats enregistres :", output)


if __name__ == "__main__":
    main()
