
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_predict,
)
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from models import create_logistic_model
from poison import random_label_flipping, targeted_label_flipping
from detection import detect_label_issues

RATES = [0.01, 0.05, 0.10, 0.20]
SEED = 42


def main():
    df = pd.read_csv(ROOT / "data/raw/bank-full.csv", sep=";")
    X = df.drop(columns=["y"])
    y = df["y"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=SEED
    )

    # Validation et test restent propres et inutilisés ici.
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50,
        stratify=y_temp, random_state=SEED
    )

    cv = StratifiedKFold(
        n_splits=5, shuffle=True, random_state=SEED
    )

    results = []

    for attack in ["P1", "P2"]:
        for rate in RATES:
            print(f"\nDetection {attack} - {rate * 100:.0f} %")

            if attack == "P1":
                y_poisoned, mask = random_label_flipping(
                    y_train, poison_rate=rate, random_state=SEED
                )
            else:
                y_poisoned, mask = targeted_label_flipping(
                    y_train, poison_rate=rate,
                    target_class="yes", random_state=SEED
                )

            probabilities = cross_val_predict(
                create_logistic_model(),
                X_train,
                y_poisoned,
                cv=cv,
                method="predict_proba",
                n_jobs=1,
            )

            detected = detect_label_issues(
                y_poisoned, probabilities
            )

            truth = mask.to_numpy().astype(int)
            predicted = detected.astype(int)

            result = {
                "attack": attack,
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
            print(result)

    output = ROOT / "results/csv/detection_comparison.csv"
    output.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(results).to_csv(output, index=False)

    print("\n=== COMPARAISON FINALE ===")
    print(pd.DataFrame(results).to_string(index=False))
    print("\nResultats enregistres :", output)


if __name__ == "__main__":
    main()
