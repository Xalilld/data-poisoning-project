
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from poison import feature_poisoning

SEED = 42
RATES = [0.01, 0.05, 0.10, 0.20]
CONTAMINATION = 0.05


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

        truth = np.asarray(mask, dtype=int)

        detector = IsolationForest(
            n_estimators=100,
            contamination=CONTAMINATION,
            random_state=SEED,
            n_jobs=-1
        )

        values = X_poisoned[["balance"]]
        detector.fit(values)

        predicted = (
            detector.predict(values) == -1
        ).astype(int)

        # Plus la valeur est grande, plus elle est suspecte.
        anomaly_scores = -detector.score_samples(values)

        results.append({
            "attack": "P3",
            "poison_rate": rate,
            "contamination": CONTAMINATION,
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
            "average_precision": average_precision_score(
                truth, anomaly_scores
            ),
            "threshold": detector.offset_
        })

    result_df = pd.DataFrame(results)

    output = ROOT / "results/csv/detection_p3_fixed.csv"
    output.parent.mkdir(parents=True, exist_ok=True)

    result_df.to_csv(output, index=False)

    print("\n=== P3 : CONTAMINATION FIXE 5 % ===")
    print(result_df.to_string(index=False))
    print("\nRésultats enregistrés :", output)


if __name__ == "__main__":
    main()
