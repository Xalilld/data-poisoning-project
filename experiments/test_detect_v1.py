
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from detect import detect_statistical
from poison import feature_poisoning

df = pd.read_csv(
    ROOT / "data/raw/bank-full.csv",
    sep=";"
)

X = df.drop(columns=["y"])
y = df["y"]

X_train, _, _, _ = train_test_split(
    X,
    y,
    test_size=0.30,
    stratify=y,
    random_state=42,
)

X_poisoned, truth = feature_poisoning(
    X_train,
    poison_rate=0.10,
    feature="balance",
    random_state=42,
)

for method in ["isolation_forest", "repeated_balance"]:
    predicted = detect_statistical(
        X_poisoned,
        method=method,
        clean_reference=X_train,
        contamination=0.05,
    )

    print(f"\n=== {method} ===")
    print("Observations empoisonnées :", sum(truth))
    print("Observations signalées :", sum(predicted))
    print(
        "Precision :",
        round(precision_score(truth, predicted, zero_division=0), 4)
    )
    print(
        "Recall :",
        round(recall_score(truth, predicted, zero_division=0), 4)
    )
    print(
        "F1 :",
        round(f1_score(truth, predicted, zero_division=0), 4)
    )
