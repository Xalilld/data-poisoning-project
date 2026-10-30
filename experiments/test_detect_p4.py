
import sys
from pathlib import Path

from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from detect import detect_statistical
from poison import injection_duplication

df = pd.read_csv(
    ROOT / "data/raw/bank-full.csv",
    sep=";"
)

X = df.drop(columns=["y"])
y = df["y"]

X_train, _, y_train, _ = train_test_split(
    X, y,
    test_size=0.30,
    stratify=y,
    random_state=42
)

# Attaque P4 : injection de doublons à 10 %
X_poisoned, y_poisoned, truth = injection_duplication(
    X_train,
    y_train,
    poison_rate=0.10,
    target_class="yes",
    random_state=42
)

# Détection des doublons avec detect.py
predicted = detect_statistical(
    X_poisoned,
    y=y_poisoned,
    method="duplicates"
)

print("\n=== TEST DETECT.PY - P4 ===")
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
