
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from poison import feature_poisoning

SEED = 42

# Charger les données
df = pd.read_csv(
    ROOT / "data/raw/bank-full.csv",
    sep=";"
)

X = df.drop(columns=["y"])
y = df["y"]

# Reproduire le Train des semaines précédentes
X_train, _, y_train, _ = train_test_split(
    X, y,
    test_size=0.30,
    stratify=y,
    random_state=SEED
)

# Appliquer P3 à 10 %
X_poisoned, mask = feature_poisoning(
    X_train,
    poison_rate=0.10,
    feature="balance",
    random_state=SEED
)

truth = np.asarray(mask, dtype=bool)

# Entraîner Isolation Forest
detector = IsolationForest(
    n_estimators=100,
    contamination=0.05,
    random_state=SEED,
    n_jobs=-1
)

values = X_poisoned[["balance"]]
detector.fit(values)

# Plus le score est faible, plus l'observation
# est considérée comme anormale.
scores = detector.score_samples(values)

print("\n=== ANALYSE DES SCORES P3 ===")

print("\nObservations empoisonnées :")
print(pd.Series(scores[truth]).describe())

print("\nObservations non empoisonnées :")
print(pd.Series(scores[~truth]).describe())

print("\nValeurs de balance empoisonnées :")
print(X_poisoned.loc[truth, "balance"].describe())

print("\nValeurs de balance non empoisonnées :")
print(X_poisoned.loc[~truth, "balance"].describe())

print("\nSeuil de décision Isolation Forest :")
print(detector.offset_)

print("\n=== FIN DE L'ANALYSE ===")
