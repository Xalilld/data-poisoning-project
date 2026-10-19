
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
from src.poison import feature_poisoning


print("=== TEST P3 : FEATURE POISONING ===")

X = pd.DataFrame({
    "age": [25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
    "balance": [100, 200, 300, 400, 500,
                600, 700, 800, 900, 1000]
})

X_original = X.copy(deep=True)

X_poisoned, is_poisoned = feature_poisoning(
    X,
    poison_rate=0.20,
    feature="balance",
    random_state=42
)

print("\nDonnees originales :")
print(X)

print("\nDonnees empoisonnees :")
print(X_poisoned)

print("\nObservations empoisonnees :")
print(is_poisoned.tolist())

print("\nNombre de lignes modifiees :", is_poisoned.sum())

# 20 % de 10 observations = 2
assert is_poisoned.sum() == 2

# Les donnees originales restent intactes
assert X.equals(X_original)

# La colonne age reste intacte
assert X_poisoned["age"].equals(X["age"])

# Seules les lignes selectionnees changent
changed = X_poisoned["balance"] != X["balance"]
assert changed.equals(is_poisoned)

# Les autres lignes restent intactes
assert X_poisoned.loc[~is_poisoned, "balance"].equals(
    X.loc[~is_poisoned, "balance"].astype(float)
)

print("\nTEST P3 REUSSI !")
