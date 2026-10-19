
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.poison import (
    random_label_flipping,
    targeted_label_flipping
)


# ==================================================
# TEST P1 - RANDOM LABEL FLIPPING
# ==================================================

print("=== TEST P1 : RANDOM LABEL FLIPPING ===")

y = pd.Series([
    "yes", "no", "yes", "no", "yes",
    "no", "yes", "no", "yes", "no"
])

y_original = y.copy()

y_poisoned, is_poisoned = random_label_flipping(
    y,
    poison_rate=0.20,
    random_state=42
)

print("Labels originaux :", y.tolist())
print("Labels empoisonnes :", y_poisoned.tolist())
print("Indicateur :", is_poisoned.tolist())
print("Nombre de labels modifies :", is_poisoned.sum())

assert is_poisoned.sum() == 2
assert (y_original != y_poisoned).sum() == 2
assert y.equals(y_original)

print("\nTEST P1 REUSSI !")


# ==================================================
# TEST P2 - TARGETED LABEL FLIPPING
# ==================================================

print("\n=== TEST P2 : TARGETED LABEL FLIPPING ===")

y_target = pd.Series([
    "yes", "yes", "no", "yes", "no",
    "yes", "no", "yes", "no", "yes"
])

y_target_original = y_target.copy()

y_target_poisoned, target_mask = targeted_label_flipping(
    y_target,
    poison_rate=0.50,
    target_class="yes",
    random_state=42
)

print("Labels originaux :", y_target.tolist())
print("Labels empoisonnes :", y_target_poisoned.tolist())
print("Indicateur :", target_mask.tolist())
print("Nombre de labels modifies :", target_mask.sum())

# 6 yes au depart, 50 % = 3 labels modifies
assert target_mask.sum() == 3
assert (y_target_original != y_target_poisoned).sum() == 3

# Les labels modifies etaient tous yes
assert (y_target_original[target_mask] == "yes").all()

# Ils sont devenus no
assert (y_target_poisoned[target_mask] == "no").all()

# Les autres observations n'ont pas change
assert y_target_poisoned[~target_mask].equals(
    y_target_original[~target_mask]
)

# Les donnees originales sont preservees
assert y_target.equals(y_target_original)

print("\nTEST P2 REUSSI !")


# ==================================================
# RESULTAT FINAL
# ==================================================

print("\n================================")
print("TOUS LES TESTS SONT REUSSIS !")
print("================================")
