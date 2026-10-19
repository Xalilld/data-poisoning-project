
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
from src.poison import injection_duplication


print("=== TEST P4 : INJECTION / DUPLICATION ===")

X = pd.DataFrame({
    "age": [25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
    "balance": [100, 200, 300, 400, 500,
                600, 700, 800, 900, 1000]
})

y = pd.Series([
    "yes", "no", "yes", "no", "yes",
    "no", "yes", "no", "yes", "no"
])

X_original = X.copy(deep=True)
y_original = y.copy(deep=True)

X_augmented, y_augmented, is_injected = injection_duplication(
    X,
    y,
    poison_rate=0.20,
    target_class="yes",
    random_state=42
)

print("Observations avant :", len(X))
print("Observations apres :", len(X_augmented))
print("Observations ajoutees :", is_injected.sum())

assert len(X_augmented) == 12
assert len(y_augmented) == 12
assert is_injected.sum() == 2

# Les 10 observations originales sont preservees
assert X_augmented.iloc[:10].reset_index(drop=True).equals(
    X_original.reset_index(drop=True)
)

assert y_augmented.iloc[:10].reset_index(drop=True).equals(
    y_original.reset_index(drop=True)
)

# Les nouvelles observations appartiennent a la classe yes
assert (y_augmented[is_injected] == "yes").all()

# Les donnees sources ne changent pas
assert X.equals(X_original)
assert y.equals(y_original)

print("\nTEST P4 REUSSI !")
