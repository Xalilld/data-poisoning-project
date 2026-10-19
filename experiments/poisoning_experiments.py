
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
from sklearn.model_selection import train_test_split

from src.poison import (
    random_label_flipping,
    targeted_label_flipping,
    feature_poisoning,
    injection_duplication
)


# ==========================================
# 1. CHARGEMENT DU DATASET
# ==========================================

df = pd.read_csv("data/raw/bank-full.csv", sep=";")

X = df.drop(columns=["y"])
y = df["y"]


# ==========================================
# 2. SPLIT STRATIFIE 70 / 15 / 15
# ==========================================

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

X_train_original = X_train.copy(deep=True)
y_train_original = y_train.copy(deep=True)
X_val_original = X_val.copy(deep=True)
y_val_original = y_val.copy(deep=True)
X_test_original = X_test.copy(deep=True)
y_test_original = y_test.copy(deep=True)

print("=== DATASET BANK MARKETING ===")
print("Train :", len(X_train))
print("Validation :", len(X_val))
print("Test :", len(X_test))
print("YES dans Train :", (y_train == "yes").sum())


# ==========================================
# 3. PARAMETRES
# ==========================================

poison_rates = [0.01, 0.05, 0.10, 0.20]
random_state = 42
results = []


# ==========================================
# 4. P1 - RANDOM LABEL FLIPPING
# ==========================================

print("\n=== P1 : RANDOM LABEL FLIPPING ===")

for rate in poison_rates:
    y_poisoned, mask = random_label_flipping(
        y_train,
        poison_rate=rate,
        random_state=random_state
    )

    n_modified = int((y_train != y_poisoned).sum())

    assert n_modified == int(mask.sum())
    assert n_modified == round(len(y_train) * rate)

    print(
        f"Taux : {rate * 100:.0f}% | "
        f"Labels modifies : {n_modified}"
    )

    results.append({
        "attack": "P1_random_label_flipping",
        "poison_rate": rate,
        "rate_denominator": "all_train",
        "n_train": len(y_train),
        "n_eligible": len(y_train),
        "n_poisoned": n_modified,
        "n_train_after": len(y_train),
        "random_state": random_state
    })


# ==========================================
# 5. P2 - TARGETED LABEL FLIPPING
# ==========================================

print("\n=== P2 : TARGETED LABEL FLIPPING ===")

n_target = int((y_train == "yes").sum())

for rate in poison_rates:
    y_poisoned, mask = targeted_label_flipping(
        y_train,
        poison_rate=rate,
        target_class="yes",
        random_state=random_state
    )

    n_modified = int((y_train != y_poisoned).sum())

    assert n_modified == int(mask.sum())
    assert n_modified == round(n_target * rate)
    assert (y_train[mask] == "yes").all()
    assert (y_poisoned[mask] == "no").all()

    print(
        f"Taux : {rate * 100:.0f}% des YES | "
        f"Labels modifies : {n_modified}"
    )

    results.append({
        "attack": "P2_targeted_label_flipping",
        "poison_rate": rate,
        "rate_denominator": "target_yes",
        "n_train": len(y_train),
        "n_eligible": n_target,
        "n_poisoned": n_modified,
        "n_train_after": len(y_train),
        "random_state": random_state
    })


# ==========================================
# 6. P3 - FEATURE POISONING
# ==========================================

print("\n=== P3 : FEATURE POISONING ===")

feature = "balance"

for rate in poison_rates:
    X_poisoned, mask = feature_poisoning(
        X_train,
        poison_rate=rate,
        feature=feature,
        random_state=random_state
    )

    changed_rows = X_train.ne(X_poisoned).any(axis=1)
    n_modified = int(changed_rows.sum())

    assert n_modified == int(mask.sum())
    assert n_modified == round(len(X_train) * rate)

    other_columns = X_train.columns.drop(feature)

    assert X_poisoned[other_columns].equals(
        X_train[other_columns]
    )

    print(
        f"Taux : {rate * 100:.0f}% | "
        f"Observations modifiees : {n_modified}"
    )

    results.append({
        "attack": "P3_feature_poisoning",
        "poison_rate": rate,
        "rate_denominator": "all_train",
        "n_train": len(X_train),
        "n_eligible": len(X_train),
        "n_poisoned": n_modified,
        "n_train_after": len(X_train),
        "random_state": random_state
    })


# ==========================================
# 7. P4 - INJECTION / DUPLICATION
# ==========================================

print("\n=== P4 : INJECTION / DUPLICATION ===")

for rate in poison_rates:
    X_augmented, y_augmented, mask = injection_duplication(
        X_train,
        y_train,
        poison_rate=rate,
        target_class="yes",
        random_state=random_state
    )

    n_injected = int(mask.sum())
    n_expected = round(len(X_train) * rate)

    assert n_injected == n_expected
    assert len(X_augmented) == len(X_train) + n_expected
    assert len(y_augmented) == len(X_augmented)

    # Les nouvelles observations appartiennent a yes
    assert (y_augmented[mask] == "yes").all()

    # Les observations originales restent intactes
    assert X_augmented.iloc[:len(X_train)].reset_index(
        drop=True
    ).equals(X_train.reset_index(drop=True))

    assert y_augmented.iloc[:len(y_train)].reset_index(
        drop=True
    ).equals(y_train.reset_index(drop=True))

    print(
        f"Taux : {rate * 100:.0f}% | "
        f"Observations ajoutees : {n_injected} | "
        f"Train apres : {len(X_augmented)}"
    )

    results.append({
        "attack": "P4_injection_duplication",
        "poison_rate": rate,
        "rate_denominator": "all_train",
        "n_train": len(X_train),
        "n_eligible": n_target,
        "n_poisoned": n_injected,
        "n_train_after": len(X_augmented),
        "random_state": random_state
    })


# ==========================================
# 8. VERIFICATIONS FINALES
# ==========================================

assert X_train.equals(X_train_original)
assert y_train.equals(y_train_original)
assert X_val.equals(X_val_original)
assert y_val.equals(y_val_original)
assert X_test.equals(X_test_original)
assert y_test.equals(y_test_original)

print("\nDonnees originales preservees.")
print("Validation et Test restent propres.")


# ==========================================
# 9. ENREGISTREMENT DES RESULTATS
# ==========================================

results_df = pd.DataFrame(results)

assert len(results_df) == 16

output_path = Path(
    "results/csv/poisoning_generation.csv"
)

output_path.parent.mkdir(
    parents=True,
    exist_ok=True
)

results_df.to_csv(output_path, index=False)

print("\n=== TABLEAU DES RESULTATS ===")
print(results_df.to_string(index=False))

print("\nCSV enregistre :", output_path)
print("\n16 EXPERIENCES TERMINEES !")
