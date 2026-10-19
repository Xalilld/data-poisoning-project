
import numpy as np
import pandas as pd


# ==========================================
# P1 - RANDOM LABEL FLIPPING
# ==========================================

def random_label_flipping(y, poison_rate=0.10, random_state=42):
    """Inverse aleatoirement une proportion des labels."""

    if not 0 <= poison_rate <= 1:
        raise ValueError("Le taux doit etre entre 0 et 1.")

    if not y.isin(["yes", "no"]).all():
        raise ValueError("Les labels doivent etre yes ou no.")

    y_poisoned = y.copy()

    n_samples = len(y)
    n_poisoned = round(n_samples * poison_rate)

    rng = np.random.default_rng(random_state)

    positions = rng.choice(
        n_samples,
        size=n_poisoned,
        replace=False
    )

    is_poisoned = pd.Series(False, index=y.index)
    is_poisoned.iloc[positions] = True

    y_poisoned.iloc[positions] = (
        y_poisoned.iloc[positions]
        .map({"yes": "no", "no": "yes"})
    )

    return y_poisoned, is_poisoned


# ==========================================
# P2 - TARGETED LABEL FLIPPING
# ==========================================

def targeted_label_flipping(
    y,
    poison_rate=0.10,
    target_class="yes",
    random_state=42
):
    """Inverse les labels d'une proportion de la classe cible."""

    if not 0 <= poison_rate <= 1:
        raise ValueError("Le taux doit etre entre 0 et 1.")

    if target_class not in ["yes", "no"]:
        raise ValueError("La classe cible doit etre yes ou no.")

    if not y.isin(["yes", "no"]).all():
        raise ValueError("Les labels doivent etre yes ou no.")

    y_poisoned = y.copy()

    target_positions = np.flatnonzero(
        y.to_numpy() == target_class
    )

    n_poisoned = round(
        len(target_positions) * poison_rate
    )

    rng = np.random.default_rng(random_state)

    selected_positions = rng.choice(
        target_positions,
        size=n_poisoned,
        replace=False
    )

    is_poisoned = pd.Series(False, index=y.index)
    is_poisoned.iloc[selected_positions] = True

    replacement_class = (
        "no" if target_class == "yes" else "yes"
    )

    y_poisoned.iloc[selected_positions] = replacement_class

    return y_poisoned, is_poisoned


# ==========================================
# P3 - FEATURE POISONING
# ==========================================

def feature_poisoning(
    X,
    poison_rate=0.10,
    feature="balance",
    random_state=42
):
    """Modifie une proportion des valeurs d'une variable numerique."""

    if not 0 <= poison_rate <= 1:
        raise ValueError("Le taux doit etre entre 0 et 1.")

    if feature not in X.columns:
        raise ValueError("La variable choisie n'existe pas.")

    if not pd.api.types.is_numeric_dtype(X[feature]):
        raise ValueError("La variable doit etre numerique.")

    if X[feature].isna().any():
        raise ValueError("La variable contient des valeurs manquantes.")

    X_poisoned = X.copy(deep=True)

    n_samples = len(X)
    n_poisoned = round(n_samples * poison_rate)

    rng = np.random.default_rng(random_state)

    positions = rng.choice(
        n_samples,
        size=n_poisoned,
        replace=False
    )

    # Valeur elevee calculee sur les donnees d'entrainement
    high_value = float(X[feature].quantile(0.99))

    current_values = X[feature].iloc[positions].to_numpy()

    new_values = np.where(
        current_values == high_value,
        np.nextafter(high_value, np.inf),
        high_value
    )

    column_position = X.columns.get_loc(feature)

    X_poisoned[feature] = X_poisoned[feature].astype(float)

    X_poisoned.iloc[
        positions,
        column_position
    ] = new_values

    is_poisoned = pd.Series(False, index=X.index)
    is_poisoned.iloc[positions] = True

    return X_poisoned, is_poisoned


# ==========================================
# P4 - INJECTION / DUPLICATION
# ==========================================

def injection_duplication(
    X,
    y,
    poison_rate=0.10,
    target_class="yes",
    random_state=42
):
    """Ajoute des copies d'observations de la classe cible."""

    if not 0 <= poison_rate <= 1:
        raise ValueError("Le taux doit etre entre 0 et 1.")

    if target_class not in ["yes", "no"]:
        raise ValueError("La classe cible doit etre yes ou no.")

    if len(X) != len(y):
        raise ValueError("X et y doivent avoir la meme taille.")

    if not X.index.equals(y.index):
        raise ValueError("Les index de X et y doivent correspondre.")

    X_original = X.copy(deep=True)
    y_original = y.copy(deep=True)

    n_original = len(X)
    n_injected = round(n_original * poison_rate)

    target_positions = np.flatnonzero(
        y.to_numpy() == target_class
    )

    if n_injected > 0 and len(target_positions) == 0:
        raise ValueError("Aucune observation de la classe cible.")

    rng = np.random.default_rng(random_state)

    selected_positions = rng.choice(
        target_positions,
        size=n_injected,
        replace=True
    )

    X_copies = X.iloc[selected_positions].copy()
    y_copies = y.iloc[selected_positions].copy()

    X_augmented = pd.concat(
        [X_original, X_copies],
        ignore_index=True
    )

    y_augmented = pd.concat(
        [y_original, y_copies],
        ignore_index=True
    )

    is_injected = pd.Series(
        [False] * n_original + [True] * n_injected
    )

    return X_augmented, y_augmented, is_injected
