
import sys
from pathlib import Path

import numpy as np
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
    confusion_matrix,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from models import create_logistic_model
from poison import random_label_flipping
from detection import detect_label_issues


RANDOM_STATE = 42


def main():
    print("=== SEMAINE 5 : DETECTION DU DATA POISONING ===")

    # 1. Chargement
    df = pd.read_csv(
        PROJECT_ROOT / "data/raw/bank-full.csv",
        sep=";"
    )

    X = df.drop(columns=["y"])
    y = df["y"]

    # 2. Même séparation que les semaines précédentes
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=RANDOM_STATE,
    )

    # 3. Attaque P1 à 10 %
    y_poisoned, is_poisoned = random_label_flipping(
        y_train,
        poison_rate=0.10,
        random_state=RANDOM_STATE,
    )

    print("Observations empoisonnées :", int(is_poisoned.sum()))

    # 4. Prédictions hors échantillon sur Train
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    model = create_logistic_model()

    pred_probs = cross_val_predict(
        model,
        X_train,
        y_poisoned,
        cv=cv,
        method="predict_proba",
        n_jobs=1,
    )

    # Le modèle classe les probabilités : no, yes
    # conformément à l'encodage utilisé par Cleanlab.
    print("Probabilités calculées :", pred_probs.shape)

    # 5. Détection
    detected = detect_label_issues(
        y_poisoned,
        pred_probs
    )

    # 6. Évaluation par rapport au masque réel
    truth = is_poisoned.to_numpy().astype(int)
    predicted = detected.astype(int)

    precision = precision_score(
        truth, predicted, zero_division=0
    )
    recall = recall_score(
        truth, predicted, zero_division=0
    )
    f1 = f1_score(
        truth, predicted, zero_division=0
    )

    print("\n=== RESULTATS DE DETECTION ===")
    print("Labels empoisonnés :", truth.sum())
    print("Labels suspects détectés :", predicted.sum())
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-score  : {f1:.4f}")
    print("Matrice de confusion :")
    print(confusion_matrix(truth, predicted))

    # 7. Enregistrement des résultats
    output_dir = PROJECT_ROOT / "results/csv"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = pd.DataFrame([{
        "attack": "P1",
        "poison_rate": 0.10,
        "detector": "Cleanlab",
        "model": "Logistic Regression",
        "n_poisoned": int(truth.sum()),
        "n_detected": int(predicted.sum()),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "random_state": RANDOM_STATE,
    }])

    output_path = output_dir / "detection_p1_10_percent.csv"
    results.to_csv(output_path, index=False)

    print("\nRésultats enregistrés :", output_path)


if __name__ == "__main__":
    main()

import sys
from pathlib import Path

import numpy as np
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
    confusion_matrix,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from models import create_logistic_model
from poison import random_label_flipping
from detection import detect_label_issues


RANDOM_STATE = 42


def main():
    print("=== SEMAINE 5 : DETECTION DU DATA POISONING ===")

    # 1. Chargement
    df = pd.read_csv(
        PROJECT_ROOT / "data/raw/bank-full.csv",
        sep=";"
    )

    X = df.drop(columns=["y"])
    y = df["y"]

    # 2. Même séparation que les semaines précédentes
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=RANDOM_STATE,
    )

    # 3. Attaque P1 à 10 %
    y_poisoned, is_poisoned = random_label_flipping(
        y_train,
        poison_rate=0.10,
        random_state=RANDOM_STATE,
    )

    print("Observations empoisonnées :", int(is_poisoned.sum()))

    # 4. Prédictions hors échantillon sur Train
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    model = create_logistic_model()

    pred_probs = cross_val_predict(
        model,
        X_train,
        y_poisoned,
        cv=cv,
        method="predict_proba",
        n_jobs=1,
    )

    # Le modèle classe les probabilités : no, yes
    # conformément à l'encodage utilisé par Cleanlab.
    print("Probabilités calculées :", pred_probs.shape)

    # 5. Détection
    detected = detect_label_issues(
        y_poisoned,
        pred_probs
    )

    # 6. Évaluation par rapport au masque réel
    truth = is_poisoned.to_numpy().astype(int)
    predicted = detected.astype(int)

    precision = precision_score(
        truth, predicted, zero_division=0
    )
    recall = recall_score(
        truth, predicted, zero_division=0
    )
    f1 = f1_score(
        truth, predicted, zero_division=0
    )

    print("\n=== RESULTATS DE DETECTION ===")
    print("Labels empoisonnés :", truth.sum())
    print("Labels suspects détectés :", predicted.sum())
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-score  : {f1:.4f}")
    print("Matrice de confusion :")
    print(confusion_matrix(truth, predicted))

    # 7. Enregistrement des résultats
    output_dir = PROJECT_ROOT / "results/csv"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = pd.DataFrame([{
        "attack": "P1",
        "poison_rate": 0.10,
        "detector": "Cleanlab",
        "model": "Logistic Regression",
        "n_poisoned": int(truth.sum()),
        "n_detected": int(predicted.sum()),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "random_state": RANDOM_STATE,
    }])

    output_path = output_dir / "detection_p1_10_percent.csv"
    results.to_csv(output_path, index=False)

    print("\nRésultats enregistrés :", output_path)


if __name__ == "__main__":
    main()
