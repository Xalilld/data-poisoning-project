
"""
Semaine 7 — Nettoyage et mitigation du Data Poisoning.

Comparaison :
1. Clean
2. Poisoned
3. Cleaned

Attaque : P1 (random label flipping) à 10 %.
Détecteur : score d'incohérence label/prédiction.
Seuil : 0.50.

Évaluation sur un jeu de test propre.
"""

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
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from models import (
    create_logistic_model,
    create_random_forest_model,
)
from poison import random_label_flipping
from detection import compute_inconsistency_scores
from mitigation import clean_suspicious_data


SEED = 42
POISON_RATE = 0.10
THRESHOLD = 0.50


def evaluate_model(model, X_train, y_train, X_test, y_test):
    """Entraîne un modèle et l'évalue sur le test propre."""

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    class_order = list(model.classes_)
    yes_index = class_order.index("yes")

    probabilities = model.predict_proba(X_test)[:, yes_index]

    y_true = (y_test == "yes").astype(int)
    y_pred = (predictions == "yes").astype(int)

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(
            y_true, y_pred
        ),
        "precision": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "recall": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "f1": f1_score(
            y_true, y_pred, zero_division=0
        ),
        "roc_auc": roc_auc_score(
            y_true, probabilities
        ),
    }


def main():
    print("\n=== SEMAINE 7 : CLEAN VS POISONED VS CLEANED ===")

    data_path = ROOT / "data" / "raw" / "bank-full.csv"

    df = pd.read_csv(data_path, sep=";")

    X = df.drop(columns=["y"])
    y = df["y"]

    # Découpage stratifié : 70 % / 15 % / 15 %.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        stratify=y,
        random_state=SEED,
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=SEED,
    )

    print("Train :", len(X_train))
    print("Validation :", len(X_val))
    print("Test propre :", len(X_test))

    # Empoisonnement uniquement du train.
    y_poisoned, is_poisoned = random_label_flipping(
        y_train,
        poison_rate=POISON_RATE,
        random_state=SEED,
    )

    print("\nObservations empoisonnées :", int(is_poisoned.sum()))

    # Détection à partir des prédictions out-of-fold.
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=SEED,
    )

    detector = create_logistic_model()

    print("\nCalcul des probabilités out-of-fold...")

    pred_probs = cross_val_predict(
        detector,
        X_train,
        y_poisoned,
        cv=cv,
        method="predict_proba",
        n_jobs=-1,
    )

    scores = compute_inconsistency_scores(
        y_poisoned,
        pred_probs,
    )

    # Nettoyage.
    X_cleaned, y_cleaned, removed_mask = (
        clean_suspicious_data(
            X_train,
            y_poisoned,
            scores,
            threshold=THRESHOLD,
        )
    )

    print("Observations supprimées :", int(removed_mask.sum()))
    print("Observations conservées :", len(X_cleaned))

    # Diagnostic expérimental : nous connaissons les lignes
    # empoisonnées parce que nous avons simulé l'attaque.
    true_poisoned = is_poisoned.to_numpy(dtype=bool)
    removed = removed_mask.to_numpy(dtype=bool)

    true_positives = int(np.sum(removed & true_poisoned))
    false_positives = int(np.sum(removed & ~true_poisoned))
    false_negatives = int(np.sum(~removed & true_poisoned))

    print("\n=== QUALITÉ DU NETTOYAGE ===")
    print("Empoisonnées supprimées :", true_positives)
    print("Propres supprimées :", false_positives)
    print("Empoisonnées conservées :", false_negatives)

    training_sets = {
        "Clean": (X_train, y_train),
        "Poisoned": (X_train, y_poisoned),
        "Cleaned": (X_cleaned, y_cleaned),
    }

    model_factories = {
        "Logistic Regression": create_logistic_model,
        "Random Forest": create_random_forest_model,
    }

    results = []

    for model_name, model_factory in model_factories.items():
        for condition, (X_fit, y_fit) in training_sets.items():
            print(
                f"\nEntraînement : {model_name} — {condition}",
                flush=True,
            )

            metrics = evaluate_model(
                model_factory(),
                X_fit,
                y_fit,
                X_test,
                y_test,
            )

            results.append({
                "model": model_name,
                "condition": condition,
                "attack": "P1",
                "poison_rate": POISON_RATE,
                "threshold": THRESHOLD,
                "n_train": len(X_fit),
                **metrics,
            })

            print(
                f"Accuracy={metrics['accuracy']:.4f} | "
                f"Recall={metrics['recall']:.4f} | "
                f"F1={metrics['f1']:.4f} | "
                f"ROC-AUC={metrics['roc_auc']:.4f}"
            )

    result_df = pd.DataFrame(results)

    output_path = (
        ROOT
        / "results"
        / "csv"
        / "week7_mitigation_comparison.csv"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_df.to_csv(output_path, index=False)

    print("\n=== TABLEAU COMPARATIF ===")
    print(result_df.to_string(index=False))

    print("\n=== RÉCUPÉRATION DU F1-SCORE ===")

    for model_name in model_factories:
        subset = result_df[result_df["model"] == model_name]

        clean_f1 = subset.loc[
            subset["condition"] == "Clean", "f1"
        ].iloc[0]

        poisoned_f1 = subset.loc[
            subset["condition"] == "Poisoned", "f1"
        ].iloc[0]

        cleaned_f1 = subset.loc[
            subset["condition"] == "Cleaned", "f1"
        ].iloc[0]

        improvement = cleaned_f1 - poisoned_f1

        print(
            f"{model_name} : "
            f"Clean={clean_f1:.4f}, "
            f"Poisoned={poisoned_f1:.4f}, "
            f"Cleaned={cleaned_f1:.4f}, "
            f"gain={improvement:+.4f}"
        )

    print("\nRésultats enregistrés dans :", output_path)


if __name__ == "__main__":
    main()
