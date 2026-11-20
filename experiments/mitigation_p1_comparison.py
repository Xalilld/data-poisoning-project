"""Semaine 7 — Comparaison des mitigations P1 à 10 % sur un test propre."""

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split


SEED = 42
POISON_RATE = 0.10
THRESHOLD = 0.50
CSV_NAME = "week7_p1_score_cleanlab_comparison.csv"


def summarize_training_set(y_fit, removed, poisoned):
    """Compte les lignes supprimées et les étiquettes d'entraînement conservées."""
    return {
        "n_train": len(y_fit),
        "n_removed": int(removed.sum()),
        "removed_clean": int((removed & ~poisoned).sum()),
        "removed_poisoned": int((removed & poisoned).sum()),
        "retained_poisoned": int((~removed & poisoned).sum()),
        "retained_no": int(y_fit.eq("no").sum()),
        "retained_yes": int(y_fit.eq("yes").sum()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.project_root.resolve()
    output = args.output or root / "results" / "csv" / CSV_NAME

    sys.path[:0] = [str(root / "src"), str(root / "experiments")]
    from detection import compute_inconsistency_scores, detect_label_issues
    from mitigation import clean_suspicious_data
    from mitigation_experiments import evaluate_model
    from models import create_logistic_model, create_random_forest_model
    from poison import random_label_flipping

    data = pd.read_csv(root / "data" / "raw" / "bank-full.csv", sep=";")
    X_train, X_temp, y_train, y_temp = train_test_split(
        data.drop(columns="y"),
        data["y"],
        test_size=0.30,
        stratify=data["y"],
        random_state=SEED,
    )
    X_val, X_test, _, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=SEED
    )
    print(f"Train={len(X_train)} | Validation={len(X_val)} | Test propre={len(X_test)}")

    y_poisoned, is_poisoned = random_label_flipping(
        y_train, poison_rate=POISON_RATE, random_state=SEED
    )
    print(f"Étiquettes inversées dans le train = {is_poisoned.sum()}")

    print("Calcul des probabilités out-of-fold...", flush=True)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    # cross_val_predict ordonne les colonnes par classe : no, yes.
    pred_probs = cross_val_predict(
        create_logistic_model(),
        X_train,
        y_poisoned,
        cv=cv,
        method="predict_proba",
        n_jobs=-1,
    )
    scores = compute_inconsistency_scores(y_poisoned, pred_probs)
    X_score, y_score, score_removed = clean_suspicious_data(
        X_train, y_poisoned, scores, threshold=THRESHOLD
    )
    cleanlab_issues = detect_label_issues(y_poisoned, pred_probs)
    # Le nettoyeur accepte les décisions booléennes sous forme de valeurs 0/1.
    X_cleanlab, y_cleanlab, cleanlab_removed = clean_suspicious_data(
        X_train, y_poisoned, cleanlab_issues
    )

    no_removals = pd.Series(False, index=y_train.index)
    training_sets = [
        ("Clean", X_train, y_train, no_removals, no_removals, None, 0.0),
        ("Poisoned", X_train, y_poisoned, no_removals, is_poisoned, None, POISON_RATE),
        ("Cleaned score >0.50", X_score, y_score, score_removed, is_poisoned, THRESHOLD, POISON_RATE),
        ("Cleaned Cleanlab", X_cleanlab, y_cleanlab, cleanlab_removed, is_poisoned, None, POISON_RATE),
    ]
    model_factories = {
        "Logistic Regression": create_logistic_model,
        "Random Forest": create_random_forest_model,
    }

    summaries = []
    results = []
    for condition, X_fit, y_fit, removed, poisoned, threshold, poison_rate in training_sets:
        summary = {
            "condition": condition,
            "attack": "P1",
            "poison_rate": POISON_RATE,
            "applied_poison_rate": poison_rate,
            "threshold": threshold,
            **summarize_training_set(y_fit, removed, poisoned),
        }
        summaries.append(summary)
        for model_name, factory in model_factories.items():
            print(f"Entraînement — {model_name} / {condition}", flush=True)
            results.append({
                "model": model_name,
                **summary,
                **evaluate_model(factory(), X_fit, y_fit, X_test, y_test),
            })

    result_df = pd.DataFrame(results)
    metric_columns = [
        "model", "condition", "accuracy", "balanced_accuracy",
        "precision", "recall", "f1", "roc_auc",
    ]
    print("\n=== Nettoyage et classes conservées ===")
    print(pd.DataFrame(summaries).drop(
        columns=["attack", "poison_rate", "applied_poison_rate", "threshold"]
    ).to_string(index=False))
    print("\n=== Métriques sur le test propre ===")
    print(result_df[metric_columns].to_string(index=False, float_format="{:.6f}".format))

    output.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output, index=False)
    print(f"\nCSV enregistré dans {output.resolve()}")


if __name__ == "__main__":
    main()
