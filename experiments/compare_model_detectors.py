
"""
Semaine 6 — Comparaison des détecteurs basés sur les modèles.

Détecteurs :
1. Cleanlab
2. Incohérence label/prédiction

Attaques :
P1 : random label flipping
P2 : targeted label flipping

Métriques :
Precision, Recall, F1, Average Precision (AP).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from cleanlab.rank import get_label_quality_scores
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_predict,
)
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from models import create_logistic_model
from poison import (
    random_label_flipping,
    targeted_label_flipping,
)
from detection import (
    detect_label_issues,
    compute_inconsistency_scores,
)

SEED = 42
RATES = [0.01, 0.05, 0.10, 0.20]
THRESHOLD = 0.50


def evaluate(truth, predicted, scores):
    truth = np.asarray(truth, dtype=int)
    predicted = np.asarray(predicted, dtype=int)

    return {
        "precision": precision_score(
            truth, predicted, zero_division=0
        ),
        "recall": recall_score(
            truth, predicted, zero_division=0
        ),
        "f1": f1_score(
            truth, predicted, zero_division=0
        ),
        "average_precision": average_precision_score(
            truth, scores
        ),
    }


def main():
    df = pd.read_csv(
        ROOT / "data/raw/bank-full.csv",
        sep=";"
    )

    X = df.drop(columns=["y"])
    y = df["y"]

    X_train, _, y_train, _ = train_test_split(
        X,
        y,
        test_size=0.30,
        stratify=y,
        random_state=SEED,
    )

    results = []

    attacks = {
        "P1": random_label_flipping,
        "P2": targeted_label_flipping,
    }

    for attack_name, attack_function in attacks.items():
        for rate in RATES:
            print(
                f"\nExpérience : {attack_name}, "
                f"taux {rate * 100:.0f} %",
                flush=True,
            )

            if attack_name == "P1":
                y_poisoned, truth = attack_function(
                    y_train,
                    poison_rate=rate,
                    random_state=SEED,
                )
            else:
                y_poisoned, truth = attack_function(
                    y_train,
                    poison_rate=rate,
                    target_class="yes",
                    random_state=SEED,
                )

            model = create_logistic_model()

            cv = StratifiedKFold(
                n_splits=5,
                shuffle=True,
                random_state=SEED,
            )

            # Chaque observation est prédite par un modèle
            # qui ne l'a pas utilisée pour son entraînement.
            pred_probs = cross_val_predict(
                model,
                X_train,
                y_poisoned,
                cv=cv,
                method="predict_proba",
                n_jobs=-1,
            )

            # Ordre attendu : no, yes
            cleanlab_mask = detect_label_issues(
                y_poisoned,
                pred_probs,
            )

            inconsistency_scores = (
                compute_inconsistency_scores(
                    y_poisoned,
                    pred_probs,
                )
            )

            consistency_mask = (
                inconsistency_scores > THRESHOLD
            )

            labels = pd.Series(y_poisoned).map(
                {"no": 0, "yes": 1}
            ).to_numpy(dtype=int)

            # Score continu propre à Cleanlab.
            # Une qualité faible signifie plus de suspicion.
            label_quality = get_label_quality_scores(
                labels=labels,
                pred_probs=pred_probs,
            )
            cleanlab_scores = 1.0 - label_quality

            for method, predicted, scores in [
                (
                    "Cleanlab",
                    cleanlab_mask,
                    cleanlab_scores,
                ),
                (
                    "Inconsistency",
                    consistency_mask,
                    inconsistency_scores,
                ),
            ]:
                metrics = evaluate(
                    truth,
                    predicted,
                    scores,
                )

                results.append({
                    "attack": attack_name,
                    "poison_rate": rate,
                    "method": method,
                    "n_poisoned": int(
                        np.asarray(truth).sum()
                    ),
                    "n_detected": int(
                        np.asarray(predicted).sum()
                    ),
                    **metrics,
                })

                print(
                    f"  {method}: "
                    f"F1={metrics['f1']:.4f}, "
                    f"AP={metrics['average_precision']:.4f}",
                    flush=True,
                )

    result_df = pd.DataFrame(results)

    output = (
        ROOT
        / "results/csv/week6_model_comparison.csv"
    )
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_df.to_csv(output, index=False)

    print("\n=== TABLEAU COMPARATIF SEMAINE 6 ===")
    print(result_df.to_string(index=False))
    print("\nFichier enregistré :", output)


if __name__ == "__main__":
    main()
