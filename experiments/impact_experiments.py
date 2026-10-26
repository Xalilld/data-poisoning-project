
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from models import create_logistic_model, create_random_forest_model
from poison import (
    random_label_flipping,
    targeted_label_flipping,
    feature_poisoning,
    injection_duplication,
)


RANDOM_STATE = 42
POISON_RATES = [0.01, 0.05, 0.10, 0.20]

MODELS = {
    "Logistic Regression": create_logistic_model,
    "Random Forest": create_random_forest_model,
}


def evaluate_model(model, X_val, y_val):
    predictions = model.predict(X_val)
    positive_index = list(model.classes_).index("yes")
    probabilities = model.predict_proba(X_val)[:, positive_index]

    return {
        "accuracy": accuracy_score(y_val, predictions),
        "balanced_accuracy": balanced_accuracy_score(y_val, predictions),
        "precision": precision_score(
            y_val, predictions, pos_label="yes", zero_division=0
        ),
        "recall": recall_score(
            y_val, predictions, pos_label="yes", zero_division=0
        ),
        "f1": f1_score(
            y_val, predictions, pos_label="yes", zero_division=0
        ),
        "roc_auc": roc_auc_score(
            (y_val == "yes").astype(int), probabilities
        ),
    }


def generate_attack(attack, X_train, y_train, rate):
    if attack == "P1":
        y_poisoned, mask = random_label_flipping(
            y_train, poison_rate=rate, random_state=RANDOM_STATE
        )
        return X_train.copy(), y_poisoned, int(mask.sum())

    if attack == "P2":
        y_poisoned, mask = targeted_label_flipping(
            y_train,
            poison_rate=rate,
            target_class="yes",
            random_state=RANDOM_STATE,
        )
        return X_train.copy(), y_poisoned, int(mask.sum())

    if attack == "P3":
        X_poisoned, mask = feature_poisoning(
            X_train,
            poison_rate=rate,
            feature="balance",
            random_state=RANDOM_STATE,
        )
        return X_poisoned, y_train.copy(), int(mask.sum())

    if attack == "P4":
        X_poisoned, y_poisoned, mask = injection_duplication(
            X_train,
            y_train,
            poison_rate=rate,
            target_class="yes",
            random_state=RANDOM_STATE,
        )
        return X_poisoned, y_poisoned, int(mask.sum())

    raise ValueError(f"Attaque inconnue : {attack}")


def main():
    print("=== SEMAINE 4 : IMPACT DU DATA POISONING ===")

    data_path = PROJECT_ROOT / "data/raw/bank-full.csv"
    df = pd.read_csv(data_path, sep=";")

    X = df.drop(columns=["y"])
    y = df["y"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=y_temp,
    )

    print(f"Train : {len(X_train)}")
    print(f"Validation : {len(X_val)}")
    print(f"Test : {len(X_test)}")

    results = []

    for model_name, model_factory in MODELS.items():
        print(f"\n=== {model_name} ===")

        # Baseline propre
        clean_model = model_factory()
        clean_model.fit(X_train, y_train)

        clean_metrics = evaluate_model(clean_model, X_val, y_val)

        results.append({
            "model": model_name,
            "attack": "clean",
            "poison_rate": 0.0,
            "n_poisoned": 0,
            "n_train_after": len(X_train),
            **clean_metrics,
        })

        print("Baseline propre terminee.")

        # Attaques
        for attack in ["P1", "P2", "P3", "P4"]:
            for rate in POISON_RATES:
                print(
                    f"En cours : {attack} | "
                    f"{rate * 100:.0f} %"
                )

                X_poisoned, y_poisoned, n_poisoned = generate_attack(
                    attack, X_train, y_train, rate
                )

                model = model_factory()
                model.fit(X_poisoned, y_poisoned)

                metrics = evaluate_model(model, X_val, y_val)

                results.append({
                    "model": model_name,
                    "attack": attack,
                    "poison_rate": rate,
                    "n_poisoned": n_poisoned,
                    "n_train_after": len(X_poisoned),
                    **metrics,
                })

    results_df = pd.DataFrame(results)

    output_path = (
        PROJECT_ROOT / "results/csv/poisoning_impact.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)

    print("\n=== RESULTATS ===")
    print(results_df.to_string(index=False))

    print(f"\nResultats enregistres : {output_path}")
    print(f"Nombre de lignes : {len(results_df)}")
    print("\nEXPERIENCES TERMINEES !")


if __name__ == "__main__":
    main()
