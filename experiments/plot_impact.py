
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = PROJECT_ROOT / "results/csv/poisoning_impact.csv"
OUTPUT_DIR = PROJECT_ROOT / "results/figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT_FILE)

models = ["Logistic Regression", "Random Forest"]
metrics = {
    "recall": "Recall",
    "f1": "F1-score",
    "roc_auc": "ROC-AUC",
}
attacks = ["P1", "P2", "P3", "P4"]

for model_name in models:
    model_df = df[df["model"] == model_name]
    baseline = model_df[model_df["attack"] == "clean"].iloc[0]

    for metric, metric_label in metrics.items():
        plt.figure(figsize=(9, 5))

        for attack in attacks:
            subset = model_df[
                model_df["attack"] == attack
            ].sort_values("poison_rate")

            plt.plot(
                subset["poison_rate"] * 100,
                subset[metric],
                marker="o",
                label=attack,
            )

        plt.axhline(
            baseline[metric],
            linestyle="--",
            color="black",
            label="Baseline propre",
        )

        plt.xlabel("Taux d'empoisonnement (%)")
        plt.ylabel(metric_label)
        plt.title(f"{metric_label} — {model_name}")
        plt.xticks([1, 5, 10, 20])
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()

        model_slug = (
            "logistic_regression"
            if model_name == "Logistic Regression"
            else "random_forest"
        )

        output_file = OUTPUT_DIR / (
            f"impact_{model_slug}_{metric}.png"
        )

        plt.savefig(output_file, dpi=200)
        plt.close()

        print(f"Graphique cree : {output_file.name}")

print("\nLes 6 graphiques ont ete generes !")
