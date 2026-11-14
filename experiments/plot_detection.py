
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# Répertoire principal du projet
ROOT = Path(__file__).resolve().parents[1]

# Fichiers d'entrée et de sortie
INPUT = ROOT / "results/csv/detection_comparison.csv"
OUTPUT_DIR = ROOT / "results/figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Charger les résultats de la semaine 5
df = pd.read_csv(INPUT)

# Métriques à représenter
metrics = {
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1-score"
}

attacks = {
    "P1": "P1 - Random Label Flipping",
    "P2": "P2 - Targeted Label Flipping"
}

# Générer un graphique pour chaque métrique
for metric, title in metrics.items():

    plt.figure(figsize=(9, 5))

    for attack, label in attacks.items():

        data = df[df["attack"] == attack].sort_values(
            "poison_rate"
        )

        plt.plot(
            data["poison_rate"] * 100,
            data[metric] * 100,
            marker="o",
            linewidth=2,
            label=label
        )

    plt.title(
        f"Detection du Data Poisoning - {title}"
    )

    plt.xlabel("Taux d'empoisonnement (%)")
    plt.ylabel(f"{title} (%)")

    plt.xticks([1, 5, 10, 20])
    plt.ylim(0, 100)

    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()

    output_file = (
        OUTPUT_DIR / f"detection_cleanlab_{metric}.png"
    )

    plt.savefig(output_file, dpi=300)
    plt.close()

    print(f"Graphique enregistré : {output_file}")

print("\n=== TROIS GRAPHIQUES GENERES ===")
