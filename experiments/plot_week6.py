
"""
Semaine 6 — Graphiques comparatifs des détecteurs.

Source :
results/csv/week6_model_comparison.csv

Sorties :
results/figures/week6_f1_comparison.png
results/figures/week6_ap_comparison.png
results/figures/week6_precision_recall.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

CSV_FILE = (
    ROOT / "results" / "csv"
    / "week6_model_comparison.csv"
)

FIG_DIR = ROOT / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV_FILE)

ATTACKS = ["P1", "P2"]
METHODS = ["Cleanlab", "Inconsistency"]

COLORS = {
    "Cleanlab": "tab:blue",
    "Inconsistency": "tab:orange",
}

# Graphique 1 : F1-score
fig, axes = plt.subplots(
    1, 2, figsize=(12, 5), sharey=True
)

for ax, attack in zip(axes, ATTACKS):
    for method in METHODS:
        subset = df[
            (df["attack"] == attack)
            & (df["method"] == method)
        ].sort_values("poison_rate")

        ax.plot(
            subset["poison_rate"] * 100,
            subset["f1"],
            marker="o",
            color=COLORS[method],
            label=method,
        )

    ax.set_title(f"Attaque {attack}")
    ax.set_xlabel("Taux d'empoisonnement (%)")
    ax.set_xticks([1, 5, 10, 20])
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend()

axes[0].set_ylabel("F1-score")
fig.suptitle("Semaine 6 — Comparaison des F1-scores")
fig.tight_layout()
fig.savefig(
    FIG_DIR / "week6_f1_comparison.png",
    dpi=300,
)
plt.close(fig)


# Graphique 2 : Average Precision
fig, axes = plt.subplots(
    1, 2, figsize=(12, 5), sharey=True
)

for ax, attack in zip(axes, ATTACKS):
    for method in METHODS:
        subset = df[
            (df["attack"] == attack)
            & (df["method"] == method)
        ].sort_values("poison_rate")

        ax.plot(
            subset["poison_rate"] * 100,
            subset["average_precision"],
            marker="o",
            color=COLORS[method],
            label=method,
        )

    ax.set_title(f"Attaque {attack}")
    ax.set_xlabel("Taux d'empoisonnement (%)")
    ax.set_xticks([1, 5, 10, 20])
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend()

axes[0].set_ylabel("Average Precision (AP)")
fig.suptitle(
    "Semaine 6 — Qualité du classement des anomalies"
)
fig.tight_layout()
fig.savefig(
    FIG_DIR / "week6_ap_comparison.png",
    dpi=300,
)
plt.close(fig)


# Graphique 3 : précision et rappel à 10 %
subset = df[df["poison_rate"] == 0.10]

fig, axes = plt.subplots(
    1, 2, figsize=(12, 5), sharey=True
)

for ax, attack in zip(axes, ATTACKS):
    attack_data = subset[
        subset["attack"] == attack
    ]

    positions = range(len(METHODS))
    width = 0.35

    for index, method in enumerate(METHODS):
        row = attack_data[
            attack_data["method"] == method
        ].iloc[0]

        ax.bar(
            index - width / 2,
            row["precision"],
            width,
            color="tab:blue",
            label="Precision" if index == 0 else None,
        )

        ax.bar(
            index + width / 2,
            row["recall"],
            width,
            color="tab:green",
            label="Recall" if index == 0 else None,
        )

    ax.set_xticks(list(positions))
    ax.set_xticklabels(METHODS)
    ax.set_title(f"Attaque {attack} — 10 %")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

axes[0].set_ylabel("Score")
fig.suptitle(
    "Semaine 6 — Precision et Recall à 10 %"
)
fig.tight_layout()
fig.savefig(
    FIG_DIR / "week6_precision_recall.png",
    dpi=300,
)
plt.close(fig)


print("\n=== GRAPHIQUES SEMAINE 6 ===")
print("1. week6_f1_comparison.png")
print("2. week6_ap_comparison.png")
print("3. week6_precision_recall.png")
print("\nGraphiques enregistrés dans :", FIG_DIR)
