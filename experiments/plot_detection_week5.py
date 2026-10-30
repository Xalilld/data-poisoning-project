
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from poison import feature_poisoning

CSV_DIR = ROOT / "results" / "csv"
FIG_DIR = ROOT / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


# GRAPHIQUE 1 : Isolation Forest, contamination fixe
fixed = pd.read_csv(CSV_DIR / "detection_p3_fixed.csv")

plt.figure(figsize=(9, 5))

for metric in ["precision", "recall", "f1"]:
    plt.plot(
        fixed["poison_rate"] * 100,
        fixed[metric],
        marker="o",
        label=metric.capitalize()
    )

plt.xlabel("Taux de poisoning (%)")
plt.ylabel("Score")
plt.title("P3 - Isolation Forest (contamination fixe 5 %)")
plt.xticks([1, 5, 10, 20])
plt.ylim(0, 1.05)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    FIG_DIR / "week5_isolation_fixed.png",
    dpi=300
)
plt.close()


# GRAPHIQUE 2 : comparaison des détecteurs P3
comparison = pd.read_csv(
    CSV_DIR / "detection_p3_comparison.csv"
)

print("Colonnes du fichier de comparaison :")
print(comparison.columns.tolist())

rate_col = (
    "poison_rate"
    if "poison_rate" in comparison.columns
    else "rate"
)

method_col = (
    "method"
    if "method" in comparison.columns
    else "detector"
)

plt.figure(figsize=(9, 5))

for method, group in comparison.groupby(method_col):
    group = group.sort_values(rate_col)

    plt.plot(
        group[rate_col] * 100,
        group["f1"],
        marker="o",
        label=method
    )

plt.xlabel("Taux de poisoning (%)")
plt.ylabel("F1-score")
plt.title("P3 - Comparaison des méthodes de détection")
plt.xticks([1, 5, 10, 20])
plt.ylim(0, 1.05)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    FIG_DIR / "week5_comparison_p3.png",
    dpi=300
)
plt.close()


# GRAPHIQUE 3 : distribution des scores
df = pd.read_csv(
    ROOT / "data" / "raw" / "bank-full.csv",
    sep=";"
)

X = df.drop(columns=["y"])
y = df["y"]

X_train, _, _, _ = train_test_split(
    X,
    y,
    test_size=0.30,
    stratify=y,
    random_state=42
)

X_poisoned, mask = feature_poisoning(
    X_train,
    poison_rate=0.10,
    feature="balance",
    random_state=42
)

truth = np.asarray(mask, dtype=bool)

detector = IsolationForest(
    n_estimators=100,
    contamination=0.05,
    random_state=42,
    n_jobs=-1
)

values = X_poisoned[["balance"]]
detector.fit(values)

scores = detector.score_samples(values)

plt.figure(figsize=(9, 5))

plt.hist(
    scores[~truth],
    bins=60,
    alpha=0.65,
    label="Non empoisonnées"
)

plt.axvline(
    scores[truth][0],
    color="red",
    linestyle="--",
    linewidth=2,
    label="Score P3"
)

plt.axvline(
    detector.offset_,
    color="black",
    linestyle=":",
    linewidth=2,
    label="Seuil Isolation Forest"
)

plt.xlabel("Score d'anomalie (plus faible = plus anormal)")
plt.ylabel("Nombre d'observations")
plt.title("P3 à 10 % - Distribution des scores")
plt.legend()
plt.tight_layout()

plt.savefig(
    FIG_DIR / "week5_score_distribution.png",
    dpi=300
)
plt.close()


print("\n=== GRAPHIQUES SEMAINE 5 ===")
print("1. week5_isolation_fixed.png")
print("2. week5_comparison_p3.png")
print("3. week5_score_distribution.png")
print("\nTous les graphiques ont été enregistrés.")
