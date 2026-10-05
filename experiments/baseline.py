import sys
from pathlib import Path

# Permet d'importer les fichiers du dossier src
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from models import (
    create_logistic_model,
    create_random_forest_model
)


# ==========================================
# 1. CHARGEMENT DES DONNÉES
# ==========================================

df = pd.read_csv(
    "data/raw/bank-full.csv",
    sep=";"
)

# X = variables utilisées pour faire les prédictions
X = df.drop(columns=["y"])

# y = variable cible : yes / no
y = df["y"]


# ==========================================
# 2. SPLIT 70 % / 15 % / 15 %
# ==========================================

# 70 % entraînement
# 30 % temporaire
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# Les 30 % restants sont divisés en :
# 15 % validation
# 15 % test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)


print("=== TAILLE DES ENSEMBLES ===")
print("Train      :", X_train.shape)
print("Validation :", X_val.shape)
print("Test       :", X_test.shape)


# ==========================================
# 3. FONCTION D'ÉVALUATION
# ==========================================

def evaluate_model(name, model):

    print(f"\n=== {name} ===")

    # --------------------------------------
    # Entraînement
    # --------------------------------------

    print("Entraînement...")

    model.fit(
        X_train,
        y_train
    )

    print("Entraînement terminé !")

    # --------------------------------------
    # Prédictions sur VALIDATION
    # --------------------------------------

    y_pred = model.predict(X_val)

    # Probabilité associée à la classe "yes"
    y_prob = model.predict_proba(X_val)[:, 1]

    # Conversion :
    # no  -> 0
    # yes -> 1
    # nécessaire pour ROC-AUC
    y_val_binary = (
        y_val == "yes"
    ).astype(int)

    # --------------------------------------
    # Calcul des métriques
    # --------------------------------------

    accuracy = accuracy_score(
        y_val,
        y_pred
    )

    balanced_accuracy = balanced_accuracy_score(
        y_val,
        y_pred
    )

    precision = precision_score(
        y_val,
        y_pred,
        pos_label="yes"
    )

    recall = recall_score(
        y_val,
        y_pred,
        pos_label="yes"
    )

    f1 = f1_score(
        y_val,
        y_pred,
        pos_label="yes"
    )

    roc_auc = roc_auc_score(
        y_val_binary,
        y_prob
    )

    # --------------------------------------
    # Matrice de confusion
    # --------------------------------------

    cm = confusion_matrix(
        y_val,
        y_pred,
        labels=["no", "yes"]
    )

    # --------------------------------------
    # Affichage des résultats
    # --------------------------------------

    print(f"Accuracy          : {accuracy:.4f}")
    print(f"Balanced Accuracy : {balanced_accuracy:.4f}")
    print(f"Precision         : {precision:.4f}")
    print(f"Recall            : {recall:.4f}")
    print(f"F1-score          : {f1:.4f}")
    print(f"ROC-AUC           : {roc_auc:.4f}")

    print("\nMatrice de confusion :")
    print(cm)

    # --------------------------------------
    # Sauvegarde de la matrice en image
    # --------------------------------------

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["no", "yes"]
    )

    display.plot()

    plt.title(
        f"Matrice de confusion - {name}"
    )

    plt.tight_layout()

    # Nom propre pour le fichier
    figure_name = (
        name.lower()
        .replace(" ", "_")
        .replace("é", "e")
    )

    plt.savefig(
        f"results/figures/confusion_{figure_name}.png"
    )

    # Fermer la figure après sauvegarde
    plt.close()

    # --------------------------------------
    # Retourner les résultats
    # --------------------------------------

    return {
        "model": name,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc
    }


# ==========================================
# 4. CRÉATION DES MODÈLES
# ==========================================

logistic_model = create_logistic_model()

random_forest_model = create_random_forest_model()


# ==========================================
# 5. ÉVALUATION DES MODÈLES
# ==========================================

print("\n=== BASELINE PROPRE ===")

logistic_results = evaluate_model(
    "Régression logistique",
    logistic_model
)

random_forest_results = evaluate_model(
    "Random Forest",
    random_forest_model
)


# ==========================================
# 6. CRÉATION DU TABLEAU DE RÉSULTATS
# ==========================================

results = pd.DataFrame([
    logistic_results,
    random_forest_results
])


# ==========================================
# 7. SAUVEGARDE DU CSV
# ==========================================

results.to_csv(
    "results/csv/baseline_results.csv",
    index=False
)


# ==========================================
# 8. AFFICHAGE FINAL
# ==========================================

print("\n=== COMPARAISON DES MODÈLES ===")

print(
    results.to_string(index=False)
)

print("\n=== SAUVEGARDE TERMINÉE ===")

print(
    "CSV : results/csv/baseline_results.csv"
)

print(
    "Figures : results/figures/"
)