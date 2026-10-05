import sys
from pathlib import Path

# Permet d'importer les fichiers du dossier src
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
from sklearn.model_selection import train_test_split
from src.preprocessing import create_preprocessor


# Charger les données
df = pd.read_csv("data/raw/bank-full.csv", sep=";")

# Séparer X et y
X = df.drop(columns=["y"])
y = df["y"]

# Créer le même ensemble d'entraînement que précédemment
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# Créer le preprocessor
preprocessor = create_preprocessor()

# Apprendre les transformations sur TRAIN et transformer TRAIN
X_train_transformed = preprocessor.fit_transform(X_train)

print("=== TEST DU PREPROCESSING ===")

print("\nAvant preprocessing :")
print("Nombre de lignes :", X_train.shape[0])
print("Nombre de colonnes :", X_train.shape[1])

print("\nAprès preprocessing :")
print("Nombre de lignes :", X_train_transformed.shape[0])
print("Nombre de colonnes :", X_train_transformed.shape[1])

print("\nPreprocessing réussi !")