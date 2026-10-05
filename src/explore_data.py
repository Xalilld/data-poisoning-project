import pandas as pd

# Charger le dataset
df = pd.read_csv("data/raw/bank-full.csv", sep=";")

print("=== DIMENSIONS DU DATASET ===")
print(df.shape)

print("\n=== NOMS DES COLONNES ===")
print(df.columns.tolist())

print("\n=== TYPES DES VARIABLES ===")
print(df.dtypes)

print("\n=== VALEURS MANQUANTES ===")
print(df.isnull().sum())

print("\n=== VARIABLES NUMÉRIQUES ===")
print(df.select_dtypes(include="number").columns.tolist())

print("\n=== VARIABLES CATÉGORIELLES ===")
print(df.select_dtypes(exclude="number").columns.tolist())

print("\n=== DISTRIBUTION DE LA CIBLE y ===")
print(df["y"].value_counts())
print("\nPourcentages :")
print(df["y"].value_counts(normalize=True) * 100)