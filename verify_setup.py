import pandas as pd
import sklearn
import matplotlib
import cleanlab

print("=== Vérification de l'environnement ===")
print("Pandas :", pd.__version__)
print("Scikit-learn :", sklearn.__version__)
print("Matplotlib :", matplotlib.__version__)
print("Cleanlab :", cleanlab.__version__)

print("\n=== Chargement de Bank Marketing ===")

data_path = "data/raw/bank-full.csv"

df = pd.read_csv(data_path, sep=";")

print("Dataset chargé avec succès !")
print("Nombre de lignes :", df.shape[0])
print("Nombre de colonnes :", df.shape[1])

print("\nPremières lignes :")
print(df.head())

print("\nVariable cible y :")
print(df["y"].value_counts())