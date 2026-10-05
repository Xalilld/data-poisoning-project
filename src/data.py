import pandas as pd
from sklearn.model_selection import train_test_split


# 1. Charger le dataset
df = pd.read_csv("data/raw/bank-full.csv", sep=";")


# 2. Séparer les variables X de la cible y
X = df.drop(columns=["y"])
y = df["y"]


# 3. Séparer 70 % pour l'entraînement et 30 % temporairement
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)


# 4. Diviser les 30 % restants en 15 % validation et 15 % test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)


# 5. Afficher les résultats
print("=== TAILLE DES ENSEMBLES ===")
print("Train      :", X_train.shape)
print("Validation :", X_val.shape)
print("Test       :", X_test.shape)


# 6. Vérifier la proportion de yes/no
print("\n=== PROPORTION DE y ===")

print("\nTrain :")
print(y_train.value_counts(normalize=True) * 100)

print("\nValidation :")
print(y_val.value_counts(normalize=True) * 100)

print("\nTest :")
print(y_test.value_counts(normalize=True) * 100)