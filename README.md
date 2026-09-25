# Détection du Data Poisoning en apprentissage automatique

## Présentation du projet

Ce projet de baccalauréat porte sur la détection du **Data Poisoning** en apprentissage automatique.

Le Data Poisoning consiste à modifier ou à ajouter des données dans les données d'entraînement afin d'influencer ou de dégrader l'apprentissage d'un modèle de Machine Learning.

L'objectif de ce projet est de simuler plusieurs formes de Data Poisoning sur le dataset **Bank Marketing**, de mesurer leur impact sur des modèles de classification, puis d'essayer de détecter et de nettoyer les données contaminées.

Deux modèles seront principalement étudiés :

- Régression logistique
- Random Forest

---

## Comprendre le Data Poisoning

Un modèle de Machine Learning apprend à partir des données qu'on lui fournit.

Si les données d'entraînement sont volontairement modifiées, le modèle peut apprendre de mauvaises relations et produire des prédictions incorrectes.

Le principe général étudié dans ce projet est donc :

```text
Données propres
      ↓
Poisoning
      ↓
Données contaminées
      ↓
Entraînement du modèle
      ↓
Mesure de l'impact
      ↓
Détection des données suspectes
      ↓
Nettoyage
      ↓
Réentraînement
```

Une règle importante du projet est que le poisoning sera appliqué uniquement aux données d'entraînement. Les données de validation et de test resteront propres.

---

## Taxonomie du Data Poisoning

Une **taxonomie** est une classification organisée des différents types d'attaques.

Parmi les catégories de poisoning, on peut notamment distinguer :

### Poisoning non ciblé / Availability Poisoning

L'objectif est de dégrader plus globalement les performances du modèle.

```text
Données propres
      ↓
Contamination
      ↓
Modèle globalement moins fiable
```

### Targeted Poisoning

L'objectif est de provoquer un comportement incorrect sur une cible ou un sous-groupe particulier.

```text
Fonctionnement normal
      ↓
Cas particulier ciblé
      ↓
Erreur recherchée
```

### Backdoor Poisoning

Le modèle apprend un comportement caché qui peut être activé lorsqu'un motif particulier apparaît.

Dans ce projet, le backdoor poisoning constitue une extension facultative qui pourra être étudiée avec Fashion-MNIST si les objectifs principaux sont terminés.

Référence :
NIST — Adversarial Machine Learning: A Taxonomy and Terminology of Attacks and Mitigations.

---

## Scénarios étudiés dans le projet

Les principaux scénarios de Data Poisoning qui seront expérimentés sont :

- Random Label Flipping
- Targeted Label Flipping
- Feature Poisoning
- Injection / duplication de données

Plusieurs taux de contamination seront utilisés afin d'étudier l'évolution des performances des modèles.

---

# Mise en place du projet

## 1. Création du projet

J'ai commencé par créer le dossier principal du projet :

```powershell
mkdir data-poisoning-project
cd data-poisoning-project
```

---

## 2. Création de l'environnement virtuel

J'ai créé un environnement virtuel Python :

```powershell
python -m venv .venv
```

L'environnement virtuel permet d'isoler les bibliothèques utilisées par ce projet de celles de mes autres projets Python.

Activation sous PowerShell :

```powershell
.\.venv\Scripts\Activate.ps1
```

---

## 3. Installation des bibliothèques

Les principales bibliothèques nécessaires au projet sont :

- pandas
- scikit-learn
- Matplotlib
- Cleanlab

Installation :

```powershell
python -m pip install --upgrade pip
pip install pandas scikit-learn matplotlib cleanlab
```

---

## 4. Environnement reproductible

Afin de conserver les versions des dépendances utilisées, j'ai généré le fichier `requirements.txt` :

```powershell
pip freeze > requirements.txt
```

Une autre personne peut ensuite recréer l'environnement avec :

```powershell
pip install -r requirements.txt
```

---

## 5. Dataset Bank Marketing

Le projet utilise le dataset **Bank Marketing** provenant du UCI Machine Learning Repository.

J'ai téléchargé `bank.zip`, puis extrait le fichier :

```text
bank-full.csv
```

Le fichier original est conservé dans :

```text
data/
└── raw/
    └── bank-full.csv
```

Le dataset contient :

- 45 211 observations
- 17 colonnes
- une variable cible `y`
- `yes` : souscription au dépôt à terme
- `no` : absence de souscription

Lors de ma première vérification :

```text
no     39922
yes     5289
```

---

## 6. Vérification du dataset

J'ai créé le script :

```text
verify_setup.py
```

Il permet de vérifier :

- les versions des principales bibliothèques ;
- le chargement de Bank Marketing ;
- le nombre de lignes et de colonnes ;
- les premières observations ;
- la distribution de la variable cible `y`.

Exécution :

```powershell
python verify_setup.py
```

Résultat principal :

```text
Dataset chargé avec succès !
Nombre de lignes : 45211
Nombre de colonnes : 17
```

---

## 7. Organisation du projet

J'ai ensuite créé la structure nécessaire pour organiser les différentes étapes du projet :

```text
data-poisoning-project/
├── data/
│   ├── raw/
│   ├── processed/
│   └── poisoned/
├── notebooks/
├── src/
├── experiments/
├── results/
│   ├── csv/
│   └── figures/
├── tests/
├── .gitignore
├── README.md
├── requirements.txt
└── verify_setup.py
```

Cette organisation permettra de séparer les données originales, les données transformées, les expériences, le code source, les tests et les résultats.

---

## 8. Git et GitHub

Le projet est versionné avec Git.

Initialisation :

```powershell
git init
git add .
git commit -m "Initialisation du projet Data Poisoning"
git branch -M main
```

Le dépôt local est ensuite relié à GitHub afin de conserver l'historique du projet et de faciliter sa reproductibilité.

---

## État actuel

### Semaine 1 — Comprendre et installer

- [x] Comprendre le principe du Data Poisoning
- [x] Étudier les principales catégories de poisoning
- [x] Installer l'environnement Python
- [x] Installer pandas, scikit-learn, Matplotlib et Cleanlab
- [x] Télécharger Bank Marketing
- [x] Charger et vérifier le dataset
- [x] Créer `requirements.txt`
- [x] Créer la structure du projet
- [x] Initialiser Git
- [x] Créer le dépôt GitHub

### Prochaine étape — Baseline propre

La prochaine étape consistera à explorer les données, effectuer un découpage stratifié entraînement/validation/test, préparer le preprocessing puis entraîner une régression logistique et un Random Forest afin d'obtenir les performances de référence.
