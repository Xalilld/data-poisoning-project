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


## Semaine 2 — Baseline propre

L'objectif de la semaine 2 est de construire une baseline propre avant l'introduction du data poisoning.

### Étapes réalisées

- Exploration du dataset Bank Marketing
- Séparation de la variable cible `y`
- Split stratifié des données :
  - 70 % entraînement
  - 15 % validation
  - 15 % test
- Prétraitement des variables numériques avec `StandardScaler`
- Encodage des variables catégorielles avec `OneHotEncoder`
- Entraînement d'une régression logistique
- Entraînement d'un Random Forest
- Évaluation avec :
  - Accuracy
  - Balanced Accuracy
  - Precision
  - Recall
  - F1-score
  - ROC-AUC
  - Matrice de confusion

### Exécution du livrable

Depuis la racine du projet :

```bash
python experiments/baseline.py
```

Le script entraîne les deux modèles sur l'ensemble d'entraînement et les évalue sur l'ensemble de validation.

L'ensemble de test reste intact pour les évaluations finales.

## Semaine 3 — Génération des attaques de Data Poisoning

L'objectif de la semaine 3 est d'implémenter différentes techniques de data poisoning afin de préparer l'étude de leur impact sur les modèles d'apprentissage automatique.

### Étapes réalisées

- Implémentation de quatre attaques :
  - P1 : Random Label Flipping (inversion aléatoire des labels)
  - P2 : Targeted Label Flipping (inversion ciblée des labels `yes` vers `no`)
  - P3 : Feature Poisoning (modification de la variable `balance`)
  - P4 : Injection / Duplication (ajout de copies d'observations de la classe `yes`)
- Application des attaques uniquement sur l'ensemble d'entraînement
- Expérimentation avec quatre taux : 1 %, 5 %, 10 % et 20 %
- Réalisation de 16 expériences au total
- Vérification du nombre d'observations modifiées ou ajoutées
- Vérification de la préservation des données originales
- Conservation des ensembles de validation et de test propres
- Création de tests unitaires pour les quatre attaques
- Enregistrement des résultats dans `results/csv/poisoning_generation.csv`

### Exécution du livrable

Depuis la racine du projet :

```bash
python experiments/poisoning_experiments.py
```

Le script génère les quatre attaques avec différents taux d'empoisonnement et enregistre les résultats dans un fichier CSV.

Les ensembles de validation et de test restent intacts pour les évaluations ultérieures.

## Semaine 4 — Évaluation de l'impact du Data Poisoning

L'objectif de la semaine 4 est de mesurer l'impact des attaques sur les performances des modèles d'apprentissage automatique.

### Étapes réalisées

- Entraînement des modèles Logistic Regression et Random Forest
- Évaluation des quatre attaques P1, P2, P3 et P4
- Expérimentation avec quatre taux : 1 %, 5 %, 10 % et 20 %
- Réalisation de 34 entraînements, dont 2 baselines propres
- Comparaison des performances avec Accuracy, Balanced Accuracy, Precision, Recall, F1-score et ROC-AUC
- Analyse des effets des attaques sur les deux modèles
- Génération de six graphiques comparatifs
- Enregistrement des résultats dans `results/csv/poisoning_impact.csv`

### Exécution du livrable

Depuis la racine du projet :

```bash
python experiments/impact_experiments.py
python experiments/plot_impact.py
```

Le premier script entraîne et évalue les modèles propres et empoisonnés sur l'ensemble de validation.

Le second script génère les graphiques dans `results/figures/`.

L'ensemble de test reste intact pour les évaluations finales.



## Semaine 5 — Détection statistique du Data Poisoning

### Objectif

Développer une première version d'un système de détection statistique des observations suspectes dans un jeu de données d'apprentissage automatique.

### Méthodes implémentées

- **Isolation Forest** : détection des anomalies à partir de la variable `balance`.
- **Contrôle des valeurs répétées** : recherche de valeurs élevées anormalement fréquentes.
- **Détection des doublons** : identification des observations dupliquées.

Le fichier `src/detect.py` centralise ces trois méthodes.

### Résultats expérimentaux

Les expériences utilisent le jeu de données Bank Marketing et les attaques P3 (modification de caractéristiques) et P4 (injection de doublons).

| Attaque | Méthode | Precision | Recall | F1-score |
|---|---|---:|---:|---:|
| P3 à 10 % | Isolation Forest, contamination 5 % | 0,0000 | 0,0000 | 0,0000 |
| P3 à 10 % | Valeurs élevées répétées | 0,9994 | 1,0000 | 0,9997 |
| P4 à 10 % | Détection des doublons | 1,0000 | 1,0000 | 1,0000 |

### Analyse des scores

L'analyse des distributions de scores montre que les observations empoisonnées par P3 ne franchissent pas le seuil de décision d'Isolation Forest à 10 % d'empoisonnement et 5 % de contamination.

Les performances dépendent donc du type d'attaque et du seuil utilisé.

### Limites

- Les expériences utilisent principalement la graine aléatoire 42.
- Isolation Forest est appliqué uniquement à la variable `balance`.
- La détection des valeurs répétées utilise une référence propre et une connaissance de la caractéristique attaquée.
- La détection des doublons est évaluée sur une attaque simulée qui ajoute des copies exactes.
- Les résultats ne garantissent pas une performance équivalente sur des attaques inconnues.

### Fichiers principaux

- `src/detect.py` : interface de détection statistique.
- `src/detection_features.py` : Isolation Forest.
- `src/detection_repeated_values.py` : contrôle des valeurs élevées répétées.
- `src/detection_duplicates.py` : détection des doublons.
- `experiments/plot_detection_week5.py` : génération des graphiques.

### Reproduire les tests

```powershell
python experiments/test_detect_v1.py
python experiments/test_detect_p4.py
python experiments/detection_p3_fixed.py
python experiments/plot_detection_week5.py
```

### Graphiques

Les figures sont enregistrées dans `results/figures/` :

- `week5_isolation_fixed.png`
- `week5_comparison_p3.png`
- `week5_score_distribution.png`

**Livrable :** `detect.py` version 1, métriques de détection et analyse des distributions de scores.

LOF n'a pas été ajouté, car il est facultatif dans le plan initial.



## Semaine 6 — Détection basée sur les modèles

### Objectif

Détecter les observations potentiellement empoisonnées en analysant les probabilités prédites par un modèle de machine learning.

### Méthodes implémentées

- **Probabilités out-of-fold** : validation croisée stratifiée à cinq plis avec régression logistique.
- **Score de cohérence label/prédiction** : score de suspicion calculé à partir de la probabilité attribuée à l'étiquette observée.
- **Cleanlab** : identification des observations présentant des erreurs potentielles d'étiquetage.

### Expériences

Les deux détecteurs ont été comparés sur les attaques P1 (random label flipping) et P2 (targeted label flipping), aux taux de 1 %, 5 %, 10 % et 20 %.

### Résultats — F1-score

| Attaque | Taux | Cleanlab | Cohérence |
|---|---:|---:|---:|
| P1 | 1 % | 0,2681 | 0,1546 |
| P1 | 5 % | 0,6150 | 0,4755 |
| P1 | 10 % | 0,7396 | 0,6444 |
| P1 | 20 % | 0,7902 | 0,7829 |
| P2 | 1 % | 0,0185 | 0,0082 |
| P2 | 5 % | 0,0717 | 0,0337 |
| P2 | 10 % | 0,1368 | 0,0671 |
| P2 | 20 % | 0,1652 | 0,0907 |

Le tableau complet contenant Precision, Recall, F1 et Average Precision est disponible dans `results/csv/week6_model_comparison.csv`.

### Observations

Cleanlab obtient un meilleur F1-score dans les huit configurations étudiées. L'attaque ciblée P2 est plus difficile à détecter que P1.

Les deux méthodes obtiennent la même Average Precision, car les scores continus utilisés produisent le même classement des observations.

### Limites

Les expériences utilisent principalement une seule graine aléatoire (42), une régression logistique et un seuil de cohérence fixé à 0,50. Les résultats concernent des attaques simulées.

### Fichiers principaux

- `src/detection.py`
- `experiments/compare_model_detectors.py`
- `experiments/test_inconsistency.py`
- `experiments/plot_week6.py`
- `results/csv/week6_model_comparison.csv`

### Graphiques

- `results/figures/week6_f1_comparison.png`
- `results/figures/week6_ap_comparison.png`
- `results/figures/week6_precision_recall.png`

### Reproduction

```powershell
python experiments/test_inconsistency.py
python experiments/compare_model_detectors.py
python experiments/plot_week6.py
```

**Livrable :** tableau comparatif des détecteurs avec Precision, Recall, F1 et Average Precision (AP, résumé de la courbe précision-rappel).