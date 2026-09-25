\# Détection du Data Poisoning en apprentissage automatique



\## Description



Ce projet de baccalauréat étudie l'impact du Data Poisoning sur des modèles de classification en apprentissage automatique.



L'objectif est de simuler plusieurs scénarios de Data Poisoning, mesurer leurs effets sur les performances des modèles, détecter les observations contaminées et évaluer l'efficacité du nettoyage des données.



\## Dataset



Le projet utilise le dataset \*\*Bank Marketing\*\* du UCI Machine Learning Repository.



\- 45 211 observations

\- 17 colonnes

\- Classification binaire

\- Variable cible : `y`

&#x20; - `yes` : le client a souscrit à un dépôt à terme

&#x20; - `no` : le client n'a pas souscrit



\## Modèles



Deux modèles seront étudiés :



\- Régression logistique

\- Random Forest



\## Scénarios de Data Poisoning



Les principaux scénarios étudiés seront :



\- Random Label Flipping

\- Targeted Label Flipping

\- Feature Poisoning

\- Injection / duplication de données



\## Détection



Plusieurs méthodes de détection seront comparées :



\- Contrôles de qualité des données

\- Isolation Forest

\- Cohérence entre labels et prédictions

\- Cleanlab

\- Combinaison de détecteurs



\## Technologies



\- Python

\- pandas

\- scikit-learn

\- Matplotlib

\- Cleanlab



\## Installation



Créer et activer un environnement virtuel puis installer les dépendances :



```powershell

python -m venv .venv

.\\.venv\\Scripts\\Activate.ps1

pip install -r requirements.txt

