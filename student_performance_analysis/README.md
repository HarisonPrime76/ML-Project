# Student Performance Analysis (Large-Scale Analytics)

Ce projet de Machine Learning analyse et modélise des **millions de données d'étudiants** afin de prédire l'impact des activités et des comportements sur leur note globale finale (`total_score`). 

À partir d'un échantillon massif d'environ 1 million de lignes après nettoyage, l'application compare une **Régression Linéaire** et un **Random Forest** pour quantifier précisément l'influence des heures d'étude, de l'assiduité et de la participation en classe.

---

## Pipeline de Données & Prétraitement

Le script automatise l'ingestion de données massives, le dédoublonnage et la préparation des données :

* **Dédoublonnage :** `5 991` lignes en double supprimées automatiquement.
* **Dataset Final :** `994 009` lignes et `4` colonnes.
* **Variables Prédictives ($X$) :** * `weekly_self_study_hours` (Heures d'étude autonome par semaine)
    * `attendance_percentage` (Pourcentage de présence)
    * `class_participation` (Niveau de participation en classe)
* **Variable Cible ($y$) :** `total_score` (Note finale globale)
* **Découpage (Train/Test) :** 75% entraînement (`745 506` lignes) / 25% test (`248 503` lignes).

---

## Performance des Modèles

Deux approches algorithmiques ont été évaluées. La **Régression Linéaire** surpasse ici le Random Forest en termes de généralisation et d'erreur moyenne.

### 1. Régression Linéaire (Meilleur Modèle)
* **R² Score (Précision) :** `0.8529` (Explique 85.3% de la variance des notes)
* **Erreur Moyenne (MAE) :** `3.19` points

### 2. Random Forest Regressor
* **R² Score (Précision) :** `0.7737`
* **Erreur Moyenne (MAE) :** `3.95` points

---

## Analyse de l'Influence des Variables

Les deux modèles s'accordent sur la hiérarchie de l'impact des comportements : **le travail autonome est le facteur clé du succès**, suivi de l'assiduité, puis de la participation active.

### Coefficients de la Régression Linéaire
*La note de base (Interception) théorique est de **60.46**.*
* ➕ **Heures d'étude (`weekly_self_study_hours`) :** **+7.514 points** par heure hebdomadaire supplémentaire.
* ➕ **Présence (`attendance_percentage`) :** **+4.945 points** par palier d'assiduité.
* ➕ **Participation (`class_participation`) :** **+3.447 points** par niveau d'engagement.

### Importance des Features (Random Forest)
* **63.76%** de contribution : `weekly_self_study_hours`
* **25.74%** de contribution : `attendance_percentage`
* **10.50%** de contribution : `class_participation`

---

## Installation & Utilisation

### Prérequis
Assurez-vous d'avoir Python 3.8+ installé.
