# 📊 COMPTE RENDU DE PROJET  
## Prédiction du Niveau de Pauvreté des Ménages Marocains par Clustering  

**Université Hassan 1er – ENCG Settat**  
**Module :** Apprentissage Machine et Data Science  
**Filière :** Finance 2 — S8  

**Réalisé par :**  
- Nouhaila El Yazidi  
- Doaa Fathallah  
- Nouhaila En-Naidi  

**Encadrant :** Pr. Laghlimi  
**Période :** Février – Mars 2026  
**Année universitaire :** 2025–2026  

---

# 📑 SOMMAIRE
1. Introduction  
2. Analyse du jeu de données  
3. Développement technique  
4. Résultats et interprétation  
5. Recommandations et perspectives  
6. Conclusion  
7. Bibliographie  
8. Annexes  

---

# I. INTRODUCTION

## I.1 Contexte et problématique
L’analyse de la pauvreté est un enjeu majeur au Maroc. L’objectif est d’identifier les niveaux de vie pour mieux orienter les politiques publiques.

Le problème principal :  
➡️ Classer les ménages **sans données étiquetées** (apprentissage non supervisé).

---

## I.2 Objectifs du projet
- Segmenter les ménages marocains  
- Comparer 4 algorithmes de clustering :
  - K-Means  
  - DBSCAN  
  - HAC  
  - GMM  
- Proposer des recommandations pour les politiques publiques  

---

## I.3 Méthodologie
1. Prétraitement des données  
2. Réduction de dimension (PCA)  
3. Clustering  
4. Évaluation (Silhouette Score)  
5. Interprétation  

---

# II. ANALYSE DU JEU DE DONNÉES

## II.1 Source des données
Données du **Ministère de l’Économie et des Finances (2014)**  
➡️ Dataset socio-économique des ménages marocains  

---

## II.2 Variables principales

### 🔹 Services de base
- Eau potable  
- Électricité  
- Assainissement  

### 🔹 Équipement
- Électroménager  
- Véhicules  
- Internet / TV  

### 🔹 Socio-démographiques
- Taille du ménage  
- Revenus  
- Dépenses  
- Niveau d’éducation  

---

## II.3 Problèmes à traiter
- Différentes échelles de variables  
- Valeurs manquantes  
- Trop de dimensions  

---

# III. DÉVELOPPEMENT TECHNIQUE

## III.1 Prétraitement

### 🔹 Imputation
```python
df[col].fillna(df[col].median(), inplace=True)
```

### 🔹 Normalisation
```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

---

## III.2 PCA (Réduction de dimension)
- Projection en 2D  
- Conservation de la variance principale  

---

## III.3 Algorithmes de clustering

### 🔸 K-Means
- Basé sur la distance  
- Méthode du coude pour choisir K  

---

### 🔸 DBSCAN
- Basé sur la densité  
- Détecte les outliers  

---

### 🔸 HAC (Hiérarchique)
- Fusion progressive des clusters  
- Utilise un dendrogramme  

---

### 🔸 GMM
- Clustering probabiliste  
- Donne des probabilités d’appartenance  

---

## III.4 Évaluation

### 📊 Score de Silhouette
- Entre -1 et 1  
- Plus proche de 1 = meilleur  

---

# IV. RÉSULTATS ET INTERPRÉTATION

## IV.1 Profils identifiés

### 🟢 Cluster 0 : Ménages aisés
- Accès complet aux services  
- Fort équipement  
- Revenus élevés  

---

### 🟡 Cluster 1 : Classe moyenne
- Accès partiel  
- Niveau intermédiaire  

---

### 🔴 Cluster 2 : Ménages vulnérables
- Peu d’accès aux services  
- Faible revenu  
- Faible équipement  

---

## IV.2 Performance des modèles

| Algorithme | Score | Interprétation |
|----------|------|---------------|
| K-Means | 0.52 | Très bon |
| DBSCAN | 0.38 | Acceptable |
| HAC | 0.49 | Bon |
| GMM | 0.50 | Bon |

👉 Meilleurs modèles : **K-Means & GMM**

---

## IV.3 Validation
Résultats cohérents avec les données du HCP :
- 15–20% pauvres  
- 40–50% classe moyenne  
- 30–35% aisés  

---

# V. RECOMMANDATIONS

## V.1 Feature Engineering
Créer des indicateurs comme :
- Dépenses par personne  
- Taux d’équipement  

---

## V.2 Cartographie
➡️ Identifier les zones de pauvreté  

---

## V.3 Politiques publiques
- Cibler le **cluster 2**  
- Aider les ménages fragiles (GMM)  
- Étudier les cas atypiques (DBSCAN)  

---

## V.4 Big Data
Utiliser **MiniBatchKMeans** pour les grands datasets  

---

# VI. CONCLUSION
- Le clustering permet une bonne segmentation sociale  
- K-Means et GMM sont les plus efficaces  
- Application directe aux politiques publiques  

---

# 📚 BIBLIOGRAPHIE
1. HCP – Enquête ménages (2014)  
2. MEF – Indicateurs socio-économiques  
3. Scikit-learn (2011)  
4. Jain (2010)  
5. Rousseeuw (1987)  
6. Ester et al. (1996)  
7. Banque Mondiale (2020)  

---

# 📎 ANNEXES

## Annexe A : Code
Notebook Jupyter disponible  

## Annexe B : Visualisations
- PCA  
- Dendrogramme  
- Méthode du coude  

## Annexe C : Statistiques

| Variable | Cluster 0 | Cluster 1 | Cluster 2 |
|--------|----------|----------|----------|
| Eau potable | 98% | 75% | 32% |
| Véhicule | 72% | 28% | 5% |
| Dépenses | 8500 | 4200 | 1800 |
