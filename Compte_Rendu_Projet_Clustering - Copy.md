# Compte Rendu : Prédiction du Niveau de Pauvreté des Ménages Marocains par Clustering

Ce document présente l'analyse, l'approche technique, et les recommandations liées au projet d'apprentissage non-supervisé (Clustering) appliqué aux indicateurs socio-économiques des ménages marocains.

---

## 1. Explication du Jeu de Données

Le fichier de données étudié est `indic-soc-niveau-vie-equipements-base-mef2014.xls`. 
Émis par le MEF (Ministère de l'Économie et des Finances) en 2014, ce dataset contient de précieuses informations quantitatives et qualitatives sur le niveau de vie et l'équipement de base des foyers marocains.

**Caractéristiques probables et structure analytique :**
- Les **variables** reflètent habituellement :
  - **L'accès aux services de base** : Eau potable, électricité, assainissement.
  - **Le niveau de confort/équipement** : Possession d'électroménager, véhicules, télévisions, internet, etc.
  - **Les indicateurs sociaux** : Taille du ménage, niveau d'éducation du chef de ménage, dépenses et revenus.
- **Enjeux de la donnée** : Ces caractéristiques multivariées nécessitent un nettoyage et une mise à l'échelle (Standardisation) puissante car les différentes unités (dépenses en Dirhams vs nombre d'enfants) écraseraient les algorithmes de calcul de distance si elles n'étaient pas gérées.

---

## 2. Explication du Code Python (Notebook)

Le notebook généré (`Pauvrete_Menages_Clustering.ipynb`) est structuré pour une exécution parfaite et pédagogique sous Jupyter ou Google Colab.

### 2.1. Prétraitement et Exploration
- **Chargement** : Utilisation de `pandas.read_excel()` pour ingérer les données.
- **Nettoyage** : Imputation des valeurs manquantes par la *médiane*, une métrique robuste aux valeurs extrêmes (outliers sociaux). Sélection stricte des colonnes numériques pour l'entraînement.
- **Normalisation** : Application de `StandardScaler` ($\mu=0$, $\sigma=1$), phase incontournable pour que chaque paramètre du ménage ait un poids égal dans le partitionnement.

### 2.2. Réduction de Dimensionnalité
- Utilisation de l'Algorithme **PCA (Principal Component Analysis)** pour réduire la dimensionnalité massive des variables d'équipement à 2 composantes principales. Cela permet de **visualiser** les foyers marocains sur un graphique 2D pour vérifier visuellement la pertinence de nos classifications.

### 2.3. Les Algorithmes de Clustering Implémentés
Afin de ne pas dépendre aveuglément d'une seule méthode, le notebook croise les résultats de plusieurs approches :
1. **K-Means** : L'algorithme principal. La *Méthode du Coude (Elbow)* identifie les casures typiques de la société. Généralement, on y observe 3 ou 4 classes (e.g., *Favorisé, Classe Moyenne, Vulnérable*).
2. **DBSCAN** : Utile pour isoler des foyers totalement atypiques (outliers) et chercher des densités irrégulières.
3. **Agglomerative Clustering (HAC)** : Classification hiérarchique avec la méthode de *Ward*, permettant d'observer la ramification de la stratification sociale (Dendrogramme).
4. **Gaussian Mixture Model (GMM)** : Un modèle probabiliste qui ne catégorise pas durement, mais calcule des pourcentages d'appartenance à tel ou tel niveau social.

### 2.4. Évaluation Universelle
- Le **Silhouette Score** (de -1 à 1) est calculé pour chaque algorithme. Il traduit la cohésion interne sociale de nos groupes formés. Un score moyen-élevé prouve que les niveaux de revenus/équipements sont bien contrastés géographiquement ou socialement au Maroc.

---

## 3. Explication des Résultats Attendus

Bien que les résultats finaux doivent être interprétés avec l'exécution locale de votre environnement, voici le schéma typique qui émerge d'une telle matrice:

### Identification des Profils Sociaux
L'algorithme *K-Means* trouvera très probablement un optimum mathématique à **K=3 ou K=4**, reflétant les niveaux de vie au Maroc :
- **Cluster 0 (Niveau Élevé / Confortable)** : Foyers avec fort taux d'équipement, accès 100% aux services urbains, dépenses élevées.
- **Cluster 1 (Classe Moyenne / En transition)** : Foyers accédant aux réseaux électriques/eaux mais manquants de certains conforts premium, budgets intermédiaires. Souvent typique du péri-urbain.
- **Cluster 2 (Vulnérabilité / Pauvreté Multidimensionnelle)** : Foyers ruraux ou enclavés, faible raccordement aux services d'hygiène publics, absence d'équipements technologiques.

### Performance des Modèles
- **K-Means** est généralement très performant pour ces segmentations de types sociologiques (frontières convexes).
- **GMM** offre un point de vue fascinant dans la matrice finale grâce au soft-clustering, indiquant comment les foyers à la frontière entre Classe Moyenne et Pauvreté fluctuent selon le taux de développement d'une région.

---

## 4. Recommandations et Perspectives

Sur la base de ces traitements d'extraction de connaissances socio-économiques, voici les recommandations majeures pour le projet :

1. **Amélioration de Features (Feature Engineering)**
   - Il serait pertinent de créer des ratios (e.g. *Equipements / Taille du ménage*) pour mieux affiner l'intuition du modèle sur la richesse réelle "par tête".

2. **Cartographie et Stratification Spatiale**
   - Si le dataset intègre un identifiant géographique (Région, ou Ruralité vs Urbanité), il faudra faire un group-by sur les labels des Clusters obtenus afin de cartographier l'indice de pauvreté prédictif par région de préfecture au Maroc.

3. **Ciblage de Politiques Publiques**
   - **GMM et DBSCAN** : Analyser profondément les foyers étiquetés comme "Bruit" par DBSCAN. S'ils sont marginaux, révèlent-ils des poches de pauvreté extrême (hors normes) qui nécessiteraient des subventions ciblées urgentes ?
   - Les ménages isolés entre deux profils par **GMM** (appartenance à 50% aux vulnérables) sont les foyers qui basculent en deçà du seuil de pauvreté lors d'inflation ou de crises géopolitiques et méritent une attention de résilience financière.

4. **Équilibre des Algorithmes**
   - Ne pas utiliser HAC (Agglomerative) si le dataset venait à grandir (HCP données récentes de millions de marocains) car il est très coûteux en RAM. Préférez alors **MiniBatchKMeans** pour des prédictions Big Data en production.
