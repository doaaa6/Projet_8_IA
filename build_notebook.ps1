$cells = @(
    @{
        cell_type = "markdown"
        metadata = @{}
        source = @(
            "# Projet : Prédiction du niveau de pauvreté des ménages marocains`n",
            "Ce notebook applique diverses techniques d'apprentissage non-supervisé (Clustering) pour analyser de potentiels niveaux de pauvreté des ménages marocains, en se basant sur le fichier de données fourni.`n",
            "`n",
            "**Étapes clés :**`n",
            "1. Installation et importation des bibliothèques.`n",
            "2. Chargement et exploration du dataset.`n",
            "3. Prétraitement des données (nettoyage, gestion des valeurs manquantes, normalisation avec StandardScaler).`n",
            "4. Réduction de dimension avec PCA pour la visualisation.`n",
            "5. Clustéring des données (K-Means, DBSCAN, HAC, GMM).`n",
            "6. Évaluation avec la métrique Silhouette Score et Visualisation."
        )
    },
    @{
        cell_type = "code"
        execution_count = $null
        metadata = @{}
        outputs = @()
        source = @(
            "# 1. Installation des dépendances (utile pour Google Colab)`n",
            "!pip install pandas numpy matplotlib seaborn scikit-learn xlrd openpyxl --quiet`n",
            "print('✅ Dépendances installées')"
        )
    },
    @{
        cell_type = "code"
        execution_count = $null
        metadata = @{}
        outputs = @()
        source = @(
            "# 2. Importation des bibliothèques nécessaires`n",
            "import numpy as np`n",
            "import pandas as pd`n",
            "import matplotlib.pyplot as plt`n",
            "import matplotlib.cm as cm`n",
            "import seaborn as sns`n",
            "import warnings`n",
            "warnings.filterwarnings('ignore')`n",
            "`n",
            "from sklearn.preprocessing import StandardScaler`n",
            "from sklearn.decomposition import PCA`n",
            "from sklearn.metrics import silhouette_score, silhouette_samples`n",
            "`n",
            "print('✅ Imports OK')"
        )
    },
    @{
        cell_type = "markdown"
        metadata = @{}
        source = @(
            "---`n",
            "## 1. Chargement et Exploration du Dataset`n",
            "Assurez-vous que le fichier `indic-soc-niveau-vie-equipements-base-mef2014.xls` est bien dans le même dossier que ce Notebook, ou uploadé sur votre session Colab."
        )
    },
    @{
        cell_type = "code"
        execution_count = $null
        metadata = @{}
        outputs = @()
        source = @(
            "# Charger les données`n",
            "file_path = 'indic-soc-niveau-vie-equipements-base-mef2014.xls'`n",
            "try:`n",
            "    data = pd.read_excel(file_path)`n",
            "    print(f'✅ Dataset chargé avec succès : {data.shape[0]} lignes et {data.shape[1]} colonnes.')`n",
            "except Exception as e:`n",
            "    print('❌ Erreur lors du chargement, veuillez vérifier le chemin du fichier XLS.')`n",
            "    print(e)`n",
            "`n",
            "# Afficher les premières lignes`n",
            "data.head()"
        )
    },
    @{
        cell_type = "code"
        execution_count = $null
        metadata = @{}
        outputs = @()
        source = @(
            "# Exploration globale`n",
            "data.info()`n",
            "print('\\nRésumé statistique :')`n",
            "display(data.describe())"
        )
    },
    @{
        cell_type = "markdown"
        metadata = @{}
        source = @(
            "---`n",
            "## 2. Prétraitement des Données`n",
            "Le nettoyage dépend de la nature de vos données. Ici, nous effectuons un traitement générique :`n",
            "- Suppression des éventuelles colonnes non pertinentes ou purement textuelles n'ayant pas de sens pour le clustering (ex: un éventuel identifiant de ménage, s'il existe).`n",
            "- Imputation des valeurs manquantes par la médiane de la colonne.`n",
            "- Normalisation obligatoire avec **StandardScaler**."
        )
    },
    @{
        cell_type = "code"
        execution_count = $null
        metadata = @{}
        outputs = @()
        source = @(
            "def preprocess_data(df):`n",
            "    # 1. Sélectionner uniquement les variables numériques pour le clustering`n",
            "    df_num = df.select_dtypes(include=[np.number]).copy()`n",
            "    `n",
            "    # S'il y a une colonne 'ID', il est préférable de l'exclure du clustering.`n",
            "    # Exemple : id_cols = [col pour col dans df_num.columns if 'id' in col.lower()] ...`n",
            "    `n",
            "    # 2. Gestion des valeurs manquantes : remplacement par la médiane (robuste aux outliers)`n",
            "    col_with_na = df_num.columns[df_num.isnull().any()].tolist()`n",
            "    if len(col_with_na) > 0:`n",
            "        print(f'Valeurs manquantes trouvées dans : {col_with_na}')`n",
            "        for col in col_with_na:`n",
            "            df_num[col].fillna(df_num[col].median(), inplace=True)`n",
            "    else:`n",
            "        print('Aucune valeur manquante détectée.')`n",
            "        `n",
            "    # 3. Normalisation (StandardScaler)`n",
            "    sc = StandardScaler()`n",
            "    X_scaled = sc.fit_transform(df_num)`n",
            "    return X_scaled, df_num`n",
            "`n",
            "X, data_num = preprocess_data(data)`n",
            "print('✅ Données nettoyées et normalisées. Shape :', X.shape)"
        )
    },
    @{
        cell_type = "markdown"
        metadata = @{}
        source = @(
            "---`n",
            "## 3. Réduction de Dimension (PCA) pour la Visualisation`n",
            "Comme le jeu de données présente probablement plusieurs variables, nous réduisons les dimensions à 2 axes (composantes principales) pour permettre une visualisation en 2D des futurs clusters."
        )
    },
    @{
        cell_type = "code"
        execution_count = $null
        metadata = @{}
        outputs = @()
        source = @(
            "# Application de PCA (2 composantes)`n",
            "pca2 = PCA(n_components=2, random_state=42)`n",
            "X_pca2 = pca2.fit_transform(X)`n",
            "`n",
            "var_expl = pca2.explained_variance_ratio_ * 100`n",
            "print(f'✅ PCA appliqué. Variance expliquée par les 2 composantes : {var_expl[0]:.2f}% et {var_expl[1]:.2f}% (Total: {sum(var_expl):.2f}%)')`n",
            "`n",
            "# Visualisation brute des données (sans clusters)`n",
            "plt.figure(figsize=(7, 5))`n",
            "plt.scatter(X_pca2[:, 0], X_pca2[:, 1], alpha=0.6, s=30, color='gray')`n",
            "plt.title('Projection 2D des ménages (PCA)')`n",
            "plt.xlabel('Composante Principale 1')`n",
            "plt.ylabel('Composante Principale 2')`n",
            "plt.grid(True, linestyle='--', alpha=0.5)`n",
            "plt.show()"
        )
    },
    @{
        cell_type = "markdown"
        metadata = @{}
        source = @(
            "---`n",
            "## 4. Fonctions Utilitaires d'Évaluation et de Visualisation`n",
            "Ces fonctions permettront d'afficher les clusters et de mesurer la performance globale :`n",
            "- `eval_clustering()` : calcul du score de Silhouette (mesure de la qualité des clusters, proche de 1 est excellent).`n",
            "- `plot_clusters()` : affichage sur le plan 2D fourni par la PCA."
        )
    },
    @{
        cell_type = "code"
        execution_count = $null
        metadata = @{}
        outputs = @()
        source = @(
            "def eval_clustering(X, labels, name='Modèle'):`n",
            "    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)`n",
            "    n_noise = list(labels).count(-1)`n",
            "    `n",
            "    if n_clusters < 2:`n",
            "        print(f'{name} -> Impossible de calculer Silhouette (Clusters créés < 2)')`n",
            "        return None`n",
            "    `n",
            "    # Exclure le bruit si l'algorithme génère un cluster de bruit (-1)`n",
            "    mask = labels != -1`n",
            "    if mask.sum() > 1:`n",
            "        sil = silhouette_score(X[mask], labels[mask])`n",
            "        print(f'{name:30s} | Clusters = {n_clusters:2d} | Bruit = {n_noise:4d} | Silhouette = {sil:.4f}')`n",
            "        return sil`n",
            "    return None`n",
            "`n",
            "def plot_clusters(X_2d, labels, title):`n",
            "    plt.figure(figsize=(8, 5))`n",
            "    unique = np.unique(labels)`n",
            "    cmap = cm.get_cmap('tab10', len(unique))`n",
            "    `n",
            "    for i, lbl in enumerate(unique):`n",
            "        mask = labels == lbl`n",
            "        name = f'Cluster {lbl}' if lbl >= 0 else 'Bruit'`n",
            "        color = 'black' if lbl == -1 else cmap(i)`n",
            "        plt.scatter(X_2d[mask, 0], X_2d[mask, 1], s=30, color=color, label=name, alpha=0.7)`n",
            "        `n",
            "    plt.title(title)`n",
            "    plt.legend(fontsize=8, loc='best')`n",
            "    plt.xlabel('PC 1')`n",
            "    plt.ylabel('PC 2')`n",
            "    plt.grid(True, linestyle='--', alpha=0.5)`n",
            "    plt.tight_layout()`n",
            "    plt.show()`n",
            "`n",
            "print('Fonctions utilitaires prêtes ✅')"
        )
    },
    @{
        cell_type = "markdown"
        metadata = @{}
        source = @(
            "---`n",
            "## 5. Algorithmes de Clustering`n",
            "### 5.1 K-Means`n",
            "Déterminons le nombre de clusters optimal $K$ à l'aide de la **Méthode du Coude (Elbow)** puis entraînons K-Means."
        )
    },
    @{
        cell_type = "code"
        execution_count = $null
        metadata = @{}
        outputs = @()
        source = @(
            "from sklearn.cluster import KMeans`n",
            "`n",
            "# -- Méthode Coude --`n",
            "print('Calcul des inerties pour K allant de 1 à 10...')`n",
            "inertias = []`n",
            "K_range = range(1, 11)`n",
            "for k in K_range:`n",
            "    km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)`n",
            "    km.fit(X)`n",
            "    inertias.append(km.inertia_)`n",
            "`n",
            "plt.figure(figsize=(8, 4))`n",
            "plt.plot(list(K_range), inertias, marker='o', color='steelblue', lw=2)`n",
            "plt.xlabel('Nombre de clusters (K)')`n",
            "plt.ylabel('Inertie intra-classe (WCSS)')`n",
            "plt.title('Méthode du Coude pour K-Means')`n",
            "plt.grid(True, linestyle='--', alpha=0.5)`n",
            "plt.show()"
        )
    },
    @{
        cell_type = "code"
        execution_count = $null
        metadata = @{}
        outputs = @()
        source = @(
            "# Choix arbitraire de K=3 (selon le coude habituel, ajustez si besoin)`n",
            "k_optimal = 3`n",
            "`n",
            "km = KMeans(n_clusters=k_optimal, init='k-means++', n_init=10, random_state=42)`n",
            "labels_km = km.fit_predict(X)`n",
            "`n",
            "sil_km = eval_clustering(X, labels_km, f'K-Means (K={k_optimal})')`n",
            "plot_clusters(X_pca2, labels_km, f'Clustering K-Means K={k_optimal}')"
        )
    },
    @{
        cell_type = "markdown"
        metadata = @{}
        source = @(
            "### 5.2 DBSCAN (Density-Based Spatial Clustering of Applications with Noise)`n",
            "Idéal pour détecter des clusters de forme arbitraire et identifier les données isolées (Outliers). On utilise le graphe des *k-distances* pour trouver un epsilon $\\epsilon$ optimal."
        )
    },
    @{
        cell_type = "code"
        execution_count = $null
        metadata = @{}
        outputs = @()
        source = @(
            "from sklearn.cluster import DBSCAN`n",
            "from sklearn.neighbors import NearestNeighbors`n",
            "`n",
            "# -- K-distance graph pour trouver eps --`n",
            "min_samples_db = 5`n",
            "nbrs = NearestNeighbors(n_neighbors=min_samples_db).fit(X)`n",
            "distances, _ = nbrs.kneighbors(X)`n",
            "k_dist = np.sort(distances[:, -1])[::-1]`n",
            "`n",
            "plt.figure(figsize=(8, 4))`n",
            "plt.plot(k_dist, color='tomato')`n",
            "plt.xlabel('Points triés')`n",
            "plt.ylabel(f'{min_samples_db}-distance')`n",
            "plt.title('Graphe des distances pour l\\'aide au choix de eps (DBSCAN)')`n",
            "plt.grid(True, linestyle='--', alpha=0.5)`n",
            "plt.show()"
        )
    },
    @{
        cell_type = "code"
        execution_count = $null
        metadata = @{}
        outputs = @()
        source = @(
            "# À vue d'œil, le coude sur l'axe des Y donne eps = ~1.5 à 2.0 (à adapter selon les données réelles)`n",
            "choix_eps = 2.0`n",
            "`n",
            "db = DBSCAN(eps=choix_eps, min_samples=min_samples_db, metric='euclidean')`n",
            "labels_db = db.fit_predict(X)`n",
            "`n",
            "sil_db = eval_clustering(X, labels_db, f'DBSCAN (eps={choix_eps})')`n",
            "plot_clusters(X_pca2, labels_db, f'Clustering DBSCAN (eps={choix_eps})')"
        )
    },
    @{
        cell_type = "markdown"
        metadata = @{}
        source = @(
            "### 5.3 Clustering Hiérarchique (HAC / Agglomerative Clustering)`n",
            "Nous traçons un **dendrogramme** pour observer la structure hiérarchique, puis exécutons le modèle final."
        )
    },
    @{
        cell_type = "code"
        execution_count = $null
        metadata = @{}
        outputs = @()
        source = @(
            "from sklearn.cluster import AgglomerativeClustering`n",
            "from scipy.cluster.hierarchy import dendrogram, linkage`n",
            "`n",
            "# -- Dendrogramme --`n",
            "# Note: si le dataset est immense, le dendrogramme peut saturer la mémoire.`n",
            "try:`n",
            "    linked = linkage(X, method='ward')`n",
            "    plt.figure(figsize=(10, 5))`n",
            "    dendrogram(linked, truncate_mode='lastp', p=30, show_contracted=True)`n",
            "    plt.title('Dendrogramme (Liaison Ward)')`n",
            "    plt.xlabel('Index des clusters condensés')`n",
            "    plt.ylabel('Distance')`n",
            "    plt.show()`n",
            "except MemoryError:`n",
            "    print('Dataset trop volumineux pour afficher directement le dendrogramme.')"
        )
    },
    @{
        cell_type = "code"
        execution_count = $null
        metadata = @{}
        outputs = @()
        source = @(
            "# Application de HAC final (sur k_optimal clusters)`n",
            "hac = AgglomerativeClustering(n_clusters=k_optimal, linkage='ward')`n",
            "labels_hac = hac.fit_predict(X)`n",
            "`n",
            "sil_hac = eval_clustering(X, labels_hac, f'HAC Ward (K={k_optimal})')`n",
            "plot_clusters(X_pca2, labels_hac, f'Clustering Hiérarchique Ward (K={k_optimal})')"
        )
    },
    @{
        cell_type = "markdown"
        metadata = @{}
        source = @(
            "### 5.4 Modèles de Mélange Gaussien (Gaussian Mixture / GMM)`n",
            "Modèle probabiliste. Utile pour obtenir des probabilités d'appartenance par niveau de pauvreté. Nous nous appuyons sur le **BIC / AIC** pour aider au choix des composantes."
        )
    },
    @{
        cell_type = "code"
        execution_count = $null
        metadata = @{}
        outputs = @()
        source = @(
            "from sklearn.mixture import GaussianMixture`n",
            "`n",
            "# Tracé du BIC / AIC`n",
            "n_range = range(1, 8)`n",
            "bic_vals, aic_vals = [], []`n",
            "for n in n_range:`n",
            "    gmm_tmp = GaussianMixture(n_components=n, covariance_type='full', random_state=42)`n",
            "    gmm_tmp.fit(X)`n",
            "    bic_vals.append(gmm_tmp.bic(X))`n",
            "    aic_vals.append(gmm_tmp.aic(X))`n",
            "`n",
            "plt.figure(figsize=(8, 4))`n",
            "plt.plot(list(n_range), bic_vals, marker='o', label='BIC', color='steelblue')`n",
            "plt.plot(list(n_range), aic_vals, marker='s', label='AIC', color='tomato')`n",
            "plt.xlabel('Nombre de composants Gaussien')`n",
            "plt.ylabel('Score (min = meilleur)')`n",
            "plt.title('Sélection de K par Critères d\\'Information BIC/AIC (GMM)')`n",
            "plt.legend()`n",
            "plt.grid(True, linestyle='--', alpha=0.5)`n",
            "plt.show()"
        )
    },
    @{
        cell_type = "code"
        execution_count = $null
        metadata = @{}
        outputs = @()
        source = @(
            "gmm = GaussianMixture(n_components=k_optimal, covariance_type='full', random_state=42, n_init=3)`n",
            "labels_gmm = gmm.fit_predict(X)`n",
            "`n",
            "sil_gmm = eval_clustering(X, labels_gmm, f'GMM (K={k_optimal})')`n",
            "plot_clusters(X_pca2, labels_gmm, f'Gaussian Mixture Model K={k_optimal}')"
        )
    },
    @{
        cell_type = "markdown"
        metadata = @{}
        source = @(
            "---`n",
            "## 6. Synthèse et Comparaison des Modèles`n",
            "Tableau de bord final regroupant l'ensemble des scores Silhouette (mesure de la cohésion interne et séparation des clusters)."
        )
    },
    @{
        cell_type = "code"
        execution_count = $null
        metadata = @{}
        outputs = @()
        source = @(
            "algo_names = ['K-Means', 'DBSCAN', 'HAC', 'GMM']`n",
            "sil_scores = [sil_km, sil_db, sil_hac, sil_gmm]`n",
            "`n",
            "plt.figure(figsize=(8, 4))`n",
            "colors = sns.color_palette('viridis', len(algo_names))`n",
            "bars = plt.barh(algo_names, [s if s else 0 for s in sil_scores], color=colors)`n",
            "`n",
            "plt.xlabel('Score de Silhouette')`n",
            "plt.title('Comparaison de la Qualité des Clusters par Modèle')`n",
            "plt.xlim((0, max([s if s else 0 for s in sil_scores] + [0.1]) * 1.2))`n",
            "`n",
            "# Affichage du score sur la barre`n",
            "for bar, score in zip(bars, sil_scores):`n",
            "    if score:`n",
            "        plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, `n",
            "                 f'{score:.3f}', va='center', fontsize=10)`n",
            "    else:`n",
            "        plt.text(0.01, bar.get_y() + bar.get_height()/2, `n",
            "                 'N/A', va='center', fontsize=10, color='red')`n",
            "`n",
            "plt.tight_layout()`n",
            "plt.show()"
        )
    },
    @{
        cell_type = "markdown"
        metadata = @{}
        source = @(
            "---`n",
            "### Analyse du Profil de Pauvreté`n",
            "Une fois le modèle idéal choisi (souvent K-Means ou GMM), nous rattachons les labels trouvés aux données d'origine de `indic-soc-niveau-vie-equipements-base-mef2014.xls`. Les différences par cluster sur des colonnes critiques (équipement de base, indicateur social) définiront les différents niveaux (ex: *Favorisé, Moyen, Vulnérable*)."
        )
    },
    @{
        cell_type = "code"
        execution_count = $null
        metadata = @{}
        outputs = @()
        source = @(
            "# Exemple : Injectons le résultat de K-Means dans notre Dataset`n",
            "data['Cluster_ID'] = labels_km`n",
            "`n",
            "# Statistiques descriptives sur un cluster`n",
            "print('Taille par Cluster :')`n",
            "print(data['Cluster_ID'].value_counts())`n",
            "`n",
            "# data.groupby('Cluster_ID').mean()  # Décommentez pour analyser la moyenne de chaque variable par cluster"
        )
    }
)

$notebook = @{
    cells = $cells
    metadata = @{
        kernelspec = @{
            display_name = "Python 3"
            language = "python"
            name = "python3"
        }
        language_info = @{
            codemirror_mode = @{
                name = "ipython"
                version = 3
            }
            file_extension = ".py"
            mimetype = "text/x-python"
            name = "python"
            nbconvert_exporter = "python"
            pygments_lexer = "ipython3"
            version = "3.8.0"
        }
    }
    nbformat = 4
    nbformat_minor = 4
}

$notebook | ConvertTo-Json -Depth 10 | Set-Content -Encoding UTF8 -Path "C:\Users\DOAA\Downloads\Projet_s8\Pauvrete_Menages_Clustering.ipynb"
Write-Host "Notebook successfully created!"
