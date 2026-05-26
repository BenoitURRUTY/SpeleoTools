# SpeleoTools — Plugin QGIS pour la Spéléologie

Plugin QGIS complet pour l'analyse, la visualisation et la cartographie de données spéléologiques. SpeleoTools intègre des outils avancés pour l'import de données Therion, l'analyse de MNT, le calcul d'épaisseur de roche et la détection de dolines.

---

## 📋 Table des matières

- [Description](#description)
- [Fonctionnalités](#fonctionnalités)
- [Prérequis](#prérequis)
- [Installation](#installation)
  - [Méthode 1 : Installation manuelle](#méthode-1--installation-manuelle)
- [Utilisation](#utilisation)
  - [Onglet 1 : Import Therion](#onglet-1--import-therion)
  - [Onglet 2 : Épaisseur de roche](#onglet-2--épaisseur-de-roche)
  - [Onglet 3 : Profils topographiques](#onglet-3--profils-topographiques)
  - [Onglet 4 : Analyse MNT et Dolines](#onglet-4--analyse-mnt-et-dolines)
- [Structure du projet](#structure-du-projet)
- [Auteur](#auteur)
- [Licence](#licence)
- [Contributions](#contributions)

---

## Description

**SpeleoTools** est un plugin QGIS conçu pour faciliter le travail des spéléologues, topographes de cavités et géologues. Il automatise de nombreuses tâches d'analyse spatiale liées à l'exploration souterraine.

### Pourquoi SpeleoTools ?

- ✅ **Import Therion facilité** : conversion automatique SHP → GPKG avec styles
- ✅ **Calculs 3D avancés** : épaisseur de roche, profils topographiques
- ✅ **Analyse MNT** : hillshade, SVF, VAT pour la prospection
- ✅ **Détection de dolines** : identification automatique des dépressions

---

## Fonctionnalités

### 🗺️ Import Therion
basé sur le script développé par Xavier Robert : https://github.com/robertxa/pyThGIS

- Import des exports Therion (Shapefile) → GeoPackage
- Conversion automatique des couches 2D et 3D :
  - **2D** : points, lignes, aires, outline
  - **3D** : stations, cheminements (shots), parois (walls)
- Application de styles prédéfinis (QML)
- Organisation en groupes hiérarchiques (2D / 3D)
- Réparation automatique des géométries invalides
- Calcul d'altitude depuis MNT pour points et stations

### 📏 Épaisseur de roche

- Calcul automatique de l'épaisseur de roche au-dessus d'une cavité
- Échantillonnage du MNT pour chaque point/sommet
- Export GeoPackage avec attributs :
  - `src_elev` : altitude de surface (MNT)
  - `cave_elev` : altitude de la cavité
  - `thickness` : épaisseur calculée (m)
  - `fid_src` : identifiant source

### 📊 Profils topographiques

Deux modes de génération de profils altimétriques. Dans les deux cas : X = distance le long du profil (m), Y = altitude extraite du MNT (m). Les couches vecteur d'entrée sont automatiquement reprojetées dans le CRS du MNT.

#### Mode 1 — Profil projeté

Coupe verticale selon un angle de projection α défini dans Therion.

**Paramètres :**
- **MNT** : raster d'élévation source
- **Couche d'emprise** : polygone ou ligne définissant la zone de la coupe — la ligne de coupe passe par le **barycentre réel** des géométries
- **Angle α** : lu automatiquement depuis un fichier `.thconfig` (`-projection [elevation XX]`) ou saisi manuellement
- **Marge emprise (%)** : agrandit la zone de découpe du MNT autour de l'emprise
- **Décalages X/Y** : shift des coordonnées pour définir une origine personnalisée

**Principe géométrique :**
- La ligne de coupe est tracée à **α + 90°** (perpendiculaire à la direction de projection Therion)
- Elle passe par le barycentre des géométries de la couche d'emprise
- Sa longueur couvre la diagonale complète de l'emprise agrandie de la marge

**Options de sortie :**
- ☑ **Sauvegarder la ligne de coupe** : exporte la ligne en GPKG et la charge dans QGIS pour vérification visuelle

**Sorties :**
```
profil_projete_aXXdeg_NomMNT.csv       # X=distance, Y=altitude
profil_projete_aXXdeg_NomMNT.gpkg      # Points profil sans CRS (coordonnées profil)
profil_projete_aXXdeg_NomMNT.png       # Graphique (si matplotlib installé)
ligne_coupe_aXXdeg_NomMNT_ligne.gpkg   # Ligne de coupe géoréférencée (optionnel)
```

#### Mode 2 — Profil développé

Profil altimétrique développé le long d'une polyligne existante.

**Paramètres :**
- **MNT** : raster d'élévation source
- **Couche polyligne** : tracé du profil (cheminement, coupe manuelle…)
- **☑ Utiliser uniquement la sélection active** : si coché, seules les entités sélectionnées dans la couche sont utilisées. Si aucune sélection active, toutes les entités sont utilisées
- **Espacement points (m)** : pas d'échantillonnage le long de la ligne
- **☑ Interpoler les valeurs NoData** : comble les trous par interpolation linéaire
- **Distance max interpolation** : limite la longueur des gaps comblés
- **Décalages X/Y** : shift des coordonnées

**Sorties :**
```
profil_dev[_sel]_NomLigne_NomMNT.csv     # X=distance cumulée, Y=altitude
profil_dev[_sel]_NomLigne_NomMNT.gpkg    # Points profil sans CRS
profil_dev[_sel]_NomLigne_NomMNT.png     # Graphique (si matplotlib installé)
```
> Le suffixe `_sel` est ajouté quand la sélection est utilisée.

**Format GPKG profil (commun aux deux modes) :**

Les GPKG de profil sont exportés **sans CRS** car leurs coordonnées sont des coordonnées de profil (X = distance, Y = altitude) et non des coordonnées géographiques. Ils sont destinés à être utilisés dans un logiciel de dessin ou de mise en page pour superposer la topographie souterraine.

Attributs : `X_dist_m`, `Y_alt_m`, `pt_index`

### 🏔️ Analyse MNT

**Produits dérivés pour la prospection spéléologique :**

- **Hillshade** : ombrage du relief (azimut et angle configurables)
- **SVF (Sky View Factor)** : facteur de visibilité du ciel
- **VAT (Variance Angular Threshold)** : variance angulaire micro-relief

### 🕳️ Détection de dolines (Onglet 4)

- Analyse automatique des dépressions fermées
- Calcul morphométrique complet :
  - Profondeur (m)
  - Surface (m²)
  - Périmètre (m)
  - Circularité (0-1)
  - Pente moyenne (°)
- Filtrage par seuils configurables
- Export vectoriel (polygones + points centraux)

---

## Prérequis

### Logiciels

- **QGIS 3.10+** ([télécharger](https://qgis.org/))
- **Python 3.6+** (inclus avec QGIS)
- Installer SAGA (https://www.sigterritoires.fr/index.php/comment-integrer-saga-a-qgis-a-partir-de-la-version-3-30/)
- RVT (https://plugins.qgis.org/plugins/rvt-qgis/)

---

## Installation

### Méthode 1 : Installation manuelle

**1. Télécharger le plugin**

Téléchargez le ZIP et décompressez-le.

**2. Activer le plugin dans QGIS**

- Ouvrez QGIS
- `Extensions` → `Installer/Gérer les extensions`
- Onglet `Installer une extension à partir d'un zip`

---

### Import Therion

Convertit les exports Therion (Shapefile) en GeoPackage avec styles.

#### Prérequis

Exporter votre topographie depuis Therion en format **Shapefile** 2D (plan) et 3D (model) dans un même et unique dossier.

#### Utilisation

**1. Chemins des données**

- **Dossier SHP Therion** : chemin vers le dossier contenant les `.shp`
- **Dossier sortie GPKG** : où sauvegarder les GeoPackage

**2. Styles (optionnel)**

Le plugin pré-remplit automatiquement les chemins vers les fichiers `.qml` du dossier `styles_therion/`.

Vous pouvez personnaliser chaque style :
- Aires 2D
- Lignes 2D
- Points 2D
- Outline 2D
- Cheminements 3D
- Stations 3D
- Parois 3D

**3. Options**

- ☑ **Réparer les géométries** : corrige automatiquement les géométries invalides
- ☑ **Calculer altitude depuis MNT** : ajoute l'altitude Z aux points/stations
- ☑ **Grouper les couches** : organise en groupes 2D/3D (nom personnalisable)

**4. Lancer l'import**

Cliquez sur **"Importer Therion"**

Le plugin :
1. ✅ Lit les fichiers SHP
2. ✅ Répare les géométries si demandé
3. ✅ Fusionne les lignes/aires par type
4. ✅ Calcule les altitudes depuis le MNT
5. ✅ Convertit en GPKG
6. ✅ Applique les styles
7. ✅ Ajoute les couches à QGIS

**Résultat :**

```
Outputs/
├── areas2dMasked.gpkg     # Aires 2D découpées sur l'outline
├── lines2dMasked.gpkg     # Lignes 2D découpées sur l'outline
├── points2dAlt.gpkg       # Points 2D avec altitude
├── outline2d.gpkg         # Contour 2D
├── shots3d.gpkg           # Cheminements 3D
├── stations3dAlt.gpkg     # Stations 3D avec altitude
└── walls3d.shp            # Parois 3D (maillage — pas de conversion GPKG)
```

Les couches sont organisées dans un groupe QGIS :

```
📁 Ma Grotte
  📁 2D
    • Points 2D
    • Lignes 2D
    • Aires 2D
    • Outline 2D
  📁 3D
    • Stations 3D
    • Cheminements 3D
    • Parois 3D
```

---

### Onglet 2 : Épaisseur de roche

Calcule l'épaisseur de roche au-dessus d'une cavité.

#### Utilisation

**1. Sélectionner les couches**

- **MNT (DEM)** : modèle numérique de terrain (surface)
- **Couche cavité** : géométrie 3D de la cavité (points, lignes)

**2. Fichier de sortie (optionnel)**

- Chemin du GeoPackage de sortie
- Si vide : couche mémoire temporaire

**3. Nom de la couche**

Nom de la couche de sortie (défaut : "Thickness")

**4. Lancer le calcul**

Cliquez sur **"Calculer épaisseur"**

**Résultat :**

Une couche de points avec :
- `src_elev` : altitude de surface (m)
- `cave_elev` : altitude de la cavité (m)
- `thickness` : épaisseur de roche (m)
- `fid_src` : ID de l'entité source

---

### Onglet 3 : Profils topographiques

Voir la section [Profils topographiques](#-profils-topographiques) dans les fonctionnalités pour le détail complet des paramètres et sorties.

**Dossier de sortie commun** : sélectionnez un dossier via le bouton 📂. Si laissé vide, les fichiers sont écrits dans le dossier temporaire système.

---

### Analyse MNT (Prospection)

Génère des produits dérivés pour faciliter la détection d'indices karstiques.

**1. Sélectionner le MNT**

**2. Dossier de sortie**

Où sauvegarder les rasters générés.

**3. Choisir les analyses**

- ☑ **Hillshade** : ombrage du relief
  - Azimut : 315° (NW)
  - Altitude : 45°

- ☑ **SVF (Sky View Factor)** : visibilité du ciel (détecte dolines)
  - Directions : 16
  - Rayon : 10 pixels

- ☑ **VAT (Variance Angular Threshold)** : variance micro-relief
  - Lissage : 5

- ☑ **Hillshade + VAT** : combinaison optimale

**4. Lancer l'analyse**

Cliquez sur **"Analyser MNT"**

**Résultat :**

```
Outputs/
├── NomMNT_hillshade.tif
├── NomMNT_svf.tif
├── NomMNT_vat.tif
└── NomMNT_multidh.tif
```

**Utilisation :**

- **SVF** : zones sombres = dépressions potentielles (dolines)
- **VAT** : variations micro-topographiques
- **Hillshade + VAT** : meilleur contraste pour prospection

### Détection de dolines

Identifie automatiquement les dépressions fermées.

**1. Sélectionner le MNT**

**2. Dossier de sortie**

**3. Paramètres de détection**

- **Profondeur min** : 2m (défaut)
- **Surface min** : 100m² (défaut)
- **Circularité min** : 0.3 (défaut, 0=linéaire, 1=cercle)

**4. Lancer la détection**

Cliquez sur **"Détecter dolines"**

**Algorithme :**

1. Remplissage des dépressions (`Fill Sinks`)
2. Soustraction : MNT rempli - MNT original = profondeur
3. Vectorisation des dépressions
4. Calcul des attributs morphométriques
5. Filtrage par seuils

**Résultat :**

- `dolines_polygones.gpkg` : contours des dolines
- `dolines_points.gpkg` : points centraux

**Attributs :**

- `profondeur` : profondeur max (m)
- `surface` : aire (m²)
- `perimetre` : périmètre (m)
- `circularite` : 4π × surface / périmètre² (0-1)
- `pente_moy` : pente moyenne (°)

**Symbologie automatique :**

- Taille proportionnelle à la profondeur
- Couleur selon la circularité

---

## Structure du projet

```
SpeleoTools/
│
├── __init__.py                  # Point d'entrée du plugin
├── speleo_tools.py              # Classe principale et interface
├── speleo_utils.py              # Fonctions utilitaires
├── speleo_dialog.ui             # Interface Qt Designer
├── install_dependencies.py      # Gestionnaire de dépendances
│
├── metadata.txt                 # Métadonnées QGIS
├── icon.png                     # Icône du plugin
│
├── styles_therion/              # Styles QML Therion
│   ├── Style_Area2D.qml
│   ├── Style_Ligne2D.qml
│   ├── Style_Point2D.qml
│   ├── Style_Outline2D.qml
│   ├── Style_Shots3D.qml
│   ├── Style_Stations3D.qml
│   └── Style_Wall3D.qml
│
└── README.md                    # Ce fichier
```

---

## Auteur

**Benoît Urruty**

- Scripts Python pour QGIS
- Interface graphique Qt
- Algorithmes d'analyse MNT et karstologie
- Intégration Therion

---

## Licence

### Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)

Ce projet est sous licence **CC BY-NC-SA 4.0**.

![CC BY-NC-SA 4.0](https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png)

#### Vous êtes autorisé à :

- **Partager** — copier, distribuer et communiquer le matériel par tous moyens et sous tous formats
- **Adapter** — remixer, transformer et créer à partir du matériel

#### Selon les conditions suivantes :

- **Attribution** — Vous devez créditer l'œuvre, intégrer un lien vers la licence et indiquer si des modifications ont été effectuées.
- **Pas d'Utilisation Commerciale** — Vous n'êtes pas autorisé à faire un usage commercial de cette œuvre.
- **Partage dans les Mêmes Conditions** — Toute œuvre dérivée doit être diffusée sous la même licence.

#### Texte complet de la licence :
[https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.fr](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.fr)

#### Résumé de la licence :
[https://creativecommons.org/licenses/by-nc-sa/4.0/deed.fr](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.fr)

---

## Source

Robert X. (2025), pyThGIS, a Python code to clean the shp Therion output. DOI:10.5281/zenodo.15078040

---

## Contributions

Les contributions sont les bienvenues !

**Types de contributions :**

- 🐛 Rapporter des bugs
- 💡 Suggérer des fonctionnalités
- 📝 Améliorer la documentation
- 🔧 Corriger des bugs
- ✨ Ajouter des fonctionnalités

---

**Bonne cartographie ! 🗺️🔦**
