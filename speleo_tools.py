# -*- coding: utf-8 -*-
"""
SpeleoTools Plugin for QGIS 3
Auteur : Urruty Benoit
Description : Interface complète à 4 onglets pour outils spéléo.
"""
import csv
import math
import tempfile
import matplotlib.pyplot as plt
import os
import processing
from qgis.PyQt import QtCore
from qgis.PyQt import QtWidgets, uic, QtCore
from qgis.PyQt.QtCore import Qt
from qgis.core import (
    QgsProject, QgsRasterLayer, QgsVectorLayer, QgsPoint, QgsPointXY,
    QgsFeature, QgsFields, QgsField, QgsWkbTypes, QgsGeometry,
    QgsFeatureSink, QgsDistanceArea, QgsCoordinateTransformContext,
    QgsFeatureRequest, QgsMessageLog, Qgis
)
from PyQt5.QtCore import QVariant

from .speleo_utils import *

# Charger l'interface .ui
FORM_CLASS, _ = uic.loadUiType(os.path.join(os.path.dirname(__file__), 'speleo_dialog.ui'))

class SpeleoToolsDialog(QtWidgets.QDialog, FORM_CLASS):
    def __init__(self, parent=None):
        super(SpeleoToolsDialog, self).__init__(parent)
        self.setupUi(self)

        # Remplir les combobox avec les couches existantes
        self.populate_layers()

        # Connexions signaux → slots
        # Onglet 1
        self.btnImport.clicked.connect(self.import_data)
        self.btnApplyStyle.clicked.connect(self.apply_style)
        # Onglet 2
        self.btnBrowse.clicked.connect(self.browse_output)
        self.btnRunThickness.clicked.connect(self.run_thickness)
        # Onglet 3
        self.btnGenerateProfile.clicked.connect(self.generate_profile_with_interpolation_and_export)
        # Onglet 4
        self.btnRunProspect.clicked.connect(self.run_mnt_analysis)
        self.btnBrowseOutput.clicked.connect(self.selectOutputDir) 
        # Onglet 5
        self.btnBrowseDolines.clicked.connect(self.selectOutputDirDoline)
        self.btnRunDolines.clicked.connect(self.main_find_dolines)

        QgsProject.instance().layerWasAdded.connect(self.populate_layers)
        QgsProject.instance().layersWillBeRemoved.connect(self.populate_layers)


        # Message d’état initial
        self.textLog.setPlainText("SpeleoTools prêt à l'emploi.\n")

    # ======================================================================
    # --- MÉTHODES GÉNÉRALES ---

    def populate_layers(self):
        """Met à jour les listes de couches disponibles dans QGIS"""
        self.comboDEM.clear()
        self.comboDEM2.clear()
        self.comboCave.clear()
        self.comboProfileLayer.clear()
        self.comboProspectDEM.clear()
        self.comboDolinesDEM.clear()

        for layer in QgsProject.instance().mapLayers().values():
            if isinstance(layer, QgsRasterLayer):
                self.comboDEM.addItem(layer.name())
                self.comboDEM2.addItem(layer.name())
                self.comboProspectDEM.addItem(layer.name())
                self.comboDolinesDEM.addItem(layer.name())
            elif isinstance(layer, QgsVectorLayer):
                self.comboCave.addItem(layer.name())
                self.comboProfileLayer.addItem(layer.name())

    def get_layer_by_name(self, name):
        """Retourne une couche par son nom"""
        layers = QgsProject.instance().mapLayersByName(name)
        return layers[0] if layers else None

    def log(self, message):
        """Ajoute un message dans le journal"""
        self.textLog.append(message)
        print("[SpeleoTools] " + message)

    # ======================================================================
    # --- ONGLET 1 : IMPORT & STYLES ---


    def apply_style(self):
        """Applique un style symbolique de base"""
        style_name = self.comboStyle.currentText()
        self.log(f"Application du style '{style_name}' (non encore implémenté).")
        QtWidgets.QMessageBox.information(self, "Style", f"Le style '{style_name}' sera appliqué prochainement.")

    # ======================================================================
    # --- ONGLET 2 : ÉPAISSEUR DE ROCHE ---

    #a faire conversion dans le crs des données vecteurs
    def browse_output(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Choisir le fichier de sortie", "", "GeoPackage (*.gpkg)")
        if path:
            if not path.endswith(".gpkg"):
                path += ".gpkg"
            self.lineOutput.setText(path)

    def run_thickness(self):
        dem_name = self.comboDEM.currentText()
        cave_name = self.comboCave.currentText()
        out_path = self.lineOutput.text().strip()
        layername = self.LayerName.text()

        if not dem_name or not cave_name:
            QtWidgets.QMessageBox.warning(self, "Erreur", "Sélectionne un DEM et une couche de cavité.")
            return

        raster = self.get_layer_by_name(dem_name)
        vec = self.get_layer_by_name(cave_name)
        print(vec)
        if raster is None or vec is None:
            QtWidgets.QMessageBox.warning(self, "Erreur", "Impossible de trouver les couches sélectionnées.")
            return

        try:
            self.log(f"Début calcul d'épaisseur entre '{cave_name}' et '{dem_name}'...")
            self.progressThickness.setValue(10)
            QtWidgets.QApplication.processEvents()

            out = out_path if out_path else None
            print(out, out_path)
            mem = compute_thickness(raster, vec, out_path=out, layer_name=layername)

            self.progressThickness.setValue(80)
            QtWidgets.QApplication.processEvents()
            self.progressThickness.setValue(100)

            if out:
                QtWidgets.QMessageBox.information(self, "Succès", f"Épaisseur calculée et sauvegardée dans :\n{out}")
            else:
                QtWidgets.QMessageBox.information(self, "Succès", "Épaisseur calculée (couche ajoutée au projet).")
        except Exception as e:
            self.log(f"Erreur run_thickness : {e}")
            QtWidgets.QMessageBox.critical(self, "Erreur", f"Une erreur est survenue :\n{e}")
        finally:
            self.progressThickness.setValue(0)
            
    # ======================================================================
    # --- ONGLET 3 : PROFILS & 3D ---
    # def generate_profile_with_interpolation_and_export(self):
    #     """Génère un profil développé + MNT interpolé, trace le graphique et exporte en CSV + PNG.
    #     Robuste aux points 2D (QgsPointXY) — si pas de Z on utilise le DEM en fallback."""
    #     dem_name = self.comboDEM2.currentText()
    #     profile_layer_name = self.comboProfileLayer.currentText()

    #     if not dem_name or not profile_layer_name:
    #         QtWidgets.QMessageBox.warning(self, "Erreur", "Veuillez sélectionner un MNT et une couche de profil.")
    #         return

    #     dem_layer = self.get_layer_by_name(dem_name)
    #     profile_layer = self.get_layer_by_name(profile_layer_name)
    #     if dem_layer is None or profile_layer is None:
    #         QtWidgets.QMessageBox.warning(self, "Erreur", "Impossible de trouver les couches sélectionnées.")
    #         return
        
    #     spacing = self.doubleSpinBoxSpacing.value()
    #     interp = self.checkBoxInterpolate.isChecked()
    #     max_gap_distance = self.doubleSpinBoxMaxGap.value() if self.checkBoxMaxGap.isChecked() else None

    #     output_dir = QtWidgets.QFileDialog.getExistingDirectory(self, "Choisir un dossier de sortie")
    #     if not output_dir:
    #         return

    #     try:
    #         self.log(f"Début de la génération du profil pour '{profile_layer_name}' avec le MNT '{dem_name}'...")
    #         profile_3d_layer = create_profile_from_line(
    #             dem_layer, profile_layer,
    #             spacing=spacing,
    #             interp=interp,
    #             max_gap_distance=max_gap_distance,
    #             add_to_project=True
    #         )

    #         if profile_3d_layer is None or profile_3d_layer.featureCount() == 0:
    #             QtWidgets.QMessageBox.warning(self, "Erreur", "Aucun profil n'a pu être généré.")
    #             return

    #         # Préparer transform pour distance / sampling si nécessaire
    #         proj = QgsProject.instance()
    #         dem_crs = dem_layer.crs()
    #         prof_crs = profile_3d_layer.crs()
    #         need_transform_to_dem = (prof_crs != dem_crs)
    #         if need_transform_to_dem:
    #             xform_to_dem = QgsCoordinateTransform(prof_crs, dem_crs, proj)
    #         else:
    #             xform_to_dem = None

    #         distances = []
    #         elevations = []
    #         cum_dist = 0.0
    #         prev_xy = None

    #         for feature in profile_3d_layer.getFeatures():
    #             geom = feature.geometry()
    #             if geom is None or geom.isEmpty():
    #                 continue

    #             # récupérer la polyline (gère multipart en itérant)
    #             polylines = []
    #             if geom.isMultipart():
    #                 polylines = geom.asMultiPolyline()
    #             else:
    #                 polylines = [geom.asPolyline()]

    #             for poly in polylines:
    #                 if not poly:
    #                     continue

    #                 for i, pt in enumerate(poly):
    #                     # pt peut être QgsPoint, QgsPointXY, QgsPointZ etc.
    #                     # extraire x,y
    #                     try:
    #                         x = pt.x()
    #                         y = pt.y()
    #                     except Exception:
    #                         # sécurité: si pt est un tuple
    #                         x, y = float(pt[0]), float(pt[1])

    #                     # essayer d'obtenir z s'il existe
    #                     z = None
    #                     # quelques objets QGIS implémentent z() ; tester proprement
    #                     try:
    #                         z_candidate = pt.z()  # lèvera AttributeError pour QgsPointXY
    #                         # vérifier NaN
    #                         if z_candidate is not None and not (isinstance(z_candidate, float) and math.isnan(z_candidate)):
    #                             z = float(z_candidate)
    #                     except Exception:
    #                         z = None

    #                     # si pas de z, fallback : échantillonner le DEM au XY transformé si besoin
    #                     if z is None:
    #                         # construire QgsPointXY dans le CRS du profil puis transformer si besoin
    #                         pt_xy_prof = QgsPointXY(x, y)
    #                         if need_transform_to_dem:
    #                             try:
    #                                 pt_dem = xform_to_dem.transform(pt_xy_prof)
    #                                 sample_xy = QgsPointXY(pt_dem.x(), pt_dem.y())
    #                             except Exception:
    #                                 sample_xy = QgsPointXY(pt_xy_prof.x(), pt_xy_prof.y())
    #                                 # en cas d'erreur de transform, on tente sans transformer
    #                         else:
    #                             sample_xy = pt_xy_prof
    #                         # utiliser ta fonction sample_dem_at_point (attend point en CRS du DEM)
    #                         try:
    #                             z = sample_dem_at_point(dem_layer, sample_xy)
    #                         except Exception as e:
    #                             z = None
    #                         print(z)
    #                     # calcul de la distance cumulée (utilise XY dans le CRS de la couche profil)
    #                     cur_xy = (x, y)
    #                     if prev_xy is None:
    #                         seg_dist = 0.0
    #                     else:
    #                         dx = cur_xy[0] - prev_xy[0]
    #                         dy = cur_xy[1] - prev_xy[1]
    #                         seg_dist = math.hypot(dx, dy)
    #                     cum_dist += seg_dist
    #                     distances.append(cum_dist)
    #                     elevations.append(z if z is not None else float('nan'))  # garder position même si NaN

    #                     prev_xy = cur_xy

    #         if not distances or not elevations:
    #             QtWidgets.QMessageBox.warning(self, "Erreur", "Aucune donnée de profil extraites.")
    #             return

    #         # Nettoyage simple : si tu veux ignorer NaN pour le tracé, on crée des listes filtrées
    #         plot_dist = []
    #         plot_elev = []
    #         for d, e in zip(distances, elevations):
    #             if e is not None and not (isinstance(e, float) and math.isnan(e)):
    #                 plot_dist.append(d)
    #                 plot_elev.append(e)

    #         if not plot_dist:
    #             QtWidgets.QMessageBox.warning(self, "Erreur", "Aucune valeur d'altitude valide à tracer.")
    #             return

    #         # Tracer
    #         plt.figure(figsize=(10, 6))
    #         plt.plot(plot_dist, plot_elev, '-', label="Profil développé")
    #         plt.title(f"Profil développé - {profile_layer_name}")
    #         plt.xlabel("Distance (m)")
    #         plt.ylabel("Altitude (m)")
    #         plt.grid(True)
    #         plt.legend()

    #         png_path = os.path.join(output_dir, f"profil_{profile_layer_name}.png")
    #         plt.savefig(png_path, dpi=300, bbox_inches='tight')
    #         plt.close()

    #         # Export CSV : écrire toutes les lignes (on peut choisir d'écrire NaN)
    #         csv_path = os.path.join(output_dir, f"profil_{profile_layer_name}.csv")
    #         with open(csv_path, 'w', newline='') as csvfile:
    #             writer = csv.writer(csvfile)
    #             writer.writerow(["Distance (m)", "Altitude (m)"])
    #             for dist, elev in zip(distances, elevations):
    #                 if elev is None or (isinstance(elev, float) and math.isnan(elev)):
    #                     writer.writerow([dist, ""])
    #                 else:
    #                     writer.writerow([dist, elev])

    #         self.log(f"Profil généré avec succès !\n- Graphique exporté : {png_path}\n- Données exportées : {csv_path}")
    #         QtWidgets.QMessageBox.information(self, "Succès", f"Profil généré et exporté dans :\n{output_dir}")

    #     except Exception as e:
    #         self.log(f"Erreur lors de la génération du profil : {e}")
    #         QtWidgets.QMessageBox.critical(self, "Erreur", f"Une erreur est survenue :\n{e}")

    # ======================================================================
    # --- ONGLET 4 : Analyse MNT ---

    
    def selectOutputDir(self):
        """Slot pour choisir le dossier de sortie via un dialog."""
        start = self.editOutputDir.text().strip() or os.path.expanduser("~")
        dirpath = QtWidgets.QFileDialog.getExistingDirectory(self, "Choisir dossier de sortie", start)
        if dirpath:
            self.editOutputDir.setText(dirpath)

    # connexion (à appeler dans ton init/setup UI)
    # self.btnBrowseOutput.clicked.connect(self.selectOutputDir)

    def _safe_name(self, name):
        """Génère un nom de fichier sûr à partir du nom de la couche."""
        # enlever extension, remplacer espaces et caractères problématiques
        base = os.path.splitext(name)[0]
        safe = "".join([c if c.isalnum() or c in ('_', '-') else '_' for c in base])
        return safe



    # Assure-toi d'avoir ces imports en haut du fichier plugin

    def run_mnt_analysis(self):
        dem_name = self.comboProspectDEM.currentText()
        if not dem_name:
            QtWidgets.QMessageBox.warning(self, "Erreur", "Sélectionne un MNT.")
            return

        layers = QgsProject.instance().mapLayersByName(dem_name)
        if not layers:
            QtWidgets.QMessageBox.warning(self, "Erreur", f"Couche {dem_name} introuvable.")
            return
        dem_layer = layers[0]

        save_temp = bool(self.AddlayerMNT.isChecked())

        # options
        do_hillshade = self.chkHillshade.isChecked()
        do_multidh = self.chkMultiHillshade.isChecked()
        do_slope = self.chkSlope.isChecked()
        do_vat = self.chkVAT.isChecked()

        zfactor = float(self.spinZFactor.value())
        vat_window = int(self.spinVATWindow.value())

        # dossier de sortie
        out_dir = self.editOutputDir.text().strip()
        if not out_dir:
            out_dir = tempfile.gettempdir()
            # informer
            self.textLog.append(f"[INFO] Aucun dossier choisi — utilisation du dossier temporaire : {out_dir}")
        try:
            os.makedirs(out_dir, exist_ok=True)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Erreur dossier", f"Impossible de créer/écrire dans le dossier {out_dir} : {e}")
            return

        # helper pour construire chemins
        base_safe = self._safe_name(dem_name)
        def outpath(suffix):
            fname = f"{base_safe}_{suffix}.tif"
            return os.path.join(out_dir, fname)

        # helper pour ajouter un raster au projet et logger
        def add_raster_to_project(path, layer_title=None):
            if not path:
                log(f"[WARN] Chemin de sortie vide, impossible d'ajouter la couche : {layer_title}")
                return None
            # si c'est une sortie "TEMPORARY_OUTPUT" ou similaire, on essaye de l'ajouter quand même
            try:
                # si le path existe sur disque, on l'ouvre
                if os.path.exists(path) or path.startswith('/vsimem/'):
                    title = layer_title or os.path.basename(path)
                    rl = QgsRasterLayer(path, title)
                    if not rl.isValid():
                        log(f"[WARN] La couche raster produite est invalide et n'a pas été ajoutée : {path}")
                        return None
                    QgsProject.instance().addMapLayer(rl)
                    log(f"[INFO] Couche ajoutée au projet : {title}")
                    return rl
                else:
                    log(f"[WARN] Fichier de sortie introuvable, impossible d'ajouter : {path}")
                    return None
            except Exception as e:
                log(f"[ERROR] Erreur lors de l'ajout de la couche {path} : {e}")
                return None

        # log helper
        self.textLog.clear()
        def log(msg):
            self.textLog.append(msg)

        log(f"Analyse MNT '{dem_name}' → dossier : {out_dir}")
        total = sum([do_hillshade, do_multidh, do_slope, do_vat]) or 1
        step = 0

        try:
            # Hillshade simple
            if do_hillshade:
                step += 1
                log(" - Hillshade simple...")
                out = hillshade(dem_layer,
                                out_path=outpath('hillshade'),
                                zfactor=zfactor,
                                azimuth=315.0,
                                altitude=45.0,
                                feedback=None)
                log(f"   -> {out}")
                # ajoute la couche au projet
                if save_temp:
                    add_raster_to_project(out, f"{base_safe}_hillshade")
                self.progressProspect.setValue(int(step/total*100))

            # Multidirectional
            if do_multidh:
                step += 1
                log(" - Hillshade multidirectionnel...")
                out = multidirectional_hillshade(dem_layer,
                                                out_path=outpath('multidh'),
                                                feedback=None)
                log(f"   -> {out}")
                if save_temp:
                    add_raster_to_project(out, f"{base_safe}_multidh")
                self.progressProspect.setValue(int(step/total*100))

            # Slope
            if do_slope:
                step += 1
                log(" - Calcul de la pente...")
                out = slope(dem_layer,
                            out_path=outpath('slope'),
                            zfactor=zfactor,
                            feedback=None)
                log(f"   -> {out}")
                if save_temp:
                    add_raster_to_project(out, f"{base_safe}_slope")
                self.progressProspect.setValue(int(step/total*100))

            # VAT
            if do_vat:
                step += 1
                log(" - VAT...")
                out = VAT(dem_layer,
                        out_path=outpath('vat'),
                        type_terrain=0,
                        feedback=None)
                log(f"   -> {out}")
                if save_temp:
                    add_raster_to_project(out, f"{base_safe}_vat")
                self.progressProspect.setValue(int(step/total*100))

            self.progressProspect.setValue(100)
            QtWidgets.QMessageBox.information(self, "Prospection", f"Analyse MNT terminée.\nFichiers écrits dans : {out_dir}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Erreur", str(e))
            self.progressProspect.setValue(0)

     # ======================================================================
    # --- ONGLET 5 : Identification doline ---

    ### Methode simple
    # etape :
    # 1. MNT fill sink processing.run("sagang:fillsinksxxlwangliu", {'ELEV':,'FILLED':,'MINSLOPE':0.1})
    # 2. Sink=MNT_fill-MNT 
    # 3. Vectoriser Sink > 1 m processing.run("native:pixelstopoints", {'INPUT_RASTER':'C:/Users/burru/Downloads/test_pulgin/Sink_1m.tif','RASTER_BAND':1,'FIELD_NAME':'VALUE','OUTPUT':'TEMPORARY_OUTPUT'})
    # 4. suprrimer les pixel seuls (OPTION)
    # 5. partitionnement DBSCAN processing.run("native:dbscanclustering", {'INPUT':'memory://Point?crs=EPSG:2154&field=VALUE:double(20,8)&uid={35a59a33-7425-46c0-b321-5adf96d17bdf}','MIN_SIZE':5,'EPS':1,'DBSCAN*':False,'FIELD_NAME':'CLUSTER_ID','SIZE_FIELD_NAME':'CLUSTER_SIZE','OUTPUT':'TEMPORARY_OUTPUT'})
    # 6. geometrie d'emprise minimale (retirer la plus grande) processing.run("qgis:minimumboundinggeometry", {'INPUT':'memory://Point?crs=EPSG:2154&field=VALUE:double(20,8)&field=CLUSTER_ID:integer(0,0)&field=CLUSTER_SIZE:integer(0,0)&uid={10d8e98c-a84f-411c-8aff-4e98d1276945}','FIELD':'CLUSTER_ID','TYPE':3,'OUTPUT':'TEMPORARY_OUTPUT'})
    # 7. Statistique dans le polygone
    # 8. centroide processing.run("native:centroids", {'INPUT':'memory://Polygon?crs=EPSG:2154&field=id:integer(20,0)&field=CLUSTER_ID:integer(0,0)&field=area:double(20,6)&field=perimeter:double(20,6)&uid={70a1f0c1-ff95-45ea-be57-a10b78f03e56}','ALL_PARTS':False,'OUTPUT':'TEMPORARY_OUTPUT'})

    def selectOutputDirDoline(self):
        """Slot pour choisir le dossier de sortie via un dialog."""
        start = self.lineOutFolderDolines.text().strip() or os.path.expanduser("~")
        dirpath = QtWidgets.QFileDialog.getExistingDirectory(self, "Choisir dossier de sortie", start)
        if dirpath:
            self.lineOutFolderDolines.setText(dirpath)



    def main_find_dolines(self):
        """Fonction principale à appeler depuis l'onglet du plugin.
        dem_layer : QgsRasterLayer ou chemin
        out_folder : dossier pour écrire les sorties (optionnel). Si None, tout en mémoire.
        params : dict pour surcharger les paramètres par défaut
        Retourne un dictionnaire des sorties principales : {'polygons':..., 'centroids':...}
        """
        import os
        from pathlib import Path

        # --- dossier de sortie (None -> tout en mémoire) ---
        out_folder_text = self.lineOutFolderDolines.text().strip()
        out_folder = Path(out_folder_text) if out_folder_text else None
        if out_folder is not None:
            out_folder.mkdir(parents=True, exist_ok=True)

        # --- sauvegarde temporaire ? (bool) ---
        save_temp = bool(self.checkBox_savetemp.isChecked())

        # --- paramètres : on prend self.params si disponible, sinon dict vide ---
        params = getattr(self, 'params', {}) or {}
        # valeurs par défaut
        minslope = params.get('minslope', 0.1)
        sink_threshold = params.get('sink_threshold', 1.0)
        area_threshold = params.get('area_threshold', 1.0)
        dbscan_eps = params.get('dbscan_eps', 5.0)
        dbscan_min = params.get('dbscan_min', 5)

        outputs = {}

        # --- récupère le MNT choisi ---
        dem_name = self.comboDolinesDEM.currentText()
        if not dem_name:
            QtWidgets.QMessageBox.warning(self, "Erreur", "Sélectionne un MNT.")
            return outputs

        layers = QgsProject.instance().mapLayersByName(dem_name)
        if not layers:
            QtWidgets.QMessageBox.warning(self, "Erreur", f"Couche {dem_name} introuvable.")
            return outputs
        dem_layer = layers[0]
        step = 0
        try:
            # --- 1. remplissage des sinks ---
            step += 1
            if save_temp and out_folder:
                path_filled = str(out_folder / 'filled.tif')
                if os.path.exists(path_filled):
                    os.remove(path_filled)
            else:
                path_filled = 'TEMPORARY_OUTPUT'
            filled = fill_sinks(dem_layer, minslope=minslope, filled_output=path_filled)
            QgsMessageLog.logMessage(f"[dolines] filled: {filled}", "Speleo", Qgis.Info)
            
            self.progressDolines.setValue(int(step/8*100))

            step += 1
            # --- 2. raster des sinks ---
            if save_temp and out_folder:
                path_sink = str(out_folder / 'sink.tif')
                if os.path.exists(path_sink):
                    os.remove(path_sink)
            else:
                path_sink = 'TEMPORARY_OUTPUT'
            sink_raster = compute_sink_raster(dem_layer, filled, threshold=sink_threshold, sink_output=path_sink)
            QgsMessageLog.logMessage(f"[dolines] sink_raster: {sink_raster}", "Speleo", Qgis.Info)

            self.progressDolines.setValue(int(step/8*100))

            step += 1

            # --- 3. vectorisation des sinks (points) ---
            if save_temp and out_folder:
                path_point = str(out_folder / 'points.shp')
                if os.path.exists(path_point):
                    os.remove(path_point)
            else:
                path_point = 'memory:'
            points = vectorize_sinks(sink_raster, vector_output=path_point)
            outputs['points'] = points

            self.progressDolines.setValue(int(step/8*100))

            step += 2

            # --- 4/5. clustering DBSCAN ---
            if save_temp and out_folder:
                path_clustered = str(out_folder / 'clustered.shp')
                if os.path.exists(path_clustered):
                    os.remove(path_clustered)
            else:
                path_clustered = 'memory:'
            clustered = dbscan_partition(points, eps=dbscan_eps, min_size=dbscan_min, vector_output=path_clustered)
            outputs['clustered'] = clustered

            self.progressDolines.setValue(int(step/8*100))

            step += 1

            # --- 6. minimum bounding geometry sur clusters ---
            if save_temp and out_folder:
                path_mbg = str(out_folder / 'mbg.shp')
                if os.path.exists(path_mbg):
                    os.remove(path_mbg)
            else:
                path_mbg = 'memory:'
            mbg = minimum_bounding_geometry(clustered, field='CLUSTER_ID', keep_largest=False, vector_output=path_mbg)
            outputs['mbg_polygons'] = mbg

            self.progressDolines.setValue(int(step/8*100))

            step += 1

            # --- 7. statistiques zonales ---
            if save_temp and out_folder:
                path_stats = str(out_folder / 'stats.shp')
                if os.path.exists(path_stats):
                    os.remove(path_stats)
            else:
                path_stats = 'memory:'
            stats = zonal_statistics(mbg, sink_raster, stats_prefix='Profondeur_', vector_output=path_stats)
            outputs['stats_polygons'] = stats
            QgsMessageLog.logMessage(f"[dolines] stats: {stats}", "Speleo", Qgis.Info)


            self.progressDolines.setValue(int(step/8*100))

            step += 1

            # si on veut afficher dans QGIS (en mémoire), on ajoute toujours la couche si c'est une QgsVectorLayer
            if not out_folder:
                try:
                    if isinstance(stats, QgsVectorLayer):
                        QgsProject.instance().addMapLayer(stats)
                except Exception:
                    # certains wrappers retournent un path/objet, on ignore si on ne peut pas ajouter
                    pass

            # --- 8. extraction des centroïdes avec stats ---
            if save_temp and out_folder:
                path_centroid = str(out_folder / 'centroids.shp')
                if os.path.exists(path_centroid):
                    os.remove(path_centroid)
            else:
                path_centroid = 'memory:'
            final_centroids = extract_centroids_with_stats(stats, vector_output=path_centroid)
            outputs['centroids'] = final_centroids
            QgsMessageLog.logMessage(f"[dolines] centroids: {final_centroids}", "Speleo", Qgis.Info)
            
            self.progressDolines.setValue(int(step/8*100))

            if not out_folder:
                try:
                    if isinstance(final_centroids, QgsVectorLayer):
                        QgsProject.instance().addMapLayer(final_centroids)
                except Exception:
                    pass

            # --- écriture GPKG si on a un dossier de sortie et qu'on ne garde pas les fichiers temporaires ---
            if out_folder and not save_temp:
                gpkg_path = str(out_folder / "dolines.gpkg")

                # Supprime le GPKG existant pour partir propre (évite les conflits de layername)
                if os.path.exists(gpkg_path):
                    os.remove(gpkg_path)

                # Écrire les polygones (stats) si présents
                if 'stats_polygons' in outputs and outputs['stats_polygons'] is not None:
                    options = QgsVectorFileWriter.SaveVectorOptions()
                    options.driverName = "GPKG"
                    options.layerName = "dolines_polygons"
                    res = QgsVectorFileWriter.writeAsVectorFormatV3(
                        outputs['stats_polygons'],
                        gpkg_path,
                        QgsProject.instance().transformContext(),
                        options
                    )
                    QgsMessageLog.logMessage(f"[dolines] write polygons result: {res}", "Speleo", Qgis.Info)

                # Écrire les centroïdes si présents (ajoute comme seconde couche dans le GPKG)
                if 'centroids' in outputs and outputs['centroids'] is not None:
                    options = QgsVectorFileWriter.SaveVectorOptions()
                    options.driverName = "GPKG"
                    options.layerName = "dolines_centroids"
                    # si gpkg existe, CreateOrOverwriteLayer permet d'ajouter une nouvelle couche
                    options.actionOnExistingFile = QgsVectorFileWriter.CreateOrOverwriteLayer
                    res = QgsVectorFileWriter.writeAsVectorFormatV3(
                        outputs['centroids'],
                        gpkg_path,
                        QgsProject.instance().transformContext(),
                        options
                    )
                    QgsMessageLog.logMessage(f"[dolines] write centroids result: {res}", "Speleo", Qgis.Info)
                    # --- ouvrir automatiquement le GPKG dans QGIS ---
                gpkg_uri = f"{gpkg_path}|layername=dolines_polygons"
                gpkg_layer_poly = QgsVectorLayer(gpkg_uri, "Dolines - Polygones", "ogr")
                if gpkg_layer_poly.isValid():
                    QgsProject.instance().addMapLayer(gpkg_layer_poly)

                gpkg_uri_centroids = f"{gpkg_path}|layername=dolines_centroids"
                gpkg_layer_centroids = QgsVectorLayer(gpkg_uri_centroids, "Dolines - Centroides", "ogr")
                if gpkg_layer_centroids.isValid():
                    QgsProject.instance().addMapLayer(gpkg_layer_centroids)

            QgsMessageLog.logMessage("[dolines] Traitement terminé.", "Speleo", level=Qgis.Info)
            return outputs

        except Exception as e:
            QgsMessageLog.logMessage(f"[dolines] Erreur pendant le traitement: {e}", "Speleo", level=Qgis.Critical)
            QtWidgets.QMessageBox.critical(self, "Erreur", f"Traitement interrompu : {e}")
            return outputs



# -------------------------------------------------
# Classe Plugin QGIS standard
# -------------------------------------------------
class SpeleoTools:
    def __init__(self, iface):
        self.iface = iface
        self.plugin_dir = os.path.dirname(__file__)
        self.dialog = None
        self.action = None

    def initGui(self):
        """Ajoute le plugin dans le menu et toolbar QGIS"""
        from qgis.PyQt.QtWidgets import QAction
        from qgis.PyQt.QtGui import QIcon

        self.action = QAction(QIcon(), "SpeleoTools", self.iface.mainWindow())
        self.action.triggered.connect(self.run)

        # Ajouter au menu "Extensions"
        self.iface.addPluginToMenu("&SpeleoTools", self.action)

        # Ajouter à la toolbar
        self.iface.addToolBarIcon(self.action)

    def unload(self):
        """Supprime le plugin de QGIS"""
        if self.action:
            self.iface.removePluginMenu("&SpeleoTools", self.action)
            self.iface.removeToolBarIcon(self.action)
            self.action = None

    def run(self):
        """Ouvre la fenêtre du plugin"""
        if not self.dialog:
            self.dialog = SpeleoToolsDialog(self.iface.mainWindow())
        self.dialog.show()
        self.dialog.raise_()
        self.dialog.activateWindow()