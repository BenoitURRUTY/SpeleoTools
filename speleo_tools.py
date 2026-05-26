# -*- coding: utf-8 -*-
"""
SpeleoTools Plugin for QGIS 3
Auteur : Urruty Benoit
Description : Interface complète à 4 onglets pour outils spéléo.
"""
import csv
import math
import tempfile
import os
import processing
from qgis.PyQt import QtWidgets, uic, QtCore
from qgis.PyQt.QtCore import Qt
from qgis.core import (
    QgsProject, QgsRasterLayer, QgsVectorLayer, QgsPoint, QgsPointXY,
    QgsFeature, QgsFields, QgsField, QgsWkbTypes, QgsGeometry,
    QgsFeatureSink, QgsDistanceArea, QgsCoordinateTransformContext,
    QgsFeatureRequest, QgsMessageLog, Qgis, QgsVectorFileWriter,
    QgsCoordinateTransform
)
from PyQt5.QtCore import QVariant

import numpy as np
import heapq

from .speleo_utils import *
from .install_dependencies import requires

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
        # self.btnImport.clicked.connect(self.import_data)
        # self.btnApplyStyle.clicked.connect(self.apply_style)
        # Onglet 2
        self.btnBrowse.clicked.connect(self.browse_output)
        self.btnRunThickness.clicked.connect(self.run_thickness)
        # Onglet 3 — Profils
        self.btnBrowseThconfig.clicked.connect(self._browse_thconfig)
        self.btnReadThconfig.clicked.connect(self._read_alpha_from_thconfig)
        self.radioAngleThconfig.toggled.connect(self._toggle_alpha_source)
        self.radioAngleManual.toggled.connect(self._toggle_alpha_source)
        self.checkBoxMaxGap.toggled.connect(self.doubleSpinBoxMaxGap.setEnabled)
        self.btnBrowseProfileOutput.clicked.connect(
            lambda: self._browse_dir(self.editProfileOutputDir))
        self.btnRunProjected.clicked.connect(self.run_projected_profile)
        self.btnGenerateProfile.clicked.connect(self.run_developed_profile)
        # Ajouter emprise aux combos vecteur
        combos_vector_profile = [self.comboProjEmprise]
        # Onglet 4
        self.btnRunProspect.clicked.connect(self.run_mnt_analysis)
        self.btnBrowseOutput.clicked.connect(self.selectOutputDir)
        # Onglet 5 — Import Therion
        self.btnBrowseTherionShp.clicked.connect(lambda: self._browse_dir(self.editTherionShpPath))
        self.btnBrowseTherionGpkg.clicked.connect(lambda: self._browse_dir(self.editTherionGpkgPath))
        self.btnStyleAreas2D.clicked.connect(lambda: self._browse_qml(self.editStyleAreas2D))
        self.btnStyleLines2D.clicked.connect(lambda: self._browse_qml(self.editStyleLines2D))
        self.btnStylePoints2D.clicked.connect(lambda: self._browse_qml(self.editStylePoints2D))
        self.btnStyleOutline2D.clicked.connect(lambda: self._browse_qml(self.editStyleOutline2D))
        self.btnStyleShots3D.clicked.connect(lambda: self._browse_qml(self.editStyleShots3D))
        self.btnStyleStations3D.clicked.connect(lambda: self._browse_qml(self.editStyleStations3D))
        self.btnStyleWalls3D.clicked.connect(lambda: self._browse_qml(self.editStyleWalls3D))
        self.btnRunTherion.clicked.connect(self.run_therion_import)
        self.chkGroupLayers.toggled.connect(self.editGroupName.setEnabled)
        # Onglet 4 (dolines)
        self.btnBrowseDolines.clicked.connect(self.selectOutputDirDoline)
        self.btnRunDolines.clicked.connect(self.main_find_dolines)

        QgsProject.instance().layerWasAdded.connect(self.populate_layers)
        QgsProject.instance().layersWillBeRemoved.connect(self.populate_layers)

        # Pré-remplir les chemins de styles avec les QML du dossier styles_therion/
        self._prefill_style_paths()

        # Message d’état initial
        self.textLog.setPlainText("SpeleoTools prêt à l'emploi.\n")

    # ======================================================================
    # --- MÉTHODES GÉNÉRALES ---

    def populate_layers(self, *args):
        """Met à jour les listes de couches disponibles dans QGIS.
        Stocke l'ID de la couche comme userData pour éviter les collisions de noms."""
        combos_raster = [self.comboDEM, self.comboDEM2, self.comboProspectDEM, self.comboDolinesDEM]
        combos_vector = [self.comboCave, self.comboProfileLayer, self.comboProjEmprise]

        # Mémoriser les sélections courantes (par ID)
        def current_id(combo):
            return combo.currentData(Qt.UserRole)

        prev_ids = {c: current_id(c) for c in combos_raster + combos_vector}

        for combo in combos_raster + combos_vector:
            combo.blockSignals(True)
            combo.clear()

        for layer in QgsProject.instance().mapLayers().values():
            if isinstance(layer, QgsRasterLayer):
                for combo in combos_raster:
                    combo.addItem(layer.name(), layer.id())
            elif isinstance(layer, QgsVectorLayer):
                for combo in combos_vector:
                    combo.addItem(layer.name(), layer.id())

        # Restaurer les sélections précédentes
        for combo in combos_raster + combos_vector:
            prev = prev_ids.get(combo)
            if prev:
                idx = combo.findData(prev, Qt.UserRole)
                if idx >= 0:
                    combo.setCurrentIndex(idx)
            combo.blockSignals(False)

    def get_layer_by_name(self, name):
        """Retourne une couche par son nom (premier résultat)."""
        layers = QgsProject.instance().mapLayersByName(name)
        return layers[0] if layers else None

    def get_layer_by_combo(self, combo):
        """Retourne la couche sélectionnée dans un combobox via son ID stocké."""
        layer_id = combo.currentData(Qt.UserRole)
        if layer_id:
            layer = QgsProject.instance().mapLayer(layer_id)
            if layer:
                return layer
        # Fallback par nom
        return self.get_layer_by_name(combo.currentText())

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

    def selectOutputDir(self):
        """Dossier de sortie pour l'onglet Traitement MNT."""
        d = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Choisir le dossier de sortie",
            self.editOutputDir.text().strip() or os.path.expanduser("~"))
        if d:
            self.editOutputDir.setText(d)

    def selectOutputDirProfile(self):
        """Dossier de sortie pour l'onglet Profils."""
        d = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Choisir le dossier de sortie profil",
            self.editProfileOutputDir.text().strip() or os.path.expanduser("~"))
        if d:
            self.editProfileOutputDir.setText(d)

    def _safe_name(self, name):
        """Génère un nom de fichier sûr : ASCII, alphanum + tirets, sans underscores multiples."""
        import unicodedata, re as _re
        base = os.path.splitext(name)[0]
        base = unicodedata.normalize('NFKD', base).encode('ascii', 'ignore').decode('ascii')
        safe = "".join(c if c.isalnum() or c == '-' else '_' for c in base)
        safe = _re.sub(r'_+', '_', safe).strip('_')
        return safe or "dem"

    def run_thickness(self):
        dem_name = self.comboDEM.currentText()
        cave_name = self.comboCave.currentText()
        out_path = self.lineOutput.text().strip()
        layername = self.LayerName.text()

        if not dem_name or not cave_name:
            QtWidgets.QMessageBox.warning(self, "Erreur", "Sélectionne un DEM et une couche de cavité.")
            return

        raster = self.get_layer_by_combo(self.comboDEM)
        vec = self.get_layer_by_combo(self.comboCave)
        if raster is None or vec is None:
            QtWidgets.QMessageBox.warning(self, "Erreur", "Impossible de trouver les couches sélectionnées.")
            return

        try:
            self.log(f"Début calcul d'épaisseur entre '{cave_name}' et '{dem_name}'...")
            self.progressThickness.setValue(10)
            QtWidgets.QApplication.processEvents()

            out = out_path if out_path else None
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
    # ======================================================================
    # --- ONGLET 3 : PROFILS ---

    def _plog(self, msg):
        self.textLogProfile.append(msg)
        QtWidgets.QApplication.processEvents()

    def _toggle_alpha_source(self):
        """Active/désactive les widgets selon la source de l'angle alpha."""
        from_file = self.radioAngleThconfig.isChecked()
        self.editThconfigPath.setEnabled(from_file)
        self.btnBrowseThconfig.setEnabled(from_file)
        self.btnReadThconfig.setEnabled(from_file)
        self.spinAlpha.setEnabled(not from_file)

    def _browse_thconfig(self):
        f, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Choisir un fichier .thconfig",
            os.path.expanduser("~"), "Therion config (*.thconfig *.th);;Tous (*)")
        if f:
            self.editThconfigPath.setText(f)

    def _read_alpha_from_thconfig(self):
        """Parse le fichier .thconfig et extrait l'angle de projection."""
        import re
        path = self.editThconfigPath.text().strip()
        if not path or not os.path.isfile(path):
            QtWidgets.QMessageBox.warning(self, "Fichier introuvable",
                "Sélectionnez d'abord un fichier .thconfig valide.")
            return
        try:
            with open(path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
            # Cherche : -projection [elevation XX] ou -projection [elevation XX.X]
            match = re.search(
                r'-projection\s+\[\s*elevation\s+([\d.]+)\s*\]', content)
            if match:
                alpha = float(match.group(1))
                self.spinAlpha.setValue(alpha)
                self._plog(f"α = {alpha}° lu depuis {os.path.basename(path)}")
                # Basculer sur saisie manuelle pour montrer la valeur lue
                self.radioAngleManual.setChecked(True)
            else:
                QtWidgets.QMessageBox.warning(self, "Angle introuvable",
                    "Aucune ligne '-projection [elevation XX]' trouvée dans le fichier.")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Erreur lecture", str(e))

    def _profile_output_dir(self):
        """Retourne le dossier de sortie des profils (crée si besoin)."""
        d = self.editProfileOutputDir.text().strip() or tempfile.gettempdir()
        os.makedirs(d, exist_ok=True)
        return d

    def _reproject_to_dem_crs(self, vector_layer, dem_layer):
        """Reprojette une couche vecteur dans le CRS du MNT si nécessaire.
        Retourne la couche reprojetée (en mémoire) ou la couche originale."""
        from qgis.core import QgsCoordinateReferenceSystem
        dem_crs  = dem_layer.crs()
        vec_crs  = vector_layer.crs()
        if vec_crs == dem_crs:
            return vector_layer
        self._plog(
            f"   Reprojection {vector_layer.name()} : "
            f"{vec_crs.authid()} → {dem_crs.authid()}")
        try:
            res = processing.run("native:reprojectlayer", {
                'INPUT':      vector_layer,
                'TARGET_CRS': dem_crs,
                'OUTPUT':     'TEMPORARY_OUTPUT'
            })
            out = res.get('OUTPUT')
            if out and (isinstance(out, QgsVectorLayer) and out.isValid()):
                return out
        except Exception as e:
            self._plog(f"   ⚠ Reprojection impossible : {e} — couche originale utilisée")
        return vector_layer

    def _export_profile_csv_png(self, distances, elevations, name, out_dir,
                                  offset_x=0.0, offset_y=0.0):
        """Exporte le profil en :
          - CSV  (X_distance_m, Y_altitude_m)
          - GPKG (points sans CRS — coordonnées X=distance, Y=altitude)
          - PNG  (si matplotlib disponible)
        Retourne (csv_path, gpkg_path, png_path_or_None).
        """
        import math, csv as _csv

        xs = [d + offset_x for d in distances]
        ys = [e + offset_y if (e is not None and not math.isnan(e)) else float('nan')
              for e in elevations]

        # ── CSV ──────────────────────────────────────────────────────
        csv_path = os.path.join(out_dir, name + ".csv")
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            w = _csv.writer(csvfile)
            w.writerow(["X_distance_m", "Y_altitude_m"])
            for x, y in zip(xs, ys):
                w.writerow([round(x, 3),
                             "" if (isinstance(y, float) and math.isnan(y))
                             else round(y, 3)])
        self._plog("CSV : " + csv_path)

        # ── GPKG sans CRS (coordonnées profil X/Y) ───────────────────
        gpkg_path = None
        try:
            from qgis.core import (QgsVectorLayer, QgsFeature, QgsGeometry,
                                   QgsPointXY, QgsField, QgsFields,
                                   QgsCoordinateReferenceSystem,
                                   QgsVectorFileWriter, QgsProject,
                                   QgsWkbTypes)
            from PyQt5.QtCore import QVariant

            # CRS vide (non géographique) pour un profil X/Y
            no_crs = QgsCoordinateReferenceSystem()

            fields = QgsFields()
            fields.append(QgsField("X_dist_m",  QVariant.Double))
            fields.append(QgsField("Y_alt_m",   QVariant.Double))
            fields.append(QgsField("pt_index",  QVariant.Int))

            mem_layer = QgsVectorLayer(
                "Point?crs=", name + "_profil", "memory")
            mem_layer.setCrs(no_crs)
            pr = mem_layer.dataProvider()
            pr.addAttributes(fields)
            mem_layer.updateFields()

            feats = []
            for idx, (x, y) in enumerate(zip(xs, ys)):
                if isinstance(y, float) and math.isnan(y):
                    continue
                f = QgsFeature()
                f.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(x, y)))
                f.setAttributes([round(x, 3), round(y, 3), idx])
                feats.append(f)
            pr.addFeatures(feats)

            gpkg_path = os.path.join(out_dir, name + ".gpkg")
            opts = QgsVectorFileWriter.SaveVectorOptions()
            opts.driverName    = "GPKG"
            opts.fileEncoding  = "UTF-8"
            opts.layerName     = name
            # Pas de CRS cible → coordonnées profil brutes
            err = QgsVectorFileWriter.writeAsVectorFormatV3(
                mem_layer, gpkg_path,
                QgsProject.instance().transformContext(), opts)
            if err[0] == QgsVectorFileWriter.NoError:
                self._plog("GPKG (sans CRS) : " + gpkg_path)
            else:
                self._plog("⚠ GPKG : erreur " + str(err))
                gpkg_path = None
        except Exception as e:
            self._plog("⚠ GPKG non créé : " + str(e))

        # ── PNG (optionnel) ───────────────────────────────────────────
        png_path = None
        try:
            import matplotlib.pyplot as plt
            px2 = [x for x, y in zip(xs, ys)
                   if not (isinstance(y, float) and math.isnan(y))]
            py2 = [y for y in ys
                   if not (isinstance(y, float) and math.isnan(y))]
            if px2:
                fig, ax = plt.subplots(figsize=(14, 5))
                ax.plot(px2, py2, '-b', linewidth=1.2)
                ax.fill_between(px2, py2, min(py2), alpha=0.12, color='steelblue')
                ax.set_title(name, fontsize=12)
                ax.set_xlabel("Distance (m)")
                ax.set_ylabel("Altitude (m)")
                ax.grid(True, linestyle='--', alpha=0.4)
                fig.tight_layout()
                png_path = os.path.join(out_dir, name + ".png")
                fig.savefig(png_path, dpi=200, bbox_inches='tight')
                plt.close(fig)
                self._plog("PNG : " + png_path)
        except ImportError:
            self._plog("[INFO] matplotlib absent — PNG ignoré.")

        return csv_path, gpkg_path, png_path

    # ── Cas 1 : Profil projeté ────────────────────────────────────────
    def run_projected_profile(self):
        import math
        dem_layer     = self.get_layer_by_combo(self.comboDEM2)
        emprise_layer = self.get_layer_by_combo(self.comboProjEmprise)
        if not dem_layer or not dem_layer.isValid():
            QtWidgets.QMessageBox.warning(self, "Erreur", "Sélectionnez un MNT valide.")
            return
        if not emprise_layer or not emprise_layer.isValid():
            QtWidgets.QMessageBox.warning(self, "Erreur",
                "Sélectionnez une couche d'emprise valide.")
            return

        alpha        = float(self.spinAlpha.value())
        margin_pct   = float(self.spinProfileMargin.value()) / 100.0
        offset_x     = float(self.spinProjOffsetX.value())
        offset_y     = float(self.spinProjOffsetY.value())
        save_line    = self.chkSaveCutLine.isChecked()
        out_dir      = self._profile_output_dir()

        self.textLogProfile.clear()
        self._plog("Profil projete alpha=" + str(alpha) + "° — " + emprise_layer.name())

        try:
            from qgis.core import (QgsRectangle, QgsPointXY, QgsVectorLayer,
                                   QgsFeature, QgsGeometry, QgsFields, QgsField,
                                   QgsVectorFileWriter, QgsProject,
                                   QgsCoordinateReferenceSystem)
            from PyQt5.QtCore import QVariant

            # Reprojeter l'emprise dans le CRS du MNT
            emprise_repr = self._reproject_to_dem_crs(emprise_layer, dem_layer)
            dem_crs      = dem_layer.crs()

            # Emprise + marge
            bbox   = emprise_repr.extent()
            margin = max(bbox.width(), bbox.height()) * margin_pct
            bbox_exp = QgsRectangle(
                bbox.xMinimum() - margin, bbox.yMinimum() - margin,
                bbox.xMaximum() + margin, bbox.yMaximum() + margin)

            # Barycentre réel des géométries (union puis centroïde)
            from qgis.core import QgsGeometry
            all_geoms = [f.geometry() for f in emprise_repr.getFeatures()
                         if f.geometry() and not f.geometry().isEmpty()]
            if all_geoms:
                union_geom = all_geoms[0]
                for g in all_geoms[1:]:
                    union_geom = union_geom.combine(g)
                centroid  = union_geom.centroid().asPoint()
                cx, cy    = centroid.x(), centroid.y()
                self._plog("Barycentre geometries : (" +
                           str(round(cx,1)) + ", " + str(round(cy,1)) + ")")
            else:
                # Fallback : centre de la bounding box
                cx = (bbox_exp.xMinimum() + bbox_exp.xMaximum()) / 2.0
                cy = (bbox_exp.yMinimum() + bbox_exp.yMaximum()) / 2.0
                self._plog("Fallback centre bbox : (" +
                           str(round(cx,1)) + ", " + str(round(cy,1)) + ")")

            half_diag = math.hypot(bbox_exp.width(), bbox_exp.height()) / 2.0

            # Ligne de coupe à α+90°
            cut_az  = (alpha + 90.0) % 360.0
            rad     = math.radians(cut_az)
            dx, dy  = math.sin(rad), math.cos(rad)
            x0, y0  = cx - dx * half_diag, cy - dy * half_diag
            x1, y1  = cx + dx * half_diag, cy + dy * half_diag
            self._plog("Coupe az=" + str(round(cut_az,1)) +
                       "° : (" + str(round(x0,1)) + "," + str(round(y0,1)) +
                       ") -> (" + str(round(x1,1)) + "," + str(round(y1,1)) + ")")

            # ── Sauvegarder la ligne de coupe en GPKG (optionnel) ────
            line_gpkg_path = None
            if save_line:
                try:
                    fields = QgsFields()
                    fields.append(QgsField("alpha_deg",   QVariant.Double))
                    fields.append(QgsField("cut_az_deg",  QVariant.Double))
                    fields.append(QgsField("longueur_m",  QVariant.Double))

                    mem_line = QgsVectorLayer(
                        "LineString?crs=" + dem_crs.authid(),
                        "ligne_coupe", "memory")
                    mem_line.setCrs(dem_crs)
                    pr = mem_line.dataProvider()
                    pr.addAttributes(fields)
                    mem_line.updateFields()

                    feat = QgsFeature()
                    feat.setGeometry(QgsGeometry.fromPolylineXY(
                        [QgsPointXY(x0, y0), QgsPointXY(x1, y1)]))
                    feat.setAttributes([
                        round(alpha, 3),
                        round(cut_az, 3),
                        round(math.hypot(x1-x0, y1-y0), 3)
                    ])
                    pr.addFeatures([feat])

                    name_base   = ("ligne_coupe_a" + str(int(alpha)) + "deg_" +
                                   self._safe_name(dem_layer.name()))
                    line_gpkg_path = os.path.join(out_dir, name_base + "_ligne.gpkg")
                    opts = QgsVectorFileWriter.SaveVectorOptions()
                    opts.driverName   = "GPKG"
                    opts.fileEncoding = "UTF-8"
                    opts.layerName    = "ligne_coupe"
                    err = QgsVectorFileWriter.writeAsVectorFormatV3(
                        mem_line, line_gpkg_path,
                        QgsProject.instance().transformContext(), opts)
                    if err[0] == QgsVectorFileWriter.NoError:
                        # Charger dans QGIS
                        lyr = QgsVectorLayer(line_gpkg_path, name_base, "ogr")
                        if lyr.isValid():
                            QgsProject.instance().addMapLayer(lyr)
                        self._plog("Ligne de coupe GPKG : " + line_gpkg_path)
                    else:
                        self._plog("⚠ Sauvegarde ligne : erreur " + str(err))
                        line_gpkg_path = None
                except Exception as e:
                    self._plog("⚠ Sauvegarde ligne impossible : " + str(e))

            # ── Echantillonnage MNT ──────────────────────────────────
            dem_res  = (dem_layer.rasterUnitsPerPixelX() +
                        dem_layer.rasterUnitsPerPixelY()) / 2.0
            line_len = math.hypot(x1-x0, y1-y0)
            n_pts    = max(int(line_len / max(dem_res, 0.01)), 2)

            distances, elevations = [], []
            for i in range(n_pts + 1):
                t  = i / n_pts
                px = x0 + t * (x1 - x0)
                py = y0 + t * (y1 - y0)
                try:
                    z = sample_dem_at_point(dem_layer, QgsPointXY(px, py))
                except Exception:
                    z = float('nan')
                distances.append(t * line_len)
                elevations.append(z if z is not None else float('nan'))

            valid = sum(1 for e in elevations if not math.isnan(e))
            self._plog(str(n_pts+1) + " points, " + str(valid) + " valides.")

            name = ("profil_projete_a" + str(int(alpha)) + "deg_" +
                    self._safe_name(dem_layer.name()))
            csv_p, gpkg_p, png_p = self._export_profile_csv_png(
                distances, elevations, name, out_dir, offset_x, offset_y)

            msg = "Profil projete genere.\nCSV : " + csv_p
            if gpkg_p:      msg += "\nGPKG profil : " + gpkg_p
            if line_gpkg_path: msg += "\nGPKG ligne : " + line_gpkg_path
            if png_p:       msg += "\nPNG : " + png_p
            QtWidgets.QMessageBox.information(self, "Termine", msg)

        except Exception as e:
            self._plog("[ERROR] " + str(e))
            QtWidgets.QMessageBox.critical(self, "Erreur", str(e))

    # ── Cas 2 : Profil développé ──────────────────────────────────────
    def run_developed_profile(self):
        import math
        dem_layer     = self.get_layer_by_combo(self.comboDEM2)
        profile_layer = self.get_layer_by_combo(self.comboProfileLayer)
        if not dem_layer or not dem_layer.isValid():
            QtWidgets.QMessageBox.warning(self, "Erreur", "Sélectionnez un MNT valide.")
            return
        if not profile_layer or not profile_layer.isValid():
            QtWidgets.QMessageBox.warning(self, "Erreur", "Sélectionnez une polyligne valide.")
            return

        spacing   = float(self.doubleSpinBoxSpacing.value())
        do_interp = self.checkBoxInterpolate.isChecked()
        max_gap   = (float(self.doubleSpinBoxMaxGap.value())
                     if self.checkBoxMaxGap.isChecked() else None)
        offset_x  = float(self.spinDevOffsetX.value())
        offset_y  = float(self.spinDevOffsetY.value())
        out_dir   = self._profile_output_dir()

        self.textLogProfile.clear()

        # Utiliser la sélection si la case est cochée ET qu'il y a une sélection
        use_sel    = self.chkUseSelection.isChecked()
        n_selected = profile_layer.selectedFeatureCount()

        if use_sel and n_selected > 0:
            self._plog("Selection active : " + str(n_selected) + " entite(s) utilisee(s).")
            request = QgsFeatureRequest().setFilterFids(
                profile_layer.selectedFeatureIds())
        elif use_sel and n_selected == 0:
            self._plog("⚠ 'Utiliser la selection' coché mais aucune entite selectionnee — toutes utilisees.")
            request = QgsFeatureRequest()
        else:
            self._plog("Toutes les entites de la couche utilisees.")
            request = QgsFeatureRequest()

        # Reprojeter la polyligne dans le CRS du MNT
        profile_repr = self._reproject_to_dem_crs(profile_layer, dem_layer)
        dem_crs      = dem_layer.crs()
        self._plog("Profil developpe — " + profile_layer.name() +
                   " / MNT : " + dem_layer.name())

        try:
            from qgis.core import QgsPointXY

            distances, elevations = [], []
            cum_dist = 0.0
            prev_xy  = None

            # Si on a reprojeté, on itère la couche reprojetée sur la sélection d'origine
            if n_selected > 0 and profile_repr is not profile_layer:
                # Récupérer les fids sélectionnés — après reprojection les fids sont conservés
                iter_request = QgsFeatureRequest().setFilterFids(
                    profile_layer.selectedFeatureIds())
            else:
                iter_request = request

            for feat in profile_repr.getFeatures(iter_request):
                geom = feat.geometry()
                if not geom or geom.isEmpty():
                    continue
                polylines = (geom.asMultiPolyline() if geom.isMultipart()
                             else [geom.asPolyline()])

                for poly in polylines:
                    if len(poly) < 2:
                        continue

                    dense_pts = []
                    for i in range(len(poly) - 1):
                        try:
                            ax, ay = poly[i].x(),   poly[i].y()
                            bx, by = poly[i+1].x(), poly[i+1].y()
                        except Exception:
                            ax, ay = float(poly[i][0]),   float(poly[i][1])
                            bx, by = float(poly[i+1][0]), float(poly[i+1][1])
                        seg_len = math.hypot(bx-ax, by-ay)
                        if seg_len == 0:
                            continue
                        n_sub = max(1, int(seg_len / spacing))
                        for k in range(n_sub):
                            t = k / n_sub
                            dense_pts.append((ax + t*(bx-ax), ay + t*(by-ay)))
                    try:
                        dense_pts.append((poly[-1].x(), poly[-1].y()))
                    except Exception:
                        dense_pts.append((float(poly[-1][0]), float(poly[-1][1])))

                    for px, py in dense_pts:
                        seg = (math.hypot(px - prev_xy[0], py - prev_xy[1])
                               if prev_xy else 0.0)
                        cum_dist += seg
                        prev_xy   = (px, py)
                        try:
                            z = sample_dem_at_point(dem_layer, QgsPointXY(px, py))
                        except Exception:
                            z = float('nan')
                        distances.append(cum_dist)
                        elevations.append(z if z is not None else float('nan'))

            if not distances:
                QtWidgets.QMessageBox.warning(self, "Erreur",
                    "Aucun point extrait. Verifiez la couche polyligne.")
                return

            # Interpolation NoData
            if do_interp:
                import numpy as np
                arr  = np.array(elevations, dtype=float)
                nans = np.isnan(arr)
                if nans.any() and not nans.all():
                    idx      = np.arange(len(arr))
                    arr_fill = np.interp(idx[nans], idx[~nans], arr[~nans])
                    if max_gap is not None:
                        dist_arr = np.array(distances)
                        valid_idx = idx[~nans]
                        for j, ni in enumerate(np.where(nans)[0]):
                            left = np.searchsorted(valid_idx, ni) - 1
                            right = left + 1
                            if 0 <= left and right < len(valid_idx):
                                gap = dist_arr[valid_idx[right]] - dist_arr[valid_idx[left]]
                                if gap > max_gap:
                                    arr_fill[j] = float('nan')
                    arr[nans] = arr_fill
                    elevations = arr.tolist()

            valid = sum(1 for e in elevations if not math.isnan(float(e)))
            self._plog(str(len(distances)) + " points, " + str(valid) + " valides.")

            sel_suffix = "_sel" if (use_sel and n_selected > 0) else ""
            name = ("profil_dev" + sel_suffix + "_" +
                    self._safe_name(profile_layer.name()) + "_" +
                    self._safe_name(dem_layer.name()))
            csv_p, gpkg_p, png_p = self._export_profile_csv_png(
                distances, elevations, name, out_dir, offset_x, offset_y)

            msg = "Profil developpe genere.\nCSV : " + csv_p
            if gpkg_p: msg += "\nGPKG : " + gpkg_p
            if png_p:  msg += "\nPNG : " + png_p
            QtWidgets.QMessageBox.information(self, "Termine", msg)

        except Exception as e:
            self._plog("[ERROR] " + str(e))
            QtWidgets.QMessageBox.critical(self, "Erreur", str(e))

    def run_mnt_analysis(self):
        dem_layer = self.get_layer_by_combo(self.comboProspectDEM)
        if not dem_layer or not dem_layer.isValid():
            QtWidgets.QMessageBox.warning(self, "Erreur", "Sélectionne un MNT valide.")
            return

        add_to_project = self.AddlayerMNT.isChecked()
        zfactor        = float(self.spinZFactor.value())
        vat_window     = int(self.spinVATWindow.value())

        do_hillshade = self.chkHillshade.isChecked()
        hs_azimuth   = float(self.spinHsAzimuth.value())
        hs_elevation = float(self.spinHsElevation.value())

        do_multidh   = self.chkMultiHillshade.isChecked()
        mhs_dirs     = int(self.spinMhsDirections.value())
        mhs_elev     = float(self.spinMhsElevation.value())

        do_slope     = self.chkSlope.isChecked()
        slope_units  = self.comboSlopeUnit.currentText()

        do_svf       = self.chkSVF.isChecked()
        svf_radius   = int(self.spinSvfRadius.value())
        svf_dirs     = int(self.spinSvfDirs.value())

        do_openness  = self.chkOpenness.isChecked()
        do_opns_neg  = self.chkOpennessNeg.isChecked()

        do_slrm      = self.chkSLRM.isChecked()
        slrm_radius  = int(self.spinSlrmRadius.value())

        do_vat       = self.chkVAT.isChecked()

        # Azimuts custom
        custom_azimuths = []
        txt = self.editAzimuths.text().strip()
        if txt:
            try:
                custom_azimuths = [float(a.strip()) for a in txt.split(',') if a.strip()]
            except ValueError:
                self.log("[WARN] Azimuts invalides — ignorés.")

        # Dossier de sortie
        out_dir = self.editOutputDir.text().strip() or tempfile.gettempdir()
        try:
            os.makedirs(out_dir, exist_ok=True)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Erreur dossier", str(e))
            return

        # Nom de base propre pour les fichiers de sortie
        base = self._safe_name(dem_layer.name())
        def outpath(suffix):
            return os.path.join(out_dir, f"{base}_{suffix}.tif")

        self.textLog.clear()
        tasks = sum([do_hillshade, do_multidh, do_slope, do_svf,
                     do_openness, do_opns_neg, do_slrm, do_vat,
                     bool(custom_azimuths)])
        total = max(tasks, 1)
        step  = 0

        self.log(f"Analyse MNT '{dem_layer.name()}' → {out_dir}")

        try:
            if do_hillshade:
                self.log(f" ☀ Hillshade (az={hs_azimuth}°, élév={hs_elevation}°)…")
                hillshade(dem_layer, out_path=outpath('hillshade'), zfactor=zfactor,
                          azimuth=hs_azimuth, altitude=hs_elevation,
                          addProject=add_to_project)
                step += 1; self.progressProspect.setValue(int(step/total*100))

            if do_multidh:
                self.log(f" 🌐 Multi-hillshade ({mhs_dirs} directions)…")
                multidirectional_hillshade(dem_layer, out_path=outpath('multidh'),
                          nr_directions=mhs_dirs, sun_elevation=mhs_elev,
                          zfactor=zfactor, addProject=add_to_project)
                step += 1; self.progressProspect.setValue(int(step/total*100))

            if do_slope:
                self.log(f" 📐 Pente ({slope_units})…")
                slope(dem_layer, out_path=outpath('slope'), zfactor=zfactor,
                      output_units=slope_units, addProject=add_to_project)
                step += 1; self.progressProspect.setValue(int(step/total*100))

            if do_svf:
                self.log(f" 🌌 SVF (r={svf_radius}px, {svf_dirs} dir.)…")
                sky_view_factor(dem_layer, out_path=out_dir,
                    svf_n_dir=svf_dirs, svf_r_max=svf_radius,
                    zfactor=zfactor, compute_svf=True,
                    compute_opns=do_openness, addProject=add_to_project)
                step += 1; self.progressProspect.setValue(int(step/total*100))
            elif do_openness:
                self.log(f" 🔭 Ouverture positive (r={svf_radius}px)…")
                sky_view_factor(dem_layer, out_path=out_dir,
                    svf_n_dir=svf_dirs, svf_r_max=svf_radius,
                    zfactor=zfactor, compute_svf=False,
                    compute_opns=True, addProject=add_to_project)
                step += 1; self.progressProspect.setValue(int(step/total*100))

            if do_opns_neg:
                self.log(f" 🔭 Ouverture négative…")
                openness_negative(dem_layer, out_path=outpath('opns_neg'),
                    svf_n_dir=svf_dirs, svf_r_max=svf_radius,
                    zfactor=zfactor, addProject=add_to_project)
                step += 1; self.progressProspect.setValue(int(step/total*100))

            if do_slrm:
                self.log(f" 📊 SLRM (rayon={slrm_radius}px)…")
                slrm(dem_layer, out_path=outpath('slrm'),
                     radius_cell=slrm_radius, zfactor=zfactor,
                     addProject=add_to_project)
                step += 1; self.progressProspect.setValue(int(step/total*100))

            if do_vat:
                self.log(f" 🏛 VAT (fenêtre={vat_window})…")
                VAT(dem_layer, out_path=outpath('vat'),
                    vat_window=vat_window, zfactor=zfactor,
                    addProject=add_to_project)
                step += 1; self.progressProspect.setValue(int(step/total*100))

            for az in custom_azimuths:
                self.log(f" ☀ Hillshade azimut {az}°…")
                hillshade(dem_layer, out_path=outpath(f'hs_az{int(az)}'),
                          zfactor=zfactor, azimuth=az, altitude=hs_elevation,
                          addProject=add_to_project)
            if custom_azimuths:
                step += 1; self.progressProspect.setValue(int(step/total*100))

            self.progressProspect.setValue(100)
            self.log("✓ Analyse MNT terminée.")
            QtWidgets.QMessageBox.information(
                self, "Terminé",
                f"Analyse MNT terminée.\nFichiers : {out_dir}")

        except Exception as e:
            self.log(f"[ERROR] {e}")
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

        # --- paramètres depuis l'UI (avec fallback si les widgets n'existent pas) ---
        def _spin(attr, default):
            w = getattr(self, attr, None)
            return float(w.value()) if w is not None else default

        minslope       = _spin('spinDolineMinSlope',   0.1)
        sink_threshold = _spin('spinDolineSinkThresh',  1.0)
        dbscan_eps     = _spin('spinDolineEps',         5.0)
        dbscan_min     = int(_spin('spinDolineMinPts',  5.0))

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
        dem_layer = self.get_layer_by_combo(self.comboDolinesDEM) or layers[0]
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



    # ======================================================================
    # --- ONGLET 5 : Import Therion ---

    def _browse_dir(self, line_edit):
        start = line_edit.text().strip() or os.path.expanduser("~")
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Choisir un dossier", start)
        if d:
            line_edit.setText(d)

    def _browse_qml(self, line_edit):
        start = os.path.dirname(line_edit.text().strip()) or os.path.expanduser("~")
        f, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Choisir un style QML", start, "Fichiers QML (*.qml)")
        if f:
            line_edit.setText(f)

    def _prefill_style_paths(self):
        """Pré-remplit les champs de style avec les QML inclus dans le plugin
        (dossier styles_therion/). L'utilisateur peut les modifier librement.
        Un champ déjà rempli n'est pas écrasé."""
        plugin_dir = os.path.dirname(__file__)
        styles_dir = os.path.join(plugin_dir, "styles_therion")

        # Correspondance widget → nom de fichier QML
        mapping = {
            self.editStyleAreas2D:    "Style_Area2D.qml",
            self.editStyleLines2D:    "Style_Ligne2D.qml",
            self.editStylePoints2D:   "Style_Point2D.qml",
            self.editStyleOutline2D:  "Style_Outline2D.qml",
            self.editStyleShots3D:    "Style_Shots3D.qml",
            self.editStyleStations3D: "Style_Stations3D.qml",
            self.editStyleWalls3D:    "Style_Wall3D.qml",
        }

        for widget, filename in mapping.items():
            # Ne pas écraser si déjà rempli par l'utilisateur
            if widget.text().strip():
                continue
            full_path = os.path.join(styles_dir, filename)
            if os.path.isfile(full_path):
                widget.setText(full_path)
            else:
                # Fichier absent : mettre le chemin en placeholder (grisé)
                widget.setPlaceholderText(
                    f"{os.path.join('styles_therion', filename)} (introuvable)"
                )

    def _tlog(self, msg):
        self.textLogTherion.append(msg)
        QtWidgets.QApplication.processEvents()

    def _fix_shp(self, shp_path):
        """Répare les géométries d'un SHP avec native:fixgeometries.
        Retourne le chemin de la couche réparée (TEMPORARY_OUTPUT en mémoire QGIS).
        En cas d'échec retourne le chemin original non réparé."""
        try:
            result = processing.run(
                "native:fixgeometries",
                {'INPUT': shp_path, 'METHOD': 1, 'OUTPUT': 'TEMPORARY_OUTPUT'}
            )
            fixed = result.get('OUTPUT')
            if fixed and (isinstance(fixed, QgsVectorLayer) and fixed.isValid() or
                          isinstance(fixed, str) and fixed):
                self._tlog(f"   ✔ Géométries réparées : {os.path.basename(shp_path)}")
                return fixed
        except Exception as e:
            self._tlog(f"   ⚠ fixgeometries échoué ({e}) — SHP original utilisé")
        return shp_path

    @requires("geopandas", "pandas")
    def run_therion_import(self):
        """Import complet des sorties Therion :
        1. Réparation géométries
        2. Découpe lignes/aires sur outline
        3. Ajout altitudes aux points
        4. Conversion SHP → GPKG
        5. Import QGIS dans un groupe avec styles QML
        """
        shp_path = self.editTherionShpPath.text().strip()
        if not shp_path or not os.path.isdir(shp_path):
            QtWidgets.QMessageBox.warning(
                self, "Erreur", "Dossier SHP Therion invalide ou non renseigné.")
            return

        shp_path = os.path.normpath(shp_path) + os.sep

        gpkg_path_txt = self.editTherionGpkgPath.text().strip()
        if gpkg_path_txt and os.path.isdir(gpkg_path_txt):
            outputs_path = os.path.normpath(gpkg_path_txt) + os.sep
        else:
            outputs_path = os.path.join(
                os.path.dirname(shp_path.rstrip(os.sep)), "GPKG") + os.sep
        os.makedirs(outputs_path, exist_ok=True)

        repair_geom = self.chkRepairGeom.isChecked()
        add_alt     = self.chkAddAlt.isChecked()
        use_group   = self.chkGroupLayers.isChecked()
        group_name  = self.editGroupName.text().strip() or \
                      os.path.basename(shp_path.rstrip(os.sep))

        def _qml(w): return w.text().strip() or None
        styles = {
            'areas2d':    _qml(self.editStyleAreas2D),
            'lines2d':    _qml(self.editStyleLines2D),
            'points2d':   _qml(self.editStylePoints2D),
            'outline2d':  _qml(self.editStyleOutline2D),
            'shots3d':    _qml(self.editStyleShots3D),
            'stations3d': _qml(self.editStyleStations3D),
            'walls3d':    _qml(self.editStyleWalls3D),
        }

        self.textLogTherion.clear()
        self.progressTherion.setValue(0)
        self._tlog(f"▶ Import Therion depuis : {shp_path}")
        self._tlog(f"  Sortie GPKG : {outputs_path}")

        for req in ['outline2d.shp', 'lines2d.shp']:
            if not os.path.isfile(shp_path + req):
                QtWidgets.QMessageBox.critical(
                    self, "Fichier manquant",
                    f"Fichier obligatoire absent : {shp_path + req}")
                return

        try:
            import geopandas as gpd
            import pandas as pd
        except ImportError:
            QtWidgets.QMessageBox.critical(
                self, "Dépendance manquante",
                "geopandas et pandas sont requis.\n"
                "pip install geopandas pandas  (dans l'interpréteur Python de QGIS)")
            return

        # ── 1 : Outline ───────────────────────────────────────────────────
        self._tlog("1/6 Lecture de l'outline…")
        self.progressTherion.setValue(5)
        try:
            outline_src = shp_path + 'outline2d.shp'
            if repair_geom:
                outline_src = self._fix_shp(outline_src)
            # Lire depuis la couche fixée (QgsVectorLayer ou chemin)
            if isinstance(outline_src, QgsVectorLayer):
                tmp_outline = os.path.join(outputs_path, '_tmp_outline.gpkg')
                outline_src.selectAll()
                processing.run("native:savefeatures",
                    {'INPUT': outline_src, 'OUTPUT': tmp_outline})
                outline_src = tmp_outline
            outlines = gpd.read_file(outline_src)
            outlines = outlines[outlines.geometry.notnull() & ~outlines.geometry.is_empty]
            self._tlog(f"   outline2d : {len(outlines)} entité(s)")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Erreur outline", str(e))
            return

        total_steps = 6
        gpkg_map = {}

        # ── 2 : Lignes ────────────────────────────────────────────────────
        self._tlog("2/6 Lignes (lines2d)…")
        self.progressTherion.setValue(15)
        lines_gpkg = None
        try:
            lines_src = shp_path + 'lines2d.shp'
            if repair_geom:
                lines_src = self._fix_shp(lines_src)
            if isinstance(lines_src, QgsVectorLayer):
                tmp_lines = os.path.join(outputs_path, '_tmp_lines.gpkg')
                processing.run("native:savefeatures",
                    {'INPUT': lines_src, 'OUTPUT': tmp_lines})
                lines_src = tmp_lines
            lines = gpd.read_file(lines_src)
            lines = lines[lines.geometry.notnull() & ~lines.geometry.is_empty]

            linesOUT = pd.concat([
                lines[lines['_TYPE'] == 'centerline'],
                lines[lines['_TYPE'] == 'water_flow'],
                lines[lines['_TYPE'] == 'label'],
                lines[lines['_CLIP'] == 'off'],
            ], ignore_index=True)

            linesIN = lines[
                (lines['_CLIP'] != 'off') &
                (~lines['_TYPE'].isin(['centerline', 'water_flow', 'label']))
            ]

            try:
                linesIN = linesIN.overlay(outlines, how='intersection', keep_geom_type=True)
                if {'_SCRAP_ID', '_ID'}.issubset(linesIN.columns):
                    linesIN = linesIN[linesIN['_SCRAP_ID'] == linesIN['_ID']]
            except Exception as e_ov:
                self._tlog(f"   ⚠ Intersection lignes : {e_ov} — lignes brutes conservées")

            linesTOT = pd.concat([linesOUT, linesIN], ignore_index=True)
            lines_gpkg = outputs_path + 'lines2dMasked.gpkg'
            linesTOT.to_file(lines_gpkg, driver='GPKG')
            self._tlog(f"   ✔ lines2dMasked.gpkg : {len(linesTOT)} entité(s)")
        except Exception as e:
            self._tlog(f"   ❌ Lignes : {e}")

        # ── 3 : Aires ─────────────────────────────────────────────────────
        self._tlog("3/6 Aires (areas2d)…")
        self.progressTherion.setValue(30)
        areas_gpkg = None
        if os.path.isfile(shp_path + 'areas2d.shp'):
            try:
                areas_src = shp_path + 'areas2d.shp'
                if repair_geom:
                    areas_src = self._fix_shp(areas_src)
                if isinstance(areas_src, QgsVectorLayer):
                    tmp_areas = os.path.join(outputs_path, '_tmp_areas.gpkg')
                    processing.run("native:savefeatures",
                        {'INPUT': areas_src, 'OUTPUT': tmp_areas})
                    areas_src = tmp_areas
                areas = gpd.read_file(areas_src)
                areas = areas[areas.geometry.notnull() & ~areas.geometry.is_empty]
                try:
                    areasIN = areas.overlay(outlines, how='intersection')
                    if {'_SCRAP_ID', '_ID'}.issubset(areasIN.columns):
                        areasIN = areasIN[areasIN['_SCRAP_ID'] == areasIN['_ID']]
                except Exception as e_ov:
                    self._tlog(f"   ⚠ Intersection aires : {e_ov} — aires brutes conservées")
                    areasIN = areas
                areas_gpkg = outputs_path + 'areas2dMasked.gpkg'
                areasIN.to_file(areas_gpkg, driver='GPKG')
                self._tlog(f"   ✔ areas2dMasked.gpkg : {len(areasIN)} entité(s)")
            except Exception as e:
                self._tlog(f"   ⚠ Aires ignorées : {e}")
        else:
            self._tlog("   Pas d'areas2d.shp — étape ignorée.")

        # ── 4 : Points + altitudes ────────────────────────────────────────
        self._tlog("4/6 Points et stations (ajout altitude)…")
        self.progressTherion.setValue(50)
        for fname in ['points2d', 'stations3d']:
            shp_file = shp_path + fname + '.shp'
            if not os.path.isfile(shp_file):
                self._tlog(f"   {fname}.shp absent — ignoré.")
                continue
            try:
                pts_src = shp_file
                if repair_geom:
                    pts_src = self._fix_shp(pts_src)
                if isinstance(pts_src, QgsVectorLayer):
                    tmp_pts = os.path.join(outputs_path, f'_tmp_{fname}.gpkg')
                    processing.run("native:savefeatures",
                        {'INPUT': pts_src, 'OUTPUT': tmp_pts})
                    pts_src = tmp_pts
                gdf = gpd.read_file(pts_src)
                gdf = gdf[gdf.geometry.notnull() & ~gdf.geometry.is_empty]
                if add_alt:
                    gdf['_ALT']      = gdf.geometry.apply(
                        lambda g: str(round(g.z)) if (g is not None and g.has_z) else '')
                    gdf['_EASTING']  = gdf.geometry.apply(
                        lambda g: g.x if g is not None else None)
                    gdf['_NORTHING'] = gdf.geometry.apply(
                        lambda g: g.y if g is not None else None)
                out_gpkg = outputs_path + fname + 'Alt.gpkg'
                gdf.to_file(out_gpkg, driver='GPKG')
                gpkg_map[fname] = out_gpkg
                self._tlog(f"   ✔ {fname}Alt.gpkg : {len(gdf)} entité(s)")
            except Exception as e:
                self._tlog(f"   ⚠ {fname} : {e}")

        # ── 5 : shots3d + outline2d + walls3d ────────────────────────────
        self._tlog("5/6 Conversion shots3d, outline, walls3d…")
        self.progressTherion.setValue(70)
        for fname in ['shots3d', 'outline2d']:
            shp_file = shp_path + fname + '.shp'
            if not os.path.isfile(shp_file):
                continue
            try:
                vec_src = shp_file
                if repair_geom:
                    vec_src = self._fix_shp(vec_src)
                if isinstance(vec_src, QgsVectorLayer):
                    tmp_vec = os.path.join(outputs_path, f'_tmp_{fname}.gpkg')
                    processing.run("native:savefeatures",
                        {'INPUT': vec_src, 'OUTPUT': tmp_vec})
                    vec_src = tmp_vec
                gdf = gpd.read_file(vec_src)
                gdf = gdf[gdf.geometry.notnull() & ~gdf.geometry.is_empty]
                out_gpkg = outputs_path + fname + '.gpkg'
                gdf.to_file(out_gpkg, driver='GPKG')
                gpkg_map[fname] = out_gpkg
                self._tlog(f"   ✔ {fname}.gpkg : {len(gdf)} entité(s)")
            except Exception as e:
                self._tlog(f"   ⚠ {fname} : {e}")

        # walls3d : copie SHP (format maillage 3D non supporté en GPKG)
        walls_dest = None
        if os.path.isfile(shp_path + 'walls3d.shp'):
            import shutil
            for ext in ['.shp', '.dbf', '.prj', '.shx']:
                src = shp_path + 'walls3d' + ext
                if os.path.isfile(src):
                    shutil.copy2(src, outputs_path + 'walls3d' + ext)
            walls_dest = outputs_path + 'walls3d.shp'
            gpkg_map['walls3d'] = walls_dest
            self._tlog("   ✔ walls3d.shp copié (maillage 3D → SHP conservé)")

        # ── 6 : Import QGIS ───────────────────────────────────────────────
        self._tlog("6/6 Import dans QGIS avec styles…")
        self.progressTherion.setValue(85)

        root  = QgsProject.instance().layerTreeRoot()
        group = None
        if use_group:
            group = root.findGroup(group_name) or root.addGroup(group_name)

        def _subgroup(parent, name):
            if parent is None:
                return None
            sg = parent.findGroup(name)
            return sg or parent.addGroup(name)

        grp2d = _subgroup(group, "2D")
        grp3d = _subgroup(group, "3D")

        def _load(path, name, subgrp, qml=None):
            if not path or not os.path.isfile(path):
                return None
            lyr = QgsVectorLayer(path, name, "ogr")
            if not lyr.isValid():
                self._tlog(f"   ⚠ Invalide : {path}")
                return None
            if qml and os.path.isfile(qml):
                lyr.loadNamedStyle(qml)
                self._tlog(f"   🎨 Style {os.path.basename(qml)} → {name}")
            QgsProject.instance().addMapLayer(lyr, False)
            (subgrp or root).addLayer(lyr)
            return lyr

        # ── Ordre d'affichage 2D ─────────────────────────────────────────
        # Ordre visuel (haut → bas dans le panneau) : Points · Lignes · Aires · Outline
        # QGIS place chaque couche ajoutée EN HAUT du groupe
        # → on ajoute dans l'ordre INVERSE : Outline, Aires, Lignes, Points

        # Outline — fond, tout en bas
        _load(gpkg_map.get('outline2d', outputs_path + 'outline2d.gpkg'),
              "Outline 2D", grp2d, styles.get('outline2d'))

        # Aires — au-dessus de l'outline
        if areas_gpkg:
            _load(areas_gpkg, "Aires 2D", grp2d, styles['areas2d'])

        # Lignes — au-dessus des aires
        if lines_gpkg:
            _load(lines_gpkg, "Lignes 2D", grp2d, styles['lines2d'])

        # Points — tout en haut du groupe 2D
        _load(gpkg_map.get('points2d', outputs_path + 'points2dAlt.gpkg'),
              "Points 2D", grp2d, styles['points2d'])

        # ── 3D (Parois en bas, Cheminements, Stations en haut) ───────────
        if walls_dest:
            _load(walls_dest, "Parois 3D", grp3d, styles['walls3d'])
        _load(gpkg_map.get('shots3d', outputs_path + 'shots3d.gpkg'),
              "Cheminements 3D", grp3d, styles['shots3d'])
        _load(gpkg_map.get('stations3d', outputs_path + 'stations3dAlt.gpkg'),
              "Stations 3D", grp3d, styles['stations3d'])

        self.progressTherion.setValue(100)
        self._tlog("✅ Import Therion terminé.")

        # Nettoyage des fichiers temporaires intermédiaires
        for tmp_name in ['_tmp_outline.gpkg', '_tmp_lines.gpkg', '_tmp_areas.gpkg',
                          '_tmp_points2d.gpkg', '_tmp_stations3d.gpkg',
                          '_tmp_shots3d.gpkg', '_tmp_outline2d.gpkg']:
            tmp_path = os.path.join(outputs_path, tmp_name)
            if os.path.isfile(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

        QtWidgets.QMessageBox.information(
            self, "Import Therion terminé",
            f"Couches importées dans le groupe « {group_name} ».\n"
            f"GPKG dans : {outputs_path}")


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

        # Action secondaire : vérifier / réinstaller les dépendances
        self.action_deps = QAction(
            "🔧 SpeleoTools — Vérifier les dépendances",
            self.iface.mainWindow()
        )
        self.action_deps.triggered.connect(self.check_dependencies)

        # Ajouter au menu "Extensions"
        self.iface.addPluginToMenu("&SpeleoTools", self.action)
        self.iface.addPluginToMenu("&SpeleoTools", self.action_deps)

        # Ajouter à la toolbar
        self.iface.addToolBarIcon(self.action)

    def unload(self):
        """Supprime le plugin de QGIS"""
        if self.action:
            self.iface.removePluginMenu("&SpeleoTools", self.action)
            self.iface.removeToolBarIcon(self.action)
            self.action = None
        if hasattr(self, 'action_deps') and self.action_deps:
            self.iface.removePluginMenu("&SpeleoTools", self.action_deps)
            self.action_deps = None

    def check_dependencies(self):
        """Lance la vérification/installation manuelle des dépendances."""
        try:
            from .install_dependencies import check_and_install
            check_and_install(
                parent_widget=self.iface.mainWindow(),
                silent_if_ok=False     # toujours afficher le résumé
            )
        except Exception as e:
            from qgis.PyQt import QtWidgets
            QtWidgets.QMessageBox.critical(
                self.iface.mainWindow(),
                "Erreur",
                f"Impossible de vérifier les dépendances :\n{e}"
            )

    def run(self):
        """Ouvre la fenêtre du plugin"""
        if not self.dialog:
            self.dialog = SpeleoToolsDialog(self.iface.mainWindow())
        self.dialog.show()
        self.dialog.raise_()
        self.dialog.activateWindow()    # ======================================================================
    # --- ONGLET 3 : PROFILS ---

    def _plog(self, msg):
        self.textLogProfile.append(msg)
        QtWidgets.QApplication.processEvents()

    def _toggle_alpha_source(self):
        from_file = self.radioAngleThconfig.isChecked()
        self.editThconfigPath.setEnabled(from_file)
        self.btnBrowseThconfig.setEnabled(from_file)
        self.btnReadThconfig.setEnabled(from_file)
        self.spinAlpha.setEnabled(not from_file)

    def _browse_thconfig(self):
        f, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Choisir un fichier .thconfig",
            os.path.expanduser("~"), "Therion config (*.thconfig *.th);;Tous (*)")
        if f:
            self.editThconfigPath.setText(f)

    def _read_alpha_from_thconfig(self):
        import re
        path = self.editThconfigPath.text().strip()
        if not path or not os.path.isfile(path):
            QtWidgets.QMessageBox.warning(self, "Fichier introuvable",
                "Selectionnez d'abord un fichier .thconfig valide.")
            return
        try:
            with open(path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
            match = re.search(r'-projection\s+\[\s*elevation\s+([\d.]+)\s*\]', content)
            if match:
                alpha = float(match.group(1))
                self.spinAlpha.setValue(alpha)
                self._plog("alpha = " + str(alpha) + " deg lu depuis " + os.path.basename(path))
                self.radioAngleManual.setChecked(True)
            else:
                QtWidgets.QMessageBox.warning(self, "Angle introuvable",
                    "Aucune ligne '-projection [elevation XX]' trouvee dans le fichier.")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Erreur lecture", str(e))

    def _profile_output_dir(self):
        d = self.editProfileOutputDir.text().strip() or tempfile.gettempdir()
        os.makedirs(d, exist_ok=True)
        return d

    def _export_profile_csv_png(self, distances, elevations, name, out_dir,
                                  offset_x=0.0, offset_y=0.0):
        import math, csv as _csv
        xs = [d + offset_x for d in distances]
        ys = [e + offset_y if (e is not None and not math.isnan(e)) else float('nan')
              for e in elevations]

        csv_path = os.path.join(out_dir, name + ".csv")
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            w = _csv.writer(csvfile)
            w.writerow(["X_distance_m", "Y_altitude_m"])
            for x, y in zip(xs, ys):
                w.writerow([round(x, 3), "" if math.isnan(y) else round(y, 3)])
        self._plog("CSV : " + csv_path)

        png_path = None
        try:
            import matplotlib.pyplot as plt
            px = [x for x, y in zip(xs, ys) if not math.isnan(y)]
            py = [y for y in ys if not math.isnan(y)]
            if px:
                fig, ax = plt.subplots(figsize=(14, 5))
                ax.plot(px, py, '-b', linewidth=1.2)
                ax.fill_between(px, py, min(py), alpha=0.12, color='steelblue')
                ax.set_title(name, fontsize=12)
                ax.set_xlabel("Distance (m)")
                ax.set_ylabel("Altitude (m)")
                ax.grid(True, linestyle='--', alpha=0.4)
                fig.tight_layout()
                png_path = os.path.join(out_dir, name + ".png")
                fig.savefig(png_path, dpi=200, bbox_inches='tight')
                plt.close(fig)
                self._plog("PNG : " + png_path)
        except ImportError:
            self._plog("[INFO] matplotlib absent - PNG ignore.")
        return csv_path, png_path

    # ----------------------------------------------------------------
    # Cas 1 : Profil projete
    # ----------------------------------------------------------------
    def run_projected_profile(self):
        import math
        dem_layer     = self.get_layer_by_combo(self.comboDEM2)
        emprise_layer = self.get_layer_by_combo(self.comboProjEmprise)
        if not dem_layer or not dem_layer.isValid():
            QtWidgets.QMessageBox.warning(self, "Erreur", "Selectionnez un MNT valide.")
            return
        if not emprise_layer or not emprise_layer.isValid():
            QtWidgets.QMessageBox.warning(self, "Erreur", "Selectionnez une couche d'emprise valide.")
            return

        alpha      = float(self.spinAlpha.value())
        margin_pct = float(self.spinProfileMargin.value()) / 100.0
        offset_x   = float(self.spinProjOffsetX.value())
        offset_y   = float(self.spinProjOffsetY.value())
        out_dir    = self._profile_output_dir()

        self.textLogProfile.clear()
        self._plog("Profil projete alpha=" + str(alpha) + " deg - emprise : " + emprise_layer.name())

        try:
            from qgis.core import QgsRectangle, QgsPointXY

            bbox = emprise_layer.extent()
            w = bbox.width()
            h = bbox.height()
            margin = max(w, h) * margin_pct
            bbox_exp = QgsRectangle(
                bbox.xMinimum() - margin, bbox.yMinimum() - margin,
                bbox.xMaximum() + margin, bbox.yMaximum() + margin)

            cx = (bbox_exp.xMinimum() + bbox_exp.xMaximum()) / 2.0
            cy = (bbox_exp.yMinimum() + bbox_exp.yMaximum()) / 2.0
            half_diag = math.hypot(bbox_exp.width(), bbox_exp.height()) / 2.0
            self._plog("Centre : (" + str(round(cx,1)) + ", " + str(round(cy,1)) + ")")

            cut_azimuth = (alpha + 90.0) % 360.0
            rad = math.radians(cut_azimuth)
            dx = math.sin(rad)
            dy = math.cos(rad)

            x0 = cx - dx * half_diag
            y0 = cy - dy * half_diag
            x1 = cx + dx * half_diag
            y1 = cy + dy * half_diag
            self._plog("Ligne de coupe : (" + str(round(x0,1)) + "," + str(round(y0,1)) +
                       ") -> (" + str(round(x1,1)) + "," + str(round(y1,1)) + ")")

            dem_res   = (dem_layer.rasterUnitsPerPixelX() + dem_layer.rasterUnitsPerPixelY()) / 2.0
            line_len  = math.hypot(x1-x0, y1-y0)
            n_pts     = max(int(line_len / max(dem_res, 0.01)), 2)

            distances  = []
            elevations = []
            for i in range(n_pts + 1):
                t  = i / n_pts
                px = x0 + t * (x1 - x0)
                py = y0 + t * (y1 - y0)
                try:
                    z = sample_dem_at_point(dem_layer, QgsPointXY(px, py))
                except Exception:
                    z = float('nan')
                distances.append(t * line_len)
                elevations.append(z if z is not None else float('nan'))

            valid = sum(1 for e in elevations if e is not None and not math.isnan(e))
            self._plog(str(n_pts+1) + " points, " + str(valid) + " valides.")

            name = ("profil_projete_a" + str(int(alpha)) + "deg_" +
                    self._safe_name(dem_layer.name()))
            csv_p, png_p = self._export_profile_csv_png(
                distances, elevations, name, out_dir, offset_x, offset_y)

            msg = "Profil projete genere.\nCSV : " + csv_p
            if png_p:
                msg += "\nPNG : " + png_p
            QtWidgets.QMessageBox.information(self, "Termine", msg)

        except Exception as e:
            self._plog("[ERROR] " + str(e))
            QtWidgets.QMessageBox.critical(self, "Erreur", str(e))

    # ----------------------------------------------------------------
    # Cas 2 : Profil developpe
    # ----------------------------------------------------------------
    def run_developed_profile(self):
        import math
        dem_layer     = self.get_layer_by_combo(self.comboDEM2)
        profile_layer = self.get_layer_by_combo(self.comboProfileLayer)
        if not dem_layer or not dem_layer.isValid():
            QtWidgets.QMessageBox.warning(self, "Erreur", "Selectionnez un MNT valide.")
            return
        if not profile_layer or not profile_layer.isValid():
            QtWidgets.QMessageBox.warning(self, "Erreur", "Selectionnez une polyligne valide.")
            return

        spacing   = float(self.doubleSpinBoxSpacing.value())
        do_interp = self.checkBoxInterpolate.isChecked()
        max_gap   = (float(self.doubleSpinBoxMaxGap.value())
                     if self.checkBoxMaxGap.isChecked() else None)
        offset_x  = float(self.spinDevOffsetX.value())
        offset_y  = float(self.spinDevOffsetY.value())
        out_dir   = self._profile_output_dir()

        self.textLogProfile.clear()
        self._plog("Profil developpe - " + profile_layer.name() + " / " + dem_layer.name())

        try:
            from qgis.core import QgsPointXY, QgsCoordinateTransform, QgsProject

            dem_crs    = dem_layer.crs()
            prof_crs   = profile_layer.crs()
            need_xform = (prof_crs != dem_crs)
            xform = (QgsCoordinateTransform(prof_crs, dem_crs, QgsProject.instance())
                     if need_xform else None)

            distances  = []
            elevations = []
            cum_dist   = 0.0
            prev_xy    = None

            for feat in profile_layer.getFeatures():
                geom = feat.geometry()
                if not geom or geom.isEmpty():
                    continue
                polylines = (geom.asMultiPolyline() if geom.isMultipart()
                             else [geom.asPolyline()])

                for poly in polylines:
                    if len(poly) < 2:
                        continue
                    dense_pts = []
                    for i in range(len(poly) - 1):
                        try:
                            ax, ay = poly[i].x(), poly[i].y()
                            bx, by = poly[i+1].x(), poly[i+1].y()
                        except Exception:
                            ax, ay = float(poly[i][0]), float(poly[i][1])
                            bx, by = float(poly[i+1][0]), float(poly[i+1][1])
                        seg_len = math.hypot(bx-ax, by-ay)
                        if seg_len == 0:
                            continue
                        n_sub = max(1, int(seg_len / spacing))
                        for k in range(n_sub):
                            t = k / n_sub
                            dense_pts.append((ax + t*(bx-ax), ay + t*(by-ay)))
                    try:
                        dense_pts.append((poly[-1].x(), poly[-1].y()))
                    except Exception:
                        dense_pts.append((float(poly[-1][0]), float(poly[-1][1])))

                    for px, py in dense_pts:
                        seg = (math.hypot(px - prev_xy[0], py - prev_xy[1])
                               if prev_xy else 0.0)
                        cum_dist += seg
                        prev_xy   = (px, py)

                        sample_pt = QgsPointXY(px, py)
                        if xform:
                            try:
                                t2 = xform.transform(sample_pt)
                                sample_pt = QgsPointXY(t2.x(), t2.y())
                            except Exception:
                                pass
                        try:
                            z = sample_dem_at_point(dem_layer, sample_pt)
                        except Exception:
                            z = float('nan')

                        distances.append(cum_dist)
                        elevations.append(z if z is not None else float('nan'))

            if not distances:
                QtWidgets.QMessageBox.warning(self, "Erreur",
                    "Aucun point extrait. Verifiez la couche polyligne.")
                return

            if do_interp:
                import numpy as np
                arr  = np.array(elevations, dtype=float)
                nans = np.isnan(arr)
                if nans.any() and not nans.all():
                    idx      = np.arange(len(arr))
                    arr_fill = np.interp(idx[nans], idx[~nans], arr[~nans])
                    if max_gap is not None:
                        dist_arr = np.array(distances)
                        for j, ni in enumerate(np.where(nans)[0]):
                            left  = np.searchsorted(idx[~nans], ni, 'left') - 1
                            right = left + 1
                            if left >= 0 and right < len(idx[~nans]):
                                gap = dist_arr[idx[~nans][right]] - dist_arr[idx[~nans][left]]
                                if gap > max_gap:
                                    arr_fill[j] = float('nan')
                        arr[nans] = arr_fill
                    else:
                        arr[nans] = arr_fill
                elevations = arr.tolist()

            valid = sum(1 for e in elevations if e is not None and not math.isnan(e))
            self._plog(str(len(distances)) + " points, " + str(valid) + " valides.")

            name = ("profil_dev_" + self._safe_name(profile_layer.name()) +
                    "_" + self._safe_name(dem_layer.name()))
            csv_p, png_p = self._export_profile_csv_png(
                distances, elevations, name, out_dir, offset_x, offset_y)

            msg = "Profil developpe genere.\nCSV : " + csv_p
            if png_p:
                msg += "\nPNG : " + png_p
            QtWidgets.QMessageBox.information(self, "Termine", msg)

        except Exception as e:
            self._plog("[ERROR] " + str(e))
            QtWidgets.QMessageBox.critical(self, "Erreur", str(e))


# -*- coding: utf-8 -*-
"""
SpeleoTools Plugin for QGIS 3
Auteur : Urruty Benoit
Description : Interface complète à 4 onglets pour outils spéléo.
"""
import csv
import math
import tempfile
import os
import processing
from qgis.PyQt import QtWidgets, uic, QtCore
from qgis.PyQt.QtCore import Qt
from qgis.core import (
    QgsProject, QgsRasterLayer, QgsVectorLayer, QgsPoint, QgsPointXY,
    QgsFeature, QgsFields, QgsField, QgsWkbTypes, QgsGeometry,
    QgsFeatureSink, QgsDistanceArea, QgsCoordinateTransformContext,
    QgsFeatureRequest, QgsMessageLog, Qgis, QgsVectorFileWriter,
    QgsCoordinateTransform
)
from PyQt5.QtCore import QVariant

import numpy as np
import heapq

from .speleo_utils import *
from .install_dependencies import requires

# Charger l'interface .ui
FORM_CLASS, _ = uic.loadUiType(os.path.join(os.path.dirname(__file__), 'speleo_dialog.ui'))

