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

from .speleo_utils import compute_thickness, sample_raster_at_point, layer_feature_elevation, run_prospection_real, create_profile_from_line

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
        self.btnRunProspect.clicked.connect(self.run_prospection)

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

        for layer in QgsProject.instance().mapLayers().values():
            if isinstance(layer, QgsRasterLayer):
                self.comboDEM.addItem(layer.name())
                self.comboDEM2.addItem(layer.name())
                self.comboProspectDEM.addItem(layer.name())
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

    def import_data(self):
        """Import d'un fichier shapefile, Therion, etc."""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Importer une donnée spéléo", "", 
            "Fichiers SIG (*.shp *.geojson *.dxf *.th *.th2 *.3d);;Tous les fichiers (*)"
        )
        if not path:
            return
        name = os.path.basename(path)
        layer = None

        if path.lower().endswith(".shp"):
            layer = QgsVectorLayer(path, name, "ogr")
        elif path.lower().endswith(".geojson"):
            layer = QgsVectorLayer(path, name, "ogr")
        elif path.lower().endswith((".th", ".th2", ".3d")):
            QtWidgets.QMessageBox.information(self, "Import Therion", "Conversion Therion non encore implémentée.")
            return
        else:
            QtWidgets.QMessageBox.warning(self, "Type non pris en charge", "Ce format n'est pas reconnu.")
            return

        if layer and layer.isValid():
            QgsProject.instance().addMapLayer(layer)
            self.populate_layers()
            self.log(f"Couche importée : {name}")
        else:
            QtWidgets.QMessageBox.warning(self, "Erreur", "Impossible de charger le fichier.")

    def apply_style(self):
        """Applique un style symbolique de base"""
        style_name = self.comboStyle.currentText()
        self.log(f"Application du style '{style_name}' (non encore implémenté).")
        QtWidgets.QMessageBox.information(self, "Style", f"Le style '{style_name}' sera appliqué prochainement.")

    # ======================================================================
    # --- ONGLET 2 : ÉPAISSEUR DE ROCHE ---

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
    def generate_profile_with_interpolation_and_export(self):
        """Génère un profil développé + MNT interpolé, trace le graphique et exporte en CSV + PNG."""
        # Récupérer les couches sélectionnées
        dem_name = self.comboDEM2.currentText()
        profile_layer_name = self.comboProfileLayer.currentText()

        if not dem_name or not profile_layer_name:
            QtWidgets.QMessageBox.warning(self, "Erreur", "Veuillez sélectionner un MNT et une couche de profil.")
            return

        dem_layer = self.get_layer_by_name(dem_name)
        profile_layer = self.get_layer_by_name(profile_layer_name)

        if dem_layer is None or profile_layer is None:
            QtWidgets.QMessageBox.warning(self, "Erreur", "Impossible de trouver les couches sélectionnées.")
            return

        # Paramètres de l'utilisateur
        spacing = self.doubleSpinBoxSpacing.value()
        interp = self.checkBoxInterpolate.isChecked()
        max_gap_distance = self.doubleSpinBoxMaxGap.value() if self.checkBoxMaxGap.isChecked() else None

        # Chemin de sortie pour les exports
        output_dir = QtWidgets.QFileDialog.getExistingDirectory(self, "Choisir un dossier de sortie")
        if not output_dir:
            return

        try:
            self.log(f"Début de la génération du profil pour '{profile_layer_name}' avec le MNT '{dem_name}'...")

            # Générer le profil 3D
            profile_3d_layer = create_profile_from_line(
                dem_layer, profile_layer,
                spacing=spacing,
                interp=interp,
                max_gap_distance=max_gap_distance,
                add_to_project=True
            )

            if profile_3d_layer is None or profile_3d_layer.featureCount() == 0:
                QtWidgets.QMessageBox.warning(self, "Erreur", "Aucun profil n'a pu être généré.")
                return

            # Extraire les données pour le graphique et le CSV
            distances = []
            elevations = []
            for feature in profile_3d_layer.getFeatures():
                geom = feature.geometry()
                if geom.isEmpty():
                    continue
                # Calculer la distance cumulée le long de la ligne
                length = geom.length()
                points = geom.asPolyline()
                for i, point in enumerate(points):
                    # Distance cumulée depuis le début de la ligne
                    if i == 0:
                        dist = 0.0
                    else:
                        prev_point = points[i-1]
                        dx = point.x() - prev_point.x()
                        dy = point.y() - prev_point.y()
                        dist = distances[-1] + math.hypot(dx, dy)
                    distances.append(dist)
                    elevations.append(point.z())

            # Générer le graphique avec matplotlib
            plt.figure(figsize=(10, 6))
            plt.plot(distances, elevations, 'b-', label="Profil développé")
            plt.title(f"Profil développé - {profile_layer_name}")
            plt.xlabel("Distance (m)")
            plt.ylabel("Altitude (m)")
            plt.grid(True)
            plt.legend()

            # Exporter le graphique en PNG
            png_path = os.path.join(output_dir, f"profil_{profile_layer_name}.png")
            plt.savefig(png_path, dpi=300, bbox_inches='tight')
            plt.close()

            # Exporter les données en CSV
            csv_path = os.path.join(output_dir, f"profil_{profile_layer_name}.csv")
            with open(csv_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["Distance (m)", "Altitude (m)"])
                for dist, elev in zip(distances, elevations):
                    writer.writerow([dist, elev])

            self.log(f"Profil généré avec succès !\n- Graphique exporté : {png_path}\n- Données exportées : {csv_path}")
            QtWidgets.QMessageBox.information(self, "Succès", f"Profil généré et exporté dans :\n{output_dir}")

        except Exception as e:
            self.log(f"Erreur lors de la génération du profil : {e}")
            QtWidgets.QMessageBox.critical(self, "Erreur", f"Une erreur est survenue :\n{e}")
    # ======================================================================
    # --- ONGLET 4 : PROSPECTION (MNT) ---

    def run_prospection(self):
        dem_name = self.comboProspectDEM.currentText()
        if not dem_name:
            QtWidgets.QMessageBox.warning(self, "Erreur", "Sélectionne un MNT.")
            return

        do_hillshade = self.chkHillshade.isChecked()
        do_slope = self.chkSlope.isChecked()
        do_low = self.chkLowPoints.isChecked()

        msg = f"Analyse MNT '{dem_name}':\n"
        msg += f" - Hillshade : {'Oui' if do_hillshade else 'Non'}\n"
        msg += f" - Pente : {'Oui' if do_slope else 'Non'}\n"
        msg += f" - Points bas : {'Oui' if do_low else 'Non'}\n"
        self.log(msg)
        self.progressProspect.setValue(100)
        QtWidgets.QMessageBox.information(self, "Prospection", "Analyse MNT terminée (simulation).")


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