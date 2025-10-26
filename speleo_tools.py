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
from qgis.PyQt import QtWidgets, uic
from qgis.PyQt.QtCore import Qt
from qgis.core import QgsProject, QgsRasterLayer, QgsVectorLayer
import processing
from qgis.PyQt import QtCore
from qgis.core import (
    QgsPointXY, QgsFeature, QgsFields, QgsField, QgsWkbTypes, QgsVectorLayer,
    QgsProject, QgsGeometry, QgsFeatureSink, QgsDistanceArea, QgsCoordinateTransformContext,QgsFeatureRequest
)
from PyQt5.QtCore import QVariant

from .speleo_utils import compute_thickness, sample_raster_at_point, layer_feature_elevation, compute_thickness, generate_profile_points, run_prospection_real

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

        if not dem_name or not cave_name:
            QtWidgets.QMessageBox.warning(self, "Erreur", "Sélectionne un DEM et une couche de cavité.")
            return

        raster = self.get_layer_by_name(dem_name)
        vec = self.get_layer_by_name(cave_name)

        if raster is None or vec is None:
            QtWidgets.QMessageBox.warning(self, "Erreur", "Impossible de trouver les couches sélectionnées.")
            return

        try:
            self.log(f"Début calcul d'épaisseur entre '{cave_name}' et '{dem_name}'...")
            self.progressThickness.setValue(10)
            QtWidgets.QApplication.processEvents()

            out = out_path if out_path else None
            mem = self.compute_thickness(raster, vec, out_path=out)

            self.progressThickness.setValue(80)
            QtWidgets.QApplication.processEvents()
            self.progressThickness.setValue(100)

            self.log(f"Résultat disponible : {mem.name()}")
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
        """
        À partir d'une polyligne 3D existante -> crée profil développé + MNT interpolé,
        trace le graphique et exporte CSV + PNG.

        - utilise self.sample_raster_at_point(dem_layer, QgsPointXY) pour lire le DEM
        - si valeur DEM manquante -> interpolation locale moyenne sur une fenêtre (nxn)
        - export CSV et PNG dans le dossier projet (si défini) ou dossier temporaire
        Retourne la liste des tuples (pts_layer, profile_layer, mnt_layer, csv_path, png_path)
            """
        layer_name = self.comboProfileLayer.currentText()
        dem_name = self.comboDEM2.currentText()
        if not layer_name or not dem_name:
            QtWidgets.QMessageBox.warning(self, "Erreur", "Sélectionne une polyligne 3D et un DEM.")
            return

        project = QgsProject.instance()
        line_layers = project.mapLayersByName(layer_name)
        dem_layers = project.mapLayersByName(dem_name)
        if not line_layers:
            QtWidgets.QMessageBox.warning(self, "Erreur", f"Couche vecteur '{layer_name}' introuvable.")
            return
        if not dem_layers:
            QtWidgets.QMessageBox.warning(self, "Erreur", f"Raster '{dem_name}' introuvable.")
            return

        line_layer = line_layers[0]
        dem_layer = dem_layers[0]

        # Vérifs basiques
        if QgsWkbTypes.geometryType(line_layer.wkbType()) != QgsWkbTypes.LineGeometry:
            QtWidgets.QMessageBox.warning(self, "Erreur", "La couche choisie ne contient pas de géométries linéaires.")
            return
        if not hasattr(dem_layer, 'rasterUnitsPerPixelX'):
            QtWidgets.QMessageBox.warning(self, "Erreur", "Le DEM sélectionné ne semble pas être un raster valide.")
            return

        # spacing par défaut = résolution X du raster (sinon 1.0)
        try:
            res_x = dem_layer.rasterUnitsPerPixelX()
            spacing_default = res_x if (res_x and res_x > 0) else 1.0
        except Exception:
            spacing_default = 1.0

        # fonction utilitaire : échantillonne DEM, et si None -> moyenne fenêtre (size = odd integer)
        def sample_dem_with_window(dem_lyr, point_xy, window=3):
            """
            window: taille en pixels (3 -> 3x3). Doit être impair >=1.
            Retourne float ou None si aucune valeur trouvée.
            """
            # tentative directe
            val = None
            try:
                val = self.sample_raster_at_point(dem_lyr, point_xy)
            except Exception:
                val = None

            if val is not None:
                try:
                    return float(val)
                except Exception:
                    return None

            # si pas de valeur, appliquer moyenne sur fenêtre autour du point
            try:
                # résolution raster
                px = dem_lyr.rasterUnitsPerPixelX()
                py = dem_lyr.rasterUnitsPerPixelY()
                if px is None or py is None:
                    return None
            except Exception:
                return None

            half = int((window - 1) / 2)
            values = []
            # parcourir la grille de pixels centrée sur point (en coordonnées)
            for i in range(-half, half + 1):
                for j in range(-half, half + 1):
                    # créer point décalé d'i*px en X et j*py en Y
                    x_offset = point_xy.x() + (i * px)
                    y_offset = point_xy.y() + (j * py)
                    try:
                        v = self.sample_raster_at_point(dem_lyr, QgsPointXY(x_offset, y_offset))
                        if v is not None:
                            # vérifier que c'est numérique
                            try:
                                fv = float(v)
                                # filtrer valeurs non valides (NaN, extremes improbables)
                                if not math.isnan(fv) and math.isfinite(fv):
                                    values.append(fv)
                            except Exception:
                                continue
                    except Exception:
                        continue
            if not values:
                return None
            # moyenne arithmétique (simple)
            return float(sum(values) / len(values))

        # récupération des entités (sélection si présente sinon toutes)
        sel_ids = line_layer.selectedFeatureIds()
        if sel_ids:
            feats = [f for f in line_layer.getFeatures(QgsFeatureRequest().setFilterFids(sel_ids))]
            self.log(f"{len(feats)} entité(s) sélectionnée(s) dans {layer_name}.")
        else:
            feats = list(line_layer.getFeatures())
            self.log(f"Aucune sélection : {len(feats)} entité(s) dans {layer_name} traitées.")

        if not feats:
            QtWidgets.QMessageBox.information(self, "Profil", "Aucune entité à traiter.")
            return

        exports = []  # résultat à retourner

        # dossier d'export : dossier du projet si disponible sinon dossier temporaire
        proj_dir = project.homePath()
        if proj_dir and os.path.isdir(proj_dir):
            export_dir = proj_dir
        else:
            export_dir = tempfile.gettempdir()

        for feat in feats:
            try:
                geom = feat.geometry()
                if geom is None or geom.isEmpty():
                    self.log(f"Entité {feat.id()} vide -> ignorée.")
                    continue
                length = geom.length()
                if length <= 0:
                    self.log(f"Entité {feat.id()} longueur nulle -> ignorée.")
                    continue

                spacing = spacing_default

                dist = 0.0
                list_dist = []
                list_z_poly = []
                list_z_dem = []

                while dist <= length + 1e-6:
                    interp = geom.interpolate(dist)
                    if interp is None or interp.isEmpty():
                        dist += spacing
                        continue
                    p = interp.asPoint()

                    # récupérer Z de la polyligne (priorité à p.z())
                    z_poly = None
                    try:
                        if hasattr(p, 'z'):
                            z_poly = p.z()
                    except Exception:
                        z_poly = None

                    # si pas de Z, tenter quelques heuristiques (vertex proche, attribut...)
                    if z_poly is None:
                        # parcours des vertices pour trouver Z si présent
                        try:
                            for v in geom.vertices():
                                try:
                                    if hasattr(v, 'z') and v.z() is not None:
                                        if abs(v.x() - p.x()) < 1e-6 and abs(v.y() - p.y()) < 1e-6:
                                            z_poly = v.z()
                                            break
                                except Exception:
                                    continue
                        except Exception:
                            pass

                    if z_poly is None:
                        # attributs courants
                        for key in ['z', 'Z', 'alt', 'elev', 'elevation']:
                            if key in feat.fields().names():
                                try:
                                    val = feat[key]
                                    if val is not None:
                                        z_poly = float(val)
                                        break
                                except Exception:
                                    continue

                    if z_poly is None:
                        z_poly = 0.0
                        self.log(f"Avertissement : pas de Z trouvé pour dist={dist:.2f} sur entité {feat.id()} -> 0.0 utilisé.")

                    # lire DEM ; si None -> essayer fenêtre 3x3 puis 5x5
                    ptxy = QgsPointXY(p.x(), p.y())
                    z_dem = sample_dem_with_window(dem_layer, ptxy, window=3)
                    if z_dem is None:
                        z_dem = sample_dem_with_window(dem_layer, ptxy, window=5)
                    # si toujours None -> NaN
                    if z_dem is None:
                        z_dem_val = float('nan')
                        self.log(f"Avertissement : pas de valeur MNT (même après interpolation locale) à dist={dist:.2f} entité {feat.id()}.")
                    else:
                        z_dem_val = float(z_dem)

                    list_dist.append(float(dist))
                    list_z_poly.append(float(z_poly))
                    list_z_dem.append(z_dem_val)

                    dist += spacing

                if len(list_dist) < 2:
                    self.log(f"Trop peu de points pour l'entité {feat.id()} -> ignorée.")
                    continue

                # --- création couche points (distance, z_poly, z_dem) ---
                points_name = f"profile_points_{line_layer.name()}_{feat.id()}"
                prov_spec = f"Point?crs={project.crs().authid()}" if project.crs().isValid() else f"Point?crs={line_layer.crs().authid()}"
                pts_layer = QgsVectorLayer(prov_spec, points_name, "memory")
                dp = pts_layer.dataProvider()
                fields = QgsFields()
                fields.append(QgsField("source_id", QVariant.Int))
                fields.append(QgsField("distance", QVariant.Double))
                fields.append(QgsField("z_poly", QVariant.Double))
                fields.append(QgsField("z_dem", QVariant.Double))
                dp.addAttributes(fields)
                pts_layer.updateFields()

                feats_to_add = []
                for d, zp, zd in zip(list_dist, list_z_poly, list_z_dem):
                    fpt = QgsFeature()
                    # géométrie (distance, z_poly) pour affichage/CSV ; CRS conceptuel
                    pt_geom = QgsGeometry.fromPointXY(QgsPointXY(d, zp))
                    fpt.setGeometry(pt_geom)
                    fpt.setFields(pts_layer.fields())
                    fpt['source_id'] = feat.id()
                    fpt['distance'] = float(d)
                    fpt['z_poly'] = float(zp)
                    try:
                        fpt['z_dem'] = float(zd)
                    except Exception:
                        fpt['z_dem'] = None
                    feats_to_add.append(fpt)
                dp.addFeatures(feats_to_add)
                pts_layer.updateExtents()
                project.addMapLayer(pts_layer)

                # --- polyligne profil (distance, z_poly) ---
                profile_name = f"profile_poly_{line_layer.name()}_{feat.id()}"
                profile_layer = QgsVectorLayer(f"LineString?crs={pts_layer.crs().authid()}", profile_name, "memory")
                dp_prof = profile_layer.dataProvider()
                fields_prof = QgsFields()
                fields_prof.append(QgsField("source_id", QVariant.Int))
                fields_prof.append(QgsField("npts", QVariant.Int))
                dp_prof.addAttributes(fields_prof)
                profile_layer.updateFields()

                poly_pts = [QgsPoint(d, z) for d, z in zip(list_dist, list_z_poly)]
                geom_profile = QgsGeometry.fromPolyline(poly_pts)
                feat_prof = QgsFeature()
                feat_prof.setGeometry(geom_profile)
                feat_prof.setFields(profile_layer.fields())
                feat_prof['source_id'] = feat.id()
                feat_prof['npts'] = len(poly_pts)
                dp_prof.addFeatures([feat_prof])
                profile_layer.updateExtents()
                project.addMapLayer(profile_layer)

                # --- polyligne MNT (distance, z_dem) : découpage en segments continus si NaN ---
                mnt_name = f"profile_mnt_{line_layer.name()}_{feat.id()}"
                mnt_layer = QgsVectorLayer(f"LineString?crs={pts_layer.crs().authid()}", mnt_name, "memory")
                dp_mnt = mnt_layer.dataProvider()
                fields_mnt = QgsFields()
                fields_mnt.append(QgsField("source_id", QVariant.Int))
                fields_mnt.append(QgsField("npts", QVariant.Int))
                dp_mnt.addAttributes(fields_mnt)
                mnt_layer.updateFields()

                # découper en segments continus
                segments = []
                cur = []
                for d, zd in zip(list_dist, list_z_dem):
                    if zd is None or (isinstance(zd, float) and (zd != zd)):  # NaN check
                        if cur:
                            segments.append(cur)
                            cur = []
                    else:
                        cur.append(QgsPoint(d, float(zd)))
                if cur:
                    segments.append(cur)

                feats_mnt = []
                for seg in segments:
                    if len(seg) < 2:
                        continue
                    fseg = QgsFeature()
                    fseg.setGeometry(QgsGeometry.fromPolyline(seg))
                    fseg.setFields(mnt_layer.fields())
                    fseg['source_id'] = feat.id()
                    fseg['npts'] = len(seg)
                    feats_mnt.append(fseg)
                if feats_mnt:
                    dp_mnt.addFeatures(feats_mnt)
                    mnt_layer.updateExtents()
                    project.addMapLayer(mnt_layer)
                else:
                    mnt_layer = None
                    self.log(f"Aucun segment MNT valide pour entité {feat.id()} (trop de valeurs manquantes).")

                # --- Export CSV ---
                safe_layer_name = "".join(c if c.isalnum() or c in "_-" else "_" for c in line_layer.name())
                csv_name = f"profile_{safe_layer_name}_{feat.id()}.csv"
                csv_path = os.path.join(export_dir, csv_name)
                try:
                    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow(['source_id', 'distance', 'z_poly', 'z_dem'])
                        for d, zp, zd in zip(list_dist, list_z_poly, list_z_dem):
                            # écrire NaN pour les valeurs manquantes
                            if zd is None or (isinstance(zd, float) and (zd != zd)):
                                zd_write = ''
                            else:
                                zd_write = f"{zd:.6f}"
                            writer.writerow([feat.id(), f"{d:.3f}", f"{zp:.6f}", zd_write])
                except Exception as e:
                    self.log(f"Erreur export CSV pour entité {feat.id()}: {e}")
                    csv_path = None

                # --- Graphique matplotlib (profil polyligne + MNT) ---
                png_name = f"profile_{safe_layer_name}_{feat.id()}.png"
                png_path = os.path.join(export_dir, png_name)
                try:
                    # préparer données pour tracé : gérer NaN pour MNT séparément
                    xs = list_dist
                    ys_poly = list_z_poly
                    ys_mnt = [None if (zd is None or (isinstance(zd, float) and (zd != zd))) else zd for zd in list_z_dem]

                    plt.figure(figsize=(10, 4))
                    plt.plot(xs, ys_poly, label='Polyligne 3D (profil)', linewidth=1.6)
                    # tracer MNT en ignorant None (création de segments continus)
                    # on trace segments où mnt n'est pas None
                    seg_x = []
                    seg_y = []
                    for x, y in zip(xs, ys_mnt):
                        if y is None:
                            if seg_x:
                                plt.plot(seg_x, seg_y, linestyle='--', label='MNT' if not plt.gca().get_legend_handles_labels()[1].count('MNT') else "", linewidth=1.2)
                                seg_x = []
                                seg_y = []
                        else:
                            seg_x.append(x)
                            seg_y.append(y)
                    if seg_x:
                        plt.plot(seg_x, seg_y, linestyle='--', label='MNT' if not plt.gca().get_legend_handles_labels()[1].count('MNT') else "", linewidth=1.2)

                    plt.xlabel('Distance (m)')
                    plt.ylabel('Altitude (m)')
                    plt.title(f"Profil développé - {line_layer.name()} / id {feat.id()}")
                    plt.grid(True, linestyle=':', linewidth=0.5)
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(png_path, dpi=150)
                    plt.close()
                except Exception as e:
                    self.log(f"Erreur création graphique pour entité {feat.id()}: {e}")
                    png_path = None

                # informer l'utilisateur des fichiers exportés
                msg = "Export :\n"
                if csv_path:
                    msg += f"- CSV : {csv_path}\n"
                if png_path:
                    msg += f"- PNG  : {png_path}\n"
                QtWidgets.QMessageBox.information(None, "Export profil", msg)

                self.log(f"Entité {feat.id()} traitée : {len(list_dist)} échantillons -> CSV: {csv_path}, PNG: {png_path}")
                exports.append((pts_layer, profile_layer, mnt_layer, csv_path, png_path))

            except Exception as e:
                self.log(f"Erreur lors du traitement de l'entité {feat.id()}: {e}")
                # continuer avec les autres entités

        if exports:
            QtWidgets.QMessageBox.information(None, "Profil développé", f"{len(exports)} profil(s) généré(s) et exportés.")
        else:
            QtWidgets.QMessageBox.information(None, "Profil développé", "Aucun profil généré.")

        return exports

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