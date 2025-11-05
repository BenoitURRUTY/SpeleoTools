from qgis.core import (
    QgsProject, QgsVectorLayer, QgsFields, QgsField, QgsFeature,
    QgsGeometry, QgsPointXY, QgsPoint, QgsDistanceArea, QgsMessageLog, Qgis,QgsWkbTypes,QgsVectorFileWriter,QgsApplication,QgsRasterLayer,
    QgsCoordinateTransform,
    QgsCoordinateReferenceSystem)
from PyQt5.QtCore import QVariant
from qgis.PyQt.QtCore import QMetaType

import processing
from processing.core.Processing import Processing
import os
import tempfile

# ensure processing is initialized
Processing.initialize()


import math
# ---------- UTILITAIRES ----------
def sample_raster_at_point(raster_layer, qgs_point):
    from qgis.core import QgsPointXY, QgsCoordinateTransform, QgsProject
    
    # convert 3D point en 2D
    point_xy = QgsPointXY(qgs_point.x(), qgs_point.y())
    
    # transformation CRS si nécessaire
    if raster_layer.crs() != QgsProject.instance().crs():
        transform = QgsCoordinateTransform(QgsProject.instance().crs(), raster_layer.crs(), QgsProject.instance())
        point_xy = transform.transform(point_xy)
    
    val, ok = raster_layer.dataProvider().sample(point_xy, 1)
    if ok:
        return float(val)
    else:
        return None


def layer_feature_elevation(feat):
    """
    Retourne une altitude pour une feature vectorielle :
    - si géométrie Z : retourne la Z minimale des sommets (ou moyenne)
    - sinon cherche un champ commun ('elev','z','alt','altitude')
    - si rien, retourne None
    """
    geom = feat.geometry()
    # si géométrie avec Z
    if geom.isMultipart():
        parts = geom.asMultiPolyline() if geom.type() == QgsWkbTypes.LineGeometry else None
    # On essaye d'extraire z depuis les vertices
    try:
        # récupère tous les vertices z si présents
        zs = []
        for p in geom.vertices():
            if p.z() is not None:
                zs.append(p.z())
        if zs:
            return float(min(zs))  # on prend min (profondeur)
    except Exception:
        pass

    # sinon check champs usuels
    for fld in ('elev', 'z', 'alt', 'altitude', 'depth'):
        if fld in [f.name().lower() for f in feat.fields()]:
            try:
                return float(feat.attribute(fld))
            except Exception:
                pass

    return None

# ---------- 1) ÉPAISSEUR ----------
def compute_thickness(dem_layer, cave_layer, out_path=None, layer_name="Thickness"):
    """
    Échantillonne le DEM pour chaque sommet de la couche cave_layer,
    récupère l'altitude de la cavité (géométrie z ou champ), calcule
    surface_elev - cave_elev et renvoie une couche mémoire contenant
    des points avec l'attribut 'thickness'.
    Si out_path (chemin .gpkg) fourni, sauvegarde la couche.
    """
    # Préparation couche sortie (points)
    fields = QgsFields()
    fields.append(QgsField("src_elev", QVariant.Double))
    fields.append(QgsField("cave_elev", QVariant.Double))
    fields.append(QgsField("thickness", QVariant.Double))
    fields.append(QgsField("fid_src", QVariant.Int))

    mem_layer = QgsVectorLayer("Point?crs=" + dem_layer.crs().authid(), "thickness_points", "memory")
    mem_dp = mem_layer.dataProvider()
    mem_dp.addAttributes(fields)
    mem_layer.updateFields()

    da = QgsDistanceArea()
    features_added = 0

    # Ensemble pour stocker les points déjà ajoutés (arrondis à 10 cm)
    added_points = set()

    for feat in cave_layer.getFeatures():
        geom = feat.geometry()
        geom_type = QgsWkbTypes.geometryType(geom.wkbType())

        if geom_type == QgsWkbTypes.PointGeometry:
            pts = [QgsPointXY(geom.asPoint())]
        elif geom_type == QgsWkbTypes.LineGeometry:
            pts = [QgsPointXY(v) for v in geom.vertices()]
        else:
            continue  # ignorer autres types

        # Boucle principale sur les vertices
        for p in pts:
            # Arrondir les coordonnées à 10 cm (0.1 m)
            x_rounded = round(p.x(), 2)
            y_rounded = round(p.y(), 2)

            # Si le point arrondi existe déjà, ignorer
            if (x_rounded, y_rounded) in added_points:
                continue

            # Ajouter le point arrondi à l'ensemble
            added_points.add((x_rounded, y_rounded))

            # Calcul thickness
            surf_elev = sample_raster_at_point(dem_layer, QgsPointXY(p.x(), p.y()))
            try:
                cave_elev = float(p.z()) if p.z() is not None else None
            except Exception:
                cave_elev = None

            if cave_elev is None:
                cave_elev = layer_feature_elevation(feat)
            if surf_elev is None or cave_elev is None:
                continue

            thickness = surf_elev - cave_elev
            new_feat = QgsFeature()
            new_feat.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(p.x(), p.y())))
            new_feat.setFields(mem_layer.fields())
            new_feat['src_elev'] = surf_elev
            new_feat['cave_elev'] = cave_elev
            new_feat['thickness'] = thickness
            new_feat['fid_src'] = feat.id()
            mem_dp.addFeatures([new_feat])
            features_added += 1

    mem_layer.updateExtents()

    # Sauvegarde si demandé
    if out_path:
        try:
            options = QgsVectorFileWriter.SaveVectorOptions()
            options.driverName = "GPKG"
            options.layerName = layer_name
            options.fileEncoding = "UTF-8"
            options.actionOnExistingFile = QgsVectorFileWriter.CreateOrOverwriteLayer
            options.sourceCrs = mem_layer.crs()

            error = QgsVectorFileWriter.writeAsVectorFormatV3(
                mem_layer,
                out_path,
                QgsProject.instance().transformContext(),
                options
            )

            if error[0] == QgsVectorFileWriter.NoError:
                QgsMessageLog.logMessage(
                    f"Couche sauvegardée avec succès : {out_path}",
                    "SpeleoTools",
                    Qgis.Info
                )

                saved_layer = QgsVectorLayer(
                    out_path,
                    f"{layer_name}_saved",
                    "ogr"
                )

                if saved_layer.isValid():
                    QgsProject.instance().addMapLayer(saved_layer)
                    QgsMessageLog.logMessage(
                        "Couche sauvegardée chargée dans le projet QGIS.",
                        "SpeleoTools",
                        Qgis.Info
                    )
                else:
                    QgsMessageLog.logMessage(
                        f"Erreur : Impossible de charger la couche sauvegardée depuis {out_path}.",
                        "SpeleoTools",
                        Qgis.Critical
                    )
            else:
                QgsMessageLog.logMessage(
                    f"Erreur lors de la sauvegarde : {error[1]}",
                    "SpeleoTools",
                    Qgis.Critical
                )

        except Exception as e:
            QgsMessageLog.logMessage(
                f"Exception lors de la sauvegarde : {str(e)}",
                "SpeleoTools",
                Qgis.Critical
            )
    else:
        QgsProject.instance().addMapLayer(mem_layer)
        QgsMessageLog.logMessage(
            "Couche mémoire ajoutée au projet (non sauvegardée).",
            "SpeleoTools",
            Qgis.Info
        )


# ---------- 2) PROFIL (version améliorée) ----------

def transform_point_to_dem_crs(point_xy, line_crs, dem_crs, proj):
    """Transforme un point du CRS de la ligne vers le CRS du MNT.
    Retourne un QgsPointXY dans le CRS du DEM."""
    if line_crs != dem_crs:
        xform = QgsCoordinateTransform(line_crs, dem_crs, proj)
        pt = xform.transform(point_xy)
        return QgsPointXY(pt.x(), pt.y())
    return QgsPointXY(point_xy.x(), point_xy.y())

def sample_dem_at_point(dem_layer, point_xy):
    """Échantillonne le MNT au point donné.
    ATTENTION : point_xy doit être dans le CRS du DEM (QgsPointXY)."""
    dp = dem_layer.dataProvider()
    # Convertir en point adapté
    sample_point = QgsPointXY(point_xy.x(), point_xy.y())
    # Essayer dataProvider.sample (si disponible)
    try:
        samp = dp.sample(sample_point, 1)
    except Exception:
        samp = None

    # si sample a renvoyé quelque chose d'utilisable
    if samp is not None:
        # dp.sample peut renvoyer tuple (val, ok) ou simplement la valeur
        if isinstance(samp, (tuple, list)):
            val = samp[0] if samp else None
            ok = samp[1] if len(samp) > 1 else True
            try:
                return float(val) if ok and val is not None and not math.isnan(float(val)) else None
            except Exception:
                return None
        else:
            try:
                return float(samp) if samp is not None and not math.isnan(float(samp)) else None
            except Exception:
                return None

    # fallback : utiliser identify (plus lent mais parfois nécessaire)
    try:
        ident = dp.identify(sample_point, dp.IdentifyFormatValue)
        if ident.isValid():
            results = list(ident.results().values())
            if results:
                val = results[0]
                try:
                    return float(val) if val is not None and not math.isnan(float(val)) else None
                except Exception:
                    return None
    except Exception:
        pass

    return None


def interpolate_z_values(z_list, sample_points, spacing, max_gap_distance):
    """Interpole les valeurs Z manquantes entre deux points valides.
    Renvoie une liste de segments; chaque segment est une liste de QgsPoint (avec Z)."""
    points_3d_segments = []
    i = 0
    n = len(z_list)
    while i < n:
        if z_list[i] is not None:
            seg = [QgsPoint(sample_points[i].x(), sample_points[i].y(), z_list[i])]
            i += 1
            while i < n and z_list[i] is not None:
                seg.append(QgsPoint(sample_points[i].x(), sample_points[i].y(), z_list[i]))
                i += 1
            points_3d_segments.append(seg)
        else:
            j = i + 1
            while j < n and z_list[j] is None:
                j += 1
            if j < n:
                prev_idx = i - 1
                next_idx = j
                if prev_idx >= 0 and z_list[prev_idx] is not None:
                    hole_len = calculate_hole_length(sample_points, prev_idx, next_idx, spacing)
                    if max_gap_distance is None or hole_len <= max_gap_distance:
                        interp_pts = create_interpolated_points(z_list, sample_points, prev_idx, next_idx)
                        if points_3d_segments and is_continuous(points_3d_segments[-1], sample_points[prev_idx]):
                            points_3d_segments[-1].extend(interp_pts)
                        else:
                            segstart = QgsPoint(sample_points[prev_idx].x(), sample_points[prev_idx].y(), z_list[prev_idx])
                            seg = [segstart] + interp_pts
                            points_3d_segments.append(seg)
                        i = next_idx
                        continue
            i = j
    return points_3d_segments


def calculate_hole_length(sample_points, prev_idx, next_idx, spacing):
    """Calcule la longueur d'un trou entre deux points."""
    if spacing and spacing > 0:
        return (next_idx - prev_idx) * spacing
    else:
        return sum(math.hypot(p2.x() - p1.x(), p2.y() - p1.y())
                   for p1, p2 in zip(sample_points[prev_idx:next_idx], sample_points[prev_idx+1:next_idx]))


def create_interpolated_points(z_list, sample_points, prev_idx, next_idx):
    """Crée des points interpolés entre deux indices (exclut les extrémités)."""
    z0 = z_list[prev_idx]
    z1 = z_list[next_idx]
    steps = next_idx - prev_idx
    return [QgsPoint(sample_points[prev_idx + step].x(),
                     sample_points[prev_idx + step].y(),
                     z0 + (step / float(steps)) * (z1 - z0))
            for step in range(1, steps)]


def is_continuous(segment, point):
    """Vérifie si un segment se termine au point donné (tolérance pour floats)."""
    return (math.isclose(segment[-1].x(), point.x(), abs_tol=1e-6) and
            math.isclose(segment[-1].y(), point.y(), abs_tol=1e-6))


def create_profile_from_line(dem_layer, line_layer, spacing=None, output_path=None, interp=True, max_gap_distance=None, add_to_project=True):
    """Crée un profil 3D (LineStringZ) à partir d'une polyligne 2D/3D."""
    proj = QgsProject.instance()
    dem_crs = dem_layer.crs()
    line_crs = line_layer.crs()
    need_transform = (dem_crs != line_crs)

    # on met la couche de sortie dans le CRS du DEM pour garder cohérence Z+XY
    crs_authid = dem_crs.authid()
    geom_type = f"LineStringZ?crs={crs_authid}"
    out_layer = QgsVectorLayer(geom_type, "profiles", "memory")
    pr = out_layer.dataProvider()
    fields = QgsFields()
    fields.append(QgsField("orig_id", QMetaType.Type.Int))
    fields.append(QgsField("length_m", QMetaType.Type.Double))

    # fields.append(QgsField("orig_id", QVariant.Int))
    # fields.append(QgsField("length_m", QVariant.Double))
    pr.addAttributes(fields)
    out_layer.updateFields()
    # distance calculator (utile si spacing non fourni)
    d_area = QgsDistanceArea()
    if hasattr(proj, "transformContext"):
        d_area.setSourceCrs(dem_crs, proj.transformContext())
    else:
        d_area.setSourceCrs(dem_crs)
    # itérer features
    for feat in line_layer.getFeatures():
        geom = feat.geometry()
        if geom is None or geom.isEmpty():
            continue

        # si besoin, on travaille sur une copie transformée dans le CRS du DEM
        geom_dem = QgsGeometry(geom)  # copie
        if need_transform:
            try:
                xform = QgsCoordinateTransform(line_crs, dem_crs, proj)
                geom_dem.transform(xform)
            except Exception as e:
                print(f"Erreur de transformation pour feature {feat.id()}: {e}")
                continue

        # gérer multipart ou singlepart
        if geom_dem.isMultipart():
            raw_parts = geom_dem.asMultiPolyline()
        else:
            raw_parts = [geom_dem.asPolyline()]

        # pour chaque partie
        for raw_part in raw_parts:
            if not raw_part:
                continue

            # Normaliser tous les points en QgsPointXY (dans le CRS du DEM maintenant)
            part_xy = [QgsPointXY(p.x(), p.y()) for p in raw_part]

            # construire une geometry 2D pour interpolation (fromPolylineXY)
            line_geom = QgsGeometry.fromPolylineXY(part_xy)

            # longueur de la partie
            length = line_geom.length()

            # déterminer points d'échantillonnage (liste de QgsPointXY)
            sample_points = []
            if spacing is None or spacing <= 0:
                # utiliser les sommets
                sample_points = part_xy[:]
            else:
                # échantillonnage régulier le long de la ligne (0..length)
                dist = 0.0
                while dist <= length + 1e-9:
                    interp_geom = line_geom.interpolate(dist)
                    if interp_geom is None or interp_geom.isEmpty():
                        break
                    p = interp_geom.asPoint()
                    sample_points.append(QgsPointXY(p.x(), p.y()))
                    dist += spacing
                # s'assurer d'avoir le dernier point exact
                end_p = line_geom.interpolate(length).asPoint()
                lastxy = QgsPointXY(end_p.x(), end_p.y())
                if (not sample_points) or (not math.isclose(sample_points[-1].x(), lastxy.x(), abs_tol=1e-6) or
                                           not math.isclose(sample_points[-1].y(), lastxy.y(), abs_tol=1e-6)):
                    sample_points.append(lastxy)

            if not sample_points:
                continue

            # échantillonner DEM pour chaque sample point (point déjà en CRS DEM)
            z_list = []
            for pxy in sample_points:
                z = sample_dem_at_point(dem_layer, pxy)
                z_list.append(z)

            points_3d_segments = []
            cur_seg = []
            for idx, z in enumerate(z_list):
                if z is None:
                    if len(cur_seg) >= 2:
                        points_3d_segments.append(cur_seg)
                    cur_seg = []
                else:
                    cur_seg.append(QgsPoint(sample_points[idx].x(), sample_points[idx].y(), z))
            if len(cur_seg) >= 2:
                points_3d_segments.append(cur_seg)

            # créer features de sortie pour chaque segment
            feats_to_add = []
            for seg in points_3d_segments:
                if len(seg) < 2:
                    continue
                feat_out = QgsFeature(out_layer.fields())
                # seg est une liste de QgsPoint (avec Z) -> fromPolyline crée LineStringZ si points ont Z
                feat_geom = QgsGeometry.fromPolyline(seg)
                feat_out.setGeometry(feat_geom)
                feat_out.setAttribute("orig_id", int(feat.id()))
                feat_out.setAttribute("length_m", float(feat_geom.length()))
                feats_to_add.append(feat_out)

            if feats_to_add:
                pr.addFeatures(feats_to_add)

    out_layer.updateExtents()

    # sauvegarder si demandé (ex: shapefile)
    if output_path:
        error = QgsVectorFileWriter.writeAsVectorFormat(out_layer, output_path, "UTF-8", out_layer.crs(), "ESRI Shapefile")
        if error == QgsVectorFileWriter.NoError:
            disk_layer = QgsVectorLayer(output_path, "profiles_saved", "ogr")
            if disk_layer.isValid():
                out_layer = disk_layer

    if add_to_project:
        QgsProject.instance().addMapLayer(out_layer)

    return out_layer

# ---------- 3) Traitement MNT ----------
"""
Module contenant fonctions de traitement des MNT pour le plugin QGIS.
Fonctions exposées :
- hillshade(dem_layer, out_path=None, params..., context=None, feedback=None)
- multidirectional_hillshade(dem_layer, out_path=None, azimuths=None, context=None, feedback=None)
- slope(dem_layer, out_path=None, params..., context=None, feedback=None)
- vat(dem_layer, out_path=None, window_size=5, context=None, feedback=None)

Chaque fonction tente de choisir automatiquement un algorithme de processing disponible
(GDAL / SAGA / GRASS) et renvoie le chemin du raster de sortie (ou None en cas d'erreur).
"""



def _available_algorithms():
    """Renvoie l'ensemble des ids d'algorithmes disponibles."""
    return {alg.id() for alg in QgsApplication.processingRegistry().algorithms()}


def _choose_alg(possible_ids):
    """Choisit le premier algorithme disponible dans possible_ids."""
    avail = _available_algorithms()
    for pid in possible_ids:
        if pid in avail:
            return pid
    return None


def _layer_input(layer):
    """Accepte soit un QgsRasterLayer soit un nom (str) et renvoie l'identifiant attendu par processing."""
    from qgis.core import QgsRasterLayer
    if isinstance(layer, str):
        return layer
    elif isinstance(layer, QgsRasterLayer):
        return layer.dataProvider().dataSourceUri()
    else:
        return layer
    
class SafeFeedback:
    """Wrapper pour feedback afin d'éviter les erreurs si feedback=None"""
    def __init__(self, fb=None):
        self.fb = fb

    def pushInfo(self, msg):
        if self.fb:
            self.fb.pushInfo(str(msg))
        else:
            print("[INFO]", msg)

    def reportError(self, msg):
        if self.fb and hasattr(self.fb, "reportError"):
            self.fb.reportError(str(msg))
        else:
            print("[ERROR]", msg)

def hillshade(dem_layer, out_path=None, zfactor=1.0, azimuth=315.0, altitude=45.0, context=None, feedback=None, addProject=True):
    """
    Calcule un hillshade simple à partir d’un MNT.
    Retourne le chemin du fichier de sortie ou None.
    """
    fb = SafeFeedback(feedback)  # safe feedback

    if not dem_layer or not dem_layer.isValid():
        fb.reportError("Le MNT spécifié est invalide.")
        return None

    dem_in = dem_layer.source()
    dem_crs = dem_layer.crs()
    fb.pushInfo(f"CRS détecté : {dem_crs.authid()}")

    # Choix de l’algorithme disponible
    candidates = ['gdal:hillshade', 'grass7:r.hillshade', 'saga:hillshade']
    alg = _choose_alg(candidates)
    if alg is None:
        fb.pushInfo("Aucun algorithme de hillshade disponible.")
        return None

    # Fichier de sortie
    if out_path is None:
        out_path = os.path.join(tempfile.gettempdir(), f"hillshade_{os.path.basename(str(dem_in))}.tif")

    # Construction des paramètres selon l'algorithme
    params = {}
    if alg == 'gdal:hillshade':
        params = {
            'INPUT': dem_in,
            'BAND': 1,
            'Z_FACTOR': float(zfactor),
            'AZIMUTH': float(azimuth),
            'ALTITUDE': float(altitude),
            'COMPUTE_EDGES': True,
            'OUTPUT': out_path,
        }
    elif alg == 'grass7:r.hillshade':
        params = {
            'elevation': dem_in,
            'scale': float(zfactor),
            'azimuth': float(azimuth),
            'altitude': float(altitude),
            'output': out_path,
        }
    else:  # saga
        params = {
            'ELEVATION': dem_in,
            'AZIMUTH': float(azimuth),
            'ALTITUDE': float(altitude),
            'METHOD': 0,
            'RESULT': out_path,
        }

    try:
        # Exécution du traitement
        res = processing.run(alg, params, context=context, feedback=feedback)

        # Récupère le raster produit
        output_file = None
        for v in res.values():
            if isinstance(v, str) and os.path.exists(v):
                output_file = v
                break
        if not output_file and os.path.exists(out_path):
            output_file = out_path
         
        # Ajoute au projet et applique le CRS
        if output_file:
            rlayer = QgsRasterLayer(output_file, os.path.basename(output_file), "gdal")
            if rlayer.isValid():
                rlayer.setCrs(dem_crs)  # applique le CRS du DEM
                # IMPORTANT : pour que le CRS soit reconnu, on doit réenregistrer le raster ou le notifier au projet
                QgsProject.instance().addMapLayer(rlayer, False)  # ajoute sans sélectionner
                if addProject:
                
                    # ajoute dans un groupe si souhaité
                    group = QgsProject.instance().layerTreeRoot().findGroup("Traitement MNT")
                    if not group:
                        group = QgsProject.instance().layerTreeRoot().addGroup("Traitement MNT")
                    group.addLayer(rlayer)

        return output_file

    except Exception as e:
        fb.reportError(f"Hillshade error: {e}")
        return None


# --- Multidirectional Hillshade ---
def multidirectional_hillshade(dem_layer, out_path=None, context=None, feedback=None):
    fb = SafeFeedback(feedback)

    if not dem_layer or not dem_layer.isValid():
        fb.reportError("Le MNT spécifié est invalide.")
        return None

    dem_in = _layer_input(dem_layer)
    dem_crs = dem_layer.crs()
    fb.pushInfo(f"CRS détecté : {dem_crs.authid()}")
    # tentative alg unique RVT
    alg = _choose_alg(["rvt:rvt_multi_hillshade"])
    if alg:
        if out_path is None:
            out_path = os.path.join(
                tempfile.gettempdir(),
                f'multidh_{os.path.basename(str(dem_in))}.tif'
            )

        try:
            # Exécution de l'algorithme avec les paramètres
            res = processing.run(
                alg,
                {
                    'INPUT': dem_in,
                    'VE_FACTOR': 1,
                    'NUM_DIRECTIONS': 16,
                    'SUN_ELEVATION': 35,
                    'SAVE_AS_8BIT': False,
                    'OUTPUT': out_path
                }
            )

            # Recherche du chemin de sortie valide
            output_file = None
            for v in res.values():
                if isinstance(v, str) and os.path.exists(v):
                    output_file = v
                    break

            if not output_file and os.path.exists(out_path):
                output_file = out_path
            return output_file

        except Exception as e:
            fb.reportError(f'MD Hillshade error (saga): {e}')
            # fallback manuel

# --- Slope ---
def slope(dem_layer, out_path=None, zfactor=1.0, context=None, feedback=None):
    fb = SafeFeedback(feedback)
    if not dem_layer or not dem_layer.isValid():
        fb.reportError("Le MNT spécifié est invalide.")
        return None

    dem_in = dem_layer.source()
    dem_crs = dem_layer.crs()
    fb.pushInfo(f"CRS détecté : {dem_crs.authid()}")

    alg = _choose_alg(['gdal:slope', 'grass7:r.slope.aspect', 'saga:slopeaspectcurvature'])
    if alg is None:
        fb.pushInfo('Aucun algorithme de pente disponible.')
        return None

    if out_path is None:
        out_path = os.path.join(tempfile.gettempdir(), f'slope_{os.path.basename(str(dem_in))}.tif')

    if alg == 'gdal:slope':
        params = {'INPUT': dem_in, 'BAND':1, 'SCALE': float(zfactor), 'AS_PERCENT': False, 'COMPUTE_EDGES': True, 'Z_FACTOR':1.0, 'OUTPUT': out_path}
    elif alg == 'grass7:r.slope.aspect':
        params = {'elevation': dem_in, 'slope': out_path, 'format': 0, 'zfactor': float(zfactor)}
    else:
        params = {'ELEVATION': dem_in, 'SCALE': float(zfactor), 'RESULT': out_path}

    try:
        res = processing.run(alg, params, context=context, feedback=feedback)
        for v in res.values():
            if isinstance(v, str) and os.path.exists(v):
                output_file = v
                rlayer = QgsRasterLayer(output_file, os.path.basename(output_file), "gdal")
                if rlayer.isValid():
                    rlayer.setCrs(dem_crs)  # applique le CRS du DEM
                    # IMPORTANT : pour que le CRS soit reconnu, on doit réenregistrer le raster ou le notifier au projet
                    QgsProject.instance().addMapLayer(rlayer, False)  # ajoute sans sélectionner
                    # ajoute dans un groupe si souhaité
                    group = QgsProject.instance().layerTreeRoot().findGroup("Traitement MNT")
                    if not group:
                        group = QgsProject.instance().layerTreeRoot().addGroup("Traitement MNT")
                    group.addLayer(rlayer)
                return output_file
        return None
    except Exception as e:
        fb.reportError(f"Slope error: {e}")
        return None

# --- VAT ---
def VAT(dem_layer, out_path=None, context=None, type_terrain=0, feedback=None):
    fb = SafeFeedback(feedback)

    if not dem_layer or not dem_layer.isValid():
        fb.reportError("Le MNT spécifié est invalide.")
        return None
    dem_in = _layer_input(dem_layer)
    dem_crs = dem_layer.crs()
    fb.pushInfo(f"CRS détecté : {dem_crs.authid()}")
    # tentative alg unique RVT
    alg = _choose_alg(["rvt:rvt_blender"])
    if alg:
        if out_path is None:
            out_path = os.path.join(
                tempfile.gettempdir(),
                f'VAT_{os.path.basename(str(dem_in))}.tif'
            )

        try:
            # Exécution de l'algorithme avec les paramètres
            res = processing.run(
                alg,
                {
                    'INPUT': dem_in,
                    'BLEND_COMBINATION':0,
                    'TERRAIN_TYPE':type_terrain,
                    'SAVE_AS_8BIT':False,
                    'OUTPUT': out_path
                }
            )

            # Recherche du chemin de sortie valide
            output_file = None
            for v in res.values():
                if isinstance(v, str) and os.path.exists(v):
                    output_file = v
                    break

            if not output_file and os.path.exists(out_path):
                output_file = out_path
            return output_file

        except Exception as e:
            fb.reportError(f'VAT error (saga): {e}')
            # fallback manuel




# ---------- 4) PROSPECTION Auto ----------

"""
Collection de fonctions pour détecter des dolines (sinkholes) dans QGIS.
Usage:
- import find_dolines
- find_dolines.main_find_dolines(dem_layer, out_folder=None, params={})

Ce fichier contient des fonctions indépendantes pour chaque étape :
 1) comblement des sinks (SAGA XXL Wang & Liu)
 2) calcul du raster `sink = filled_dem - dem`
 3) vectorisation des zones de sink > seuil
 4) suppression des petites entités (option)
 5) clustering DBSCAN sur les centroïdes (optionnel)
 6) géométrie d'emprise minimale + suppression de la plus grande
 7) statistiques zonales (z_min, z_max, z_mean, z_median)
 8) centroids (barycentres) avec attributs

Remarques:
- Les sorties intermédiaires sont créées en mémoire (TEMPORARY_OUTPUT) sauf si out_folder est fourni.
- Le code utilise processing.run ; il doit être exécuté dans l'environnement QGIS (console Python ou plugin).
"""

from qgis.core import QgsProject, QgsVectorLayer, QgsFeature, QgsGeometry
import processing
import uuid


def _temp_path(name_prefix):
    return 'memory:' + name_prefix + '_' + str(uuid.uuid4())


def fill_sinks(dem_layer, minslope=0.1, filled_output=None):
    """Etape 1 : remplit les sinks avec SAGA (sagang:fillsinksxxlwangliu)
    dem_layer : QgsRasterLayer ou chemin
    minslope : float
    retourne : chemin/objet du raster rempli
    """
    if filled_output is None:
        filled_output = _temp_path('filled')
    params = {
        'ELEV': dem_layer,
        'FILLED': filled_output,
        'MINSLOPE': minslope,
    }
    res = processing.run('sagang:fillsinksxxlwangliu', params)
    return res['FILLED']


def compute_sink_raster(dem_layer, filled_dem, threshold=1.0, sink_output=None):
    """Etape 2 : sink = filled_dem - dem_layer (GDAL Raster calculator)
    Retourne un raster (path or memory id)
    """
    if sink_output is None:
        sink_output = 'TEMPORARY_OUTPUT'

    # gdal:rastercalculator requiert des noms de couche A,B,... on utilisera A-B

    # Valeur temporaire pour NoData
    sentinel = -9999.0

    # Étape 1 : calculer (A - B), mais remplacer les valeurs <= threshold par sentinel
    expr = f"(A - B) - ((A - B) <= {threshold}) * ((A - B) - {float(sentinel)})"

    res = processing.run(
        "gdal:rastercalculator",
        {
            'INPUT_A': filled_dem,
            'BAND_A': 1,
            'INPUT_B': dem_layer,
            'BAND_B': 1,
            'FORMULA': expr,
            'NO_DATA': None,
            'RTYPE': 5,  # Float32
            'NO_DATA': sentinel,
            'OPTIONS': '',
            'OUTPUT': sink_output
            }
        )


    return res.get('OUTPUT') or res.get('RESULT')


def vectorize_sinks(sink_raster, vector_output=None):
    """Etape 3 : polygonize les pixels 
    Retourne une couche vectorielle (polygons)
    """
    if vector_output is None:
        vector_output = 'TEMPORARY_OUTPUT'
    # polygonize
    poly_params = {
        'INPUT_RASTER': sink_raster,
        'RASTER_BAND': 1,
        'FIELD_NAME': 'VALUE',
        'OUTPUT': vector_output
    }
    res = processing.run('native:pixelstopoints', poly_params)

    out_path = res.get('OUTPUT')
    print(f"[DEBUG] Résultat pixelstopoints : {out_path}")


    return res['OUTPUT']


def dbscan_partition(point_layer, eps=1.0, min_size=5, field_name='CLUSTER_ID', vector_output=None):
    """Etape 5 : partitionnement DBSCAN sur une couche de points
    retourne la couche annotée avec field_name (et size field)
    """
    if vector_output is None:
        vector_output = 'TEMPORARY_OUTPUT'
    params = {
        'INPUT': point_layer,
        'EPS': eps,
        'MIN_SIZE': min_size,
        'FIELD_NAME': field_name,
        'SIZE_FIELD_NAME': 'CLUSTER_SIZE',
        'OUTPUT': vector_output
    }
    res = processing.run('native:dbscanclustering', params)
    return res['OUTPUT']


def minimum_bounding_geometry(polygons_layer, field='CLUSTER_ID', keep_largest=False, vector_output=None):
    """Etape 6 : calcule l'emprise minimale pour chaque cluster et retire la plus grande (si demandé)
    Retourne la couche de polygones filtrée
    """
    if vector_output is None:
        vector_output = 'TEMPORARY_OUTPUT'
    res = processing.run('qgis:minimumboundinggeometry', {'INPUT': polygons_layer, 'FIELD': field, 'TYPE': 3, 'OUTPUT': vector_output})
    mbg = res['OUTPUT']

    if not keep_largest:
        # supprimer la plus grande emprise
        # charger et parcourir pour trouver la plus grande
        vl = mbg
        if isinstance(mbg, str):
            vl = QgsVectorLayer(mbg, 'mbg', 'ogr')
        max_area = 0
        max_id = None
        for feat in vl.getFeatures():
            a = feat.geometry().area()
            if a > max_area:
                max_area = a
                max_id = feat.id()
        if max_id is not None:
            try:
                vl.dataProvider().deleteFeatures([max_id])
            except Exception:
                # fallback to extract by expression
                expr = f"$id != {max_id}"
                res2 = processing.run('native:extractbyexpression', {'INPUT': vl, 'EXPRESSION': expr, 'OUTPUT': vector_output})
                vl = res2['OUTPUT']
        return vl
    return mbg


def zonal_statistics(polygons_layer, dem_layer, stats_prefix='dz_', stats=[2,3,5,6], vector_output=None):
    """Etape 7 : calcule des statistiques zonales (z_min, z_max, z_mean, z_median)
    Ajoute les champs sur la couche polygons_layer et retourne la couche.
    """
    if vector_output is None:
        vector_output = 'TEMPORARY_OUTPUT'
    params = {
        'INPUT': polygons_layer,
        'INPUT_RASTER': dem_layer,
        'COLUMN_PREFIX': stats_prefix,
        'STATISTICS':stats,
        'OUTPUT': vector_output
    }

    res = processing.run('native:zonalstatisticsfb', params)

    return res.get('OUTPUT', polygons_layer)


def extract_centroids_with_stats(polygons_layer, vector_output=None):
    """Etape 8 : créé la couche de centroïdes et copie les champs statistiques
    Retourne la couche point (centroides) avec les attributs présents dans polygons_layer
    """
    if vector_output is None:
        vector_output = 'TEMPORARY_OUTPUT'
    cents = processing.run('native:centroids', {'INPUT': polygons_layer, 'ALL_PARTS': False, 'OUTPUT': vector_output})
    return cents['OUTPUT']


def cleanup_layers(layers_list):
    """Supprime les couches temporaires de la légende (si elles ont été ajoutées)
    layers_list: liste d'objets ou ids
    """
    proj = QgsProject.instance()
    for l in layers_list:
        try:
            if isinstance(l, str):
                # attempt to find layer by id or path
                layer = proj.mapLayersByName(l)
                if layer:
                    proj.removeMapLayer(layer[0])
            else:
                proj.removeMapLayer(l.id())
        except Exception:
            pass


