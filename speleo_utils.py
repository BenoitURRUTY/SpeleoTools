from qgis.core import (
    QgsProject, QgsVectorLayer, QgsFields, QgsField, QgsFeature,
    QgsGeometry, QgsPointXY, QgsPoint, QgsDistanceArea, QgsMessageLog, Qgis,QgsWkbTypes,QgsVectorFileWriter,
    QgsCoordinateTransform,
    QgsCoordinateReferenceSystem
)
from PyQt5.QtCore import QVariant

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



# ---------- 2) PROFIL ----------
def transform_point_to_dem_crs(point_xy, line_crs, dem_crs, proj):
    """Transforme un point du CRS de la ligne vers le CRS du MNT."""
    if line_crs != dem_crs:
        xform = QgsCoordinateTransform(line_crs, dem_crs, proj)
        return xform.transform(point_xy)
    return point_xy

def sample_dem_at_point(dem_layer, point_xy, line_crs, dem_crs, proj):
    """Échantillonne le MNT au point donné (dans le CRS de la ligne)."""
    pt_dem = transform_point_to_dem_crs(point_xy, line_crs, dem_crs, proj)
    sample_point = QgsPointXY(pt_dem.x(), pt_dem.y())
    dp = dem_layer.dataProvider()
    try:
        samp = dp.sample(sample_point, 1)
    except Exception:
        try:
            ident = dp.identify(sample_point, dp.IdentifyFormatValue)
            if ident.isValid():
                val = list(ident.results().values())[0]
                try:
                    return float(val) if val is not None and not math.isnan(float(val)) else None
                except:
                    return None
        except Exception:
            pass
        return None
    if isinstance(samp, (tuple, list)):
        val, ok = samp[0], samp[1] if len(samp) > 1 else True
        try:
            return float(val) if ok and not math.isnan(float(val)) else None
        except:
            return None
    else:
        try:
            return float(samp) if not math.isnan(float(samp)) else None
        except:
            return None

def interpolate_z_values(z_list, sample_points, spacing, max_gap_distance):
    """Interpole les valeurs Z manquantes entre deux points valides."""
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
    """Crée des points interpolés entre deux indices."""
    z0 = z_list[prev_idx]
    z1 = z_list[next_idx]
    steps = next_idx - prev_idx
    return [QgsPoint(sample_points[prev_idx + step].x(),
                     sample_points[prev_idx + step].y(),
                     z0 + (step / float(steps)) * (z1 - z0))
            for step in range(1, steps)]

def is_continuous(segment, point):
    """Vérifie si un segment se termine au point donné."""
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
    fields.append(QgsField("orig_id", QVariant.Int))
    fields.append(QgsField("length_m", QVariant.Double))
    pr.addAttributes(fields)
    out_layer.updateFields()
    
    # distance calculator (utile si spacing non fourni)
    d_area = QgsDistanceArea()
    d_area.setSourceCrs(dem_crs, proj.transformContext() if hasattr(proj, "transformContext") else proj)
    
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
            parts = geom_dem.asMultiPolyline()
        else:
            parts = [geom_dem.asPolyline()]

        # pour chaque partie
        for part in parts:
            if not part:
                continue

            # --- IMPORTANT FIX ---
            # part peut contenir des QgsPoint ou des QgsPointXY suivant la source.
            # Pour l'interpolation on utilise fromPolylineXY si ce sont des QgsPointXY,
            # sinon fromPolyline. Cela évite l'erreur "index 0 has type 'QgsPointXY' but 'QgsPoint' is expected".
            first_pt = part[0]
            try:
                is_qgspointxy = isinstance(first_pt, QgsPointXY)
            except Exception:
                # fallback : considérer comme QgsPoint si doute
                is_qgspointxy = False

            if is_qgspointxy:
                line_geom = QgsGeometry.fromPolylineXY(part)
            else:
                # si ce sont des QgsPoint (ou QgsPointZ), convertir en liste de QgsPoint 2D
                # asPolyline renvoie souvent QgsPoint, on peut l'utiliser directement
                line_geom = QgsGeometry.fromPolyline([QgsPoint(p.x(), p.y()) for p in part])

            # longueur de la partie
            length = line_geom.length()
            # déterminer points d'échantillonnage (liste de QgsPointXY)
            sample_points = []
            if spacing is None or spacing <= 0:
                # utiliser les sommets (part est une liste de QgsPoint/QgsPointXY)
                for p in part:
                    # normaliser en QgsPointXY pour le sampling
                    sample_points.append(QgsPointXY(p.x(), p.y()))
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
                if not sample_points or (not math.isclose(sample_points[-1].x(), lastxy.x(), abs_tol=1e-6) or
                                         not math.isclose(sample_points[-1].y(), lastxy.y(), abs_tol=1e-6)):
                    sample_points.append(lastxy)

            if not sample_points:
                continue

            # échantillonner DEM pour chaque sample point
            z_list = []
            # NOTE: sample_dem_at_point doit exister dans ton module (ne l'ai pas redéfini ici).
            for pxy in sample_points:
                # sample_dem_at_point(dem_layer, point_xy, src_crs, dem_crs, project)
                z = sample_dem_at_point(dem_layer, pxy, dem_crs, dem_crs, proj)
                z_list.append(z)

            # interpolation / découpage en segments 3D
            if interp:
                # interpolate_z_values doit renvoyer une liste de segments où chaque seg est une liste de QgsPoint (avec Z)
                points_3d_segments = interpolate_z_values(z_list, sample_points, spacing, max_gap_distance)
            else:
                # sans interpolation, on transforme directement les points valides en segments contigus
                points_3d_segments = []
                cur_seg = []
                for idx, z in enumerate(z_list):
                    if z is None:
                        if len(cur_seg) >= 2:
                            points_3d_segments.append(cur_seg)
                        cur_seg = []
                    else:
                        # Ici on crée un QgsPoint (avec Z) — correct pour fromPolyline (LineStringZ)
                        cur_seg.append(QgsPoint(sample_points[idx].x(), sample_points[idx].y(), z))
                if len(cur_seg) >= 2:
                    points_3d_segments.append(cur_seg)

            # créer features de sortie pour chaque segment
            for seg in points_3d_segments:
                if len(seg) < 2:
                    continue
                feat_out = QgsFeature(out_layer.fields())
                # seg est une liste de QgsPoint (avec Z), on peut créer une geom 3D
                feat_geom = QgsGeometry.fromPolyline(seg)
                feat_out.setGeometry(feat_geom)
                feat_out.setAttribute("orig_id", feat.id())
                # longueur en unités du CRS du DEM (souvent mètres si CRS métrique)
                feat_out.setAttribute("length_m", feat_geom.length())
                pr.addFeatures([feat_out])

    out_layer.updateExtents()

    # sauvegarder si demandé (ex: shapefile)
    if output_path:
        error = QgsVectorFileWriter.writeAsVectorFormat(out_layer, output_path, "UTF-8", out_layer.crs(), "ESRI Shapefile")
        # si succès, remplacer out_layer par couche sur disque
        if error == QgsVectorFileWriter.NoError:
            disk_layer = QgsVectorLayer(output_path, "profiles_saved", "ogr")
            if disk_layer.isValid():
                out_layer = disk_layer

    if add_to_project:
        QgsProject.instance().addMapLayer(out_layer)

    return out_layer

# ---------- 3) PROSPECTION (hillshade / slope / bas-fonds) ----------
def run_prospection_real(dem_layer, do_hillshade=True, do_slope=True, do_low=True, out_folder=None):
    """
    Exécute hillshade (GDAL), slope (GDAL) et détection simple de zones basses
    (seuil = mean - k * stddev). Sauvegarde les rasters dans out_folder si fourni.
    Retourne un dict avec les layers créés.
    """
    results = {}
    dem_path = dem_layer.source()

    # hillshade via GDAL
    if do_hillshade:
        out_hill = (out_folder + "/hillshade.tif") if out_folder else "memory:hillshade"
        params = {
            'INPUT': dem_path,
            'Z_FACTOR': 1.0,
            'AZIMUTH': 315.0,
            'ZENITH': 45.0,
            'OUTPUT': out_hill
        }
        res = processing.run("gdal:hillshade", params)
        results['hillshade'] = res['OUTPUT']
        self.log("Hillshade généré.")

    # slope via GDAL
    if do_slope:
        out_slope = (out_folder + "/slope.tif") if out_folder else "memory:slope"
        params = {
            'INPUT': dem_path,
            'Z_FACTOR': 1.0,
            'SCALE': 1.0,
            'OUTPUT': out_slope
        }
        res = processing.run("gdal:slope", params)
        results['slope'] = res['OUTPUT']
        self.log("Pente générée.")

    # détection zones basses : statistique simple + calcul mask
    if do_low:
        stats = dem_layer.dataProvider().bandStatistics(1)
        mean = stats.mean
        stddev = stats.stdDev
        # seuil : moyenne - 0.5 * stddev (paramétrable)
        k = 0.5
        threshold = mean - k * stddev
        self.log(f"Stats DEM: mean={mean:.2f}, stddev={stddev:.2f}, seuil bas={threshold:.2f}")

        # création masque : pixels < threshold -> 1 else 0 via raster calculator (GDAL calc)
        out_mask = (out_folder + "/low_mask.tif") if out_folder else "memory:low_mask"
        # formule GDAL: "A < threshold"
        calc_params = {
            'INPUT_A': dem_path,
            'BAND_A': 1,
            'EXPRESSION': f"A < {threshold}",
            'OUTPUT': out_mask
        }
        # utilise GDAL raster calculator
        res = processing.run("gdal:rastercalculator", calc_params)
        results['low_mask'] = res['OUTPUT']
        self.log("Masque zones basses généré.")

        # polygoniser le masque si on veut vecteur
        out_poly = (out_folder + "/low_areas.gpkg") if out_folder else "memory:low_areas"
        poly_params = {
            'INPUT': res['OUTPUT'],
            'BAND': 1,
            'FIELD': 'VALUE',
            'EIGHT_CONNECTEDNESS': False,
            'EXTRA': '',
            'OUTPUT': out_poly
        }
        pres = processing.run("gdal:polygonize", poly_params)
        results['low_areas'] = pres['OUTPUT']
        self.log("Zones basses polygonisées.")

    # ajouter au projet si sorties sont fichiers ou memory
    for k, pth in results.items():
        try:
            lyr = QgsProject.instance().mapLayersByName(os.path.basename(pth))[0] if isinstance(pth, str) and QgsProject.instance().mapLayersByName(os.path.basename(pth)) else None
        except Exception:
            lyr = None
        # si path is in-memory type returned by processing, on le récupère différemment
        try:
            if isinstance(pth, QgsRasterLayer) or isinstance(pth, QgsVectorLayer):
                QgsProject.instance().addMapLayer(pth)
        except Exception:
            pass

    return results