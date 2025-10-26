
# ---------- UTILITAIRES ----------
def sample_raster_at_point(self, raster_layer, qgs_point):
    """
    Echantillonne la valeur du raster au point (QgsPointXY).
    Retourne la valeur (float) ou None si échec.
    """
    provider = raster_layer.dataProvider()
    # sample attend QgsPointXY ou QPoint; band index commence à 1
    val = provider.sample(qgs_point, 1)
    try:
        return float(val) if val is not None else None
    except Exception:
        return None

def layer_feature_elevation(self, feat):
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
def compute_thickness(self, dem_layer, cave_layer, out_path=None):
    """
    Echantillonne le DEM pour chaque sommet de la couche cave_layer,
    récupère l'altitude de la cavité (géométrie z ou champ), calcule
    surface_elev - cave_elev et renvoie une couche mémoire contenant
    des points avec l'attribut 'thickness'.
    Si out_path (chemin .gpkg) fourni, sauvegarde la couche.
    """
    # préparation couche sortie (points)
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
    for feat in cave_layer.getFeatures():
        geom = feat.geometry()
        # itérer sommets
        for v in geom.vertices():
            p = QgsPointXY(v.x(), v.y())
            surf_elev = self.sample_raster_at_point(dem_layer, p)
            cave_elev = None
            # si vertex a z() utilisable
            try:
                z = v.z()
                if z is not None:
                    cave_elev = float(z)
            except Exception:
                cave_elev = None

            # si pas de z, essaye d'extraire depuis feature
            if cave_elev is None:
                cave_elev = self.layer_feature_elevation(feat)

            if surf_elev is None or cave_elev is None:
                # on saute si donnée manquante
                continue

            thickness = surf_elev - cave_elev
            new_feat = QgsFeature()
            new_feat.setGeometry(QgsGeometry.fromPointXY(p))
            new_feat.setFields(mem_layer.fields())
            new_feat['src_elev'] = surf_elev
            new_feat['cave_elev'] = cave_elev
            new_feat['thickness'] = thickness
            new_feat['fid_src'] = feat.id()
            mem_dp.addFeatures([new_feat])
            features_added += 1

    mem_layer.updateExtents()
    QgsProject.instance().addMapLayer(mem_layer)

    # sauvegarde si demandé
    if out_path:
        error = QgsVectorLayer.exportLayer(mem_layer, out_path, "GPKG", mem_layer.crs(), False)
        if error[0] != QgsVectorLayer.NoError:
            self.log(f"Erreur sauvegarde épaisseur: {error}")
        else:
            self.log(f"Couche d'épaisseur sauvegardée dans : {out_path}")

    self.log(f"Calcul épaisseur terminé : {features_added} points créés.")
    return mem_layer

# ---------- 2) PROFIL ----------
def generate_profile_points(self, dem_layer, line_layer, spacing=None, out_name="profile_points"):
    """
    Pour chaque ligne sélectionnée dans line_layer, on génère
    des points le long de la ligne espacés de `spacing` mètres
    (si spacing None -> on prend la résolution du raster en X).
    On échantillonne le DEM pour créer points (distance, elevation).
    Retourne la couche mémoire ajoutée au projet.
    """
    # trouve spacing par défaut : résolution raster en X
    if spacing is None:
        try:
            # provider extent pour récupérer taille pixel
            res_x = dem_layer.rasterUnitsPerPixelX()
            spacing = res_x if res_x and res_x > 0 else 1.0
        except Exception:
            spacing = 1.0

    # création layer points
    fields = QgsFields()
    fields.append(QgsField("line_id", QVariant.Int))
    fields.append(QgsField("distance", QVariant.Double))
    fields.append(QgsField("elev", QVariant.Double))

    mem = QgsVectorLayer("Point?crs=" + dem_layer.crs().authid(), out_name, "memory")
    dp = mem.dataProvider()
    dp.addAttributes(fields)
    mem.updateFields()

    for feat in line_layer.getFeatures():
        geom = feat.geometry()
        length = geom.length()
        if length <= 0:
            continue
        dist = 0.0
        count = 0
        while dist <= length:
            pt = geom.interpolate(dist)
            p = pt.asPoint()
            surf_elev = self.sample_raster_at_point(dem_layer, QgsPointXY(p.x(), p.y()))
            if surf_elev is not None:
                newf = QgsFeature()
                newf.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(p.x(), p.y())))
                newf.setFields(mem.fields())
                newf['line_id'] = feat.id()
                newf['distance'] = dist
                newf['elev'] = surf_elev
                dp.addFeatures([newf])
                count += 1
            dist += spacing
        self.log(f"Profil : ligne {feat.id()} -> {count} points.")
    mem.updateExtents()
    QgsProject.instance().addMapLayer(mem)
    return mem

# ---------- 3) PROSPECTION (hillshade / slope / bas-fonds) ----------
def run_prospection_real(self, dem_layer, do_hillshade=True, do_slope=True, do_low=True, out_folder=None):
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