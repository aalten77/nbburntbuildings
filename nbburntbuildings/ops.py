from functools import partial
import pyproj
from shapely.geometry import shape
from shapely import ops, geometry


def reproject(geom, from_proj='EPSG:4326', to_proj='EPSG:26942'):
    tfm = partial(pyproj.transform, pyproj.Proj(init=from_proj), pyproj.Proj(init=to_proj))
    return ops.transform(tfm, geom)

def km2_area(polygons):
    reprojected_polygons = [reproject(p) for p in polygons]
    return ops.cascaded_union(reprojected_polygons).area * 1e-6
