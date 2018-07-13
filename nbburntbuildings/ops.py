"""""
    GBDX Notebook: "Identifying Destroyed Buildings with Multispectral Imagery"
    Link: https://notebooks.geobigdata.io/hub/notebooks/5b47cfb82486966ea89b75fd?tab=code
    Author: Ai-Linh Alten
    Date created: 7/5/2018
    Date last modified: 7/13/2018
    Python Version: 2.7.15

"""

from branca.element import Element, Figure
import cPickle
import folium
from functools import partial
from gbdxtools import CatalogImage, IdahoImage
import geojson
from IPython.display import HTML, display
import jinja2
import json
from matplotlib import pyplot as plt, colors
import numpy as np
import os
from past.utils import old_div
import pickle
import plotly.graph_objs as go
from plotly.graph_objs import Line
from plotly.offline.offline import _plot_html
import pyproj
from rasterio import features
import requests
from scipy import ndimage as ndi
from shapely import geometry, ops
from shapely.geometry import shape, geo, box
from skimage import filters, morphology, measure, color, segmentation, exposure
from skimage.measure import label, regionprops
from sklearn.metrics import confusion_matrix, recall_score, precision_score, accuracy_score

#CONSTANTS
buildings_geojson_link = 'https://s3.amazonaws.com/gbdx-training/burnt_areas/Nuns_SonomaCounty_Glenn_selected_labelled.geojson'
RF_model_link = 'https://s3.amazonaws.com/gbdx-training/burnt_areas/rf_allseg_model.pkl'

"""Helper functions for the GBDX Notebook."""

def pixels_as_features(image, include_gabors=True):
    """Calculates remote sensing indices and gabor filters(optional).
    Returns image features of image bands, remote sensing indices, and gabor filters."""

    # roll axes to conventional row,col,depth
    img = np.rollaxis(image, 0, 3)
    rsi = calc_rsi(image)
    if include_gabors is True:
        gabors = calc_gabors(image)
        stack = np.dstack([img, rsi, gabors])
    else:
        stack = np.dstack([img, rsi])

    feats = stack.ravel().reshape(stack.shape[0] * stack.shape[1], stack.shape[2])

    return feats


def calc_rsi(image):
    """Remote sensing indices for vegetation, built-up, and bare soil."""

    # roll axes to conventional row,col,depth
    img = np.rollaxis(image, 0, 3)

    # bands: Coastal(0), Blue(1), Green(2), Yellow(3), Red(4), Red-edge(5), NIR1(6), NIR2(7)) Multispectral
    COAST = img[:, :, 0]
    B = img[:, :, 1]
    G = img[:, :, 2]
    Y = img[:, :, 3]
    R = img[:, :, 4]
    RE = img[:, :, 5]
    NIR1 = img[:, :, 6]
    NIR2 = img[:, :, 7]

    arvi = old_div((NIR1 - (R - (B - R))), (NIR1 + (R - (B - R))))
    dd = (2 * NIR1 - R) - (G - B)
    gi2 = (B * -0.2848 + G * -0.2434 + R * -0.5436 + NIR1 * 0.7243 + NIR2 * 0.0840) * 5
    gndvi = old_div((NIR1 - G), (NIR1 + G))
    ndre = old_div((NIR1 - RE), (NIR1 + RE))
    ndvi = old_div((NIR1 - R), (NIR1 + R))
    ndvi35 = old_div((G - R), (G + R))
    ndvi84 = old_div((NIR2 - Y), (NIR2 + Y))
    nirry = old_div((NIR1), (R + Y))
    normnir = old_div(NIR1, (NIR1 + R + G))
    psri = old_div((R - B), RE)
    rey = old_div((RE - Y), (RE + Y))
    rvi = old_div(NIR1, R)
    sa = old_div(((Y + R) * 0.35), 2) + old_div((0.7 * (NIR1 + NIR2)), 2) - 0.69
    vi1 = old_div((10000 * NIR1), (RE) ** 2)
    vire = old_div(NIR1, RE)
    br = (old_div(R, B)) * (old_div(G, B)) * (old_div(RE, B)) * (old_div(NIR1, B))
    gr = old_div(G, R)
    rr = (old_div(NIR1, R)) * (old_div(G, R)) * (old_div(NIR1, RE))

    ###Built-Up indices
    wvbi = old_div((COAST - RE), (COAST + RE))
    wvnhfd = old_div((RE - COAST), (RE + COAST))

    ###SIs
    evi = old_div((2.5 * (NIR2 - R)), (NIR2 + 6 * R - 7.5 * B + 1))
    L = 0.5  # some coefficient for Soil Adjusted Vegetation Index (SAVI) DO NOT INCLUDE IN FEATURES
    savi = old_div(((1 + L) * (NIR2 - R)), (NIR2 + R + L))
    msavi = old_div((2 * NIR2 + 1 - ((2 * NIR2 + 1) ** 2 - 8 * (NIR2 - R)) ** 0.5), 2)
    bai = old_div(1.0, ((0.1 + R) ** 2 + 0.06 + NIR2))
    rgi = old_div(R, G)
    bri = old_div(B, R)

    rsi = np.stack(
        [arvi, dd, gi2, gndvi, ndre, ndvi, ndvi35, ndvi84, nirry, normnir, psri, rey, rvi, sa, vi1, vire, br, gr, rr,
         wvbi, wvnhfd, evi, savi, msavi, bai, rgi, bri],
        axis=2)

    return rsi


def power(image, kernel):
    """Normalize images for better comparison."""

    image = old_div((image - image.mean()), image.std())
    return np.sqrt(ndi.convolve(image, np.real(kernel), mode='wrap') ** 2 +
                   ndi.convolve(image, np.imag(kernel), mode='wrap') ** 2)


def calc_gabors(image, frequency=1, theta_vals=[0, 1, 2, 3]):
    """Calculate gabor."""

    # convert to gray scale
    img = exposure.equalize_hist(color.rgb2gray(image.rgb(blm=True)))
    results_list = []
    for theta in theta_vals:
        theta = theta / 4. * np.pi
        kernel = filters.gabor_kernel(frequency, theta=theta)
        # Save kernel and the power image for each image
        results_list.append(power(img, kernel))

    gabors = np.rollaxis(np.dstack([results_list]), 0, 3)

    return gabors



def get_link(model_url):
    """Fetch the RF model pickle file."""
    response = requests.get(model_url)
    # if sys.version_info[0] == 2:
    #     pickle_opts = {}
    # else:
    #     pickle_opts = {'encoding': 'latin1'}

    #model = cPickle.load(response.content, **pickle_opts)
    #model = pickle.loads(response.content, **pickle_opts)

    return response.content

#partials
get_model = partial(get_link, model_url=RF_model_link)
get_geojson = partial(get_link, model_url=buildings_geojson_link)

def reproject(geom, from_proj='EPSG:4326', to_proj='EPSG:26942'):
    """Project from ESPG:4326 to ESPG:26942."""

    tfm = partial(pyproj.transform, pyproj.Proj(init=from_proj), pyproj.Proj(init=to_proj))
    return ops.transform(tfm, geom)

def km2_area(polygons):
    """Get area in km^2 after reprojection."""

    reprojected_polygons = [reproject(p) for p in polygons]
    return ops.cascaded_union(reprojected_polygons).area * 1e-6

def clean(img):
    """Clean the binary image by removing small holes and objects."""

    label_img = label(img, connectivity=2)
    props = sorted(regionprops(label_img), key=lambda x: x.area)
    clean = morphology.binary_closing(img)

    clean = morphology.remove_small_holes(clean)
    return morphology.remove_small_objects(clean,
                                           int(np.floor(props[-1].area) / 10), connectivity=2)


def to_geojson(shapes, buildings):
    """Converts the shapes into geojson.
    This function will combine the burn scar region and buildings into geojson.
    Burn scar polygon in red, buildings polygon all in blue."""

    #append burn scar region polygons to geojson
    if type(shapes) == list:
        results = ({
            'type': 'Feature',
            'properties': {'raster_val': v, 'color': 'red'},
            'geometry': s.__geo_interface__}
            for i, (s, v)
            in enumerate(shapes))
    else:
        results = ({
            'type': 'Feature',
            'properties': {'raster_val': v, 'color': 'red'},
            'geometry': s}
            for i, (s, v)
            in enumerate(shapes))

    list_results = list(results)

    # append the building footprints to geojson
    results_buildings = ({
        'type': 'Feature',
        'properties': {'BuildingID': b['properties']['BuildingID'], 'color': 'blue'},
        'geometry': b['geometry']}
        for i, b
        in enumerate(buildings['features']))

    list_results_buildings = list(results_buildings)

    collection = {
        'type': 'FeatureCollection',
        'features': list_results + list_results_buildings}

    return collection


def geojson_to_polygons(js_):
    """Convert the geojson into Shapely Polygons.
    Keep burn scar polygons as red.
    Mark all building polygons labelled as ('yellow', False) and will be changed later."""

    burnt_polys = []
    building_polys = []
    for i, feat in enumerate(js_['features']):
        o = {
            "coordinates": feat['geometry']['coordinates'],
            "type": feat['geometry']['type']
        }
        s = json.dumps(o)

        # convert to geojson.geometry.Polygon
        g1 = geojson.loads(s)

        # covert to shapely.geometry.polygon.Polygon
        g2 = shape(g1)

        if feat['properties']['color'] == 'red':  # red for the burnt region
            burnt_polys.append(g2)
        else:  # for the building poly
            building_polys.append([g2, [feat['properties']['BuildingID'], 'yellow',
                                        False]])  # mark building polygons as 'yellow' for non-burnt for now
    return burnt_polys, building_polys


def label_building_polys(burnt_polys, building_polys):
    """Labels the building polygons as ('blue', True) if the building is destroyed."""

    for b in building_polys:
        for r in burnt_polys:
            if b[0].intersects(r):
                b[1] = [b[1][0], 'blue', True]  # mark building polygon as 'blue' if found in burnt region
                continue


def to_geojson_burnt(burnt_polys, building_polys):
    """Convert shapes into geojson with new labelled building footprints. """

    results = ({
        'type': 'Feature',
        'properties': {'color': 'red'},
        'geometry': geo.mapping(r)}
        for r in burnt_polys)

    list_results = list(results)

    # append the building footprints to geojson
    results_buildings = ({
        'type': 'Feature',
        'properties': {'BuildingID': b[1][0], 'color': b[1][1]},
        'geometry': geo.mapping(b[0])}
        for b in building_polys)

    list_results_buildings = list(results_buildings)

    collection = {
        'type': 'FeatureCollection',
        'features': list_results + list_results_buildings}

    return collection


def to_geojson_groundtruth(burnt_polys, data_labelled):
    """Convert shapes into geojson for the groundtruth."""

    results = ({
        'type': 'Feature',
        'properties': {'color': 'red'},
        'geometry': geo.mapping(r)}
        for r in burnt_polys)

    list_results = list(results)

    # append the building footprints to geojson
    results_buildings = ({
        'type': 'Feature',
        'properties': {'BuildingID': b['properties']['BuildingID'], 'color': b['properties']['color'],
                       'Burnt_Label': b['properties']['Burnt_Label']},
        'geometry': b['geometry']}
        for b in data_labelled['features'])

    list_results_buildings = list(results_buildings)

    collection = {
        'type': 'FeatureCollection',
        'features': list_results + list_results_buildings}

    return collection



def geojson_to_polygons_groundtruth(js_):
    """Convert geojson to polygons for the groundtruth map."""

    burnt_polys = []
    building_polys = []
    for i, feat in enumerate(js_['features']):
        o = {
            "coordinates": feat['geometry']['coordinates'],
            "type": feat['geometry']['type']
        }
        s = json.dumps(o)

        # convert to geojson.geometry.Polygon
        g1 = geojson.loads(s)

        # covert to shapely.geometry.polygon.Polygon
        g2 = shape(g1)

        if feat['properties']['color'] == 'red':  # red for the burnt region
            burnt_polys.append(g2)
        else:  # for the building poly
            if feat['properties']['Burnt_Label']:
                building_polys.append([g2, [feat['properties']['BuildingID'], 'blue',
                                            True]])  # mark building polygons as 'blue' for burnt for now
            else:
                building_polys.append([g2, [feat['properties']['BuildingID'], 'yellow',
                                            False]])  # mark building polygons as 'yellow' for non-burnt for now
    return burnt_polys, building_polys


def accuracy_measures(predictions, trues):
    """Accuracy measures for the predictions of the method vs the groundtruth.
    Prints a confusion matrix, accuracy, misclassifcation rate, true positieve rate, false positive rate, specificity, precision, prevalence.
    Returns the accuracy score, precision score, and recall score."""

    tn, fp, fn, tp = confusion_matrix(trues, predictions).ravel()
    print "\t(tn, fp, fn, tp) =", (tn, fp, fn, tp)

    # how often is classifier correct?
    print "\tAccuracy = {:.2%}".format(float(tp + tn) / len(trues))

    # how often is it wrong?
    print "\tMisclassification Rate = {:.2%}".format(float(fp + fn) / len(trues))

    # when actually yes, how often does it predict yes?
    print "\tTrue Positive Rate = {:.2%}".format(float(tp) / trues.count(True))

    # when actually no, how often does it predict yes?
    print "\tFalse Positive Rate = {:.2%}".format(float(fp) / trues.count(False))

    # when actually no, how often does it predict no?
    print "\tSpecificity = {:.2%}".format(float(tn) / trues.count(False))

    # when it predicts yes, how often is it correct?
    print "\tPrecision = {:.2%}".format(float(tp) / predictions.count(True))

    # how often does yes condition occur in our sample?
    print "\tPrevalence = {:.2%}\n".format(float(trues.count(True)) / len(trues))

    # return accuracy, precision, and recall score
    return accuracy_score(trues, predictions), precision_score(trues, predictions, average='binary'), recall_score(
        trues, predictions, average='binary')


def create_mask(predictions_2d, sizeX, sizeY, chip_shape):
    """Create a new binary mask of burn scar with the tiles."""

    # reshape predictions_2d
    predictions_2d_res = np.array(predictions_2d)
    predictions_2d_res = predictions_2d_res.reshape(sizeX, sizeY)

    # create new mask of area of interest
    new_mask = np.zeros((chip_shape[1], chip_shape[2]))
    for x in range(0, chip_shape[1], 256):
        for y in range(0, chip_shape[2], 256):
            new_mask[x:x + 256, y:y + 256] = predictions_2d_res[x / 256][y / 256]

    return new_mask


"""Functions for plots."""

def folium_map(geojson_to_overlay, layer_name, location, style_function=None, tiles='Stamen Terrain', zoom_start=16,
               show_layer_control=True, width='100%', height='75%', attr=None, map_zoom=18, max_zoom=20, tms=False,
               zoom_beyond_max=None, base_tiles='OpenStreetMap', opacity=1):
    """Folium map with Geojson layer and TMS tiles layer.
    This function requires geojson_to_overlay (geojson), layer_name (String), and location (map center tuple).
    You can also set tiles to the TMS URL and control map zoom."""

    m = folium.Map(location=location, zoom_start=zoom_start, width=width, height=height, max_zoom=map_zoom,
                   tiles=base_tiles)
    tiles = folium.TileLayer(tiles=tiles, attr=attr, name=attr, max_zoom=max_zoom)
    if tms is True:
        options = json.loads(tiles.options)
        options.update({'tms': True})
        tiles.options = json.dumps(options, sort_keys=True, indent=2)
        tiles._template = jinja2.Template(u"""
        {% macro script(this, kwargs) %}
            var {{this.get_name()}} = L.tileLayer(
                '{{this.tiles}}',
                {{ this.options }}
                ).addTo({{this._parent.get_name()}});
        {% endmacro %}
        """)
    if zoom_beyond_max is not None:
        options = json.loads(tiles.options)
        options.update({'maxNativeZoom': zoom_beyond_max, 'maxZoom': max_zoom})
        tiles.options = json.dumps(options, sort_keys=True, indent=2)
        tiles._template = jinja2.Template(u"""
        {% macro script(this, kwargs) %}
            var {{this.get_name()}} = L.tileLayer(
                '{{this.tiles}}',
                {{ this.options }}
                ).addTo({{this._parent.get_name()}});
        {% endmacro %}
        """)
    if opacity < 1:
        options = json.loads(tiles.options)
        options.update({'opacity': opacity})
        tiles.options = json.dumps(options, sort_keys=True, indent=2)
        tiles._template = jinja2.Template(u"""
        {% macro script(this, kwargs) %}
            var {{this.get_name()}} = L.tileLayer(
                '{{this.tiles}}',
                {{ this.options }}
                ).addTo({{this._parent.get_name()}});
        {% endmacro %}
        """)

    tiles.add_to(m)
    if style_function is not None:
        gj = folium.GeoJson(geojson_to_overlay, overlay=True, name=layer_name, style_function=style_function)
    else:
        gj = folium.GeoJson(geojson_to_overlay, overlay=True, name=layer_name)
    gj.add_to(m)

    if show_layer_control is True:
        folium.LayerControl().add_to(m)

    return m


def plot_array(array, subplot_ijk, title="", font_size=18, cmap=None):
    """Plot image with subplot.
    Requires image and subplot location (ie. (1,2,1)).
    You can also set title."""

    sp = plt.subplot(*subplot_ijk)
    sp.set_title(title, fontsize=font_size)
    plt.axis('off')
    plt.imshow(array, cmap=cmap)

def displayHTMLtable(acc_sent2, acc_wv03, acc, prec_sent2, prec_wv03, prec, recall_sent2, recall_wv03, recall):
    """Display accuracy scores in a table."""

    methods = ['Sent2 NBR', 'WV03 NBR', 'WV03 RF']
    accuracies = ["{:.2%}".format(acc_sent2), "{:.2%}".format(acc_wv03), "{:.2%}".format(acc)]
    precisions = ["{:.2%}".format(prec_sent2), "{:.2%}".format(prec_wv03), "{:.2%}".format(prec)]
    recalls = ["{:.2%}".format(recall_sent2), "{:.2%}".format(recall_wv03), "{:.2%}".format(recall)]

    data = methods + accuracies + precisions + recalls

    data = np.reshape(data, (4, 3)).T

    display(HTML(
        '<table style="width:100%;"><th>Method</th><th>Accuracy</th><th>Precision</th><th>Recall</th><tr>{}</tr></table>'.format(
            '</tr><tr>'.join(
                '<td>{}</td>'.format('</td><td>'.join(str(_) for _ in row)) for row in data)
            )
     ))

