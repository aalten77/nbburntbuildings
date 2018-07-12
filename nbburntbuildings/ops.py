from past.utils import old_div
from shapely.geometry import shape, geo
from shapely import geometry
import requests
import os
from skimage import filters, morphology, measure, color, segmentation, exposure
from skimage.measure import label, regionprops
from scipy import ndimage as ndi
from functools import partial
import pyproj
from shapely import ops
import json
from rasterio import features
import numpy as np
import pickle
import jinja2
import json
import folium
from shapely.geometry import box
from gbdxtools import CatalogImage
from gbdxtools import IdahoImage
import numpy as np
from matplotlib import pyplot as plt, colors
import plotly.graph_objs as go
from branca.element import Element, Figure
from plotly.offline.offline import _plot_html
from plotly.graph_objs import Line
from IPython.display import HTML, display
from sklearn.metrics import confusion_matrix, recall_score, precision_score, accuracy_score
import geojson



# FUNCTIONS
def pixels_as_features(image, include_gabors=True):
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
    # Normalize images for better comparison.
    image = old_div((image - image.mean()), image.std())
    return np.sqrt(ndi.convolve(image, np.real(kernel), mode='wrap') ** 2 +
                   ndi.convolve(image, np.imag(kernel), mode='wrap') ** 2)


def calc_gabors(image, frequency=1, theta_vals=[0, 1, 2, 3]):
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

def reproject(geom, from_proj='EPSG:4326', to_proj='EPSG:26942'):
    tfm = partial(pyproj.transform, pyproj.Proj(init=from_proj), pyproj.Proj(init=to_proj))
    return ops.transform(tfm, geom)

def km2_area(polygons):
    reprojected_polygons = [reproject(p) for p in polygons]
    return ops.cascaded_union(reprojected_polygons).area * 1e-6


###### From plots.py ##############


# CONSTANTS
TMS_1040010039BAAF00 = 'https://s3.amazonaws.com/notebooks-small-tms/1040010039BAAF00/{z}/{x}/{y}.png'

COLORS = {'gray'       : '#8F8E8E',
          'white'      : '#FFFFFF',
          'brightgreen': '#00FF17',
          'red'        : '#FF0000',
          'cyan'       : '#1FFCFF'}


def bldg_styler(x):
    return {'fillOpacity': .25,
            'color'      : COLORS['cyan'] if x['properties']['blue'] == True else COLORS['white'],
            'fillColor'  : COLORS['gray'],
            'weight'     : 1}

# FUNCTIONS
def folium_map(geojson_to_overlay, layer_name, location, style_function=None, tiles='Stamen Terrain', zoom_start=16,
               show_layer_control=True, width='100%', height='75%', attr=None, map_zoom=18, max_zoom=20, tms=False,
               zoom_beyond_max=None, base_tiles='OpenStreetMap', opacity=1):
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


def get_idaho_tms_ids(image):
    ms_parts = {str(p['properties']['attributes']['idahoImageId']): str(
            p['properties']['attributes']['vendorDatasetIdentifier'].split(':')[1])
        for p in image._find_parts(image.cat_id, 'MS')}

    pan_parts = {str(p['properties']['attributes']['vendorDatasetIdentifier'].split(':')[1]): str(
            p['properties']['attributes']['idahoImageId'])
        for p in image._find_parts(image.cat_id, 'pan')}

    ms_idaho_ids = [(k, box(*IdahoImage(k).bounds).intersection(box(*image.bounds)).area) for k in ms_parts.keys() if
                    box(*IdahoImage(k).bounds).intersects(box(*image.bounds))]
    min_area = 0
    for ms_idaho_id in ms_idaho_ids:
        if ms_idaho_id[1] >= min_area:
            min_area = ms_idaho_id[1]
            the_ms_idaho_id = ms_idaho_id[0]

    pan_idaho_id = pan_parts[ms_parts[the_ms_idaho_id]]

    idaho_ids = {'ms_id' : the_ms_idaho_id,
                 'pan_id': pan_idaho_id}
    return idaho_ids


def get_idaho_tms_url(source_catid_or_image, gbdx):
    if type(source_catid_or_image) == str:
        image = CatalogImage(source_catid_or_image)
    elif '_ipe_op' in source_catid_or_image.__dict__.keys():
        image = source_catid_or_image
    else:
        err = "Invalid type for source_catid_or_image. Must be either a Catalog ID (string) or CatalogImage object"
        raise TypeError(err)

    url_params = get_idaho_tms_ids(image)
    url_params['token'] = str(gbdx.gbdx_connection.access_token)
    url_params['z'] = '{z}'
    url_params['x'] = '{x}'
    url_params['y'] = '{y}'
    url_params['bucket'] = str(image.ipe.metadata['image']['tileBucketName'])
    url_template = 'https://idaho.geobigdata.io/v1/tile/{bucket}/{ms_id}/{z}/{x}/{y}?bands=4,2,1&token={token}&panId={pan_id}'
    url = url_template.format(**url_params)

    return url


def plot_array(array, subplot_ijk, title="", font_size=18, cmap=None):
    sp = plt.subplot(*subplot_ijk)
    sp.set_title(title, fontsize=font_size)
    plt.axis('off')
    plt.imshow(array, cmap=cmap)


def plot_plotly(chart, width='100%', height=525):
    # produce the html in Ipython compatible format
    plot_html, plotdivid, width, height = _plot_html(chart, {'showLink': False}, True, width, height, True)
    # define the plotly js library source url
    head = '<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>'
    # extract the div element from the ipython html
    div = plot_html[0:plot_html.index('<script')]
    # extract the script element from the ipython html
    script = plot_html[plot_html.index('Plotly.newPlot'):plot_html.index('});</script>')] + ';'
    # combine div and script to build the body contents
    body = '<body>{div}<script>{script}</script></body>'.format(div=div, script=script)
    # instantiate a figure object
    figure = Figure()
    # add the head
    figure.header.add_child(Element(head))
    # add the body
    figure.html.add_child(Element(body))

    return figure


def plot_ribbon(df, x, ylower, yupper, name, ylab, ymax_factor=1, fillcolor='rgba(21,40,166,0.2)'):
    # Create a trace
    trace1 = go.Scatter(x=df[x],
                        y=df[yupper],
                        fill='tonexty',
                        fillcolor=fillcolor,
                        line=Line(color='transparent'),
                        showlegend=False,
                        name=name)

    trace2 = go.Scatter(x=df[x],
                        y=df[ylower],
                        fill='tonexty',
                        fillcolor='transparent',
                        line=Line(color='transparent'),
                        showlegend=False,
                        name=name)

    graph_data = [trace2, trace1]
    yaxis = dict(title=ylab,
                 range=(0, max(df[yupper]) * ymax_factor))
    graph_layout = go.Layout(yaxis=yaxis, showlegend=False)
    fig = go.Figure(data=graph_data, layout=graph_layout)

    return fig


def plot_results(df, x, y, name, ylab, ymax_factor=1):
    # Create a trace
    trace = go.Scatter(x=df[x],
                       y=df[y],
                       name=name)
    graph_data = [trace]
    yaxis = dict(title=ylab,
                 range=(0, max(df[y])*ymax_factor))
    graph_layout = go.Layout(yaxis=yaxis, showlegend=False)
    fig = go.Figure(data=graph_data, layout=graph_layout)

    return fig


def plot_multi_trace(df, x, y, factor_var, ymax_factor=1.):
    graph_layout = go.Layout(showlegend=True)

    graph_data = []
    factor_vals = df[factor_var].unique()
    domain_breaks = np.linspace(0, 1, len(factor_vals) + 1)
    for i, factor_val in enumerate(factor_vals):
        df_subset = df[df[factor_var] == factor_val]
        # Create a trace
        x_anchor = 'x1'
        y_anchor = 'y{}'.format(i + 1)
        new_trace = go.Scatter(x=df_subset[x],
                               y=df_subset[y],
                               name=factor_val,
                               mode='lines+markers',
                               xaxis=x_anchor,
                               yaxis=y_anchor)
        graph_data.append(new_trace)
        yaxis = dict(range=(0, max(df_subset[y]) * ymax_factor),
                     anchor=x_anchor,
                     domain=(domain_breaks[i], domain_breaks[i + 1] - 0.03))
        if i == 0:
            y_axis_name = 'yaxis'
        else:
            y_axis_name = 'yaxis{}'.format(i + 1)
        graph_layout[y_axis_name] = yaxis.copy()
    fig = go.Figure(data=graph_data, layout=graph_layout)

    return fig

def displayHTMLtable():
    display(HTML(
        '<table style="width:100%;"><th>Method</th><th>Total Area Burnt (km<sup>2</sup>)</th><th>Total Bldgs Burnt</th><tr>{}</tr></table>'.format(
            '</tr><tr>'.join(
                '<td>{}</td>'.format('</td><td>'.join(str(_) for _ in row)) for row in data)
            )
     ))

#### more functions from Notebooks ######
def clean(img):
    label_img = label(img, connectivity=2)
    props = sorted(regionprops(label_img), key=lambda x: x.area)
    clean = morphology.binary_closing(img)

    clean = morphology.remove_small_holes(clean)
    return morphology.remove_small_objects(clean,
                                           int(np.floor(props[-1].area) / 10), connectivity=2)


# convert shapes to geojson
def to_geojson(shapes, buildings):
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


# convert geojson into Shapely Polygons
def geojson_to_polygons(js_):
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
    for b in building_polys:
        for r in burnt_polys:
            if b[0].intersects(r):
                b[1] = [b[1][0], 'blue', True]  # mark building polygon as 'blue' if found in burnt region
                continue


# convert shapes to geojson
def to_geojson_burnt(burnt_polys, building_polys):
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


# convert shapes to geojson
def to_geojson_groundtruth(burnt_polys, data_labelled):
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


# convert geojson into Shapely Polygons
def geojson_to_polygons_groundtruth(js_):
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
