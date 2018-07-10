from past.utils import old_div
from shapely.geometry import shape
from shapsely import sgeometry
import requests
import os
from skimage import filters, morphology, measure, color, segmentation, exposure
from scipy import ndimage as ndi
from functools import partial
import pyproj
from shapely import ops
import json
from rasterio import features
import numpy as np
import pickle

# FUNCTIONS
def calc_rsi(image):
    # roll axes to conventional row,col,depth
    img = np.rollaxis(image, 0, 3)

    # bands: Coastal(0), Blue(1), Green(2), Yellow(3), Red(4), Red-edge(5), NIR1(6), NIR2(7)) Multispectral
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

    rsi = np.stack([arvi, dd, gi2, gndvi, ndre, ndvi, ndvi35, ndvi84, nirry, normnir, psri, rey, rvi,
                    sa, vi1, vire, br, gr, rr], axis=2)

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

def reproject(geom, from_proj='EPSG:4326', to_proj='EPSG:26942'):
    tfm = partial(pyproj.transform, pyproj.Proj(init=from_proj), pyproj.Proj(init=to_proj))
    return ops.transform(tfm, geom)

def km2_area(polygons):
    reprojected_polygons = [reproject(p) for p in polygons]
    return ops.cascaded_union(reprojected_polygons).area * 1e-6
