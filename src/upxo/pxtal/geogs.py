# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 21:25:32 2024

@author: Dr. Sunil Anandatheertha
"""
import numpy as np
import rasterio
from copy import deepcopy
import matplotlib.pyplot as plt
from rasterio.features import shapes
from shapely.strtree import STRtree
from shapely.geometry import Point
from upxo._sup import dataTypeHandlers as dth
from shapely.geometry import LineString, MultiLineString
from shapely.geometry import shape as ShShape
from shapely.geometry import Polygon, MultiPolygon
from shapely.geometry.collection import GeometryCollection


class geogs2d():
    __slots__ = ('mslines2d', 'jnp', 'grains', 'neigh_gid', 'gid',
                 'ea', 'tess', 'bounds', 'gid')

    def __init__(self, bounds, mslines2d):
        self.bounds = bounds
        self.mslines2d = mslines2d
        self.gid = list(self.mslines2d.keys())
        #self.are_all_grains_closed()

    def __iter__(self):
        pass

    def __next__(self):
        pass

    def __eq__(self):
        """Representativeness qualificatipon."""
        pass

    def check_closures(self):
        closures = []
        for g in gs.mslines2d:
            g = gs.mslines2d[2]
            first, last = g[0], g[-1]
            closures.append(first.nodes[0].eq_fast(last.nodes[-1]))


from descartes import PolygonPatch  # For easier plotting of Shapely polygons

def plot_polygon(polygon, ax=None, color="blue", alpha=0.5, **kwargs):
    """Plots a Shapely Polygon.

    Args:
        polygon (Polygon): The Polygon object to plot.
        ax (matplotlib.axes.Axes, optional): The axes to plot on. If None, the current axes will be used.
        color (str, optional): The color of the polygon. Defaults to "blue".
        alpha (float, optional): The transparency of the polygon. Defaults to 0.5.
        **kwargs: Additional keyword arguments to pass to PolygonPatch.
    """
    if ax is None:
        ax = plt.gca()  # Get the current axes if not provided

    # Use PolygonPatch for easy plotting
    patch = PolygonPatch(polygon, fc=color, ec="black", alpha=alpha, **kwargs)
    ax.add_patch(patch)

    # Set plot limits based on the polygon's extent
    minx, miny, maxx, maxy = polygon.bounds
    ax.set_xlim(minx - 0.1, maxx + 0.1)  # Add a bit of padding
    ax.set_ylim(miny - 0.1, maxy + 0.1)

    ax.set_aspect('equal')  # Ensure equal aspect ratio




gid = 4
g = gs.mslines2d[gid]
first, last = g[0], g[-1]
fig, ax = plt.subplots()
first.plot(ax)
last.plot(ax)

gid = 1
g = gs.mslines2d[gid]
coords = g[0].get_node_coords()
for _g_ in g[1:]:
    coords = np.vstack((coords, _g_.get_node_coords()))


gid = 1
g = sorted_segs[gid]

fig, ax = plt.subplots()
for _g_ in g:
    _g_.plot(ax)

for _g_ in g:
    print(_g_.nodes)


gid = 2
g = sorted_segs[gid]
coords = g[0].get_node_coords()
for _g_ in g[1:]:
    coords = np.vstack((coords, _g_.get_node_coords()))

plt.plot(coords[:, 0], coords[:, 1], '-k.')
for i, coord in enumerate(coords):
    plt.text(coord[0], coord[1], i)






polygon = Polygon(coords)
polygon
#type(polygon)


#plot_polygon(polygon)
for gid in reordering_needed:
    g = sorted_segs[gid]
    coords = g[0].get_node_coords()
    for _g_ in g[1:]:
        coords = np.vstack((coords, _g_.get_node_coords()))

    plt.plot(coords[:, 0], coords[:, 1], '-k.')
    for i, coord in enumerate(coords):
        plt.text(coord[0], coord[1], i)


no_reordering_needed = list(set(list(geom.gid))-set(reordering_needed))
plt.figure()
for gid in no_reordering_needed:
    g = sorted_segs[gid]
    coords = g[0].get_node_coords()
    for _g_ in g[1:]:
        coords = np.vstack((coords, _g_.get_node_coords()))

    plt.plot(coords[:, 0], coords[:, 1], '-k.')
    for i, coord in enumerate(coords):
        plt.text(coord[0], coord[1], i)
