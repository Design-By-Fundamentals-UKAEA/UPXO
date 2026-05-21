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
        """Initialise from a bounding box and a dict of grain multi-lines."""
        self.bounds = bounds
        self.mslines2d = mslines2d
        self.gid = list(self.mslines2d.keys())
        #self.are_all_grains_closed()

    def __iter__(self):
        """Iterate over grain ids in ``self.gid``."""
        pass

    def __next__(self):
        """Return the next grain in the iteration sequence."""
        pass

    def __eq__(self):
        """Representativeness qualificatipon."""
        pass

    def check_closures(self):
        """Check whether each grain boundary forms a closed loop."""
        closures = []
        for g in gs.mslines2d:
            g = gs.mslines2d[2]
            first, last = g[0], g[-1]
            closures.append(first.nodes[0].eq_fast(last.nodes[-1]))
