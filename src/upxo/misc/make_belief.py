# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 13:50:18 2025

@author: rg5749
"""
import numpy as np

class uigrid:
    __slots__ = ('dim', 'npixels_max', 'vox_size', 'type', 'xbound', 'xinc',
                 'xls', 'xmax', 'xmin', 'ybound', 'yinc', 'yls', 'ymax',
                 'ymin', 'zbound', 'zinc', 'zls', 'zmax', 'zmin',)
    def __init__(self, dim=3, npixels_max=1.01E9, xmin=0.0, xinc=1.0, xmax=100.0,
                 ymin=0.0, yinc=1.0, ymax=100.0, zmin=0.0, zinc=1.0, zmax=100.0):
        self.dim = dim
        self.npixels_max = npixels_max
        self.vox_size = (xinc, yinc, zinc)
        self.type = 'hexahedral'  # No need for transformation unless it's dynamic
        self.xbound = (xmin, xmax, xinc)
        self.xinc = xinc
        # print(xmin, xmax, xinc)
        self.xls = np.arange(xmin, xmax, xinc)
        self.xmax = xmax
        self.xmin = xmin
        self.ybound = (ymin, ymax, yinc)
        self.yinc = yinc
        self.yls = np.arange(ymin, ymax, yinc)
        self.ymax = ymax
        self.ymin = ymin
        self.zbound = (zmin, zmax, zinc)
        self.zinc = zinc
        self.zls = np.arange(zmin, zmax, zinc)
        self.zmax = zmax
        self.zmin = zmin
