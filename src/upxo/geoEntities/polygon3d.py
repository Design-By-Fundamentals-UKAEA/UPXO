# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 05:42:54 2024

@author: rg5749
"""
import numpy as np
import shapely

class Polygon3d():
    __slots__ = ('points', 'be')
    def __init__(self, points, be):
        self.points = points
        self.be = be

    def project(self, plane='xy'):
        pass
