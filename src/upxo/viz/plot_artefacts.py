# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 22:30:23 2025

@author: rg5749
"""
import numpy as np

def cuboid_data(center, size):
    """Create the data for plotting a cuboid with a given center and size."""
    o = [a - b / 2 for a, b in zip(center, size)]
    l, w, h = size
    x = [[o[0], o[0] + l, o[0] + l, o[0], o[0]]] * 4
    y = [[o[1], o[1], o[1] + w, o[1] + w, o[1]],
         [o[1], o[1], o[1] + w, o[1] + w, o[1]],
         [o[1], o[1], o[1], o[1], o[1]],
         [o[1] + w, o[1] + w, o[1] + w, o[1] + w, o[1] + w]]
    z = [[o[2], o[2], o[2], o[2], o[2]],
         [o[2] + h, o[2] + h, o[2] + h, o[2] + h, o[2] + h],
         [o[2], o[2], o[2] + h, o[2] + h, o[2]],
         [o[2], o[2], o[2] + h, o[2] + h, o[2]]]
    return np.array(x), np.array(y), np.array(z)
