# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 18:38:11 2025

@author: rg5749
"""

import numpy as np
from collections import defaultdict

def get_voxel_locations(lgi):
    unique_labels, inv_indices = np.unique(lgi, return_inverse=True)
    coords = np.array(np.where(np.ones_like(lgi))).T
    grain_locs = defaultdict(list)

    for i, label_id in enumerate(unique_labels):
        if label_id != 0:  # Usually skip background label 0
            grain_mask = (inv_indices.reshape(lgi.shape) == i)
            grain_locs[label_id] = coords[grain_mask]
    return dict(grain_locs)

# Example (using the same lgi as before)
lgi = np.array([
    [0, 0, 1, 1],
    [0, 1, 1, 1],
    [0, 0, 2, 2],
    [2, 2, 2, 0]
])

voxel_locations = get_voxel_locations(lgi)
print(voxel_locations)
