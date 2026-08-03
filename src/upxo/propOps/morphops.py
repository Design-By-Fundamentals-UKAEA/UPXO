"""Morphological property helpers for labelled grain structures."""

from math import floor
import numpy as np


def volumes_with_bincount(label_image, nlabels=None):
    """Return voxel counts indexed by label ID."""
    label_image = np.asarray(label_image)
    if nlabels is None:
        nlabels = int(label_image.max())
    return np.bincount(label_image.ravel(), minlength=int(nlabels)+1)


def volume_dict(label_image, labels):
    """Return ``{label: voxel_count}`` for requested labels."""
    counts = volumes_with_bincount(label_image, max(labels))
    return {int(label): counts[int(label)] for label in labels}


def values_array(prop_dict):
    """Return property dictionary values as an array."""
    return np.array(list(prop_dict.values()))


def gids_with_value(prop_dict, value):
    """Return one-based IDs whose property equals ``value``."""
    vals = values_array(prop_dict)
    return np.where(vals == value)[0] + 1


def gids_with_min(prop_dict):
    """Return IDs whose property equals the minimum value."""
    vals = values_array(prop_dict)
    return np.where(vals == vals.min())[0] + 1


def gids_with_max(prop_dict):
    """Return IDs whose property equals the maximum value."""
    vals = values_array(prop_dict)
    return np.where(vals == vals.max())[0] + 1


def gids_le(prop_dict, threshold):
    """Return IDs whose property is less than or equal to threshold."""
    return np.where(values_array(prop_dict) <= threshold)[0] + 1


def gids_ge(prop_dict, threshold):
    """Return IDs whose property is greater than or equal to threshold."""
    return np.where(values_array(prop_dict) >= threshold)[0] + 1


def prop_values_for_gids(prop_dict, gids):
    """Return floored property values for requested IDs."""
    return [floor(prop_dict[gid]) for gid in gids]


def gids_in_range(prop_dict, low=10, high=15, low_ineq='ge',
                  high_ineq='le'):
    """Return IDs whose property values fall in the requested range."""
    low_ineq = low_ineq if low_ineq in ('ge', 'gt') else 'ge'
    high_ineq = high_ineq if high_ineq in ('le', 'lt') else 'le'
    prop = values_array(prop_dict)

    low_mask = prop >= low if low_ineq == 'ge' else prop > low
    high_mask = prop <= high if high_ineq == 'le' else prop < high
    gids = np.argwhere(np.logical_and(low_mask, high_mask)).squeeze() + 1
    if gids.ndim == 0:
        gids = np.expand_dims(gids, 0)
    return gids


def voxel_volume(spacing):
    """Return voxel volume from spacing."""
    return np.prod(spacing)


def voxel_surface_areas(spacing, ret_metric='mean'):
    """Return voxel face areas for a spacing tuple."""
    areas = [spacing[0]*spacing[1], spacing[1]*spacing[2],
             spacing[2]*spacing[0]]
    if ret_metric == 'mean':
        return sum(areas)/3.0
    if ret_metric == 'min':
        return min(areas)
    if ret_metric == 'max':
        return max(areas)
    if ret_metric == 'all':
        return areas
    raise ValueError("ret_metric must be 'mean', 'min', 'max' or 'all'.")
