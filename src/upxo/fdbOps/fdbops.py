"""Small helpers for UPXO feature databases."""

from copy import deepcopy
import numpy as np


def add_feature_database_entry(fdb, fname, dnames, datas, info,
                               iterable_types=(list, tuple)):
    """Add a feature database entry to an existing FDB dictionary."""
    if not isinstance(info, dict):
        raise ValueError('info must be a dictionary')
    if not all(isinstance(key, str) for key in info.keys()):
        raise ValueError('infokey_list are not all strings.')
    if not isinstance(dnames, iterable_types):
        dnames = (dnames,)
    if not isinstance(datas, iterable_types):
        datas = (datas,)
    fdb[fname] = {'data': {}, 'info': info}
    for dname, data in zip(dnames, datas):
        fdb[fname]['data'][dname] = data
    return fdb


def validate_instance_name(instance_name):
    """Return whether an instance name is recognised."""
    if instance_name in ('base', 'lgi'):
        return True
    if isinstance(instance_name, str) and instance_name[:4] == 'twin':
        return True
    return False


def validate_fids(fids, reference_fids, number_types=(int, float, np.integer,
                                                     np.floating)):
    """Validate and filter feature IDs against a reference ID list."""
    validated = False
    if not isinstance(fids, (list, tuple, np.ndarray, set)):
        if not isinstance(fids, number_types):
            return validated, fids
        fids = [int(fids)]
    else:
        fids = np.array([int(fid) for fid in fids
                         if isinstance(fid, number_types)])
    reference_fids = set(reference_fids)
    valid_fids = np.array([fid for fid in fids if fid in reference_fids])
    validated = len(valid_fids) > 0
    return validated, valid_fids


def mask_feature_ids(fid_array, target_ids, fid_mask_value=-32,
                     non_fid_mask=False, non_fid_mask_value=-31):
    """Mask selected feature IDs in a labelled feature image."""
    if fid_mask_value >= 0:
        fid_mask_value = -fid_mask_value
    if non_fid_mask_value >= 0:
        non_fid_mask_value = -non_fid_mask_value

    data = deepcopy(fid_array)
    for fid in target_ids:
        data[np.where(data == fid)] = fid_mask_value

    if non_fid_mask:
        data[np.where(data != fid_mask_value)] = non_fid_mask_value
    else:
        data[np.where(data != fid_mask_value)] = 0
    return data


def parent_minus_child_coordinates(parent_ids, parent_coords_by_id,
                                   child_coords_by_parent_id,
                                   valid_parent_ids=None):
    """Return parent coordinate sets after removing child feature coordinates."""
    pc_rem = {parent_id: -1 for parent_id in parent_ids}
    valid_parent_ids = (
        set(parent_coords_by_id)
        if valid_parent_ids is None else set(valid_parent_ids)
    )

    for parent_id in parent_ids:
        if parent_id not in valid_parent_ids:
            continue
        child_coords = child_coords_by_parent_id.get(parent_id, {})
        if not child_coords:
            continue

        parent_coords = np.ascontiguousarray(parent_coords_by_id[parent_id])
        child_coords_acc = np.ascontiguousarray(
            np.vstack(tuple(child_coords.values()))
        )
        ncols = parent_coords.shape[1]
        mask = ~np.in1d(
            parent_coords.view([('', parent_coords.dtype)]*ncols),
            child_coords_acc.view([('', child_coords_acc.dtype)]*ncols)
        )
        pc_rem[parent_id] = parent_coords[mask]
    return pc_rem


def extract_nested_feature_coordinates(feature_coords_by_parent,
                                       use_parent_ids=True,
                                       parent_ids=None,
                                       child_ids=None,
                                       child_parent_map=None):
    """Extract nested feature coordinates by parent IDs or child IDs."""
    if use_parent_ids:
        return {
            parent_id: feature_coords_by_parent[parent_id]
            for parent_id in parent_ids
        }

    parent_ids_by_child = {
        child_id: child_parent_map[child_id] for child_id in child_ids
    }
    return {
        child_id: feature_coords_by_parent[parent_ids_by_child[child_id]][child_id]
        for child_id in child_ids
    }
