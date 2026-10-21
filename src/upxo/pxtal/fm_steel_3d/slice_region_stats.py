"""Region properties of PAGs and packets on 2-D slices of a structure.

A structure is cut by evenly spaced slices in one of the planes ``'xy'``,
``'yz'`` and ``'xz'``. On each slice the regions of the PAG label map (grains
merged into their PAG) and of the grain label map (the packets) are measured
and the values of all slices are collected per group and property.
"""
from typing import Callable, Dict, Iterable, List, Optional

import numpy as np

__all__ = ['REGION_PROPERTIES', 'SLICE_PLANES', 'GROUPS', 'pag_label_grid',
           'periodic_neighbour_counts', 'region_property_values',
           'pag_slice_distributions']

REGION_PROPERTIES = ('area', 'aspect_ratio', 'perimeter', 'solidity', 'n_neighbours')
SLICE_PLANES = ('xy', 'yz', 'xz')
GROUPS = ('pag', 'packet')


def pag_label_grid(lgi: np.ndarray, clusters_dict: dict, dtype=np.int32) -> np.ndarray:
    """Grain label grid with every grain relabelled by the id of its PAG.

    Grains in no cluster, and void, become 0. The result has type ``dtype``.
    """
    max_gid = int(lgi.max())
    lut = np.zeros(max_gid + 1, dtype=dtype)
    for pag_id, gids in clusters_dict.items():
        for gid in gids:
            if 0 < gid <= max_gid:
                lut[gid] = int(pag_id)
    return lut[lgi]


def periodic_neighbour_counts(label_img: np.ndarray) -> Dict[int, int]:
    """Number of distinct regions edge-adjacent to each region of a 2-D label
    image. Edges wrap around, so regions on opposite edges of the image are
    adjacent. Void (label 0 or less) is not a neighbour."""
    labels = np.unique(label_img)
    labels = labels[labels > 0]
    neighbours = {int(lbl): set() for lbl in labels}
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        shifted = np.roll(np.roll(label_img, dr, 0), dc, 1)
        mask = (label_img != shifted) & (label_img > 0) & (shifted > 0)
        ys, xs = np.where(mask)
        for y, x in zip(ys.tolist(), xs.tolist()):
            a, b = int(label_img[y, x]), int(shifted[y, x])
            neighbours[a].add(b)
            neighbours[b].add(a)
    return {lbl: len(s) for lbl, s in neighbours.items()}


def region_property_values(label_img: np.ndarray, voxel_size: float,
                           properties: Iterable[str]) -> Dict[str, List[float]]:
    """One value per region of a 2-D label image for each requested property.

    Properties (``REGION_PROPERTIES``): 'area' (voxel_size^2), 'aspect_ratio'
    (major / minor axis of the fitted ellipse, 1 when the minor axis is 0),
    'perimeter' (voxel_size), 'solidity' and 'n_neighbours'
    (``periodic_neighbour_counts``). Regions are taken in order of label; an
    image without regions gives empty lists.
    """
    from skimage.measure import regionprops

    properties = list(properties)
    unknown = set(properties) - set(REGION_PROPERTIES)
    if unknown:
        raise ValueError(f"unknown properties {sorted(unknown)}; choose from {REGION_PROPERTIES}")
    out: Dict[str, List[float]] = {p: [] for p in properties}
    label_img = label_img.astype(np.int32)
    if np.count_nonzero(label_img) == 0:
        return out
    neighbour_count = periodic_neighbour_counts(label_img) if 'n_neighbours' in properties else {}
    for region in regionprops(label_img):
        if 'area' in properties:
            out['area'].append(region.area * voxel_size * voxel_size)
        if 'aspect_ratio' in properties:
            minor = region.axis_minor_length
            out['aspect_ratio'].append(region.axis_major_length / minor if minor > 0 else 1.0)
        if 'perimeter' in properties:
            out['perimeter'].append(region.perimeter * voxel_size)
        if 'solidity' in properties:
            out['solidity'].append(region.solidity)
        if 'n_neighbours' in properties:
            out['n_neighbours'].append(float(neighbour_count.get(int(region.label), 0)))
    return out


def pag_slice_distributions(lgi: np.ndarray, clusters_dict: dict, voxel_size: float,
                            plane: str, n_slices: int, groups: Iterable[str],
                            properties: Iterable[str],
                            progress: Optional[Callable[[str, int, int, int], None]] = None
                            ) -> Dict[str, Dict[str, List[float]]]:
    """Property values of the regions on evenly spaced slices, per group.

    Parameters
    ----------
    lgi : ndarray
        Grain label grid.
    clusters_dict : dict
        PAG id -> grain ids.
    voxel_size : float
    plane : {'xy', 'yz', 'xz'}
        Plane of the slices.
    n_slices : int
        Number of slices, from the second to the last-but-one voxel layer.
    groups : iterable of {'pag', 'packet'}
        'pag' measures the regions of the PAG label map and 'packet' those of
        the grain label map.
    properties : iterable of str
        From ``REGION_PROPERTIES``.
    progress : callable(plane, index, done, total), optional
        Called before each slice is measured.

    Returns
    -------
    dict
        ``{group: {property: [value, ...]}}`` over all slices.
    """
    if plane not in SLICE_PLANES:
        raise ValueError(f"plane must be one of {SLICE_PLANES}, got {plane!r}.")
    groups = list(groups)
    unknown = set(groups) - set(GROUPS)
    if unknown:
        raise ValueError(f"unknown groups {sorted(unknown)}; choose from {GROUPS}")
    properties = list(properties)

    nx, ny, nz = lgi.shape
    pag_lgi = pag_label_grid(lgi, clusters_dict)
    size, cut = {
        'xy': (nz, lambda i: (pag_lgi[:, :, i], lgi[:, :, i])),
        'yz': (nx, lambda i: (pag_lgi[i, :, :], lgi[i, :, :])),
        'xz': (ny, lambda i: (pag_lgi[:, i, :], lgi[:, i, :])),
    }[plane]

    collected = {group: {p: [] for p in properties} for group in groups}
    for done, index in enumerate(np.linspace(1, size - 2, n_slices, endpoint=True, dtype=int), start=1):
        if progress is not None:
            progress(plane, int(index), done, n_slices)
        pag_slice, packet_slice = cut(int(index))
        if 'pag' in collected:
            for p, values in region_property_values(pag_slice, voxel_size, properties).items():
                collected['pag'][p].extend(values)
        if 'packet' in collected:
            for p, values in region_property_values(packet_slice, voxel_size, properties).items():
                collected['packet'][p].extend(values)
    return collected
