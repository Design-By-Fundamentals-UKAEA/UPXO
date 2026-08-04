"""
twin3d.py
=========
3D voxel-space twin lamella introduction for representative grain
structure generation.

This module is the 3D counterpart of the polygon-space twin introduction
functions in :mod:`upxo.pxtalops.gssmooth2d`.  It uses the
:class:`~upxo.geoEntities.plane.Plane` class for perpendicular-distance
voxel selection and crystallographically-derived habit plane normals
(computed by
:func:`~upxo.xtalphy.crystal_orientation.compute_s3_habit_plane_3d`) to
ensure lamella geometry is physically consistent with the host grain's
crystal orientation.
"""

import numpy as np
from upxo.geoEntities.plane import Plane


def introduce_twin_lamella_3d(
        lgi,
        host_gid,
        host_centroid,
        normal,
        half_width,
        next_gid,
        abrupt=False,
        rng=None,
        host_coords=None,
):
    """
    Carve a single Sigma3 twin lamella out of a host grain in a 3D lgi
    array.

    Uses :class:`~upxo.geoEntities.plane.Plane` to compute perpendicular
    distances from each host-grain voxel to the habit plane, then selects
    voxels within *half_width* as the twin lamella region.

    This mirrors the production morphological pipeline
    (``identify_twins_gid``) but drives the plane normal from the host's
    crystallographic {111} habit plane instead of three randomly chosen
    voxel points, ensuring lamella geometry and crystal orientation remain
    consistent.

    Parameters
    ----------
    lgi : ndarray (nx, ny, nz), int
        3D labelled grain image.  Modified in place.
    host_gid : int
        Grain ID of the host grain to twin.
    host_centroid : array-like (3,)
        Centroid of the host grain's voxel coordinates; used as the
        plane origin.
    normal : array-like (3,)
        Unit normal of the habit plane in the sample frame (typically
        from
        :func:`~upxo.xtalphy.crystal_orientation.compute_s3_habit_plane_3d`).
    half_width : float
        Lamella half-thickness in voxels.
    next_gid : int
        Grain ID to assign to the new twin region.
    abrupt : bool
        If True, truncate the lamella to one half of the host grain
        along an in-plane axis, mimicking EBSD-measured abrupt twins.
    rng : numpy.random.Generator or None
        Required when *abrupt* is True.
    host_coords : ndarray (M, 3) or None
        Pre-built coordinate array of host-grain voxels.  When provided
        the O(N_total_voxels) ``np.argwhere`` scan is skipped entirely,
        giving a large speedup for large domains.  If None the scan is
        performed as a fallback.

    Returns
    -------
    twin_voxels : ndarray (N, 3) or None
        Global voxel coordinates of the twin region, or None if no
        voxels were selected.
    """
    coords = host_coords if host_coords is not None else np.argwhere(lgi == host_gid)
    if coords.shape[0] == 0:
        return None

    plane = Plane(point=host_centroid, normal=normal)
    dist = plane.calc_perp_distances(coords.astype(float), signed=True)
    sel = np.abs(dist) <= half_width

    if abrupt and np.any(sel) and rng is not None:
        arbitrary = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(arbitrary, normal)) > 0.9:
            arbitrary = np.array([0.0, 1.0, 0.0])
        in_plane = np.cross(normal, arbitrary)
        in_plane = in_plane / np.linalg.norm(in_plane)
        extent = (coords[sel].astype(float) - host_centroid) @ in_plane
        cutoff = np.median(extent)
        side = rng.choice([-1.0, 1.0])
        keep_local = side * (extent - cutoff) >= 0
        sel_idx = np.where(sel)[0]
        sel[sel_idx[~keep_local]] = False

    if not np.any(sel):
        return None

    twin_voxels = coords[sel]
    lgi[twin_voxels[:, 0], twin_voxels[:, 1], twin_voxels[:, 2]] = next_gid
    return twin_voxels
