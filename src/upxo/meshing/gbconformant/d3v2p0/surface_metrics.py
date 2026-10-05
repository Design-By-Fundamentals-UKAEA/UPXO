"""Shape measures of the grain-boundary surface."""
import numpy as np


def voxel_like_fraction(points, triangles, exterior=None, grain_pairs=None, grain_ids=(), tolerance_deg=5.):
    """Share of internal grain-boundary area whose triangles lie within
    ``tolerance_deg`` of a voxel face plane (axis-aligned normal).

    exterior marks RVE-face triangles, which are excluded. With grain_pairs and
    grain_ids, also returns the share on interfaces touching those grains.
    Returns dict(voxel_like, smooth, on_given_grains).
    """
    if not 0 < tolerance_deg < 45:
        raise ValueError('tolerance_deg must lie in (0, 45)')
    x = np.asarray(points, float)[np.asarray(triangles)]
    n = np.cross(x[:, 1] - x[:, 0], x[:, 2] - x[:, 0])
    area = np.linalg.norm(n, axis=1) / 2
    aligned = np.abs(n).max(axis=1) > np.cos(np.radians(tolerance_deg)) * 2 * np.maximum(area, 1e-300)
    inner = np.ones(len(x), bool) if exterior is None else ~np.asarray(exterior, bool)
    total = area[inner].sum()
    if total <= 0:
        raise ValueError('No internal grain-boundary area')
    share = float(area[aligned & inner].sum() / total)
    out = dict(voxel_like=share, smooth=1. - share, on_given_grains=None)
    if grain_pairs is not None and len(grain_ids):
        given = np.isin(np.asarray(grain_pairs), list(grain_ids)).any(axis=1)
        out['on_given_grains'] = float(area[aligned & inner & given].sum() / total)
    return out
