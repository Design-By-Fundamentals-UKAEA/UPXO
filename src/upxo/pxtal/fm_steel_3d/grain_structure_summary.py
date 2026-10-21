"""Summary numbers of a labelled grain structure.

``voxel_spacing`` gives the voxel size along each axis for a domain of given
physical dimensions and reports how far the voxels are from being cubic.
``grain_structure_summary`` gives grain-size statistics and the means of
equivalent-sphere diameter, sphericity and neighbour count.
"""
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from .geom_metrics_3d import feature_distribution

__all__ = ['voxel_spacing', 'grain_structure_summary']


def voxel_spacing(physical_dimensions: Sequence[float], shape: Sequence[int]
                  ) -> Tuple[float, float, float, float]:
    """Voxel size along x, y and z, and the relative deviation from cubic voxels.

    Parameters
    ----------
    physical_dimensions : (lx, ly, lz)
    shape : (nx, ny, nz)

    Returns
    -------
    (vx, vy, vz, deviation)
        ``deviation`` is the largest of |vx - vy| / vx, |vx - vz| / vx and
        |vy - vz| / vy, and 0 when ``vx`` is not positive. An axis with no
        voxels has the spacing 1.
    """
    lx, ly, lz = physical_dimensions
    nx, ny, nz = shape
    vx = lx / nx if nx > 0 else 1.0
    vy = ly / ny if ny > 0 else 1.0
    vz = lz / nz if nz > 0 else 1.0
    deviation = 0.0
    if vx > 0:
        deviation = max(abs(vx - vy) / vx, abs(vx - vz) / vx, abs(vy - vz) / max(vy, 1e-12))
    return vx, vy, vz, deviation


def grain_structure_summary(lfi: np.ndarray, voxel_size: float,
                            metrics: Optional[dict] = None) -> Dict[str, float]:
    """Grain-size statistics and mean morphology values of a label grid.

    Parameters
    ----------
    lfi : ndarray
        Grain label grid; 0 is void.
    voxel_size : float
    metrics : dict, optional
        ``geom_metrics_3d.all_metrics(lfi)`` computed earlier.

    Returns
    -------
    dict
        n_grains; mean_vox, min_vox, max_vox, std_vox (voxels per grain);
        mean_volume, min_volume, max_volume, std_volume (the same in physical
        units); mean_eqdia, mean_sphericity, mean_nnb (means over grains).
        All are 0 for a grid without grains.
    """
    ids, counts = np.unique(lfi, return_counts=True)
    if len(ids) and ids[0] == 0:
        counts = counts[1:]
    n_grains = len(counts)

    def stat(function, scale=1.0):
        return float(function(counts)) * scale if n_grains > 0 else 0.0

    volume_scale = voxel_size ** 3
    summary = {
        'n_grains': n_grains,
        'mean_vox': stat(np.mean), 'min_vox': stat(np.min),
        'max_vox': stat(np.max), 'std_vox': stat(np.std),
        'mean_volume': stat(np.mean, volume_scale), 'min_volume': stat(np.min, volume_scale),
        'max_volume': stat(np.max, volume_scale), 'std_volume': stat(np.std, volume_scale),
    }
    for name, key in (('mean_eqdia', 'eqdia'), ('mean_sphericity', 'sphericity'),
                      ('mean_nnb', 'nnb')):
        values = feature_distribution(lfi, voxel_size, key, metrics=metrics)
        summary[name] = float(values.mean()) if len(values) else 0.0
    return summary
