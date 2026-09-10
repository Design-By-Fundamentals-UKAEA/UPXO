"""
labeling.py
===========
Sample grain labels from a voxel array onto BCC lattice vertices.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from upxo.meshing.cleaving.config import LabelingConfig
from upxo.meshing.cleaving.lattice import BCCLattice


def sample_labels(
        lattice: BCCLattice,
        lgi: np.ndarray,
        config: Optional[LabelingConfig] = None,
) -> np.ndarray:
    """
    Sample the nearest voxel's label onto every lattice vertex.

    Vertices outside the real (unpadded) domain -- i.e. in the padding
    region added by ``build_bcc_lattice`` -- receive ``config.wall_label``.

    Parameters
    ----------
    lattice : BCCLattice
    lgi     : ndarray (nx, ny, nz) int -- labelled grain image; must match
              ``lattice.grid_shape``.
    config  : LabelingConfig or None

    Returns
    -------
    ndarray (V,) int64 -- one label per lattice vertex.
    """
    cfg = config or LabelingConfig()
    nx, ny, nz = lattice.grid_shape
    if lgi.shape != (nx, ny, nz):
        raise ValueError(
            f'lgi.shape {lgi.shape} != lattice.grid_shape {(nx, ny, nz)}')

    # Vertices sit at axis-index * voxel_size. Primal vertices land exactly
    # on integer axis indices; dual vertices sit at half-integer offsets,
    # so round-to-nearest gives the correct nearest-voxel index for both.
    idx = np.rint(lattice.vertices / lattice.voxel_size).astype(np.int64)

    inside = (
        (idx[:, 0] >= 0) & (idx[:, 0] < nx) &
        (idx[:, 1] >= 0) & (idx[:, 1] < ny) &
        (idx[:, 2] >= 0) & (idx[:, 2] < nz)
    )

    labels = np.full(len(lattice.vertices), cfg.wall_label, dtype=np.int64)
    ii, jj, kk = idx[inside, 0], idx[inside, 1], idx[inside, 2]
    labels[inside] = lgi[ii, jj, kk]
    return labels
