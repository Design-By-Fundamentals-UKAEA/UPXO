"""
config.py
=========
Configuration dataclasses for the cleaving conformal tet-meshing module.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class LatticeConfig:
    """Controls for BCC background lattice construction."""
    voxel_size: float = 1.0
    # Layers of padding (in primal-grid units) added beyond the real domain
    # on each side, so every long edge inside the real domain has full
    # 4-vertex flanking support. 1 is the minimum sufficient value -- do
    # not lower it.
    pad: int = 1


@dataclass
class LabelingConfig:
    """Controls for sampling grain labels onto lattice vertices."""
    # Label assigned to any lattice vertex whose nearest voxel falls outside
    # the real (unpadded) domain. Must not coincide with any real grain ID.
    wall_label: int = 32767
