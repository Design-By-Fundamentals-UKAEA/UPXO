"""
Polycrystal (pxtal) core package for UPXO grain-structure data models.

Houses:

* MCGS temporal slices — ``mcgs2_temporal_slice``, ``mcgs3_temporal_slice``
* Voronoi tessellation — ``vortess2d`` (``gtess2d``), ``vortess3d`` (``gtess3d``)
* Hierarchical FM steel — ``fm_steel_3d``
* Twinned FCC — ``twinned_simple_3d``
* Supporting geometry / grid / image utilities

Generation drivers live in ``upxo.ggrowth``; this package focuses on structure
objects, specialised pipelines, and related operations.
"""

from upxo.pxtal.vortess2d import gtess2d
from upxo.pxtal.geotess import geotess2d
from upxo.pxtal.voronoi_tessellation_2d import (
    generate_voronoi_2d,
    compute_standard_voronoi_2d,
    compute_power_diagram_2d,
    cvt_relax_2d,
    perturb_interfaces_2d,
)

__all__ = [
    'gtess2d',
    'geotess2d',
    'generate_voronoi_2d',
    'compute_standard_voronoi_2d',
    'compute_power_diagram_2d',
    'cvt_relax_2d',
    'perturb_interfaces_2d',
]

