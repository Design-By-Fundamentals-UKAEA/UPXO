"""
twinned_simple_3d
=================
3D representative grain structure generation for simple twinned FCC
polycrystals (e.g. OFHC copper, Cu-Cr-Zr).

Hierarchy: host grain -> primary twin -> secondary twin (2-3 levels).
This is the 3D counterpart of ``twinned_simple_2d``, and the FCC
analogue of ``pxtal.fm_steel_3d`` for FM steels.

Public API
----------
TwinnedSimple3DBase         -- core grain structure, host allocation
TwinGenerator3D             -- primary and secondary twin introduction
OrientationAssigner3D       -- conflict-free EBSD-sampled orientations
StructureCleaner3D          -- voxel spike removal and lobe splitting
RepresentativenessValidator3D -- multi-axis 2D slice MDF validation
"""

from upxo.pxtal.twinned_simple_3d.base_3d import TwinnedSimple3DBase
from upxo.pxtal.twinned_simple_3d.twin_generator_3d import TwinGenerator3D
from upxo.pxtal.twinned_simple_3d.orientation_3d import OrientationAssigner3D
from upxo.pxtal.twinned_simple_3d.cleaning_3d import StructureCleaner3D
from upxo.pxtal.twinned_simple_3d.repr_validator_3d import RepresentativenessValidator3D

__all__ = [
    "TwinnedSimple3DBase",
    "TwinGenerator3D",
    "OrientationAssigner3D",
    "StructureCleaner3D",
    "RepresentativenessValidator3D",
]
