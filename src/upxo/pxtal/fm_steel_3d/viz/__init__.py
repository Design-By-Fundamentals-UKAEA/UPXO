"""
fm_steel_3d.viz: Visualization subpackage for FM steel structures.

Contains three main visualization worker classes:
- GrainStructureViz3D: Grain and block structure (PyVista)
- OrientationViz3D: Orientation-based visualizations (PyVista + matplotlib)
- DataViz3D: Statistical data plots (matplotlib)
"""

from upxo.pxtal.fm_steel_3d.viz.grain_structure_viz_3d import GrainStructureViz3D
from upxo.pxtal.fm_steel_3d.viz.orientation_viz_3d import OrientationViz3D
from upxo.pxtal.fm_steel_3d.viz.data_viz_3d import DataViz3D

__all__ = [
    'GrainStructureViz3D',
    'OrientationViz3D',
    'DataViz3D',
]
