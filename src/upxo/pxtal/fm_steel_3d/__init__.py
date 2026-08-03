"""
fm_steel_3d: Complete 3D ferrite-martensite steel microstructure generation framework.

This package provides a chainable, stateful pipeline for generating realistic
FM steel microstructures from labeled grain images (LFI), with full hierarchical
decomposition into PAGs, packets, and blocks, plus Kurdjumov-Sachs orientation
assignment and comprehensive visualization.

Main Classes (Pipeline Chain)
=============================
- FMSteel3DBase: Foundation grain structure from LFI
- FMSteel3DWithPAGs: Base + PAG partitioning
- FMSteel3DWithBlocks: PAGs + martensitic blocks
- FMSteel3DWithOrientations: Blocks + BCC orientations
- FMSteel3DWithSubBlocks: Sub-blocks (laths) + per-sub-block orientation spread (complete)

Worker Classes (Stateless Utilities)
====================================
- BlockGenerator3D: Block slicing logic
- SubBlockGenerator3D: Sub-block slicing logic
- OrientationAssigner3D: KS variant assignment + sub-block orientation spread
- MeshExporter3D: Abaqus mesh generation (deferred)

Visualization Classes (Viz Workers)
===================================
- GrainStructureViz3D: Grain and block PyVista plots
- OrientationViz3D: IPF maps, pole figures, KS verification
- DataViz3D: Statistical distributions (matplotlib)

Usage Example
=============
>>> from upxo.pxtal.fm_steel_3d import FMSteel3DBase
>>> lfi = np.random.randint(1, 100, (50, 50, 50))
>>> fm = FMSteel3DBase.from_lfi(lfi, physical_dimensions=(100, 100, 100))
>>> 
>>> # Generate PAGs
>>> fm_pag = fm.generate_pag_clusters(
...     pag_size_distribution={'sizes': [3, 4, 6], 'probs': [0.25, 0.5, 0.25]},
...     pag_grain_fraction=0.8,
...     random_seed=42
... )
>>> 
>>> # Assign PAG orientations then generate blocks
>>> fm_pag.assign_pag_orientations(pag_ori_mode='random', random_seed=42)
>>> fm_blk = fm_pag.generate_blocks(block_thickness_range=(2.0, 5.0), random_seed=42)
>>>
>>> # Assign KS block orientations
>>> fm_ori = fm_blk.assign_orientations(ks_variant_selection='random_per_block', random_seed=42)
>>> 
>>> # Visualize
>>> fm_ori.visualize_block_ipf_map()
>>> fm_ori.plot_pag_map_pyvista()
"""

from upxo.pxtal.fm_steel_3d.base_3d import FMSteel3DBase, PhysicalDimensions
from upxo.pxtal.fm_steel_3d.with_pags_3d import FMSteel3DWithPAGs
from upxo.pxtal.fm_steel_3d.with_blocks_3d import FMSteel3DWithBlocks
from upxo.pxtal.fm_steel_3d.with_orientations_3d import FMSteel3DWithOrientations
from upxo.pxtal.fm_steel_3d.with_subblocks_3d import FMSteel3DWithSubBlocks

from upxo.pxtal.fm_steel_3d.block_generator_3d import BlockGenerator3D
from upxo.pxtal.fm_steel_3d.subblock_generator_3d import SubBlockGenerator3D
from upxo.pxtal.fm_steel_3d.orientation_assigner_3d import OrientationAssigner3D
from upxo.pxtal.fm_steel_3d.mesh_exporter_3d import MeshExporter3D

__all__ = [
    'FMSteel3DBase',
    'FMSteel3DWithPAGs',
    'FMSteel3DWithBlocks',
    'FMSteel3DWithOrientations',
    'FMSteel3DWithSubBlocks',
    'PhysicalDimensions',
    'BlockGenerator3D',
    'SubBlockGenerator3D',
    'OrientationAssigner3D',
    'MeshExporter3D',
]

__version__ = '0.0.1-alpha'
__author__ = 'UPXO Development Team'
