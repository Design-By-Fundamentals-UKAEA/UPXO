"""
Finite-element meshing package for UPXO grain structures.

Includes:

* 2D conformal meshing (``confMesh2dGMSH`` / ``gsmesh2d.mesh_gs``)
* Non-conformal structured meshes (``nonConformalMesher``)
* Grain-boundary-conformant pipelines (``gbconformant.d3v2p0``
  and its faster stages ``gbconformant.d3v2p1``)
* Abaqus keyword helpers (``writer_ABQ``)
* Element utilities (``elemOps``)

New voxel-interface development lives under ``gbconformant``. Pipeline
exporters in ``fm_steel_3d`` / ``twinned_simple_3d`` provide hierarchy/twin-aware INP export.
"""
