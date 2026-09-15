"""
Finite-element meshing package for UPXO grain structures.

Includes:

* 2D conformal meshing (``confMesh2dGMSH`` / ``gsmesh2d.mesh_gs``;
  ``confMesh2d`` pygmsh path is deprecated)
* Non-conformal structured meshes (``nonConformalMesher``)
* 3D conformal tet pipeline (``confMesh3d`` — surface nets → gmsh → Abaqus)
* Grain-boundary-conformant pipelines (``gbconformant.d3v2p0``
  and ``gbconformant.cleaving3dV1P0``)
* Abaqus keyword helpers (``writer_ABQ``)
* Element utilities (``elemOps``)

New voxel-interface and cleaving development lives under ``gbconformant``;
the separate ``confMesh3d`` SurfaceNets pipeline remains available. Pipeline
exporters in ``fm_steel_3d`` / ``twinned_simple_3d`` provide hierarchy/twin-aware INP export.
"""
