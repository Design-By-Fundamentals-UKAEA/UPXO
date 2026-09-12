"""
Finite-element meshing package for UPXO grain structures.

Includes:

* 2D conformal meshing (``confMesh2dGMSH`` / ``gsmesh2d.mesh_gs``;
  ``confMesh2d`` pygmsh path is deprecated)
* Non-conformal structured meshes (``nonConformalMesher``)
* 3D conformal tet pipeline (``confMesh3d`` — surface nets → gmsh → Abaqus)
* Abaqus keyword helpers (``writer_ABQ``)
* Element utilities (``elemOps``)

Prefer ``confMesh3d`` for modern 3D conformal work and pipeline exporters in
``fm_steel_3d`` / ``twinned_simple_3d`` for hierarchy/twin-aware INP export.
"""
