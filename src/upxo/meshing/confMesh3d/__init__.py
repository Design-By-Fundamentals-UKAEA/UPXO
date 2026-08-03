"""
confMesh3d
==========
Conformal multi-grain tetrahedral meshing for 3D voxelated grain structures.

Pipeline
--------
1. surface_nets.run_surface_nets(lgi, voxel_size)
       → SurfaceNetsResult  (vtkSurfaceNets3D, wall-pad, face projection)

2. complex_builder.build_conformal_surface_complex(result)
       → ConformalSurfaceComplex  (shared vertex table + per-grain triangle lists)

3. validation.validate_surface_complex(complex_, lgi)
       → ValidationReport  (watertight, volume, bounds checks)
   validation.fix_winding(complex_)
       → ConformalSurfaceComplex  (consistent outward normals)

4. volume_mesh.generate_conformal_tet_mesh(complex_, config)
       → GmshVolumeResult  (gmsh discrete-entity model, tets generated)

5. export.export_conformal_mesh(gmsh_result, all_quats, ...)
       → Abaqus INP (partitioned) + optional meshio formats

Visualisation helpers at each stage accept show=True for interactive display.
"""

from upxo.meshing.confMesh3d.config import (
    SurfaceNetsConfig, ComplexConfig, ValidationConfig,
    GmshMeshConfig, ExportConfig, ConfMesh3DConfig,
)
from upxo.meshing.confMesh3d.surface_nets import (
    SurfaceNetsResult, run_surface_nets,
    apply_volume_correction, visualize_surface_nets,
)
from upxo.meshing.confMesh3d.complex_builder import (
    ConformalSurfaceComplex, GmshVolumeResult,
    build_conformal_surface_complex, visualize_complex,
)
from upxo.meshing.confMesh3d.validation import (
    ValidationReport, validate_surface_complex,
    fix_winding, visualize_validation,
)
from upxo.meshing.confMesh3d.volume_mesh import (
    generate_conformal_tet_mesh, visualize_tet_mesh,
)
from upxo.meshing.confMesh3d.export import export_conformal_mesh

__all__ = [
    # Config
    'SurfaceNetsConfig', 'ComplexConfig', 'ValidationConfig',
    'GmshMeshConfig', 'ExportConfig', 'ConfMesh3DConfig',
    # Stage 1
    'SurfaceNetsResult', 'run_surface_nets',
    'apply_volume_correction', 'visualize_surface_nets',
    # Stage 2
    'ConformalSurfaceComplex', 'GmshVolumeResult',
    'build_conformal_surface_complex', 'visualize_complex',
    # Stage 3
    'ValidationReport', 'validate_surface_complex',
    'fix_winding', 'visualize_validation',
    # Stage 4
    'generate_conformal_tet_mesh', 'visualize_tet_mesh',
    # Stage 5
    'export_conformal_mesh',
]
