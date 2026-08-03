"""
config.py
=========
Configuration dataclasses for the confMesh3d conformal meshing pipeline.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List


@dataclass
class SurfaceNetsConfig:
    """Controls for the vtkSurfaceNets3D surface extraction stage."""
    smoothing: bool          = True
    # Label assigned to the 1-voxel padding layer on all 6 RVE faces.
    # Must not coincide with any real grain ID. uint16 max-safe default.
    wall_label: int          = 32767
    # Volume conservation tolerance: |V_mesh - V_vox| / V_vox
    volume_tol: float        = 0.01
    max_correction_iters: int = 15
    # Proximity guard: vertex cannot be displaced closer than this fraction
    # of voxel_size to any neighbouring grain surface.
    min_clearance_frac: float = 0.5
    # Crease angle for detecting thin-twin fold edges (degrees).
    crease_angle_deg: float  = 30.0


@dataclass
class ComplexConfig:
    """Controls for building the ConformalSurfaceComplex."""
    # Vertices closer than this (in voxel units) are welded to one node.
    # At RVE domain edges (where two face planes meet), SurfaceNets3D generates
    # y=0 and z=0 cap patches independently. Their boundary vertices at the
    # RVE edge can have float32 x-positions differing by ~0.001 vox due to
    # independent smoothing. 1e-2 merges these safely: minimum inter-vertex
    # spacing along a grain boundary surface is ~0.5 vox >> 0.01.
    vertex_weld_tol: float  = 1e-2


@dataclass
class ValidationConfig:
    """Tolerances for the pre-tet surface validation gate."""
    zero_area_tol: float    = 1e-12
    rve_bounds_tol: float   = 1e-6
    # Fraction of voxel volume: flag grain if |V_mesh-V_vox|/V_vox > tol
    volume_tol: float       = 0.05
    warn_min_voxels: int    = 8


@dataclass
class GmshMeshConfig:
    """Controls for the gmsh tet-mesh generation stage."""
    # Element type: 'C3D4' (linear) or 'C3D10' (quadratic)
    element_type: str       = 'C3D4'
    # Characteristic length bounds as multiples of voxel_size
    char_length_min_frac: float = 0.5
    char_length_max_frac: float = 3.0
    # gmsh 3D algorithm: 1=Delaunay, 4=Frontal-Delaunay, 10=HXT (fast)
    algorithm_3d: int       = 4
    optimize: bool          = True
    verbose: int            = 0
    # Show gmsh GUI after meshing (blocks until window is closed)
    show_gui: bool          = False


@dataclass
class ExportConfig:
    """Controls for the Abaqus INP export."""
    out_dir: str            = ''
    # 'bunge_euler': *User Material with 3 Euler constants (CPFEM stub)
    # 'orientation': *Orientation + *Elastic (elastic-only)
    material_format: str    = 'bunge_euler'
    n_depvar: int           = 100
    # Additional formats via meshio: [] or ['vtk', 'xdmf', ...]
    extra_formats: List[str] = field(default_factory=list)


@dataclass
class ConfMesh3DConfig:
    """Top-level config bundling all stage configs."""
    surface_nets : SurfaceNetsConfig = field(default_factory=SurfaceNetsConfig)
    complex_cfg  : ComplexConfig     = field(default_factory=ComplexConfig)
    validation   : ValidationConfig  = field(default_factory=ValidationConfig)
    gmsh         : GmshMeshConfig    = field(default_factory=GmshMeshConfig)
    export       : ExportConfig      = field(default_factory=ExportConfig)
