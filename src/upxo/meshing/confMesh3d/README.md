# `upxo.meshing.confMesh3d`

**Grain-boundary-conformant tetrahedral meshing** for 3D voxelated grain structures (multi-label LFI).

Not a single mesher class: a **5-stage functional pipeline**. Public docs:  
[Meshing](https://github.com/Design-By-Fundamentals-UKAEA/UPXO/wiki/Meshing) ·  
[Data I/O](https://github.com/Design-By-Fundamentals-UKAEA/UPXO/wiki/Data-IO)

Requires **gmsh** (Python package + system libraries as appropriate). Optional **meshio** for extra export formats.

---

## Pipeline

```text
1. run_surface_nets(lfi, voxel_size)     → SurfaceNetsResult
2. build_conformal_surface_complex(...)  → ConformalSurfaceComplex
3. validate_surface_complex(...) + fix_winding(...)
4. generate_conformal_tet_mesh(...)      → GmshVolumeResult  (gmsh)
5. export_conformal_mesh(..., all_quats) → Abaqus INP (+ optional meshio)
```

```python
from upxo.meshing.confMesh3d import (
    run_surface_nets,
    build_conformal_surface_complex,
    validate_surface_complex,
    fix_winding,
    generate_conformal_tet_mesh,
    export_conformal_mesh,
)

sn = run_surface_nets(lfi_3d, voxel_size=1.0)
complex_ = build_conformal_surface_complex(sn)
report = validate_surface_complex(complex_, lfi_3d)
complex_ = fix_winding(complex_)
gmsh_result = generate_conformal_tet_mesh(complex_)
# all_quats: {grain_id: quaternion(4,)}
export_conformal_mesh(gmsh_result, all_quats, voxel_size=1.0)
```

Configs: `SurfaceNetsConfig`, `ComplexConfig`, `ValidationConfig`, `GmshMeshConfig`, `ExportConfig`, `ConfMesh3DConfig` in `config.py`.

Export layout follows the same partitioned Abaqus pattern as the twin exporter (`model_master.inp`, nodes, elements, ELSETs, materials, …). **Interaction / step files may be stubs** — not simulation-ready without manual edits.

---

## Layout

| File | Stage |
|---|---|
| `surface_nets.py` | Multi-label surface (VTK surface nets, wall pad, volume correction hooks) |
| `complex_builder.py` | Shared-vertex surface complex |
| `validation.py` | Watertight / volume / bounds; winding fix |
| `volume_mesh.py` | gmsh tet generation |
| `export.py` | Abaqus (+ meshio options) |
| `config.py` | Dataclass configs |

Demos: `src/upxo/demos/confMesh/confMesh3d*.ipynb` (when present in the clone).

Install: `pip install upxo`; ensure `gmsh` is available. Prefer this package over older one-off mesher scripts under `demos/confMesh/`.
