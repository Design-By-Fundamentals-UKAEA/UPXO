# d3v2p1: faster stages of the d3v2p0 conformal tet pipeline

`d3v2p1` contains faster versions of the slow stages of `d3v2p0`. Every function keeps its `d3v2p0` name and arguments and adds `backend` and `n_workers`. Stages without a faster version are not repeated here: import them from `d3v2p0`.

## Stages

| Module | Function(s) | Output compared with `d3v2p0` |
|---|---|---|
| `gmsh_interfaces` | `remesh_interfaces_gmsh` | Same |
| `gmsh_closed` | `remesh_closed_rve_gmsh` | Same |
| `tet_validation` | `validate_tet_surfaces` | Same report, plus `report['backend']` |
| `surface_intersections` | `find_surface_intersections` | Same pairs |
| `surface_angles` | `improve_surface_angles` | Same rules, different order: equivalent surface |
| `gmsh_tets` | `mesh_repaired_rve_gmsh`, `save_grain_tetrahedra` | One Gmsh model per grain: equivalent mesh |
| `tet_smoothing` | `smooth_grain_tetrahedra`, `smooth_tet_dihedrals`, `surface_after_smoothing` | Same rules, block order: equivalent mesh |
| `tet_swaps` | `swap_grain_tetrahedra`, `swap_tets`, `insert_grain_tetrahedra` | Same swaps |

"Same" means identical arrays. "Equivalent" means the same rules and checks hold and the quality statistics agree closely; individual nodes or triangles differ. Results do not depend on the worker count.

## Backend and workers

| Argument | Values | Default |
|---|---|---|
| `backend` | `'auto'`, `'numpy'`, `'parallel'`, `'numba'` | `'auto'` |
| `n_workers` | `None` or a positive integer | `None` |

- `'numpy'`: serial, pure numpy, in the calling process. Always available.
- `'parallel'`: worker processes.
- `'numba'`: compiled kernels (intersection search, smoothing trial evaluation and surface checks, swap acceptance). Stages without kernels use `'parallel'`.
- `'auto'`: `'numba'` where the stage has kernels and numba compiles, otherwise `'parallel'` when more than one worker is available, otherwise `'numpy'`.
- `n_workers=None`: the number of physical cores this process may use (psutil when installed, `/proc/cpuinfo` on Linux), otherwise the usable logical CPUs.

Order of precedence: function arguments, then the environment variables `UPXO_BACKEND` and `UPXO_N_WORKERS`, then the automatic choice.

If worker processes cannot start or a worker dies, the stage reruns serially, warns, and records the reason. Errors raised by the stage itself are not retried. Each stage report records the tier requested and used, the worker count, any fallback reason and the detected cores (`report[...]['backend']`).

## Usage

```python
from upxo.meshing.gbconformant.d3v2p0.rve_caps import close_rve_faces          # unchanged stage
from upxo.meshing.gbconformant.d3v2p1.gmsh_closed import remesh_closed_rve_gmsh
from upxo.meshing.gbconformant.d3v2p1.gmsh_tets import mesh_repaired_rve_gmsh
from upxo.meshing.gbconformant.d3v2p1.tet_smoothing import smooth_grain_tetrahedra

tets = mesh_repaired_rve_gmsh(surface, mesh_size=1.5, n_workers=8)
tets = smooth_grain_tetrahedra(tets, len(surface.points), surface=surface, target=30., backend='numpy')
```

On Windows, worker processes are started with `spawn`: scripts need an `if __name__ == '__main__':` guard around the calls. Notebooks do not.

## Tests

```bash
python -m pytest tests/meshing/gbconformant/d3v2p1
```

The tests compare every stage with `d3v2p0` on each tier, check that results do not depend on the worker count, and check the serial fallback.
