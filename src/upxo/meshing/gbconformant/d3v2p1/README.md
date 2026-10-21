# d3v2p1: faster stages of the d3v2p0 conformal tet pipeline

`d3v2p1` contains faster versions of the slow stages of `d3v2p0`. Every function keeps its `d3v2p0` name and arguments and, where the stage has tiers, adds `backend` and `n_workers`. Stages without a faster version are not repeated here: import them from `d3v2p0`.

## Stages

| Module | Function(s) | Output compared with `d3v2p0` |
|---|---|---|
| `gmsh_interfaces` | `remesh_interfaces_gmsh` | Same; with `retry_failed_only=True` equivalent (see below) |
| `gmsh_closed` | `remesh_closed_rve_gmsh` | Same; with `retry_failed_only=True` equivalent (see below) |
| `tet_validation` | `validate_tet_surfaces` | Same report, plus `report['backend']` |
| `surface_intersections` | `find_surface_intersections` | Same pairs |
| `interfaces` | `smooth_interfaces` (no `backend` or `n_workers`: numpy only; `Interfaces` and `upscale_labels` are d3v2p0's) | Same |
| `geometry_guard` | `stabilize_interfaces` | Same: d3v2p0's rules, intersection search on the d3v2p1 tiers |
| `guard_relaxation` | `relax_guarded_interfaces` (`fold_counts`, `check_nodes_outside` are d3v2p0's) | Same |
| `intersection_repair` | `repair_interface_crossings` | Same |
| `surface_quality` | `improve_surface_quality` (`triangle_quality` is d3v2p0's) | Same |
| `surface_angles` | `improve_surface_angles` | `'numpy'` and `'parallel'`: same rules, different order, equivalent surface. `'numba'`: d3v2p0's order of work, same surface up to float round-off (`numba_surface_angles`) |
| `gmsh_tets` | `mesh_repaired_rve_gmsh`, `save_grain_tetrahedra` | One Gmsh model per grain: equivalent mesh |
| `tet_smoothing` | `smooth_grain_tetrahedra`, `smooth_tet_dihedrals`, `surface_after_smoothing` | Same rules, block order: equivalent mesh |
| `tet_swaps` | `swap_grain_tetrahedra`, `swap_tets`, `insert_grain_tetrahedra` | Same swaps |

`mesh_tables` (`unique_rows`, `EdgeIncidence`), `facet_angles` (`small_facet_angles`) and `surface_charts` (`is_disk_chart`) are the integer-key versions of d3v2p0's edge tables: they return d3v2p0's results element for element, and `unique_rows` hands anything it cannot pack (non-integer rows, keys that could overflow) to `np.unique`.

"Same" means identical arrays. "Equivalent" means the same rules and checks hold and the quality statistics agree closely; individual nodes or triangles differ. Results do not depend on the worker count.

## Retry passes of the Gmsh remeshers

`remesh_interfaces_gmsh` and `remesh_closed_rve_gmsh` retry with a rebuilt model when charts fail to parametrize or mesh, or when the assembled surface has intersections or sharp openings. By default every retry meshes all charts again. With `retry_failed_only=True` (pipeline setting `GMSH_RETRY_FAILED_ONLY`, default off) the charts that came out of a pass without a Gmsh warning keep their mesh (`chart_reuse.ChartCache`) and a retry parametrizes and meshes only the charts that failed or changed. The kept meshes are put back after `generate(2)`; every check on the assembled surface (boundary conformity, orientation, intersections, openings, volumes) runs as before. The triangles of a chart are those of a full re-mesh up to Gmsh's own run-to-run variation, node numbering differs, so the result is equivalent, not identical, to the default's. `report['chart_reuse']` lists, per pass, the charts reused and meshed. The setting is part of a stage's saved-results signature only when it is on, so results saved before it existed stay valid.

## Backend and workers

| Argument | Values | Default |
|---|---|---|
| `backend` | `'auto'`, `'numpy'`, `'parallel'`, `'numba'` | `'auto'` |
| `n_workers` | `None` or a positive integer | `None` |

- `'numpy'`: serial, pure numpy, in the calling process. Always available.
- `'parallel'`: worker processes.
- `'numba'`: compiled kernels (intersection search, smoothing trial evaluation and surface checks, swap acceptance). Stages without kernels use `'parallel'`.
- `'auto'`: `'numba'` where the stage has kernels and numba compiles, otherwise `'parallel'` when more than one worker is available, otherwise `'numpy'`.
- `improve_surface_angles` is the one stage where `'auto'` does not pick `'numba'`: the compiled tier (`numba_surface_angles`) runs only when `'numba'` is requested (argument, `UPXO_BACKEND` or the pipeline's backend setting). It searches in the order d3v2p0 uses, so its result is d3v2p0's and differs from the `'numpy'`/`'parallel'` result. If numba is missing or its kernels do not compile or fail a small self-test on the machine, the stage runs the `'parallel'`/`'numpy'` tier and records why in `report['surface_angle_repair']['backend']['fallback']`.
- `n_workers=None`: the number of physical cores this process may use (psutil when installed, `/proc/cpuinfo` on Linux), otherwise the usable logical CPUs.

Order of precedence: function arguments, then the environment variables `UPXO_BACKEND` and `UPXO_N_WORKERS`, then the automatic choice.

Worker processes are started through `_pool_init`, a module that imports nothing heavy: a worker's start-up message then names no stage module, so the parent does not wait for the worker's imports before it can start the next one. Data that every worker reads (the angle-repair surface and its per-triangle constants) is passed in shared memory.

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
