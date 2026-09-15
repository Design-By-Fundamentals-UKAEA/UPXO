# Conforming voxel tetrahedral meshes

```python
import numpy as np
from upxo.meshing.gbconformant.d3v2p0 import mesh_voxels

labels = np.load("data/conformalMeshing3DData/samples/blk_lgi.npy")
mesh = mesh_voxels(labels, spacing=1.0, iterations=20, min_quality=0.5,
                   quality_iterations=60, quality_displacement=0.35)
report = mesh.save("data/conformalMeshing3DData/runs/legacy/blk_mesh")
```

Run the sample from the repository with `PYTHONPATH=src` and
`python -m upxo.meshing.gbconformant.d3v2p0`. Run checks with
`python -m unittest discover -s tests/meshing/gbconformant/d3v2p0`.
Only NumPy is required. CLI options include `--output`, `--iterations`,
`--spacing`, `--min-quality`, and `--quality-iterations`.

Each voxel is split into six tetrahedra using a consistent global diagonal.
Shared nodes and faces are retained across all grains, including disconnected
components of the same label. Every integer label is included, including zero.
Array axes are x/y/z; a shape (nx, ny, nz) fills [0,nx] × [0,ny] × [0,nz],
scaled by spacing (scalar or three positive values in the Python API).

Laplacian relaxation uses surface adjacency for two-grain nodes and junction
edges for three-grain nodes. Four-or-more-grain nodes and junction endpoints
or branches remain frozen. Volume nodes also relax. Each coordinate lying on
an RVE face is fixed, allowing tangential motion while keeping caps planar,
edges straight, and corners fixed. Constraints use the original voxel topology.
Node kinds are 0: volume, 1: surface, 2: junction line, 3: frozen junction point.

Moves that violate positive volume or the mean-ratio quality floor are locally
suppressed; global backtracking handles residual violations. Mean ratio is 1
for a regular tetrahedron. This is a quality-constrained structured baseline,
not an adaptive tetrahedral remesher: smoothing can stop locally at the quality
floor, voxel diagonal bias remains, and grain volumes may change. Total RVE
volume is preserved. Raising `min_quality` preserves better-shaped elements
at the expense of smoothing. An infeasible initial quality floor raises an error.

After Laplacian smoothing, constrained gradient optimization minimizes weighted
inverse-square tetrahedral mean-ratio energy. Tetrahedra touching grain boundary
nodes receive triple weight. Surface nodes move in local tangent planes and
junction-line nodes in local tangent directions computed at the post-Laplacian
geometry. These subspaces stay fixed during optimization. The displacement from
that geometry is bounded by `quality_displacement` times the smallest spacing.
Tangential movement approximates surface preservation; curved interfaces may
still change slightly. Frozen junction points and RVE coordinates remain exact.
Accepted steps preserve positive volume, do not reduce global minimum quality,
and decrease the weighted energy. Individual element qualities can decrease.
Optimization may stop early when constraints prevent further improvement.

The default quality floor is now 0.50 (previously 0.30), so Laplacian smoothing
accepts less distortion. Set `quality_iterations=0` to disable optimization;
combine this with `min_quality=0.3` to reproduce the previous algorithm.
`mesh.quality_before_optimization` stores per-element quality after Laplacian
smoothing; `mesh.quality_history` records accepted optimization steps.
`mesh.boundary_quality()` separately reports tetrahedra touching internal
boundary nodes and interface triangle quality (1 for an equilateral triangle).

Validation checks positive tetrahedral volumes, face incidence, closed grain
surface edge parity, fixed junction points, RVE face constraints, and total
volume. Edge/corner-only voxel contacts can create nonmanifold grain surfaces;
these are preserved and reported, not silently repaired. Closure does not imply
that every grain is a manifold. No claim of globally optimal mesh quality or
exhaustive geometric self-intersection testing is made.

Outputs: NPZ contains shared connectivity, boundary triangles, adjacent grain
labels, exterior flags, original coordinates and constraints; VTU contains the
volume mesh, grain IDs, node kinds and element quality for ParaView/PyVista;
JSON contains validation statistics. For exterior triangles, both adjacent label
slots contain the owning grain; use `exterior` to identify caps. Boundary triangle
indices are sorted for matching and are not oriented surface normals.

## CM02 discrete interface remeshing

`gmsh_interfaces.remesh_interfaces_gmsh` remeshes the smoothed shared interface
triangles. Difficult surface components are automatically partitioned along
existing triangle edges into disk charts when parametrization fails. Retries
construct shared discrete curves explicitly and retain connected components
separately. `report['parametrization_repairs']`, when present, records the
subdivision stages. Captured Gmsh invalid-element warnings also trigger targeted
subdivision. The function starts logging when it initializes Gmsh itself; when
using a caller-owned Gmsh session, start its logger to enable this warning check.
`surface_warning_capture_available` records whether messages were captured, and
`surface_meshing_warnings` contains remaining warnings from the final generation.
These chart seams preserve source edges and therefore limit
coarsening across them; subdivision alone does not guarantee tetrahedral quality
or remove geometric intersections. The downstream validation remains necessary.

CM02 also includes a read-only feature audit after the smoothed-interface plot.
It traces an original grain ID through colour shuffling and cleaning to its
surface triangles. Small retained voxel grains can appear rounded after
smoothing; the interface plot does not insert circular healing plugs or point
markers. Interface colours identify grain pairs, not individual grains.

## CM03 topology and geometry safeguards

`voxel_topology.clean_voxel_topology` checks the cubical boundary link of every
grain at every lattice vertex, including the box boundary. A valid boundary link
is one cycle; disconnected cycles or branching links identify point/edge defects.
The 256 binary octant masks are classified explicitly. Targeted one-voxel edits
and bounded adjacent two-voxel edits reduce this defect count while retaining all
grain IDs. Net per-grain volume budgets, small-grain protection and optional
component-count preservation constrain the changes. A zero net-volume budget
still permits count-preserving swaps. Unresolved defects are reported; stopping
does not establish successful cleaning.

`geometry_guard.stabilize_interfaces` checks the smoothed triangles and restores
local voxel detail where intersections, source-relative reversal/collapse or cap
tangencies appear. Junction points and existing RVE-plane coordinates stay fixed.
Correction limits and rollback controls are exposed in CM03. Incremental checks
cover every changed triangle against the whole surface, followed by a full scan.
The intersection predicates use floating-point tolerances, not exact arithmetic.
`MIN_FACET_OPENING_DEGREES` also prevents nearly closed shared-edge wedges that
can be rejected by Gmsh even without a mathematical intersection. CM03 defaults
to 1 degree and applies this threshold before and after remeshing. Surface
validation separately flags openings below 0.1 degree by default, matching the
overlap tolerance reported by Gmsh on this sample. Increasing the threshold
restores more voxel detail; zero disables the angular check.

CM03 enables `check_intersections` for both Gmsh remeshing stages and supplies
`rve_dimensions` for detecting internal triangles flattened onto cap planes.
`GMSH_CHART_TRIANGLES` (128 by default) partitions surfaces into bounded disks
before Gmsh parametrization. Smaller values retain more source seams and limit
remeshing freedom; `None` selects the automatic topology path. Every seam uses
one shared curve mesh. The API option is `max_chart_triangles` in both remeshers.
Affected charts can be restored to the exact source triangulation by remeshing
individual source triangles, retaining shared boundaries. Bounded retry failures
raise. Genuine multi-grain junctions remain connected. These stages do not
guarantee a requested tetrahedral quality threshold; the final volume-meshing
verification is still required. CM03 writes its own output and audit filenames.

`surface_quality.improve_surface_quality` repairs surface slivers by flipping
interior diagonals only when the pair's minimum quality improves. Grain junction
edges, RVE patch boundaries and all coordinates stay fixed. Each batch is checked
against the whole complex, and intersecting proposals are rejected. Curved flips
can change local grain shape/volume; reported enclosed volumes are refreshed.
CM03 exposes the enable switch, quality target and pass limit and runs a full
validation afterwards. Remaining constrained poor triangles are reported.
