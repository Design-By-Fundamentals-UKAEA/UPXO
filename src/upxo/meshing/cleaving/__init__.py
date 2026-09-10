"""
Grain-boundary-conformal tetrahedral meshing via lattice cleaving.

Standalone module: does not import or depend on ``upxo.meshing.confMesh3d``
or any other UPXO meshing pipeline.

Reference: Bronson, Levine, Whitaker, "Lattice Cleaving: A Multimaterial
Tetrahedral Meshing Algorithm with Guarantees," IEEE TVCG 2013.

Milestone 1: BCC background lattice construction + grain-label sampling +
visualisation (done).

Milestone 2: single-interface (2-label) stencil cleaving -- classify each
tet's corner labels, cut it along the boundary, and assemble a conformal
per-grain tet mesh (done).

Milestone 3: triple- and quadruple-junction (3- and 4-label) stencil
cleaving via "face triple point" and "tet quadruple point" entities (done)
-- every corner-label pattern a tetrahedron can have is now handled, and
verified to hold even where these junctions coincide with the RVE's own
face/edge/corner boundary.

Milestone 4: quality refinement -- cut/triple/quadruple points are placed
at simple centroids (edge midpoint, face/tet centroid), which conserves
volume and keeps every tet valid but doesn't know where the true sub-voxel
boundary sits, so the reconstructed surface "staircases" across roughly a
voxel's width. ``relax.relax_boundary_points`` Laplacian-smooths those
points along the grain-boundary surface, each move clipped to a per-tet
safe distance (an exact computation, not an approximation) so no tet can
ever invert or collapse -- this remains the right tool for ordinary
interior grain-boundary quality, which is expected to stay approximate.

``relax.snap_to_rve_boundary`` originally targeted the same wobble on the
RVE's own outer faces specifically (coordinate-only snapping with an
added edge-length floor). Milestone 5 replaces that purpose with an exact
alternative (below) -- snap_to_rve_boundary still works and is still
exported, but for guaranteed-flat RVE faces, boundary_clip is the one that
actually delivers it.

Milestone 5: exact RVE-face flatness for CPFEM, where the outer box must
be exactly flat, not merely close. A patch-after-the-fact approach (merge
a wobbly cut point onto an exact anchor, delete whatever tet becomes
degenerate) was tried and directly falsified: every one of 1359 candidate
merges tried, across every grain structure tested, individually broke a
neighbouring tet -- not a conflict a smarter merge selection could
resolve, since the neighbouring tet's own topology (not just that one
corner's coordinate) was built around the wobbly position.
``boundary_clip.clip_lattice_to_domain`` sidesteps this by never creating
the wobble: the RVE box's shape is exact by definition (unlike an interior
grain boundary, never inferred from label data), so it clips
boundary-straddling tets directly against the domain's known-exact planes
before grain labels are even sampled. Verified exactly flat (bit-exact,
not tolerance-based) on ordinary grain structures, a grain boundary
running exactly along an RVE edge, a single voxel at an RVE corner, and a
40-grain Voronoi structure -- with volume conservation, all-positive
tets, no dangling nodes, and single-component connectivity holding
throughout.

Milestone 6 (current): element-shape quality, as a standard first pass.
``relax.relax_boundary_points`` smooths the grain-boundary SURFACE (move
toward neighbours' centroid) but has no notion of tet shape, so it improves
quality only as an incidental side effect. ``relax.optimize_boundary_points``
targets shape directly: every tet touching a movable point below
``OptimizeConfig.quality_threshold`` pulls that point toward whichever
direction (found via a small finite-difference probe) improves ITS OWN
quality, weighted by how bad it currently is, so a point shared by several
tets moves toward a compromise dominated by its worst offender(s). Two
independent floors gate every step, both fixed to each tet's own state
before optimization started: volume (unchanged from relax_boundary_points'
reasoning) AND shape quality itself -- confirmed directly that the volume
floor alone is not sufficient, since a tet can trade volume for a worse
shape (more sheared/skewed) while staying comfortably above the volume
floor; an early version without the quality floor let the single worst
tet's min_angle regress (13.9 deg -> 10.7 deg). The quality floor cannot
be near-zero-tolerance either -- confirmed directly that requiring literal
non-regression everywhere freezes the optimizer completely (0 tets change
at all), since helping one tet reliably costs some other tet sharing that
point a little quality; ``OptimizeConfig.quality_regression_tolerance``
(default 0.05) is the knob trading how much incidental regression
elsewhere is acceptable against how much improvement can happen at all.
With that tolerance, on a real 500-grain test mesh: the worst tet's
min_angle improved (13.9 deg -> 14.8 deg, no longer a regression), and the
fraction of tets below 20 deg dropped from 3.50% to 2.31% -- real,
verified improvement, not a guarantee every sliver disappears (some are
baked into the topology decided at cleave time, which only actual
re-triangulation -- not repositioning -- could fix).

Milestone 7: second-pass quality optimization, extending the zone that can
move. ``relax.optimize_boundary_points`` alone has a ceiling: every original
lattice vertex around a boundary point is perfectly rigid, which caps how
much the boundary point itself can be improved. ``relax.optimize_boundary_
zone`` lifts that ceiling by also allowing a tapering ring of nearby
original vertices to move -- found via a multi-source breadth-first
expansion (shared-vertex tet adjacency) out to ``ZoneOptimizeConfig.
max_ring`` hops from the boundary, with mobility decaying linearly from hop
1 (strongest pull) to ``max_ring`` (faintest), and anything further out
staying exactly as frozen as it always was. Both the adjacency definition
(``adjacency='vertex'``) and the decay shape (``decay='linear'``) are
config fields with only one implemented option each -- reserved for a
future GUI to expose alternatives (face/edge adjacency, other decay
curves) without needing a signature change; passing anything else raises
rather than silently behaving like the implemented option. Reuses
optimize_boundary_points' exact mechanism and safety guarantees (RVE-plane
lock, volume + quality floors) rather than re-deriving them -- the two
functions' movable sets are disjoint, so running both (recommended order:
optimize_boundary_points, then optimize_boundary_zone) never risks a
double-move race, and neither function's safety depends on that order.
"""
from upxo.meshing.cleaving.config import LatticeConfig, LabelingConfig
from upxo.meshing.cleaving.lattice import BCCLattice, build_bcc_lattice
from upxo.meshing.cleaving.labeling import sample_labels
from upxo.meshing.cleaving.stencils import TooManyLabelsError, cleave_tet
from upxo.meshing.cleaving.cleave import CleaveResult, cleave_lattice
from upxo.meshing.cleaving.relax import (
    RelaxConfig, relax_boundary_points, SnapConfig, snap_to_rve_boundary,
    OptimizeConfig, optimize_boundary_points,
    ZoneOptimizeConfig, optimize_boundary_zone,
)
from upxo.meshing.cleaving.boundary_clip import (
    clip_lattice_to_domain, cleave_lattice_exact_boundary,
)
from upxo.meshing.cleaving.viz import visualize_lattice, visualize_cleaved_mesh

__all__ = [
    'LatticeConfig', 'LabelingConfig',
    'BCCLattice', 'build_bcc_lattice',
    'sample_labels',
    'TooManyLabelsError', 'cleave_tet',
    'CleaveResult', 'cleave_lattice',
    'RelaxConfig', 'relax_boundary_points',
    'SnapConfig', 'snap_to_rve_boundary',
    'OptimizeConfig', 'optimize_boundary_points',
    'ZoneOptimizeConfig', 'optimize_boundary_zone',
    'clip_lattice_to_domain', 'cleave_lattice_exact_boundary',
    'visualize_lattice', 'visualize_cleaved_mesh',
]
