"""
test_mesh_integrity.py
=======================
Exhaustive (not point-sampling-based) structural integrity checks on
cleave_lattice's output:

1. No dangling/orphaned nodes -- every row of node_coords must be
   referenced by at least one tet.
2. No internal holes -- every face shared by exactly one tet (i.e. a true
   exterior face of the whole meshed region) must lie entirely outside the
   real domain. A hole INSIDE the region that matters would show up as an
   "exterior" face with vertices strictly inside the real domain bounds --
   the disposable padded-lattice fringe is allowed to have its own
   (already-established, harmless) incomplete coverage; the real domain
   and the wall-padding shell are not.
3. No disjoint entities -- the whole mesh, and EVERY individual grain
   label's own tets, must each form exactly one connected component (via
   shared-face adjacency), never split into separate islands.
"""
import numpy as np
import pytest
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from upxo.meshing.cleaving.config import LatticeConfig, LabelingConfig
from upxo.meshing.cleaving.lattice import build_bcc_lattice
from upxo.meshing.cleaving.labeling import sample_labels
from upxo.meshing.cleaving.cleave import cleave_lattice
from upxo.meshing.cleaving.relax import relax_boundary_points, RelaxConfig

WALL = 32767
_FACE_DEFS = ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))


def _two_grain_lgi(shape, split_at=None):
    lgi = np.ones(shape, dtype=np.int32)
    split_at = split_at if split_at is not None else shape[2] // 2
    lgi[:, :, split_at:] = 2
    return lgi


def _four_grain_quadrant_lgi(shape):
    nx, ny, nz = shape
    lgi = np.ones(shape, dtype=np.int32)
    hx, hy = nx // 2, ny // 2
    lgi[:hx, :hy, :] = 1
    lgi[hx:, :hy, :] = 2
    lgi[:hx, hy:, :] = 3
    lgi[hx:, hy:, :] = 4
    return lgi


def _cleave(lgi, pad=1):
    lat = build_bcc_lattice(lgi.shape, LatticeConfig(voxel_size=1.0, pad=pad))
    labels = sample_labels(lat, lgi, LabelingConfig(wall_label=WALL))
    return lat, cleave_lattice(lat, labels)


def _grouped_faces(tets):
    faces = np.concatenate([tets[:, list(idx)] for idx in _FACE_DEFS], axis=0)
    tet_of_face = np.tile(np.arange(len(tets)), 4)
    sorted_faces = np.sort(faces, axis=1)
    order = np.lexsort((sorted_faces[:, 2], sorted_faces[:, 1], sorted_faces[:, 0]))
    sf = sorted_faces[order]
    tf = tet_of_face[order]
    boundaries = np.nonzero(np.any(sf[1:] != sf[:-1], axis=1))[0] + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [len(sf)]])
    return sf, tf, starts, ends


def _n_connected_components(tets, mask=None):
    if mask is not None:
        tets = tets[mask]
    if len(tets) == 0:
        return 0
    sf, tf, starts, ends = _grouped_faces(tets)
    shared = np.nonzero((ends - starts) == 2)[0]
    rows = tf[starts[shared]]
    cols = tf[starts[shared] + 1]
    n = len(tets)
    graph = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    n_components, _ = connected_components(graph, directed=False)
    return n_components


CASES = [
    ('two_grain', _two_grain_lgi((5, 5, 6))),
    ('four_grain_quadrant', _four_grain_quadrant_lgi((6, 6, 6))),
]


@pytest.mark.parametrize('name,lgi', CASES)
def test_no_dangling_nodes(name, lgi):
    _, result = _cleave(lgi)
    used = np.unique(result.tets.ravel())
    assert len(used) == len(result.node_coords), (
        f'{name}: {len(result.node_coords) - len(used)} node(s) never referenced by any tet')


@pytest.mark.parametrize('name,lgi', CASES)
def test_no_holes_in_real_domain(name, lgi):
    """
    Any face shared by exactly one tet (a true exterior face of the whole
    meshed region, regardless of grain label) must have no vertex strictly
    inside the real domain's bounds. The padded lattice's own outer fringe
    (including the wall shell's own outer edge) is allowed to fall short of
    full tet coverage -- that's an already-established, harmless property
    of the lattice construction itself (tet coverage tapers off somewhat
    before the lattice's nominal padded extent, not exactly at it) -- but a
    hole strictly inside the real domain would be a genuine defect.
    """
    shape = lgi.shape
    _, result = _cleave(lgi)
    sf, tf, starts, ends = _grouped_faces(result.tets)
    counts = ends - starts
    assert (counts <= 2).all(), 'a face shared by >2 tets means overlapping/duplicate tets'

    ext_idx = np.nonzero(counts == 1)[0]
    ext_faces = sf[starts[ext_idx]]
    ext_coords = result.node_coords[ext_faces]  # (F, 3, 3)

    real_lo = np.array([0.0, 0.0, 0.0])
    real_hi = np.array([s - 1 for s in shape], dtype=float)
    inside_real = np.all(
        (ext_coords >= real_lo - 1e-6) & (ext_coords <= real_hi + 1e-6), axis=2)
    face_touches_real_domain = inside_real.any(axis=1)

    n_bad = int(face_touches_real_domain.sum())
    assert n_bad == 0, (
        f'{name}: {n_bad} exterior face(s) with a vertex inside the real '
        f'domain -- possible hole')


@pytest.mark.parametrize('name,lgi', CASES)
def test_whole_mesh_and_every_grain_single_component(name, lgi):
    _, result = _cleave(lgi)
    n_whole = _n_connected_components(result.tets)
    assert n_whole == 1, f'{name}: whole mesh split into {n_whole} components'

    for lab in sorted(set(result.tet_labels.tolist())):
        mask = result.tet_labels == lab
        n_comp = _n_connected_components(result.tets, mask)
        assert n_comp == 1, f'{name}: grain {lab} split into {n_comp} disconnected components'


@pytest.mark.parametrize('name,lgi', CASES)
def test_integrity_preserved_after_relaxation(name, lgi):
    """Relaxation only moves coordinates -- confirm it can't reintroduce
    dangling nodes or split a grain (connectivity depends only on tets/
    labels, which relaxation never touches, but this closes the loop)."""
    lat, result = _cleave(lgi)
    relaxed = relax_boundary_points(lat, result, RelaxConfig(n_iterations=5))

    used = np.unique(relaxed.tets.ravel())
    assert len(used) == len(relaxed.node_coords)

    assert _n_connected_components(relaxed.tets) == 1
    for lab in sorted(set(relaxed.tet_labels.tolist())):
        mask = relaxed.tet_labels == lab
        assert _n_connected_components(relaxed.tets, mask) == 1
