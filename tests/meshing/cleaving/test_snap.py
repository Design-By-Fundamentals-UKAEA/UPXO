"""
test_snap.py
============
Correctness tests for relax.snap_to_rve_boundary.

Two properties matter most, mirroring test_relax.py's discipline:

1. Neither safety floor can ever be breached -- volume (shared design with
   relax_boundary_points) AND edge length (the new concern specific to
   snapping: pulling several points toward the SAME shared plane can crowd
   them together in a way ordinary neighbour-averaging never risks, and
   volume alone doesn't catch two corners of a tet nearly coinciding).

2. The snap actually improves flatness on the RVE's own faces where it can
   safely do so, confirmed on the exact construction that first exposed the
   "protruding tets" question: a uniform single-grain domain, so the only
   interface is grain-vs-wall at the RVE's own outer boundary.
"""
import numpy as np
import pytest

from upxo.meshing.cleaving.config import LatticeConfig, LabelingConfig
from upxo.meshing.cleaving.lattice import build_bcc_lattice, signed_volumes
from upxo.meshing.cleaving.labeling import sample_labels
from upxo.meshing.cleaving.cleave import cleave_lattice
from upxo.meshing.cleaving.relax import snap_to_rve_boundary, SnapConfig

WALL = 999999
_EDGE_DEFS = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))


def _two_grain_lgi(shape, split_at=None):
    lgi = np.ones(shape, dtype=np.int32)
    split_at = split_at if split_at is not None else shape[2] // 2
    lgi[:, :, split_at:] = 2
    return lgi


def _uniform_single_grain_lgi(shape):
    return np.ones(shape, dtype=np.int32)


def _cleave(lgi, pad=1):
    lat = build_bcc_lattice(lgi.shape, LatticeConfig(voxel_size=1.0, pad=pad))
    labels = sample_labels(lat, lgi, LabelingConfig(wall_label=WALL))
    return lat, cleave_lattice(lat, labels)


def _original_vertex_mask(result):
    """
    Which node ids are ORIGINAL lattice vertices, i.e. NOT in
    new_point_gids. A plain ``index < len(lat.vertices)`` slice looks
    equivalent but silently breaks the moment ANY original vertex gets
    dropped by cleave_lattice's own dangling-node compaction (always
    exactly the padded lattice's 8 outer corners) -- that shifts later
    rows down into what used to be original-vertex index territory, so
    the tail of a "first n_base rows" slice quietly starts containing new
    points instead. new_point_gids is the compaction-robust source of
    truth (see cleave.py / relax.py's own _is_new_point_mask).
    """
    n_points = len(result.node_coords)
    is_new = np.zeros(n_points, dtype=bool)
    if result.new_point_gids:
        is_new[np.fromiter(result.new_point_gids.keys(), dtype=np.int64,
                            count=len(result.new_point_gids))] = True
    return ~is_new


def _min_edge_length(node_coords, tets):
    lens = [np.linalg.norm(node_coords[tets[:, a]] - node_coords[tets[:, b]], axis=1)
            for a, b in _EDGE_DEFS]
    return np.concatenate(lens).min()


CASES = [
    ('two_grain', _two_grain_lgi((6, 6, 8))),
    ('uniform_single_grain', _uniform_single_grain_lgi((8, 8, 8))),
]


@pytest.mark.parametrize('name,lgi', CASES)
@pytest.mark.parametrize('floor', [0.05, 0.2, 0.5])
def test_volume_and_edge_floors_never_breached(name, lgi, floor):
    lat, result = _cleave(lgi)
    original_vol = signed_volumes(result.node_coords, result.tets)
    original_min_edge = _min_edge_length(result.node_coords, result.tets)

    cfg = SnapConfig(min_volume_fraction=floor, min_edge_fraction=floor)
    snapped = snap_to_rve_boundary(lat, result, cfg)
    snapped_vol = signed_volumes(snapped.node_coords, snapped.tets)

    assert (snapped_vol > 0).all()
    slack = 1e-6
    assert (snapped_vol >= floor * original_vol - slack).all(), (
        f'{name} floor={floor}: some tet dropped below its volume floor')

    # Edge-length floor is defined relative to each TOUCHED edge's own
    # pre-snap length, not a single global minimum -- but the global
    # minimum edge length can only ever shrink to floor * (that edge's own
    # original length), so as a coarse sanity check the new global minimum
    # can't be smaller than floor * the ORIGINAL global minimum among
    # edges that could plausibly have been touched.
    new_min_edge = _min_edge_length(snapped.node_coords, snapped.tets)
    assert new_min_edge >= floor * original_min_edge - slack, (
        f'{name} floor={floor}: min edge length {new_min_edge} fell below '
        f'floor*original ({floor * original_min_edge})')


@pytest.mark.parametrize('name,lgi', CASES)
def test_topology_unchanged(name, lgi):
    lat, result = _cleave(lgi)
    snapped = snap_to_rve_boundary(lat, result, SnapConfig())
    assert np.array_equal(result.tets, snapped.tets)
    assert np.array_equal(result.tet_labels, snapped.tet_labels)
    assert np.array_equal(result.source_tet_index, snapped.source_tet_index)
    assert np.array_equal(result.deferred_tet_indices, snapped.deferred_tet_indices)


@pytest.mark.parametrize('name,lgi', CASES)
def test_original_vertices_never_move(name, lgi):
    lat, result = _cleave(lgi)
    snapped = snap_to_rve_boundary(lat, result, SnapConfig())
    original = _original_vertex_mask(result)
    assert np.array_equal(result.node_coords[original], snapped.node_coords[original])


def test_rve_face_wobble_measurably_reduced():
    """
    The actual point of this module, on the exact construction that first
    exposed the "protruding tets" question: a uniform single-grain domain
    (so the only interface anywhere is grain-vs-wall at the RVE's own
    outer boundary). The z-spread of boundary points near the nominal flat
    top face must shrink after snapping, and no point may end up on the
    WRONG side of the nominal plane (overshooting past z=7 into the wall
    region, or undershooting back into the domain beyond where it started).
    """
    shape = (8, 8, 8)
    lat, result = _cleave(_uniform_single_grain_lgi(shape))
    snapped = snap_to_rve_boundary(lat, result, SnapConfig())

    def face_zspread(node_coords, tets, tet_labels):
        face_defs = ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))
        faces = np.concatenate([tets[:, list(idx)] for idx in face_defs], axis=0)
        tet_of_face = np.tile(np.arange(len(tets)), 4)
        sf = np.sort(faces, axis=1)
        order = np.lexsort((sf[:, 2], sf[:, 1], sf[:, 0]))
        sf, tf = sf[order], tet_of_face[order]
        boundaries = np.nonzero(np.any(sf[1:] != sf[:-1], axis=1))[0] + 1
        starts = np.concatenate([[0], boundaries])
        ends = np.concatenate([boundaries, [len(sf)]])
        pair2 = np.nonzero((ends - starts) == 2)[0]
        t0 = tf[starts[pair2]]
        t1 = tf[starts[pair2] + 1]
        is_gw = tet_labels[t0] != tet_labels[t1]
        gw_faces = sf[starts[pair2]][is_gw]
        coords = node_coords[gw_faces]
        cx, cy, cz = coords[:, :, 0].mean(1), coords[:, :, 1].mean(1), coords[:, :, 2].mean(1)
        mask = (cx > 2) & (cx < 5) & (cy > 2) & (cy < 5) & (cz > 5.5)
        return coords[mask][:, :, 2].ravel()

    z_before = face_zspread(result.node_coords, result.tets, result.tet_labels)
    z_after = face_zspread(snapped.node_coords, snapped.tets, snapped.tet_labels)

    assert len(z_before) > 0 and len(z_after) > 0
    assert (z_after.max() - z_after.min()) < (z_before.max() - z_before.min()), (
        f'expected the top-face wobble to shrink: '
        f'before=[{z_before.min():.4f},{z_before.max():.4f}] '
        f'after=[{z_after.min():.4f},{z_after.max():.4f}]')
    # Nothing may overshoot past the true boundary plane (z=7) toward the
    # wall side, and nothing may retreat further from it than it started.
    assert z_after.max() <= z_before.max() + 1e-9
    assert z_after.min() >= 7.0 - 1e-9
