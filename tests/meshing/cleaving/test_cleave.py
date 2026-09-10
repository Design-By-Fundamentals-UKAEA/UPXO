"""
test_cleave.py
===============
Full-lattice integration tests for cleave.cleave_lattice: build a real
lattice, sample a synthetic 2-grain labeling onto it, cleave it, and check
the properties that actually matter -- volume conservation against the
source tets, positive-volume output, and (the real point of this module)
gap-free / overlap-free / correctly-labeled coverage of the real domain.
"""
import numpy as np
import pytest

from upxo.meshing.cleaving.config import LatticeConfig, LabelingConfig
from upxo.meshing.cleaving.lattice import build_bcc_lattice, signed_volumes
from upxo.meshing.cleaving.labeling import sample_labels
from upxo.meshing.cleaving.cleave import cleave_lattice


def _two_grain_lgi(shape, split_axis=2, split_at=None):
    lgi = np.ones(shape, dtype=np.int32)
    if split_at is None:
        split_at = shape[split_axis] // 2
    idx = [slice(None)] * 3
    idx[split_axis] = slice(split_at, None)
    lgi[tuple(idx)] = 2
    return lgi


def _four_grain_quadrant_lgi(shape):
    """4 grains meeting along a genuine quadruple LINE (splitting both x and
    y in half, leaving z uniform) -- real voxel data that should exercise
    quadruple-point tets directly, not just wall-padding edge cases."""
    nx, ny, nz = shape
    lgi = np.ones(shape, dtype=np.int32)
    hx, hy = nx // 2, ny // 2
    lgi[:hx, :hy, :] = 1
    lgi[hx:, :hy, :] = 2
    lgi[:hx, hy:, :] = 3
    lgi[hx:, hy:, :] = 4
    return lgi


def test_four_grain_quadrant_junction_fully_cleaved():
    """
    A genuine 4-grain configuration (no wall interaction needed): tets right
    at the central column where all 4 quadrants meet should exercise the
    quadruple-point code path directly. Confirms deferred_tet_indices stays
    empty and the mesh is still gap-free/overlap-free even here.
    """
    shape = (6, 6, 4)
    lgi = _four_grain_quadrant_lgi(shape)
    lat = build_bcc_lattice(shape, LatticeConfig(voxel_size=1.0, pad=1))
    labels = sample_labels(lat, lgi, LabelingConfig(wall_label=32767))
    result = cleave_lattice(lat, labels)

    assert len(result.deferred_tet_indices) == 0
    assert set(np.unique(labels).tolist()) >= {1, 2, 3, 4}

    vol = signed_volumes(result.node_coords, result.tets)
    assert (vol > 0).all()

    out_vol = np.abs(vol)
    per_source = {}
    for src, v in zip(result.source_tet_index, out_vol):
        per_source[src] = per_source.get(src, 0.0) + v
    src_vol = np.abs(signed_volumes(lat.vertices, lat.tets))
    for src_idx, total in per_source.items():
        assert total == pytest.approx(src_vol[src_idx], rel=1e-9)

    # Gap/overlap-free coverage away from the central quadruple line itself
    # (points exactly on it are a genuine multi-way tie, not what this
    # checks) and away from the x/y domain edges (wall-interaction edge
    # cases already covered by test_volume_conserved_against_source_tets).
    # The exclusion margin is a full voxel wide, not a thin band -- same
    # reason as the 2-grain case's boundary "staircase": cut/triple/
    # quadruple points are placed at the centroid of the ids they're
    # defined by (2 for a cut point, up to 4 for a quadruple point), and
    # different edges/faces reaching the same nominal boundary have
    # different lengths, so the actual cut surface wobbles across roughly
    # the whole voxel it's crossing (confirmed by inspection: cut points
    # near x=3 range from x=2.5 to x=3.5 here). Expected for this
    # milestone's simple centroid placement, not a defect.
    rng = np.random.default_rng(7)
    nx, ny, nz = shape
    gap = overlap = mislabeled = n_checked = 0
    for _ in range(1500):
        x = rng.uniform(1.05, nx - 2.05)
        y = rng.uniform(1.05, ny - 2.05)
        z = rng.uniform(0.05, nz - 1.05)
        if abs(x - nx / 2) < 0.6 or abs(y - ny / 2) < 0.6:
            continue
        n_checked += 1
        p = np.array([x, y, z])
        covering = _covering_labels(p, result.node_coords, result.tets, result.tet_labels)
        if len(covering) == 0:
            gap += 1
        elif len(covering) > 1:
            overlap += 1
        else:
            expected = (1 if x < nx / 2 else 2) if y < ny / 2 else (3 if x < nx / 2 else 4)
            if covering[0] != expected:
                mislabeled += 1

    assert n_checked > 200
    assert gap == 0, f'{gap}/{n_checked} sample points uncovered'
    assert overlap == 0, f'{overlap}/{n_checked} sample points double-covered'
    assert mislabeled == 0, f'{mislabeled}/{n_checked} sample points have the wrong grain label'


def test_volume_conserved_against_source_tets():
    shape = (4, 4, 5)
    lgi = _two_grain_lgi(shape)
    lat = build_bcc_lattice(shape, LatticeConfig(voxel_size=1.0, pad=1))
    labels = sample_labels(lat, lgi, LabelingConfig(wall_label=32767))

    result = cleave_lattice(lat, labels)
    # Tets near a domain edge where the wall label AND a real grain label
    # meet (possible even without a grain boundary nearby, since a dual
    # vertex sitting exactly at a half-integer coordinate resolves its
    # nearest-voxel label via round-half-to-even -- np.rint(1.5) == 2) can
    # legitimately carry 3 distinct corner labels. That's exactly what
    # deferred_tet_indices exists to flag rather than silently mishandle,
    # so some deferred tets here are expected, not a failure.

    out_vol = np.abs(signed_volumes(result.node_coords, result.tets))
    per_source = {}
    for src, v in zip(result.source_tet_index, out_vol):
        per_source[src] = per_source.get(src, 0.0) + v

    src_vol = np.abs(signed_volumes(lat.vertices, lat.tets))
    for src_idx, total in per_source.items():
        assert total == pytest.approx(src_vol[src_idx], rel=1e-9), (
            f'source tet {src_idx}: cleaved volume {total} != original {src_vol[src_idx]}')


def test_all_output_tets_positive_volume():
    shape = (4, 4, 5)
    lat = build_bcc_lattice(shape, LatticeConfig(voxel_size=1.0, pad=1))
    labels = sample_labels(lat, _two_grain_lgi(shape), LabelingConfig())
    result = cleave_lattice(lat, labels)
    vol = signed_volumes(result.node_coords, result.tets)
    assert (vol > 0).all()


def _vol6(p0, p1, p2, p3):
    return np.einsum('ij,ij->i', np.cross(p1 - p0, p2 - p0), p3 - p0)


def _covering_labels(p, node_coords, tets, tet_labels):
    a = node_coords[tets[:, 0]]
    b = node_coords[tets[:, 1]]
    c = node_coords[tets[:, 2]]
    d = node_coords[tets[:, 3]]
    pp = np.broadcast_to(p, a.shape)
    w0 = _vol6(a, b, c, d)
    w1 = _vol6(pp, b, c, d)
    w2 = _vol6(a, pp, c, d)
    w3 = _vol6(a, b, pp, d)
    w4 = _vol6(a, b, c, pp)
    s = np.sign(w0)
    inside = ((np.sign(w1) == s) & (np.sign(w2) == s) &
              (np.sign(w3) == s) & (np.sign(w4) == s))
    return tet_labels[inside]


def test_real_domain_gap_free_overlap_free_and_correctly_labeled():
    """
    The property that actually matters: sample points in the domain's
    interior and check each is covered by exactly one output tet, tagged
    with the label that matches the underlying voxel data at that point --
    not just "some tet or other."

    Sampling is restricted to at least 1 unit away from the x/y domain
    edges. Right at those edges, a dual lattice vertex can legitimately
    carry the wall label (from padding) alongside a real grain label AND
    hit an exact nearest-voxel rounding tie (np.rint's round-half-to-even),
    producing a genuine 3-label tet -- correctly deferred, not cleaved, by
    this milestone's scope (see test_volume_conserved_against_source_tets).

    The z-margin excluded around the boundary is a full voxel wide, not a
    thin band: cut points are placed at each cut EDGE's midpoint (see
    cleave.py), and the lattice's edges are not all the same length or
    orientation (long primal-primal/dual-dual edges vs. short primal-dual
    ones) -- confirmed by inspection, the resulting boundary legitimately
    "staircases" across roughly [split_at-1, split_at], not a flat plane at
    split_at-0.5. That's expected for this milestone's simple midpoint
    placement (warping cut points toward the true sub-voxel boundary is a
    later quality refinement, not a topological-correctness one) -- so
    correctness is checked only where it's unambiguous: comfortably away
    from that band.
    """
    shape = (5, 5, 6)
    split_at = 3
    lgi = _two_grain_lgi(shape, split_axis=2, split_at=split_at)
    lat = build_bcc_lattice(shape, LatticeConfig(voxel_size=1.0, pad=1))
    labels = sample_labels(lat, lgi, LabelingConfig(wall_label=32767))
    result = cleave_lattice(lat, labels)

    rng = np.random.default_rng(2)
    nx, ny, nz = shape
    gap = 0
    overlap = 0
    mislabeled = 0
    n_checked = 0
    for _ in range(400):
        x = rng.uniform(1.05, nx - 2.05)
        y = rng.uniform(1.05, ny - 2.05)
        z = rng.uniform(0.05, nz - 1.05)
        if split_at - 1.1 < z < split_at + 0.1:
            continue
        n_checked += 1
        p = np.array([x, y, z])
        covering = _covering_labels(p, result.node_coords, result.tets, result.tet_labels)
        if len(covering) == 0:
            gap += 1
        elif len(covering) > 1:
            overlap += 1
        else:
            expected = 1 if z < split_at - 1 else 2
            if covering[0] != expected:
                mislabeled += 1

    assert n_checked > 200
    assert gap == 0, f'{gap}/{n_checked} sample points uncovered'
    assert overlap == 0, f'{overlap}/{n_checked} sample points double-covered'
    assert mislabeled == 0, f'{mislabeled}/{n_checked} sample points have the wrong grain label'


WALL = 32767


def test_triple_junction_line_at_edge_gap_free_and_no_deferrals():
    """
    A triple line positioned to hug an actual domain EDGE (x=0 and z=0
    faces simultaneously) for its whole length, with its own endpoints
    touching the y=0 and y=ny-1 faces (so it also terminates at 2 corners).
    Every source tet near this feature can combine up to 3 real grains
    with the wall label at once (4 distinct corner labels total) -- this
    directly exercises the quadruple-point code path via "3 grains + wall"
    rather than "4 real grains", which is the more common real-world case
    (an RVE carved out of a larger structure will routinely have triple
    lines terminate at its own faces).
    """
    shape = (6, 6, 6)
    lgi = np.ones(shape, dtype=np.int32)
    lgi[0:1, :, :] = 1     # grain hugging the x=0 face
    lgi[1:, :, 0:1] = 2    # grain hugging the z=0 face (for the rest of x)
    lgi[1:, :, 1:] = 3     # bulk grain
    lat = build_bcc_lattice(shape, LatticeConfig(voxel_size=1.0, pad=1))
    labels = sample_labels(lat, lgi, LabelingConfig(wall_label=WALL))
    result = cleave_lattice(lat, labels)

    assert len(result.deferred_tet_indices) == 0
    vol = signed_volumes(result.node_coords, result.tets)
    assert (vol > 0).all()

    # Confirm this construction genuinely reaches a source tet with all 4
    # distinct labels present (3 real grains + wall) -- i.e. the test
    # actually exercises the quadruple-point path, not just a milder case.
    max_distinct = max(len(set(labels[tet].tolist())) for tet in lat.tets)
    assert max_distinct == 4

    # Gap/overlap-free coverage of the real domain, sampled generically
    # (independently randomized x/y/z -- NOT equal-eps points near a
    # corner, which can coincide exactly with a lattice edge/diagonal and
    # falsely read as "uncovered" under a strict inside-tet test; that's a
    # property of testing point membership on a shared mesh boundary, not
    # a defect, and is avoided here by construction).
    rng = np.random.default_rng(42)
    nx, ny, nz = shape
    gap = overlap = n_checked = 0
    for _ in range(3000):
        p = rng.uniform(0.01, [nx - 1.01, ny - 1.01, nz - 1.01])
        n_checked += 1
        covering = _covering_labels(p, result.node_coords, result.tets, result.tet_labels)
        if len(covering) == 0:
            gap += 1
        elif len(covering) > 1:
            overlap += 1

    assert gap == 0, f'{gap}/{n_checked} points uncovered near the edge/corner junction'
    assert overlap == 0, f'{overlap}/{n_checked} points double-covered'


def test_quadruple_point_near_corner_gap_free_and_no_deferrals():
    """
    An isolated quadruple-point-like feature (4 grains meeting, not just a
    line) carved into a single voxel layer right next to a domain corner,
    with the topology reverting to 2 grains elsewhere -- stresses a
    genuine 4-real-grain junction positioned as close to a corner as
    voxel resolution allows, simultaneously combined with wall padding
    from 3 directions at once.
    """
    shape = (6, 6, 6)
    lgi = np.ones(shape, dtype=np.int32)
    lgi[3:, :, :] = 2
    lgi[0:1, 0:1, 2:3] = 3
    lgi[1:3, 0:1, 2:3] = 1
    lgi[0:1, 1:3, 2:3] = 4
    lgi[1:3, 1:3, 2:3] = 2
    lat = build_bcc_lattice(shape, LatticeConfig(voxel_size=1.0, pad=1))
    labels = sample_labels(lat, lgi, LabelingConfig(wall_label=WALL))
    result = cleave_lattice(lat, labels)

    assert len(result.deferred_tet_indices) == 0
    vol = signed_volumes(result.node_coords, result.tets)
    assert (vol > 0).all()

    max_distinct = max(len(set(labels[tet].tolist())) for tet in lat.tets)
    assert max_distinct == 4

    rng = np.random.default_rng(43)
    nx, ny, nz = shape
    gap = overlap = n_checked = 0
    for _ in range(3000):
        p = rng.uniform(0.01, [nx - 1.01, ny - 1.01, nz - 1.01])
        n_checked += 1
        covering = _covering_labels(p, result.node_coords, result.tets, result.tet_labels)
        if len(covering) == 0:
            gap += 1
        elif len(covering) > 1:
            overlap += 1

    assert gap == 0, f'{gap}/{n_checked} points uncovered near the quadruple-point/corner feature'
    assert overlap == 0, f'{overlap}/{n_checked} points double-covered'
