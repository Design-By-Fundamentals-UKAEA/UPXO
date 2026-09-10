"""
test_stencils.py
=================
Correctness tests for the per-tet cleaving stencil (stencils.cleave_tet).

Two properties matter most, and both are tested directly rather than by
trusting the derivation on paper:

1. Volume conservation: whatever cleave_tet produces for a given tet must
   sum to exactly that tet's own volume, for EVERY non-trivial 2-label
   corner pattern, and regardless of which physical corner happens to carry
   the smallest global id (since the apex/fan choice depends on that).

2. Cross-tet consistency: if two tets share a face and that face is cut,
   both tets -- cleaved independently, knowing nothing about each other --
   must produce the exact same triangulation of that shared face. This is
   the specific failure mode (a crack at a grain boundary) the whole
   canonical-key/fan/cone design exists to prevent.
"""
import itertools

import numpy as np
import pytest

from upxo.meshing.cleaving.config import LatticeConfig
from upxo.meshing.cleaving.lattice import build_bcc_lattice
from upxo.meshing.cleaving.stencils import (
    TooManyLabelsError, cleave_tet, point_key,
)


def _tet_volume(a, b, c, d):
    return np.dot(np.cross(b - a, c - a), d - a) / 6.0


def _resolve(point, gid_to_coord):
    if point[0] == 'v':
        return gid_to_coord[point[1]]
    coords = [gid_to_coord[g] for g in point[1:]]
    return sum(coords) / len(coords)


NON_TRIVIAL_PATTERNS = [c for c in itertools.product([0, 1], repeat=4)
                        if len(set(c)) > 1]

# All (up to relabeling) 2+1+1 triple-junction patterns: pick which 2 of the
# 4 corners share the majority label (0), the other 2 each get one of the 2
# distinct minority labels (1, 2), in either order.
NON_TRIVIAL_3LABEL_PATTERNS = []
for _maj_pair in itertools.combinations(range(4), 2):
    _rest = [i for i in range(4) if i not in _maj_pair]
    for _assign in itertools.permutations([1, 2]):
        _labs = [0, 0, 0, 0]
        _labs[_rest[0]] = _assign[0]
        _labs[_rest[1]] = _assign[1]
        NON_TRIVIAL_3LABEL_PATTERNS.append(tuple(_labs))


@pytest.mark.parametrize('pattern', NON_TRIVIAL_PATTERNS)
def test_volume_conservation_all_patterns(pattern):
    rng = np.random.default_rng(hash(pattern) & 0xFFFFFFFF)
    coords = np.array([
        [0.0, 0.0, 0.0], [1.3, 0.1, 0.2], [0.2, 1.1, 0.4], [0.3, 0.2, 1.5],
    ])
    input_vol = abs(_tet_volume(*coords))

    for _ in range(30):
        gids = rng.permutation(10_000)[:4]
        gid_to_coord = {int(g): coords[i] for i, g in enumerate(gids)}
        by_label = cleave_tet(gids, np.array(pattern))

        total = 0.0
        n_tets = 0
        for lab, tets in by_label.items():
            for pts in tets:
                pcoords = [_resolve(p, gid_to_coord) for p in pts]
                total += abs(_tet_volume(*pcoords))
                n_tets += 1

        assert total == pytest.approx(input_vol, rel=1e-9), (
            f'pattern={pattern} gids={gids}: volume {total} != {input_vol}')
        assert n_tets > 0


def test_volume_conservation_4label_pattern():
    """
    All 4 corners distinctly labelled (a quadruple junction) -- structurally
    there's only one such pattern (unlike 2- or 3-label splits, which vary
    by which corners share a label), so this fuzzes hard on random
    global-id orderings instead, since the new quadruple point Q and all 6
    pairwise interface quads depend on which corner has the lowest id.
    """
    rng = np.random.default_rng(99)
    coords = np.array([
        [0.0, 0.0, 0.0], [1.3, 0.1, 0.2], [0.2, 1.1, 0.4], [0.3, 0.2, 1.5],
    ])
    input_vol = abs(_tet_volume(*coords))
    pattern = (10, 20, 30, 40)

    for _ in range(200):
        gids = rng.permutation(10_000)[:4]
        gid_to_coord = {int(g): coords[i] for i, g in enumerate(gids)}
        by_label = cleave_tet(gids, np.array(pattern))
        assert len(by_label) == 4

        total = 0.0
        n_tets = 0
        for lab, tets in by_label.items():
            for pts in tets:
                pcoords = [_resolve(p, gid_to_coord) for p in pts]
                total += abs(_tet_volume(*pcoords))
                n_tets += 1

        assert total == pytest.approx(input_vol, rel=1e-9), (
            f'pattern={pattern} gids={gids}: volume {total} != {input_vol}')
        assert n_tets > 0


@pytest.mark.parametrize('pattern', NON_TRIVIAL_3LABEL_PATTERNS)
def test_volume_conservation_all_3label_patterns(pattern):
    """Same check as the 2-label case, for every 2+1+1 (triple-junction)
    corner pattern -- this is what actually exercises the new face-triple-
    point + triple-line code path, not just the pre-existing 2-label one."""
    rng = np.random.default_rng(hash(pattern) & 0xFFFFFFFF)
    coords = np.array([
        [0.0, 0.0, 0.0], [1.3, 0.1, 0.2], [0.2, 1.1, 0.4], [0.3, 0.2, 1.5],
    ])
    input_vol = abs(_tet_volume(*coords))

    for _ in range(30):
        gids = rng.permutation(10_000)[:4]
        gid_to_coord = {int(g): coords[i] for i, g in enumerate(gids)}
        by_label = cleave_tet(gids, np.array(pattern))
        assert len(by_label) == 3

        total = 0.0
        n_tets = 0
        for lab, tets in by_label.items():
            for pts in tets:
                pcoords = [_resolve(p, gid_to_coord) for p in pts]
                total += abs(_tet_volume(*pcoords))
                n_tets += 1

        assert total == pytest.approx(input_vol, rel=1e-9), (
            f'pattern={pattern} gids={gids}: volume {total} != {input_vol}')
        assert n_tets > 0


def test_uncut_tet_returns_itself():
    gids = [5, 12, 3, 99]
    labels = [7, 7, 7, 7]
    by_label = cleave_tet(gids, labels)
    assert set(by_label.keys()) == {7}
    tets = by_label[7]
    assert len(tets) == 1
    # The single output tet must be exactly the 4 original vertices.
    got = {p[1] for p in tets[0]}
    assert got == set(gids)


def test_3_distinct_labels_handled():
    # A 2+1+1 split (3 distinct corner labels -- a triple junction).
    by_label = cleave_tet([1, 2, 3, 4], [10, 10, 20, 30])
    assert set(by_label.keys()) == {10, 20, 30}


def test_4_distinct_labels_handled():
    # A 1+1+1+1 split (4 distinct corner labels -- a quadruple junction).
    by_label = cleave_tet([1, 2, 3, 4], [10, 20, 30, 40])
    assert set(by_label.keys()) == {10, 20, 30, 40}


def test_more_than_4_labels_impossible_but_guarded():
    # Not reachable via real (4-vertex) tet data, but the guard exists.
    with pytest.raises(TooManyLabelsError):
        cleave_tet([1, 2, 3, 4, 5], [10, 20, 30, 40, 50])


def _output_faces(by_label):
    """All triangular faces of all output tets, as frozensets of Points."""
    faces = []
    for tets in by_label.values():
        for t in tets:
            for i, j, k in ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)):
                faces.append(frozenset((t[i], t[j], t[k])))
    return faces


def _face_restricted_to(faces, allowed_gids):
    """Faces whose every point (vertex or cut point) only involves gids in
    allowed_gids -- i.e. faces lying on the sub-face spanned by those ids."""
    out = set()
    for f in faces:
        ids_in_face = set()
        for p in f:
            ids_in_face.update(p[1:])
        if ids_in_face <= allowed_gids:
            out.add(f)
    return out


def _find_shared_face_pair(lattice):
    """First pair of tet row-indices in lattice.tets sharing exactly 3 vertices."""
    tets = lattice.tets
    n = len(tets)
    for i in range(min(n, 400)):
        si = set(tets[i].tolist())
        for j in range(i + 1, min(n, 400)):
            sj = set(tets[j].tolist())
            shared = si & sj
            if len(shared) == 3:
                return i, j, shared
    raise AssertionError('no adjacent tet pair found in sampled range')


def test_cross_tet_shared_face_consistency():
    """
    Pull two real, actually-adjacent tets out of a real lattice, force their
    shared face to be cut, cleave each in isolation, and require the
    triangulation restricted to that shared face to match exactly.
    """
    lat = build_bcc_lattice((3, 3, 3), LatticeConfig(voxel_size=1.0, pad=1))
    i, j, shared_gids = _find_shared_face_pair(lat)
    tet_a = lat.tets[i]
    tet_b = lat.tets[j]

    # Build a labeling: one of the shared vertices is label 2 (the "solo"
    # corner), everything else (including the non-shared 4th corner of each
    # tet) is label 1 -- this forces the shared face itself to be cut (a
    # 1-vs-2 split on that face) in both tets simultaneously.
    solo_gid = sorted(shared_gids)[0]
    all_gids = set(tet_a.tolist()) | set(tet_b.tolist())
    label_of = {g: (2 if g == solo_gid else 1) for g in all_gids}

    labels_a = [label_of[g] for g in tet_a.tolist()]
    labels_b = [label_of[g] for g in tet_b.tolist()]

    out_a = cleave_tet(tet_a, labels_a)
    out_b = cleave_tet(tet_b, labels_b)

    faces_a = _face_restricted_to(_output_faces(out_a), shared_gids)
    faces_b = _face_restricted_to(_output_faces(out_b), shared_gids)

    assert faces_a, 'expected at least one face restricted to the shared vertices'
    assert faces_a == faces_b, (
        f'shared-face triangulation mismatch:\n  tet A gave: {faces_a}\n'
        f'  tet B gave: {faces_b}')


def test_cross_tet_shared_face_consistency_many_random_labelings():
    """Same check, but fuzzed over many adjacent pairs and 2-vs-1 labelings
    on the shared face, to reduce the chance a single lucky case hides a bug."""
    lat = build_bcc_lattice((4, 4, 4), LatticeConfig(voxel_size=1.0, pad=1))
    tets = lat.tets
    rng = np.random.default_rng(0)

    pair_indices = np.arange(min(len(tets), 300))
    rng.shuffle(pair_indices)

    checked = 0
    for i in pair_indices[:60]:
        si = set(tets[i].tolist())
        for j in pair_indices[60:160]:
            sj = set(tets[j].tolist())
            shared = si & sj
            if len(shared) != 3:
                continue
            shared = sorted(shared)
            other_a = next(g for g in tets[i].tolist() if g not in shared)
            other_b = next(g for g in tets[j].tolist() if g not in shared)

            for solo_gid in shared:
                label_of = {g: 1 for g in shared + [other_a, other_b]}
                label_of[solo_gid] = 2
                labels_a = [label_of[g] for g in tets[i].tolist()]
                labels_b = [label_of[g] for g in tets[j].tolist()]
                out_a = cleave_tet(tets[i], labels_a)
                out_b = cleave_tet(tets[j], labels_b)
                fa = _face_restricted_to(_output_faces(out_a), set(shared))
                fb = _face_restricted_to(_output_faces(out_b), set(shared))
                assert fa == fb, f'mismatch for pair ({i},{j}) solo={solo_gid}'
                checked += 1
            break  # one partner per i is enough once found

    assert checked >= 5, f'only checked {checked} adjacent pairs -- test setup issue'


def test_cross_tet_shared_face_consistency_with_triple_point():
    """
    Same cross-tet consistency requirement, but forcing the shared face
    itself to carry a face triple point (its 3 corners get 3 distinct
    labels) -- the new code path added for triple junctions. Each tet's own
    4th corner reuses one of the 3 shared labels, so neither tet exceeds
    the 3-distinct-label scope.
    """
    lat = build_bcc_lattice((4, 4, 4), LatticeConfig(voxel_size=1.0, pad=1))
    i, j, shared_gids = _find_shared_face_pair(lat)
    tet_a, tet_b = lat.tets[i], lat.tets[j]
    shared_sorted = sorted(shared_gids)

    label_of = {g: lab for g, lab in zip(shared_sorted, [1, 2, 3])}
    other_a = next(g for g in tet_a.tolist() if g not in shared_gids)
    other_b = next(g for g in tet_b.tolist() if g not in shared_gids)
    label_of[other_a] = 1
    label_of[other_b] = 2

    labels_a = [label_of[g] for g in tet_a.tolist()]
    labels_b = [label_of[g] for g in tet_b.tolist()]
    assert len(set(labels_a)) == 3 and len(set(labels_b)) == 3

    out_a = cleave_tet(tet_a, labels_a)
    out_b = cleave_tet(tet_b, labels_b)

    faces_a = _face_restricted_to(_output_faces(out_a), set(shared_gids))
    faces_b = _face_restricted_to(_output_faces(out_b), set(shared_gids))

    assert faces_a, 'expected at least one face restricted to the shared vertices'
    assert faces_a == faces_b, (
        f'shared-face triangulation mismatch (triple point):\n'
        f'  tet A gave: {faces_a}\n  tet B gave: {faces_b}')


def test_cross_tet_shared_face_consistency_both_tets_quadruple():
    """
    Push the shared-face check to its most demanding combination: BOTH tets
    have 4 distinct corner labels (so each independently builds its own
    tet-private quadruple point Q), while their SHARED face still carries a
    triple point (3 distinct labels on it). The quadruple points must never
    leak into the shared face's triangulation -- they're tet-private by
    construction -- and the shared face itself must still match exactly.
    """
    lat = build_bcc_lattice((4, 4, 4), LatticeConfig(voxel_size=1.0, pad=1))
    i, j, shared_gids = _find_shared_face_pair(lat)
    tet_a, tet_b = lat.tets[i], lat.tets[j]
    shared_sorted = sorted(shared_gids)

    label_of = {g: lab for g, lab in zip(shared_sorted, [1, 2, 3])}
    other_a = next(g for g in tet_a.tolist() if g not in shared_gids)
    other_b = next(g for g in tet_b.tolist() if g not in shared_gids)
    label_of[other_a] = 4
    label_of[other_b] = 5

    labels_a = [label_of[g] for g in tet_a.tolist()]
    labels_b = [label_of[g] for g in tet_b.tolist()]
    assert len(set(labels_a)) == 4 and len(set(labels_b)) == 4

    out_a = cleave_tet(tet_a, labels_a)
    out_b = cleave_tet(tet_b, labels_b)

    faces_a = _face_restricted_to(_output_faces(out_a), set(shared_gids))
    faces_b = _face_restricted_to(_output_faces(out_b), set(shared_gids))

    assert faces_a, 'expected at least one face restricted to the shared vertices'
    assert faces_a == faces_b, (
        f'shared-face triangulation mismatch (both tets quadruple):\n'
        f'  tet A gave: {faces_a}\n  tet B gave: {faces_b}')
