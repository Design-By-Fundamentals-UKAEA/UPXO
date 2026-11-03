"""The 24 Kurdjumov-Sachs variants of the FM-steel pipeline.

What is checked
    * the 24 matrices are the KS family built independently from the two vector pairs
      (111)_g || (011)_a and [1-10]_g || [-1-11]_a (``_oracle.ks_variants``);
    * they are the same 24 orientations as mtex's (frozen JSON, ``reference/ks_mtex.json``);
    * the 276 pairs give 10 distinct disorientations with the counts below, whatever the parent;
    * 4 packets of 6 variants, 3 Bain groups of 8, and the split of the angles between the
      variants of one packet and variants of different packets.

Run from the repository root:
    pytest tests/validation_mtex/test_ks_variant_set.py -v
"""
import json
import pathlib

import numpy as np
import pytest

from upxo.pxtal.fm_steel_3d.orientation_assigner_3d import (
    OrientationAssigner3D,
    disorientation_from_g_batch,
)

from . import _oracle as ref
from . import tolerances as tol

_REF = json.loads((pathlib.Path(__file__).parent / "reference" / "ks_mtex.json").read_text(encoding="utf-8"))

# angle (deg) -> (pairs in total, pairs inside one packet, pairs between packets)
_PAIR_TABLE = {
    60.00: (36, 36, 0),
    57.21: (48, 0, 48),
    51.73: (12, 0, 12),
    50.51: (48, 0, 48),
    49.47: (24, 12, 12),
    47.11: (24, 0, 24),
    21.06: (12, 0, 12),
    20.61: (24, 0, 24),
    14.88: (24, 0, 24),
    10.53: (24, 12, 12),
}
_IU = np.triu_indices(24, 1)


@pytest.fixture(scope="module")
def K():
    return OrientationAssigner3D.get_ks_rotations()


@pytest.fixture(scope="module")
def pair_angles(K):
    return disorientation_from_g_batch(K[_IU[0]], K[_IU[1]])


def _grouped(values, decimals=2):
    out = {}
    for v in values:
        out.setdefault(round(float(v), decimals), []).append(float(v))
    return out


def test_there_are_24_proper_rotations(K):
    assert K.shape == (24, 3, 3)
    assert np.allclose(np.linalg.det(K), 1.0, atol=tol.EXACT_ELEMENT)
    assert np.allclose(K @ K.transpose(0, 2, 1), np.eye(3), atol=tol.EXACT_ELEMENT)


def test_variants_are_the_independently_built_ks_family(K):
    """Each library variant equals exactly one of the 24 built from the vector pairs."""
    oracle = ref.ks_variants()
    d = np.abs(K[:, None] - oracle[None]).max(axis=(2, 3))            # (24, 24) element differences
    assert (d.min(axis=1) < tol.EXACT_ELEMENT).all()
    assert (d.min(axis=0) < tol.EXACT_ELEMENT).all()
    assert len({int(j) for j in d.argmin(axis=1)}) == 24


def test_variants_are_the_same_orientations_as_mtex(K):
    """Parent = identity: the child orientation (crystal -> sample) of variant i is K[i].T.

    mtex's 24 children (frozen) and the library's agree as orientations of a cubic crystal,
    one to one.
    """
    mtex_R = np.array(_REF["ks"]["variant_matrices_crystal_to_sample"])
    ours_R = K.transpose(0, 2, 1)
    diff = np.array([[ref.orientation_difference(ours_R[i], mtex_R[j]) for j in range(24)] for i in range(24)])
    assert (diff.min(axis=1) < tol.SAME_ORIENTATION_ELEMENT).all()
    assert len({int(j) for j in diff.argmin(axis=1)}) == 24


def test_mtex_variant_index_is_a_bijection_with_the_library_variants(K):
    mtex_R = np.array(_REF["ks"]["variant_matrices_crystal_to_sample"])
    ours_R = K.transpose(0, 2, 1)
    mapping = [int(np.argmin([ref.orientation_difference(ours_R[i], mtex_R[j]) for j in range(24)])) + 1
               for i in range(24)]
    assert sorted(mapping) == list(range(1, 25))
    assert sorted(_REF["ks"]["mtex_to_morito_permutation"]) == list(range(1, 25))


def test_276_pairs_give_10_distinct_disorientations(pair_angles):
    assert len(pair_angles) == 276
    groups = _grouped(pair_angles)
    assert sorted(groups, reverse=True) == sorted(_PAIR_TABLE, reverse=True)


@pytest.mark.parametrize("angle, expected", list(_PAIR_TABLE.items()))
def test_pair_count_and_value_of_each_distinct_angle(pair_angles, angle, expected):
    sel = pair_angles[np.abs(pair_angles - angle) < 0.006]
    assert len(sel) == expected[0]
    assert np.abs(sel - angle).max() < tol.TABULATED_DEG


def test_pair_angles_equal_the_frozen_mtex_counts(pair_angles):
    mtex_counts = {float(k): v for k, v in _REF["ks"]["pair_angle_counts_deg"].items()}
    ours = {k: len(v) for k, v in _grouped(pair_angles).items()}
    assert ours == mtex_counts


def test_pair_angles_do_not_depend_on_the_parent(K):
    """K @ g for any parent g gives the same 10 angles with the same counts."""
    rng = np.random.default_rng(20260503)
    want = {k: v[0] for k, v in _PAIR_TABLE.items()}
    for _ in range(25):
        e = rng.uniform([0, 0, 0], [360, 180, 360])
        g_parent = ref.bunge_R(*e)[0].T
        g_child = K @ g_parent
        ang = disorientation_from_g_batch(g_child[_IU[0]], g_child[_IU[1]])
        got = {k: len(v) for k, v in _grouped(ang).items()}
        assert got == want


def test_pair_angles_agree_with_the_independent_reduction(K, pair_angles):
    mine = ref.disorientation_R(K[_IU[0]].transpose(0, 2, 1), K[_IU[1]].transpose(0, 2, 1))
    # g1 @ g2.T in the library is R1.T @ R2 with R = g.T, which is what the oracle reduces
    assert np.abs(np.sort(mine) - np.sort(pair_angles)).max() < tol.EXACT_DEG


def test_four_packets_of_six(K):
    packets = np.array([ref.packet_index(g) for g in K])
    assert sorted(np.bincount(packets)) == [6, 6, 6, 6]


def test_three_bain_groups_of_eight_and_two_per_packet_and_group(K):
    packets = np.array([ref.packet_index(g) for g in K])
    bain = np.array([ref.bain_index(g) for g in K])
    assert sorted(np.bincount(bain)) == [8, 8, 8]
    table = np.zeros((4, 3), int)
    for p, b in zip(packets, bain):
        table[p, b] += 1
    assert (table == 2).all()


def test_inside_and_between_packet_split_of_every_angle(K, pair_angles):
    packets = np.array([ref.packet_index(g) for g in K])
    same = packets[_IU[0]] == packets[_IU[1]]
    for angle, (total, inside, between) in _PAIR_TABLE.items():
        m = np.abs(pair_angles - angle) < 0.006
        assert int((m & same).sum()) == inside, angle
        assert int((m & ~same).sum()) == between, angle
    assert int(same.sum()) == 60 and int((~same).sum()) == 216


def test_only_three_angles_occur_inside_a_packet(K, pair_angles):
    packets = np.array([ref.packet_index(g) for g in K])
    same = packets[_IU[0]] == packets[_IU[1]]
    inside = sorted({round(float(a), 2) for a in pair_angles[same]}, reverse=True)
    assert inside == [60.0, 49.47, 10.53]


def test_the_set_of_angles_is_the_one_documented_for_a_parent_pag(K, pair_angles):
    documented = [10.53, 14.88, 20.61, 21.06, 47.11, 49.47, 50.51, 51.73, 57.21, 60.0]    # docs/orientation_conventions.rst
    got = sorted({round(float(a), 2) for a in pair_angles})
    assert got == documented
