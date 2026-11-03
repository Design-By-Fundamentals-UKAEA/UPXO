"""Sigma3 twin geometry: operator, variants, twin chains, habit plane, tolerance criteria.

What is checked
    * the twin operator is the 180 degree rotation about the {111} normal and is a 60 degree
      <111> disorientation (Sigma3);
    * its normal is shared by parent and twin, in the crystal frame and in the sample frame;
    * the four variants of one parent are pairwise 38.9424 degrees apart about <110> (Sigma9);
    * chains of twins give Sigma1 or Sigma9 after two, and Sigma3, Sigma27a (31.59 about <110>) and
      Sigma27b (35.43 about <210>) after three;
    * ``compute_s3_habit_plane_3d`` and ``compute_s3_lamella_angle_2d`` equal the independently
      computed plane normals and trace angles;
    * the Brandon and Palumbo-Aust tolerances quoted on the wiki, and the overlap of the Sigma5,
      Sigma7 and Sigma9 windows of the pipeline's flat tolerance.

Run from the repository root:
    pytest tests/validation_mtex/test_twin_geometry.py -v
"""
import itertools

import numpy as np
import pytest

from upxo.xtalphy.crystal_orientation import (
    CUBIC_CSL,
    compute_s3_habit_plane_3d,
    compute_s3_lamella_angle_2d,
)

from . import _oracle as ref
from . import tolerances as tol

_N = ref.FOUR_111
_T = [ref.twin_operator(n) for n in _N]
_SIGMA9_DEG = 38.9424


def _parent_R(seed):
    rng = np.random.default_rng(seed)
    return ref.bunge_R(*rng.uniform([0, 0, 0], [360, 180, 360]))[0]


def _stored_quat(R):
    """The pipeline's quaternion: the quaternion of g = R.T (sample -> crystal)."""
    return ref.quat_wxyz_from_matrix(R.T)[0]


# ---- the operator ----------------------------------------------------------------------------

@pytest.mark.parametrize("k", range(4))
def test_operator_is_a_proper_involution_that_fixes_its_normal(k):
    T = _T[k]
    assert abs(np.linalg.det(T) - 1) < tol.EXACT_ELEMENT
    assert np.allclose(T @ T, np.eye(3), atol=tol.EXACT_ELEMENT)
    assert np.allclose(T @ _N[k], _N[k], atol=tol.EXACT_ELEMENT)


def test_operator_matrix_of_the_wiki():
    want = np.array([[-1, 2, 2], [2, -1, 2], [2, 2, -1]]) / 3
    assert np.allclose(_T[0], want, atol=tol.EXACT_ELEMENT)


def test_mirror_and_rotation_differ_by_inversion():
    n = _N[0]
    M = np.eye(3) - 2 * np.outer(n, n)
    assert np.allclose(M, -_T[0], atol=tol.EXACT_ELEMENT)
    assert abs(np.linalg.det(M) + 1) < tol.EXACT_ELEMENT


@pytest.mark.parametrize("k", range(4))
def test_twin_operator_is_sigma3_60_degrees_about_111(k):
    angle, axis = ref.disorientation_axis_family(np.eye(3), _T[k])
    assert angle == pytest.approx(60.0, abs=tol.EXACT_DEG)
    assert np.allclose(axis, np.ones(3) / np.sqrt(3), atol=1e-6)


def test_unit_quaternion_of_the_60_degree_rotation():
    q = ref.quat_wxyz_from_matrix(_T[0] @ ref.OPS[np.argmax([ref.angle_of(_T[0] @ S) < 60.001 and
                                                            ref.angle_of(_T[0] @ S) > 59.999
                                                            for S in ref.OPS])])[0]
    assert abs(q[0]) == pytest.approx(0.86603, abs=1e-5)
    assert np.allclose(np.abs(q[1:]), 0.28868, atol=1e-5)


# ---- parent and twin -------------------------------------------------------------------------

@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("k", range(4))
def test_parent_and_twin_share_the_plane_normal_in_the_sample_frame(seed, k):
    Rp = _parent_R(seed)
    Rt = Rp @ _T[k]
    assert np.allclose(Rt @ _N[k], Rp @ _N[k], atol=tol.EXACT_ELEMENT)


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("k", range(4))
def test_twin_is_sigma3_of_its_parent_whatever_the_parent(seed, k):
    Rp = _parent_R(seed)
    assert ref.disorientation_R(Rp, Rp @ _T[k])[0] == pytest.approx(60.0, abs=tol.EXACT_DEG)


@pytest.mark.parametrize("seed", range(5))
def test_the_four_variants_of_one_parent_are_pairwise_sigma9(seed):
    Rp = _parent_R(seed)
    for i, j in itertools.combinations(range(4), 2):
        angle, axis = ref.disorientation_axis_family(Rp @ _T[i], Rp @ _T[j])
        assert angle == pytest.approx(_SIGMA9_DEG, abs=tol.TABULATED_DEG)
        assert np.allclose(axis, [1 / np.sqrt(2), 1 / np.sqrt(2), 0], atol=1e-6)


def test_composing_on_the_sample_side_would_lose_sigma9():
    """The wrong-side product T @ Rp does not give 38.94 for every parent: the test discriminates."""
    bad = 0
    for seed in range(20):
        Rp = _parent_R(seed)
        a = ref.disorientation_R(_T[0] @ Rp, _T[1] @ Rp)[0]
        if abs(a - _SIGMA9_DEG) > 1.0:
            bad += 1
    assert bad >= 10


# ---- chains of twins -------------------------------------------------------------------------

def _all_sigma3_descriptions():
    return [T @ S for T in _T for S in ref.OPS]


def test_two_sigma3_relationships_give_sigma1_or_sigma9_only():
    A = [T @ S for T in _T[:1] for S in ref.OPS]
    B = _all_sigma3_descriptions()
    got = {round(float(ref.disorientation_R(np.eye(3), a @ b)[0]), 2) for a in A for b in B}
    assert got == {0.0, 38.94}


def test_three_sigma3_relationships_reach_sigma3_sigma27a_and_sigma27b():
    sigma9 = [_T[i].T @ _T[j] for i, j in itertools.combinations(range(4), 2)]
    B = [T @ S for T in _T for S in ref.OPS]
    got = {}
    for a in sigma9:
        for b in B:
            angle, axis = ref.disorientation_axis_family(np.eye(3), a @ b)
            got.setdefault(round(float(angle), 2), set()).add(tuple(np.round(axis, 3)))
    assert {0.0, 31.59, 35.43, 38.94, 60.0} >= set(got) >= {31.59, 35.43, 60.0}
    assert got[31.59] == {(0.707, 0.707, 0.0)}
    assert got[35.43] == {(0.894, 0.447, 0.0)}


# ---- habit plane and lamella trace -----------------------------------------------------------

@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("k", range(4))
def test_habit_plane_is_the_rotated_111_normal(seed, k):
    Rp = _parent_R(seed)
    n = compute_s3_habit_plane_3d(_stored_quat(Rp), np.random.default_rng(0), variant_idx=k)
    assert np.allclose(n, Rp @ _N[k], atol=tol.EXACT_ELEMENT)


def test_habit_plane_of_a_variant_is_the_one_of_its_twin():
    for seed in range(5):
        Rp = _parent_R(seed)
        for k in range(4):
            Rt = Rp @ _T[k]
            n_parent = compute_s3_habit_plane_3d(_stored_quat(Rp), np.random.default_rng(0), variant_idx=k)
            n_twin_same_plane = Rt @ _N[k]
            assert np.allclose(n_parent, n_twin_same_plane, atol=tol.EXACT_ELEMENT)


def _oracle_trace_angle(R):
    n_s = (R @ _N.T).T
    best = int(np.argmax(np.linalg.norm(n_s[:, :2], axis=1)))
    nx, ny = n_s[best, 0], n_s[best, 1]
    return float(np.degrees(np.arctan2(nx, -ny)) % 180.0)


@pytest.mark.parametrize("seed", range(30))
def test_lamella_angle_is_the_trace_of_the_plane_most_inclined_to_the_section(seed):
    Rp = _parent_R(seed)
    got = compute_s3_lamella_angle_2d(_stored_quat(Rp))
    want = _oracle_trace_angle(Rp)
    diff = abs((got - want + 90) % 180 - 90)
    assert diff < 1e-6


# ---- tolerance criteria and window overlap --------------------------------------------------

@pytest.mark.parametrize("sigma, brandon, palumbo", [(3, 8.66, 6.00), (9, 5.00, 2.40), (27, 2.89, 0.96)])
def test_published_tolerance_criteria(sigma, brandon, palumbo):
    assert 15 * sigma ** -0.5 == pytest.approx(brandon, abs=0.005)
    assert 15 * sigma ** (-5 / 6) == pytest.approx(palumbo, abs=0.005)


def test_flat_tolerance_windows_of_sigma5_sigma7_sigma9_overlap_at_two_degrees():
    a5, a7, a9 = CUBIC_CSL["S5"], CUBIC_CSL["S7"], CUBIC_CSL["S9"]
    assert (a5, a7, a9) == (36.87, 38.21, 38.94)
    assert a9 - a5 == pytest.approx(2.07, abs=1e-9)
    csl_tol = 2.0
    assert a7 - csl_tol < a5 + csl_tol            # S5 and S7 windows overlap
    assert a9 - csl_tol < a7 + csl_tol            # S7 and S9 windows overlap
    assert a9 - csl_tol < a5 + csl_tol            # S5 and S9 windows overlap
