"""Invariants of the orientation maths that hold whatever mtex gives.

A wrong-side symmetry operator, a transposed matrix or a conjugated quaternion changes these
properties even when a Mackenzie check on a random pool cannot see it, so they are the first
line of defence when a convention changes.

What is checked
    * the disorientation equals the independent reduction, is symmetric, obeys the triangle
      inequality, is bounded by 62.7994 degrees, and is unchanged by a rotation of the sample;
    * the same for the g-matrix route of the FM-steel pipeline (sample rotation acts from the right);
    * the independent sample-side reduction differs from the correct one for many pairs, so the
      pairs used here discriminate;
    * quaternion conventions: the stored quaternion is the quaternion of g = R.T, and the
      conjugate is the quaternion of R.

Run from the repository root:
    pytest tests/validation_mtex/test_invariants.py -v
"""
import numpy as np
import pytest

from upxo.pxtal.fm_steel_3d.orientation_assigner_3d import disorientation_from_g_batch
from upxo.xtalphy.crystal_orientation import (
    cubic_misorientation,
    defdap_passive_to_active,
    euler_bunge_to_matrix,
    quat_to_R_batch,
    rotation_matrices_to_quats,
)

from . import _oracle as ref
from . import tolerances as tol

_RNG = np.random.default_rng(20261103)
_N_PAIRS = 300


def _random_euler(n, rng=_RNG):
    return rng.uniform([0, 0, 0], [360, 180, 360], size=(n, 3))


def _random_rotation(rng):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    from scipy.spatial.transform import Rotation
    return Rotation.from_quat(q).as_matrix()


@pytest.fixture(scope="module")
def pairs():
    ea, eb = _random_euler(_N_PAIRS), _random_euler(_N_PAIRS)
    return ea, eb, ref.bunge_R(*ea.T), ref.bunge_R(*eb.T)


def _lib(ea, eb):
    return np.array([cubic_misorientation(a, b)[0] for a, b in zip(ea, eb)])


def test_disorientation_equals_the_independent_reduction(pairs):
    ea, eb, Ra, Rb = pairs
    assert np.abs(_lib(ea, eb) - ref.disorientation_R(Ra, Rb)).max() < tol.LIVE_MTEX_DEG


def test_the_pairs_discriminate_the_sample_side_reduction(pairs):
    _, _, Ra, Rb = pairs
    right = ref.disorientation_R(Ra, Rb)
    wrong = ref.disorientation_R_sample_side(Ra, Rb)
    assert (np.abs(right - wrong) > 1.0).sum() >= 0.3 * _N_PAIRS


def test_disorientation_is_symmetric_and_bounded(pairs):
    ea, eb, *_ = pairs
    ab, ba = _lib(ea, eb), _lib(eb, ea)
    assert np.abs(ab - ba).max() < tol.LIVE_MTEX_DEG
    assert ab.min() >= 0 and ab.max() <= tol.CUBIC_MAX_DISORIENTATION_DEG + 1e-3


def test_triangle_inequality(pairs):
    ea, eb, *_ = pairs
    ec = _random_euler(_N_PAIRS)
    ab, bc, ac = _lib(ea, eb), _lib(eb, ec), _lib(ea, ec)
    assert (ac <= ab + bc + 1e-6).all()


def test_a_rotation_of_the_whole_sample_does_not_change_the_disorientation(pairs):
    _, _, Ra, Rb = pairs
    rng = np.random.default_rng(7)
    base = ref.disorientation_R(Ra, Rb)
    for _ in range(5):
        Q = _random_rotation(rng)
        ea = ref.euler_from_R(Q @ Ra)
        eb = ref.euler_from_R(Q @ Rb)
        assert np.abs(_lib(ea, eb) - base).max() < tol.LIVE_MTEX_DEG


def test_a_cubic_relabelling_of_the_crystal_axes_does_not_change_the_disorientation(pairs):
    _, _, Ra, Rb = pairs
    base = ref.disorientation_R(Ra, Rb)
    rng = np.random.default_rng(11)
    for _ in range(5):
        S = ref.OPS[rng.integers(24)]
        ea = ref.euler_from_R(Ra @ S)
        eb = ref.euler_from_R(Rb @ ref.OPS[rng.integers(24)])
        assert np.abs(_lib(ea, eb) - base).max() < tol.LIVE_MTEX_DEG


# ---- g-matrix route of the FM-steel pipeline -------------------------------------------------

def test_g_route_equals_the_independent_reduction(pairs):
    *_, Ra, Rb = pairs
    got = disorientation_from_g_batch(Ra.transpose(0, 2, 1), Rb.transpose(0, 2, 1))
    assert np.abs(got - ref.disorientation_R(Ra, Rb)).max() < tol.EXACT_DEG


def test_g_route_is_unchanged_by_a_rotation_of_the_sample(pairs):
    *_, Ra, Rb = pairs
    ga, gb = Ra.transpose(0, 2, 1), Rb.transpose(0, 2, 1)
    base = disorientation_from_g_batch(ga, gb)
    Q = _random_rotation(np.random.default_rng(3))
    # g maps sample -> crystal, so a rotation of the sample multiplies from the right: g @ Q.T
    got = disorientation_from_g_batch(ga @ Q.T, gb @ Q.T)
    assert np.abs(got - base).max() < tol.EXACT_DEG


def test_g_route_and_euler_route_agree(pairs):
    ea, eb, Ra, Rb = pairs
    got = disorientation_from_g_batch(Ra.transpose(0, 2, 1), Rb.transpose(0, 2, 1))
    assert np.abs(got - _lib(ea, eb)).max() < tol.LIVE_MTEX_DEG


# ---- conventions -----------------------------------------------------------------------------

@pytest.mark.parametrize("seed", range(10))
def test_euler_matrix_is_the_crystal_to_sample_rotation(seed):
    e = _random_euler(1, np.random.default_rng(seed))[0]
    assert np.allclose(euler_bunge_to_matrix(*e), ref.bunge_R(*e)[0], atol=tol.EXACT_ELEMENT)


@pytest.mark.parametrize("seed", range(10))
def test_stored_quaternion_is_the_quaternion_of_g_and_its_conjugate_is_that_of_R(seed):
    e = _random_euler(1, np.random.default_rng(100 + seed))[0]
    R = ref.bunge_R(*e)[0]
    q_stored = rotation_matrices_to_quats(R.T[None])
    assert np.allclose(quat_to_R_batch(q_stored)[0], R.T, atol=tol.EXACT_ELEMENT)
    q_active = defdap_passive_to_active(q_stored)
    assert np.allclose(quat_to_R_batch(q_active)[0], R, atol=tol.EXACT_ELEMENT)
    assert np.allclose(q_active[0, 1:], -q_stored[0, 1:], atol=tol.EXACT_ELEMENT)
    assert np.allclose(q_active[0, 0], q_stored[0, 0], atol=tol.EXACT_ELEMENT)


def test_a_stored_quaternion_read_as_crystal_to_sample_gives_the_transposed_orientation():
    e = np.array([37.0, 54.0, 12.0])
    R = ref.bunge_R(*e)[0]
    q_stored = rotation_matrices_to_quats(R.T[None])
    misread = quat_to_R_batch(q_stored)[0]
    assert np.allclose(misread, R.T, atol=tol.EXACT_ELEMENT)
    assert np.abs(misread - R).max() > 0.1
