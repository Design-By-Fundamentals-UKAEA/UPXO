"""UPXO against mtex computed live, on the same inputs.

These tests need mtex (pyMTEX) and are skipped without it. The same quantities are frozen in the
other modules of this folder, so a missing mtex does not leave them unchecked; this module is the
guard against a change in mtex or a change in the frozen values.

What is checked
    * mtex reads the matrix of ``euler_bunge_to_matrix`` as its own crystal -> sample rotation;
    * the cubic disorientation of random pairs;
    * the 24 KS variants of random parents: the same orientations, and the same 276 pair angles;
    * the Sigma3, Sigma9 and Sigma27 angles of the twin relationships;
    * the {111} plane normals of a grain in the sample frame (``orientation * Miller``).

Run from the repository root:
    pytest tests/validation_mtex/test_mtex_live.py -v
"""
import warnings

import numpy as np
import pytest

mtx = pytest.importorskip("mtex")

from upxo.pxtal.fm_steel_3d.orientation_assigner_3d import OrientationAssigner3D          # noqa: E402
from upxo.xtalphy.crystal_orientation import cubic_misorientation, euler_bunge_to_matrix      # noqa: E402

from . import _oracle as ref        # noqa: E402
from . import tolerances as tol     # noqa: E402

warnings.simplefilter("ignore")


@pytest.fixture(scope="module")
def cs():
    return mtx.crystalFrame("m-3m", [1, 1, 1])


def _mtex_ori(R, cs):
    return mtx.orientation.byMatrix(R, cs)


def _mtex_angle(Ra, Rb, cs):
    return float(np.degrees(np.asarray(_mtex_ori(Ra, cs).angle(_mtex_ori(Rb, cs)))).ravel()[0])


def test_mtex_reads_the_library_matrix_as_crystal_to_sample(cs):
    rng = np.random.default_rng(1)
    for _ in range(10):
        e = rng.uniform([0, 0, 0], [360, 180, 360])
        R = euler_bunge_to_matrix(*e)
        o = mtx.orientation.byEuler(*np.radians(e), cs)
        assert np.abs(np.asarray(o.matrix()).reshape(3, 3) - R).max() < tol.EXACT_ELEMENT


def test_cubic_disorientation_of_random_pairs(cs):
    rng = np.random.default_rng(2)
    ea = rng.uniform([0, 0, 0], [360, 180, 360], size=(200, 3))
    eb = rng.uniform([0, 0, 0], [360, 180, 360], size=(200, 3))
    oa = mtx.orientation.byEuler(*np.radians(ea).T, cs)
    ob = mtx.orientation.byEuler(*np.radians(eb).T, cs)
    theirs = np.degrees(np.asarray(oa.angle(ob))).ravel()
    ours = np.array([cubic_misorientation(a, b)[0] for a, b in zip(ea, eb)])
    assert np.abs(ours - theirs).max() < tol.LIVE_MTEX_DEG


@pytest.mark.parametrize("seed", range(4))
def test_ks_variants_of_a_random_parent(cs, seed):
    from mtex.geometry.standardorientations import variants
    e = np.random.default_rng(300 + seed).uniform([0, 0, 0], [360, 180, 360])
    Rp = ref.bunge_R(*e)[0]
    p2c = mtx.orientation.KurdjumovSachs(cs, cs)
    kids = variants(p2c, mtx.orientation.byEuler(*np.radians(e), cs))
    theirs = np.asarray(kids.matrix(), dtype=float).reshape(24, 3, 3)
    K = OrientationAssigner3D.get_ks_rotations()
    ours = Rp[None] @ K.transpose(0, 2, 1)                     # child crystal -> sample = R_p @ K_i.T
    diff = np.array([[ref.orientation_difference(ours[i], theirs[j]) for j in range(24)] for i in range(24)])
    assert (diff.min(axis=1) < 1e-8).all()
    assert len({int(j) for j in diff.argmin(axis=1)}) == 24
    iu = np.triu_indices(24, 1)
    their_ang = np.array([np.degrees(np.asarray(kids[i].angle(kids))).ravel() for i in range(24)])[iu]
    our_ang = ref.disorientation_R(ours[iu[0]], ours[iu[1]])
    assert np.abs(np.sort(their_ang) - np.sort(our_ang)).max() < tol.LIVE_MTEX_DEG


def test_twin_relationships_have_the_same_angles_in_mtex(cs):
    Rp = ref.bunge_R(37, 54, 12)[0]
    T = [ref.twin_operator(n) for n in ref.FOUR_111]
    for k in range(4):
        assert _mtex_angle(Rp, Rp @ T[k], cs) == pytest.approx(60.0, abs=tol.LIVE_MTEX_DEG)
    for i in range(4):
        for j in range(i + 1, 4):
            assert _mtex_angle(Rp @ T[i], Rp @ T[j], cs) == pytest.approx(38.9424, abs=tol.TABULATED_DEG)


def test_sigma27_angles_in_mtex(cs):
    T = [ref.twin_operator(n) for n in ref.FOUR_111]
    found = set()
    for i in range(4):
        for j in range(4):
            if i == j:
                continue
            for k in range(4):
                R = np.eye(3) @ T[i] @ T[j] @ T[k]
                found.add(round(_mtex_angle(np.eye(3), R, cs), 2))
    assert {31.59, 35.43, 60.0} <= found


def test_111_plane_normals_in_the_sample_frame(cs):
    Rp = ref.bunge_R(213, 101, 288)[0]
    o = _mtex_ori(Rp, cs)
    r = o * mtx.symmetrise(mtx.Miller(1, 1, 1, cs))
    theirs = np.stack([np.asarray(r.x).ravel(), np.asarray(r.y).ravel(), np.asarray(r.z).ravel()], axis=1)
    ours = (Rp @ ref.FOUR_111.T).T
    gap = (1 - np.abs(ours @ theirs.T).max(axis=1)).max()
    assert gap < tol.PARALLEL * 10
