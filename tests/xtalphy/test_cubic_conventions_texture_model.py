"""
Cubic-symmetry conventions in ``upxo.texOps.fcc`` and in
``OrientationAssigner3D.cubic_misorientation_old1``, checked against the
disorientation that matches mtex (frozen mtex values in
test_cubic_disorientation_vs_mtex.py).

* ``texOps.fcc.fcc_symmetrise_ori`` must return crystal-equivalent orientations.
* ``FCCTexture`` must generate orientations around the requested component. Crystal
  symmetry acts on the crystal side of the Euler matrix, sample symmetry (180 deg turns
  about RD, TD, ND) on the sample side.
* ``OrientationAssigner3D.cubic_misorientation_old1`` returns the cubic disorientation,
  with symmetry, and agrees with mtex.

Run from repo root:
    pytest tests/xtalphy/test_cubic_conventions_texture_model.py -v
"""

import numpy as np
import pytest

from upxo.pxtal.fm_steel_3d.orientation_assigner_3d import OrientationAssigner3D
from upxo.texOps.fcc import FCCTexture, fcc_symmetrise_ori
from upxo.xtalphy.crystal_orientation import (
    cubic_misorientation, euler_bunge_to_matrix, matrix_to_euler_bunge)

from .test_cubic_disorientation_vs_mtex import _PAIR_A, _PAIR_ANGLE_DEG, _PAIR_B

_BASES = [(35.26, 45.0, 0.0), (120.0, 60.0, 200.0), (10.0, 80.0, 330.0)]
_SAMPLE_OPS = [np.eye(3), np.diag([1.0, -1.0, -1.0]), np.diag([-1.0, 1.0, -1.0]),
               np.diag([-1.0, -1.0, 1.0])]


@pytest.mark.parametrize("bea", _BASES)
def test_fcc_symmetrise_ori_returns_the_24_crystal_equivalents(bea):
    eq = fcc_symmetrise_ori(bea).astype(np.float64)
    assert eq.shape == (24, 3)
    for e in eq:
        # float32 output: 1e-3 deg is well inside its precision, far below a wrong-side error
        assert cubic_misorientation(bea, e)[0] < 1e-2


def _sample_variants(component):
    """Distinct orientations of the component after each sample-symmetry operation (sample side).

    Some components (copper, brass) are mapped by a sample operation onto an orientation that
    is crystal-equivalent to another variant, so fewer than four are distinct.
    """
    R = euler_bunge_to_matrix(*component)
    out = []
    for M in _SAMPLE_OPS:
        e = matrix_to_euler_bunge(M @ R)
        if all(cubic_misorientation(e, v)[0] > 1.0 for v in out):
            out.append(e)
    return out


@pytest.mark.parametrize("name", ["copper", "brass", "s"])
def test_texture_without_sample_symmetry_stays_around_the_component(name):
    np.random.seed(5)
    tex = FCCTexture({name: 1.0}, spread=3.0, apply_sample_symmetries=False)
    ea = tex.generate_euler(960)
    d = np.array([cubic_misorientation(tex.components[name], e)[0] for e in ea])
    assert len(ea) == 960
    assert np.mean(d <= 15.0) >= 0.95, f"only {np.mean(d <= 15.0):.0%} within 15 deg"


@pytest.mark.parametrize("name", ["copper", "brass", "s"])
def test_texture_with_sample_symmetry_sits_on_the_sample_variants(name):
    np.random.seed(6)
    tex = FCCTexture({name: 1.0}, spread=3.0, apply_sample_symmetries=True)
    variants = _sample_variants(tex.components[name])
    ea = tex.generate_euler(1920)
    d = np.array([[cubic_misorientation(v, e)[0] for v in variants] for e in ea])
    assert np.mean(d.min(axis=1) <= 15.0) >= 0.95
    share = np.bincount(d.argmin(axis=1), minlength=len(variants)) / len(ea)
    # every distinct variant is used, none dominates beyond twice an even share
    assert (share > 0.5 / len(variants)).all() and (share < 2.0 / len(variants)).all(), share
    assert len(variants) in (2, 4)


def test_cubic_misorientation_old1_is_the_disorientation_and_matches_mtex():
    assigner = OrientationAssigner3D()
    got = np.array([assigner.cubic_misorientation_old1(a, b)[0] for a, b in zip(_PAIR_A, _PAIR_B)])
    assert np.allclose(got, _PAIR_ANGLE_DEG, atol=1e-4), \
        f"max deviation from mtex {np.abs(got - _PAIR_ANGLE_DEG).max():.4f} deg"


def test_cubic_misorientation_old1_agrees_with_disorientation_deg():
    assigner = OrientationAssigner3D()
    for a, b in list(zip(_PAIR_A, _PAIR_B))[:20]:
        assert assigner.cubic_misorientation_old1(a, b)[0] == pytest.approx(
            assigner.disorientation_deg(a, b), abs=1e-6)


def test_texOps_euler_matrix_conversions_round_trip():
    from upxo.texOps.fcc import euler_bunge_to_matrix as e2m, matrix_to_euler_bunge as m2e
    rng = np.random.default_rng(2)
    for _ in range(200):
        e = (rng.uniform(0, 360), rng.uniform(5, 175), rng.uniform(0, 360))
        back = m2e(e2m(*e))
        assert np.allclose(e2m(*back), e2m(*e), atol=1e-9)
        assert np.allclose(back, e, atol=1e-6)


@pytest.mark.parametrize("e", [(30.0, 0.0, 0.0), (120.0, 180.0, 40.0), (0.0, 0.0, 0.0)])
def test_texOps_matrix_to_euler_handles_phi_zero_and_pi(e):
    from upxo.texOps.fcc import euler_bunge_to_matrix as e2m, matrix_to_euler_bunge as m2e
    assert np.allclose(e2m(*m2e(e2m(*e))), e2m(*e), atol=1e-9)


# ---------------------------------------------------------------------------
# FM-steel assigner: its Euler matrix is the Bunge g matrix (the transpose of
# euler_bunge_to_matrix), so the crystal-frame relative rotation of two
# orientations is g1 @ g2.T. Every disorientation computed from those matrices
# must agree with mtex, and the HAGB constraints must hold in the true sense.
# ---------------------------------------------------------------------------

def test_disorientation_deg_matches_mtex():
    assigner = OrientationAssigner3D()
    got = np.array([assigner.disorientation_deg(a, b) for a, b in zip(_PAIR_A, _PAIR_B)])
    assert np.allclose(got, _PAIR_ANGLE_DEG, atol=1e-4), \
        f"max deviation from mtex {np.abs(got - _PAIR_ANGLE_DEG).max():.4f} deg"


def test_pag_hagb_constraint_holds_in_the_true_disorientation():
    n = 400
    ids = list(range(n))
    neigh = {i: [j for j in (i - 1, i + 1) if 0 <= j < n] for i in ids}
    ori = OrientationAssigner3D().assign_pag_orientations_with_hagb(
        pag_ids=ids, pag_neigh_map=neigh, orientation_pool=None,
        hagb_threshold=15.0, random_seed=3, max_attempts=2000)
    worst = min(cubic_misorientation(ori[i], ori[i + 1])[0] for i in range(n - 1))
    assert worst >= 15.0 - 1e-6, f"a neighbouring pair is only {worst:.2f} deg apart"


def test_isolated_grain_hagb_constraint_holds_in_the_true_disorientation():
    from types import SimpleNamespace
    from upxo.pxtal.fm_steel_3d.with_pags_3d import FMSteel3DWithPAGs
    n = 300
    gids = list(range(1, n + 1))
    fake = SimpleNamespace(
        retained_austenite_pag_ids=[], clusters_dict={}, pag_orientations={},
        isolated_grains=set(gids),
        neigh_gid={g: [h for h in (g - 1, g + 1) if 1 <= h <= n] for g in gids},
        _emit=lambda *a, **k: None, _verbosity=0, _log_sink=None)
    FMSteel3DWithPAGs.assign_isolated_grain_orientations(
        fake, mode='random', hagb_threshold=15.0, max_attempts=2000, random_seed=4)
    res = fake.isolated_grain_orientations
    worst = min(cubic_misorientation(res[g], res[h])[0]
                for g in gids for h in fake.neigh_gid[g] if h > g)
    assert worst >= 15.0 - 1e-6, f"a neighbouring pair is only {worst:.2f} deg apart"


def test_block_misorientation_distribution_returns_true_disorientations():
    from types import SimpleNamespace
    from upxo.pxtal.fm_steel_3d.with_orientations_3d import FMSteel3DWithOrientations
    rng = np.random.default_rng(8)
    blocks = {f"B_{p}_{p}_{k}": tuple(rng.uniform([0, 5, 0], [360, 175, 360]))
              for p in (1, 2) for k in range(4)}
    fake = SimpleNamespace(block_orientations=blocks)
    out = FMSteel3DWithOrientations.compute_misorientation_distribution(
        fake, n_sample_pairs=40, random_seed=1)
    names = list(blocks)
    truth = [cubic_misorientation(blocks[a], blocks[b])[0]
             for i, a in enumerate(names) for b in names[i + 1:]]
    assert len(out['within_pag']) and len(out['across_pag'])
    for a in np.concatenate([out['within_pag'], out['across_pag']]):
        assert min(abs(a - t) for t in truth) < 1e-3


def test_ks_variants_of_one_parent_have_the_literature_mutual_disorientations():
    """The 24 Kurdjumov-Sachs variants of any parent orientation are separated by a
    fixed set of disorientations (Kurdjumov-Sachs, tabulated in the literature)."""
    import itertools
    from upxo.pxtal.fm_steel_3d.orientation_assigner_3d import disorientation_from_g
    ks_set = np.array([10.53, 14.88, 20.61, 21.06, 47.11, 49.47, 50.51, 51.73, 57.21, 60.0])
    K = OrientationAssigner3D.get_ks_rotations()
    assigner = OrientationAssigner3D()
    rng = np.random.default_rng(1)
    for _ in range(3):
        e = (rng.uniform(0, 360), rng.uniform(5, 175), rng.uniform(0, 360))
        g = assigner.cubic_euler_bunge_to_matrix_v1(
            np.array([e[0]]), np.array([e[1]]), np.array([e[2]]))
        g = g[0] if g.ndim == 3 else g
        variants = [Ki @ g for Ki in K]
        d = np.array([disorientation_from_g(variants[i], variants[j])[0]
                      for i, j in itertools.combinations(range(24), 2)])
        assert np.abs(d[:, None] - ks_set[None, :]).min(axis=1).max() < 0.01
        # every angle of the set occurs
        assert np.abs(d[None, :] - ks_set[:, None]).min(axis=1).max() < 0.01


def test_disorientation_from_g_matches_mtex():
    from upxo.pxtal.fm_steel_3d.orientation_assigner_3d import disorientation_from_g
    assigner = OrientationAssigner3D()

    def g(e):
        m = assigner.cubic_euler_bunge_to_matrix_v1(np.array([e[0]]), np.array([e[1]]), np.array([e[2]]))
        return m[0] if m.ndim == 3 else m

    got = np.array([disorientation_from_g(g(a), g(b))[0] for a, b in zip(_PAIR_A, _PAIR_B)])
    assert np.allclose(got, _PAIR_ANGLE_DEG, atol=1e-4)
