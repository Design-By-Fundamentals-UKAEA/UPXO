"""
Sigma3 twin generation (``TwinGenerator3D._apply_sigma3_variant``) and the {111} habit plane
(``compute_s3_habit_plane_3d``, ``compute_s3_lamella_angle_2d``), checked against mtex.
The Sigma5 seeding used by the 2D Monte Carlo twin builder is checked at the end.

The numbers below were produced once with mtex 1.0.0 on a cubic m-3m crystal frame and are
frozen here, so the tests need neither mtex nor an EBSD file.

* ``_MTEX_TWIN_R``   the twin orientation mtex gives for a parent orientation: the parent rotation
  times a 60 degree rotation about <111> of the crystal (``rp * T``), one per variant.
* ``_MTEX_HABIT``    ``orientation * Miller(<111>)``, the {111} plane normals in the sample frame.
* ``_MTEX_LAMELLA_DEG``  the 2D trace angle derived from those normals with the library's rule.

Orientations in the pipeline are DefDAP-stored quaternions: the quaternion of the Bunge g matrix
(sample -> crystal), the inverse of the crystal -> sample matrix R that mtex's rotation holds.

Run from repo root:
    pytest tests/xtalphy/test_sigma3_twin_vs_mtex.py -v
"""

import numpy as np
import pytest

from upxo.pxtal.twinned_simple_3d.twin_generator_3d import TwinGenerator3D
from upxo.xtalphy.crystal_orientation import (
    _SIGMA3_Q_ALL_VARIANTS,
    _SIGMA5_Q_VARIANTS,
    _positive_w,
    _quat_mul,
    compute_s3_habit_plane_3d,
    compute_s3_lamella_angle_2d,
    euler_bunge_to_matrix,
    get_cubic_ops_np,
    quat_to_R_batch,
    rotation_matrices_to_quats,
)

# ---------------------------------------------------------------------------
# Frozen mtex 1.0.0 reference values
# ---------------------------------------------------------------------------

_EULER = [(37.0, 54.0, 12.0), (213.0, 101.0, 288.0), (305.0, 22.0, 140.0)]

# mtex rotation(euler) * rotation.byAxisAngle(<111>_k, 60 deg): [parent][variant] -> 3x3 crystal->sample matrix
_MTEX_TWIN_R = [
    [
        [[-0.0319038518861855, -0.25266241400759004, 0.9670283598647362], [0.8955744454689764, -0.4367983651613859, -0.08457896201347181], [0.44376633137019333, 0.8633474924618696, 0.24021355166193725]],
        [[-0.0235121687693175, -0.9754200429816042, 0.21909568154011938], [0.9995964429345365, -0.019443035452087587, 0.020710375299146155], [-0.015941430060874326, 0.2194942097691307, 0.9754835532624018]],
        [[0.6256592728308945, -0.6508343221814974, 0.43007576006009335], [0.13811684301511767, -0.4501828354117975, -0.8821899752185647], [0.7677722396624234, 0.6113510446307793, -0.19176988353910307]],
        [[0.3178569182645239, 0.29268186891392056, 0.9018338556078026], [0.7769006379059469, 0.4648346455092671, -0.42468123475803254], [-0.5435001180613607, 0.8356231662318422, -0.07963382273857156]],
    ],
    [
        [[-0.48183218484843704, -0.8561369296821029, -0.18672788564829762], [-0.8004393863912989, 0.34331882273537073, 0.49135422524805905], [-0.35655929991756324, 0.38621463409625134, -0.8507077771198497]],
        [[-0.027042430490087588, -0.26806186871005216, -0.9630220877512955], [-0.9186827341466592, -0.37311087749269833, 0.12965456828607097], [-0.39406946206729565, 0.8882179392695824, -0.23617398549732127]],
        [[-0.7398857397400269, -0.6244835233350217, 0.25017877850135595], [0.1789997024569668, 0.17573034080911443, 0.9680278683175553], [-0.6484814559026885, 0.7610119423518859, -0.01823800833807132]],
        [[0.5261154236016417, -0.8382538394734069, 0.1432936204321634], [-0.606328211355567, -0.25159816808948565, 0.754363613868255], [-0.596295783284457, -0.48376529683525943, -0.6406266279316442]],
    ],
    [
        [[-0.4988380771087646, -0.8545129063151959, 0.14480423255964409], [0.7972873979941488, -0.3869263885892139, 0.46327181525863226], [-0.3398431665138346, 0.34654821129832125, 0.8743059872949426]],
        [[-0.079717469352109, -0.5639248403163001, -0.8219695247114279], [0.8291925470057125, -0.4951769642701962, 0.2593057925429584], [-0.5532493516221003, -0.6608998021866764, 0.5070765291347419]],
        [[-0.4888638117531481, -0.7684980115168198, 0.41282318231038884], [0.542705193825574, -0.6384206408602656, -0.5457931457230971], [0.6829957878002827, -0.04277723247548498, 0.7291686102876411]],
        [[0.5539505749606839, -0.7034112483092847, 0.4453665639141564], [0.7497591684387712, 0.6540437214040797, 0.10043903540907534], [-0.36193915212744077, 0.278279403197357, 0.8896969281240623]],
    ],
]

# mtex orientation * Miller(<111>_k): sample-frame unit normals [parent][variant]
_MTEX_HABIT = [
    [[0.39401967366584967, 0.21604280697709152, 0.8933498767659379], [-0.4230893167047152, -0.5763855764073932, 0.6991245220987848], [0.9852882765418655, -0.1696776419648779, -0.02040857482528173], [-0.16817928617130046, 0.9621060253493625, 0.21463392949243484]],
    [[-0.8802842234858667, 0.01976481373556336, -0.47403484895087966], [-0.6951536991349946, 0.3898421582030758, 0.6039738622378468], [0.07781328440678043, 0.5607787171836659, -0.8243011119267469], [-0.26294380875765266, -0.9308560616511782, -0.2537075992619791]],
    [[-0.6977547919397606, 0.5043921464924035, 0.5086519565300884], [-0.7541215823275885, -0.6149148247145729, 0.23060875398384215], [0.39979045594280227, 0.3668094989976082, 0.8400109420609769], [-0.3434236655549745, 0.7524974722093682, -0.5619677395147306]],
]

# trace angle from the mtex normals with the library's selection rule (largest XY-projected normal)
_MTEX_LAMELLA_DEG = [80.22886793165533, 164.2263267452969, 129.19398903771872]

_SIGMA9_DEG = 38.942441269       # mtex angle between any two of the four variants of one parent
_PARENTS = range(len(_EULER))
_VARIANTS = range(4)
_S = get_cubic_ops_np()


def _parent_R(p):
    return euler_bunge_to_matrix(*_EULER[p])                  # crystal -> sample


def _parent_q(p):
    return rotation_matrices_to_quats(_parent_R(p).T[None])[0]   # DefDAP-stored: quaternion of g


def _R_of_stored(q):
    return quat_to_R_batch(np.asarray(q)[None])[0].T          # stored quaternion is g; R = g.T


def _disorientation_deg(Ra, Rb):
    """Smallest angle between two crystal -> sample matrices over the 24 x 24 cubic equivalents."""
    delta = Ra.T @ Rb
    cand = np.einsum("iab,bc,jcd->ijad", _S, delta, _S)
    tr = np.einsum("ijaa->ij", cand)
    return float(np.degrees(np.arccos(np.clip((tr.max() - 1.0) / 2.0, -1.0, 1.0))))


def _generator():
    gen = TwinGenerator3D.__new__(TwinGenerator3D)
    gen.twin_orient_scatter_deg = 0.0
    gen._rng = np.random.default_rng(0)
    return gen


@pytest.mark.parametrize("variant", _VARIANTS)
@pytest.mark.parametrize("parent", _PARENTS)
def test_twin_orientation_equals_mtex(parent, variant):
    q_twin = _generator()._apply_sigma3_variant(_parent_q(parent), variant)
    got = _disorientation_deg(_R_of_stored(q_twin), np.array(_MTEX_TWIN_R[parent][variant]))
    assert got < 1e-5


def test_the_variant_is_applied_in_the_crystal_frame_not_the_sample_frame():
    # The same variants multiplied on the other side of the parent are 38-59 degrees from mtex.
    for parent in _PARENTS:
        for variant in _VARIANTS:
            q_other = _positive_w(_quat_mul(_parent_q(parent), _SIGMA3_Q_ALL_VARIANTS[variant]))
            got = _disorientation_deg(_R_of_stored(q_other), np.array(_MTEX_TWIN_R[parent][variant]))
            assert got > 1.0


@pytest.mark.parametrize("parent", _PARENTS)
def test_every_twin_is_sigma3_from_its_parent(parent):
    gen = _generator()
    for variant in _VARIANTS:
        q_twin = gen._apply_sigma3_variant(_parent_q(parent), variant)
        delta = _parent_R(parent).T @ _R_of_stored(q_twin)
        assert _disorientation_deg(np.eye(3), delta) == pytest.approx(60.0, abs=1e-5)


@pytest.mark.parametrize("parent", _PARENTS)
def test_variants_of_one_parent_are_sigma9_apart(parent):
    gen = _generator()
    R = [_R_of_stored(gen._apply_sigma3_variant(_parent_q(parent), v)) for v in _VARIANTS]
    for i in _VARIANTS:
        for j in _VARIANTS:
            expected = 0.0 if i == j else _SIGMA9_DEG
            assert _disorientation_deg(R[i], R[j]) == pytest.approx(expected, abs=1e-5)


def test_scatter_keeps_the_twin_within_the_requested_spread_of_the_exact_twin():
    gen = _generator()
    gen.twin_orient_scatter_deg = 1.0
    exact = _R_of_stored(_generator()._apply_sigma3_variant(_parent_q(0), 0))
    angles = [_disorientation_deg(exact, _R_of_stored(gen._apply_sigma3_variant(_parent_q(0), 0)))
              for _ in range(200)]
    assert 0.0 < max(angles) < 6.0 and np.mean(angles) < 2.0


@pytest.mark.parametrize("variant", _VARIANTS)
@pytest.mark.parametrize("parent", _PARENTS)
def test_habit_plane_equals_mtex_o_times_miller(parent, variant):
    n = compute_s3_habit_plane_3d(_parent_q(parent), np.random.default_rng(0), variant_idx=variant)
    assert np.allclose(n, _MTEX_HABIT[parent][variant], atol=1e-12)


@pytest.mark.parametrize("parent", _PARENTS)
def test_lamella_trace_angle_equals_mtex(parent):
    got = compute_s3_lamella_angle_2d(_parent_q(parent))
    diff = abs(got - _MTEX_LAMELLA_DEG[parent]) % 180.0
    assert min(diff, 180.0 - diff) < 1e-9


@pytest.mark.parametrize("variant", _VARIANTS)
@pytest.mark.parametrize("parent", _PARENTS)
def test_parent_and_twin_share_the_habit_plane(parent, variant):
    rng = np.random.default_rng(0)
    q_twin = _generator()._apply_sigma3_variant(_parent_q(parent), variant)
    n_parent = compute_s3_habit_plane_3d(_parent_q(parent), rng, variant_idx=variant)
    n_twin = compute_s3_habit_plane_3d(q_twin, rng, variant_idx=variant)
    assert np.allclose(n_parent, n_twin, atol=1e-12)


# ---------------------------------------------------------------------------
# Sigma5: 36.87 degrees about a cubic <100> axis, seeded by the 2D Monte Carlo twin builder
# ---------------------------------------------------------------------------

_MTEX_SIGMA5_DEG = 36.8698976         # mtex angle between a parent and its Sigma5 child, every parent


def _crystal_frame_relative(q_parent, q_child):
    """g_parent @ g_child.T of two stored quaternions: the rotation between the two crystals."""
    gp = quat_to_R_batch(np.asarray(q_parent)[None])[0]
    gc = quat_to_R_batch(np.asarray(q_child)[None])[0]
    return gp @ gc.T


def _axis(rel):
    a = np.array([rel[2, 1] - rel[1, 2], rel[0, 2] - rel[2, 0], rel[1, 0] - rel[0, 1]])
    return a / np.linalg.norm(a)


@pytest.mark.parametrize("variant", range(3))
@pytest.mark.parametrize("parent", _PARENTS)
def test_sigma5_child_is_36p87_degrees_about_a_cubic_axis_of_its_parent(parent, variant):
    q_child = _positive_w(_quat_mul(_SIGMA5_Q_VARIANTS[variant], _parent_q(parent)))
    rel = _crystal_frame_relative(_parent_q(parent), q_child)
    assert _disorientation_deg(_parent_R(parent), _R_of_stored(q_child)) == pytest.approx(
        _MTEX_SIGMA5_DEG, abs=1e-3)
    assert np.sort(np.abs(_axis(rel)))[-1] == pytest.approx(1.0, abs=1e-9)         # along <100>


@pytest.mark.parametrize("parent", _PARENTS)
def test_sigma5_composed_on_the_parent_side_is_not_about_a_cubic_axis(parent):
    # The angle alone cannot tell the two sides apart (a rotation under 45 degrees is its own
    # disorientation); the axis does.
    q_child = _positive_w(_quat_mul(_parent_q(parent), _SIGMA5_Q_VARIANTS[0]))
    rel = _crystal_frame_relative(_parent_q(parent), q_child)
    assert np.sort(np.abs(_axis(rel)))[-1] < 0.999
