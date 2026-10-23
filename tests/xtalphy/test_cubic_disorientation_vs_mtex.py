"""
Cubic disorientation and symmetry-aware grain-mean orientation, checked
against mtex.

The reference numbers below were produced once with mtex 1.0.0
(``orientation.angle`` and ``orientation.mean`` on a cubic m-3m crystal
frame) and are frozen here, so these tests need neither mtex nor any EBSD
file. mtex is the standard: UPXO must reproduce these values.

* ``_PAIR_*``  pairs of Bunge Euler angles (degrees) and mtex's disorientation
  angle between them (degrees). Includes identical, cubic-equivalent and
  large-angle pairs.
* ``_MEAN_CASES``  small grains of pixel orientations, each pixel a noisy copy
  of one orientation expressed as a randomly chosen cubic-symmetry variant,
  with mtex's mean orientation as Bunge Euler angles.

Run from repo root:
    pytest tests/xtalphy/test_cubic_disorientation_vs_mtex.py -v
"""

import numpy as np
import pytest

from upxo.xtalphy.crystal_orientation import (
    cubic_misorientation,
    cubic_misorientation_scalar,
    cubic_symmetry_quats,
    euler_bunge_to_matrix,
    get_cubic_ops_np,
    grain_avg_quats,
    matrix_to_euler_bunge,
    quat_to_R_batch,
    rotation_matrices_to_quats,
)

# ---------------------------------------------------------------------------
# Frozen mtex 1.0.0 reference values
# ---------------------------------------------------------------------------

_PAIR_A = np.array([
    (0.000000, 0.000000, 0.000000),
    (0.000000, 0.000000, 0.000000),
    (30.000000, 0.000000, 0.000000),
    (35.260000, 45.000000, 0.000000),
    (200.000000, 70.000000, 15.000000),
    (90.000000, 0.000000, 0.000000),
    (113.543127, 129.121297, 184.012253),
    (214.763008, 131.861107, 155.439338),
    (342.034205, 137.710756, 162.129589),
    (113.898203, 54.080668, 345.075921),
    (339.439627, 121.724015, 172.969159),
    (94.260946, 130.117832, 179.732720),
    (61.325548, 89.197529, 280.275723),
    (307.258863, 60.677655, 352.894716),
    (338.302763, 61.587671, 120.051994),
    (65.081354, 95.208100, 47.670061),
    (271.476617, 84.929717, 294.824280),
    (94.298811, 135.118138, 242.983236),
    (334.037723, 92.439427, 156.314086),
    (341.611416, 72.899039, 128.199695),
    (189.569976, 153.555055, 355.568064),
    (300.307288, 115.117020, 47.704997),
    (332.834128, 112.864322, 123.350782),
    (132.286267, 106.270587, 95.647405),
    (71.274815, 156.037550, 348.523778),
    (288.604015, 122.905520, 306.245966),
    (77.248257, 73.938220, 253.003378),
    (288.902067, 48.080246, 309.102159),
    (331.888833, 37.665741, 183.583360),
    (157.382742, 50.142707, 44.828342),
    (350.029985, 129.877471, 9.025473),
    (312.745666, 134.253479, 356.250107),
    (166.058276, 127.019521, 182.832683),
    (356.220363, 100.959710, 329.088641),
    (15.195104, 75.806329, 341.495441),
    (58.517389, 54.952829, 101.976913),
    (250.031458, 46.481486, 123.370659),
    (132.155383, 77.664196, 133.371008),
    (13.198049, 37.382865, 231.092044),
    (81.664750, 122.304755, 140.813309),
    (82.164274, 117.318845, 86.051147),
    (293.268542, 111.932526, 125.636073),
    (168.513475, 151.297987, 74.602169),
    (68.965378, 43.514809, 75.948562),
    (49.185218, 19.718609, 287.002722),
    (46.783300, 58.278597, 292.141107),
])

_PAIR_B = np.array([
    (0.000000, 0.000000, 0.000000),
    (45.000000, 0.000000, 0.000000),
    (30.000000, 20.000000, 0.000000),
    (35.260000, 45.000000, 3.000000),
    (110.000000, 70.000000, 105.000000),
    (0.000000, 0.000000, 0.000000),
    (89.783627, 113.626618, 310.095818),
    (288.679931, 68.639043, 245.135183),
    (308.171508, 65.097003, 127.341321),
    (150.708735, 130.038519, 208.444876),
    (262.790505, 61.771433, 343.774286),
    (85.928250, 127.129638, 228.266279),
    (4.617477, 121.671438, 257.358763),
    (332.649550, 25.815145, 86.436055),
    (200.329275, 150.480085, 33.858432),
    (262.099930, 152.619652, 230.452935),
    (94.961071, 88.676816, 119.157886),
    (316.922577, 109.164871, 130.885133),
    (91.518925, 99.156907, 148.054247),
    (207.388502, 131.248749, 245.358787),
    (231.073697, 149.597271, 342.952225),
    (86.712367, 81.969700, 131.813932),
    (182.156293, 141.312128, 317.833096),
    (10.604209, 17.859369, 320.930534),
    (58.938428, 35.762735, 354.993303),
    (162.225525, 10.979081, 1.080388),
    (119.310630, 51.642782, 164.469858),
    (357.003108, 111.254336, 186.409071),
    (39.912752, 129.829728, 127.530696),
    (175.782328, 111.039479, 9.102188),
    (40.744982, 115.394501, 65.479328),
    (148.309269, 79.322527, 232.941594),
    (37.520059, 120.258938, 110.869482),
    (209.020008, 86.665398, 315.177659),
    (27.242589, 122.724614, 46.022723),
    (175.474254, 64.301296, 147.971760),
    (233.549662, 152.703606, 169.122587),
    (324.643886, 62.577307, 7.680170),
    (342.873750, 122.025735, 209.052193),
    (231.113181, 30.832173, 284.030161),
    (278.250302, 74.501328, 117.406601),
    (84.326404, 158.001660, 41.298945),
    (109.243571, 45.089393, 63.684169),
    (359.681559, 123.040527, 119.460920),
    (104.797280, 64.085782, 156.024496),
    (149.345570, 42.998111, 258.421094),
])

_PAIR_ANGLE_DEG = np.array([
    0.00000000, 45.00000000, 20.00000000, 3.00000000,
    34.95434316, 0.00000000, 48.18959409, 44.68516973,
    43.61358933, 42.92266812, 30.31622105, 36.93448944,
    45.15892721, 44.55417804, 53.20897491, 46.72230654,
    36.83399001, 24.45239149, 50.92824656, 44.85952364,
    44.89145745, 37.70446768, 42.98394762, 35.25067219,
    39.36067931, 49.00660136, 47.98486451, 42.33391315,
    48.49781553, 38.44466769, 51.92553027, 48.76228667,
    48.47666515, 38.03420576, 55.22382656, 26.61389662,
    44.67342617, 55.22285701, 44.94660681, 53.44711066,
    34.90952670, 33.75721071, 58.97724803, 43.06377789,
    54.56148645, 35.96064701,
])

_MEAN_CASES = [
    (
        np.array([
            (214.618101, 45.471070, 359.325697),
            (35.591320, 135.178220, 0.450132),
            (35.214880, 44.651269, 89.384127),
            (305.088222, 90.149922, 134.816618),
            (215.381217, 134.738143, 90.088243),
            (215.454103, 135.050886, 90.041737),
            (36.507947, 43.282812, 89.988341),
            (35.539046, 134.885655, 89.574569),
            (305.194220, 90.080706, 135.032260),
            (305.320459, 90.051944, 224.942551),
            (215.316961, 134.890005, 0.622315),
            (125.286255, 90.480583, 224.859422),
            (35.244380, 134.989376, 359.976008),
            (215.077345, 134.265413, 90.510566),
        ]),
        (215.141851, 45.097385, 0.104910),
    ),
    (
        np.array([
            (296.687476, 117.370073, 65.495750),
            (336.109024, 36.077778, 59.440700),
            (40.901275, 107.087584, 122.052387),
            (39.665475, 106.525290, 120.607872),
            (40.697726, 107.305116, 121.719257),
            (44.403018, 107.568365, 214.047705),
            (121.493317, 60.584917, 288.361105),
            (40.086279, 107.471045, 211.964626),
            (300.514220, 120.270007, 249.593944),
            (40.262761, 107.894255, 210.643056),
            (153.342807, 147.855748, 298.921649),
        ]),
        (300.178165, 119.712737, 69.858416),
    ),
    (
        np.array([
            (10.848592, 79.875390, 149.743737),
            (191.056616, 99.240118, 210.029362),
            (117.521743, 31.574764, 159.879425),
            (296.093347, 148.492808, 18.760101),
            (93.752558, 119.355914, 170.504152),
            (301.715805, 149.718002, 294.393715),
            (295.382733, 147.448678, 17.971050),
            (115.265512, 31.126878, 162.384362),
            (273.523907, 61.397878, 189.630313),
            (116.683357, 31.497509, 160.609719),
            (94.594747, 119.415855, 348.639275),
            (189.440194, 100.500140, 119.442186),
            (98.027860, 121.107857, 171.339047),
            (297.456753, 148.340667, 108.652113),
            (93.143532, 119.718405, 348.274815),
            (274.474912, 60.331028, 10.528834),
            (118.773211, 32.155144, 159.774492),
        ]),
        (10.175289, 80.346380, 149.989390),
    ),
]



# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stored_quat(euler_deg):
    """Quaternion (w, x, y, z) as DefDAP stores it for Bunge Euler angles in degrees."""
    R = euler_bunge_to_matrix(*euler_deg, degrees=True)
    return rotation_matrices_to_quats(R.T[None])[0]


def _quat_disorientation_deg(q1, q2):
    """Cubic disorientation (deg) between two stored quaternions, independent of the Euler route."""
    S = get_cubic_ops_np()
    R1, R2 = quat_to_R_batch(np.array([q1, q2]))
    best = 180.0
    for Si in S:
        for Sj in S:
            M = (Si @ R1) @ (Sj @ R2).T
            best = min(best, np.degrees(np.arccos(np.clip((np.trace(M) - 1) / 2, -1, 1))))
    return best


# ---------------------------------------------------------------------------
# cubic_misorientation / cubic_misorientation_scalar against mtex
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fn", [cubic_misorientation, cubic_misorientation_scalar])
def test_disorientation_angle_matches_mtex(fn):
    got = np.array([fn(a, b)[0] for a, b in zip(_PAIR_A, _PAIR_B)])
    assert np.allclose(got, _PAIR_ANGLE_DEG, atol=1e-4), \
        f"max deviation from mtex {np.abs(got - _PAIR_ANGLE_DEG).max():.4f} deg"


def test_disorientation_is_symmetric_in_its_arguments():
    for a, b in list(zip(_PAIR_A, _PAIR_B))[:15]:
        assert cubic_misorientation(a, b)[0] == pytest.approx(cubic_misorientation(b, a)[0], abs=1e-9)


def test_disorientation_ignores_which_cubic_variant_is_given():
    S = get_cubic_ops_np()
    rng = np.random.default_rng(5)
    for a, b in list(zip(_PAIR_A, _PAIR_B))[6:16]:
        Rb = euler_bunge_to_matrix(*b)
        variant = matrix_to_euler_bunge(Rb @ S[rng.integers(24)])   # symmetry acts on the crystal side
        assert cubic_misorientation(a, variant)[0] == pytest.approx(
            cubic_misorientation(a, b)[0], abs=1e-6)


def test_matrix_input_is_the_euler_matrix_of_the_module():
    a, b = _PAIR_A[10], _PAIR_B[10]
    from_euler = cubic_misorientation(a, b)[0]
    from_matrix = cubic_misorientation(euler_bunge_to_matrix(*a), euler_bunge_to_matrix(*b))[0]
    assert from_matrix == pytest.approx(from_euler, abs=1e-9)


def test_axis_is_the_misorientation_axis_in_the_crystal_frame():
    # B is A turned by 20 deg about the crystal x axis; the sample-frame axis
    # of that turn is R_A x = (cos 30, sin 30, 0), which is not what is returned.
    A = (30.0, 0.0, 0.0)
    th = np.radians(20.0)
    Rx = np.array([[1, 0, 0], [0, np.cos(th), -np.sin(th)], [0, np.sin(th), np.cos(th)]])
    B = matrix_to_euler_bunge(euler_bunge_to_matrix(*A) @ Rx)
    angle, axis, top3 = cubic_misorientation(A, B)
    assert angle == pytest.approx(20.0, abs=1e-6)
    assert abs(axis @ np.array([1.0, 0.0, 0.0])) == pytest.approx(1.0, abs=1e-6)
    assert top3[0] == pytest.approx(20.0, abs=1e-6)


def test_identical_and_cubic_equivalent_orientations_have_zero_disorientation():
    assert cubic_misorientation((0, 0, 0), (0, 0, 0))[0] == pytest.approx(0.0, abs=1e-6)
    assert cubic_misorientation((0, 0, 0), (90, 0, 0))[0] == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Quaternion helpers
# ---------------------------------------------------------------------------

def test_rotation_matrices_round_trip_through_quaternions():
    rng = np.random.default_rng(1)
    E = np.column_stack([rng.uniform(0, 360, 50), rng.uniform(0, 180, 50), rng.uniform(0, 360, 50)])
    R = np.array([euler_bunge_to_matrix(*e) for e in E])
    q = rotation_matrices_to_quats(R)
    assert np.allclose(quat_to_R_batch(q), R, atol=1e-12)
    assert (q[:, 0] >= 0).all()
    assert np.allclose(np.linalg.norm(q, axis=1), 1.0)


def test_cubic_symmetry_quats_are_the_24_operators():
    sq = cubic_symmetry_quats()
    assert sq.shape == (24, 4)
    R = quat_to_R_batch(sq)
    S = get_cubic_ops_np()
    for r in R:
        assert any(np.allclose(r, s) for s in S)
    assert len({tuple(np.round(r, 6).ravel()) for r in R}) == 24


# ---------------------------------------------------------------------------
# grain_avg_quats against mtex's mean orientation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pixels, mtex_mean", _MEAN_CASES)
def test_grain_mean_matches_mtex_mean_orientation(pixels, mtex_mean):
    quat = np.array([_stored_quat(e) for e in pixels]).reshape(1, -1, 4)
    lfi = np.ones((1, len(pixels)), dtype=int)
    gids, q_mean = grain_avg_quats(lfi, quat)
    assert list(gids) == [1]
    dev = _quat_disorientation_deg(q_mean[0], _stored_quat(mtex_mean))
    assert dev < 0.005, f"UPXO mean is {dev:.4f} deg from the mtex mean"


def test_pixels_that_are_symmetry_variants_of_one_orientation_average_to_it():
    S = get_cubic_ops_np()
    base = (35.0, 50.0, 20.0)
    Rb = euler_bunge_to_matrix(*base)
    pix = [matrix_to_euler_bunge(Rb @ S[k]) for k in (0, 5, 9, 14, 20)]
    quat = np.array([_stored_quat(e) for e in pix]).reshape(1, -1, 4)
    _, q_mean = grain_avg_quats(np.ones((1, 5), dtype=int), quat)
    assert _quat_disorientation_deg(q_mean[0], _stored_quat(base)) < 1e-4


def test_opposite_signs_of_one_quaternion_do_not_cancel():
    q = _stored_quat((20.0, 40.0, 60.0))
    quat = np.array([q, -q, q, -q]).reshape(1, 4, 4)
    _, q_mean = grain_avg_quats(np.ones((1, 4), dtype=int), quat)
    assert _quat_disorientation_deg(q_mean[0], q) < 1e-9
    assert q_mean[0, 0] >= 0


def test_each_grain_is_averaged_on_its_own_and_ids_are_sorted():
    a, b = _stored_quat((10.0, 30.0, 50.0)), _stored_quat((200.0, 80.0, 100.0))
    lfi = np.array([[7, 7, 3, 3, 0, -1]])
    quat = np.array([[a, a, b, b, a, b]])
    gids, q_mean = grain_avg_quats(lfi, quat)
    assert list(gids) == [3, 7]
    assert _quat_disorientation_deg(q_mean[0], b) < 1e-9
    assert _quat_disorientation_deg(q_mean[1], a) < 1e-9
    assert np.allclose(np.linalg.norm(q_mean, axis=1), 1.0)


def test_symmetry_can_be_switched_off_for_the_plain_quaternion_mean():
    S = get_cubic_ops_np()
    Rb = euler_bunge_to_matrix(35.0, 50.0, 20.0)
    pix = [matrix_to_euler_bunge(Rb @ S[k]) for k in (0, 7)]
    quat = np.array([_stored_quat(e) for e in pix]).reshape(1, 2, 4)
    lfi = np.ones((1, 2), dtype=int)
    _, plain = grain_avg_quats(lfi, quat, symmetry=False)
    _, aware = grain_avg_quats(lfi, quat, symmetry=True)
    base = _stored_quat((35.0, 50.0, 20.0))
    assert _quat_disorientation_deg(aware[0], base) < 1e-4
    assert _quat_disorientation_deg(plain[0], base) > 1.0


# ---------------------------------------------------------------------------
# Same convention, other consumers: fcc_symmetrise_ori and the texture
# component peaks. Both must agree with the disorientation that matches mtex.
# ---------------------------------------------------------------------------

from upxo.xtalphy.crystal_orientation import (  # noqa: E402
    FCC_TEXTURE_COMPONENTS,
    detect_texture_component_peaks,
    fcc_symmetrise_ori,
)

_BASES = [(35.26, 45.0, 0.0), (120.0, 60.0, 200.0), (10.0, 80.0, 330.0)]


@pytest.mark.parametrize("bea", _BASES)
def test_fcc_symmetrise_ori_returns_the_24_crystal_equivalents_of_the_input(bea):
    eq = fcc_symmetrise_ori(bea, dtype=np.float64)
    assert eq.shape == (24, 3)
    for e in eq:
        assert cubic_misorientation(bea, e)[0] < 1e-4
    mats = {tuple(np.round(euler_bunge_to_matrix(*e), 4).ravel()) for e in eq}
    assert len(mats) == 24


@pytest.mark.parametrize("bea", _BASES)
def test_texture_peak_reports_the_orientation_of_the_quaternions_it_was_given(bea):
    q = _stored_quat(bea)
    result = detect_texture_component_peaks(np.repeat(q[None], 6, axis=0), n_peaks=1,
                                            std_components={})
    c = result["components"][0]
    assert cubic_misorientation(bea, (c["phi1"], c["Phi"], c["phi2"]))[0] < 1e-4


@pytest.mark.parametrize("name", ["copper", "brass", "goss", "cube", "S" if "S" in FCC_TEXTURE_COMPONENTS else "s"])
def test_texture_peak_names_a_standard_component(name):
    q = _stored_quat(FCC_TEXTURE_COMPONENTS[name])
    result = detect_texture_component_peaks(np.repeat(q[None], 6, axis=0), n_peaks=1)
    assert result["components"][0]["name"] == name
