"""
Pole figures (``upxo.viz.xphy.pole_figure.PoleFigure``) against mtex.

The sample-frame pole axes below were produced once with mtex 1.0.0 (``orientation * Miller``
over the symmetrised family, normalised, folded to the upper hemisphere, v and -v one axis) for the
Bunge Euler angles in ``_EULER`` on a cubic crystal frame. They are frozen here, so the tests need
neither mtex nor an EBSD file.

Conventions the tests pin down:
* Euler angles, a rotation matrix, and the quaternion of the crystal -> sample rotation give one R.
* A quaternion stored the way DefDAP stores it (the quaternion of the transpose) gives the
  transposed poles, so it must be conjugated first (``sample_frame_quats`` does that).
* Sample symmetry replicates every orientation as ``M @ R``.

Run from repo root:
    pytest tests/viz/test_pole_figure_vs_mtex.py -v
"""

import numpy as np
import pytest

from upxo.viz.xphy.pole_figure import (
    PoleFigure, apply_sample_symmetry, compute_sample_symmetry_group)
from upxo.xtalphy.crystal_orientation import rotation_matrices_to_quats

# ---------------------------------------------------------------------------
# Frozen mtex 1.0.0 reference values
# ---------------------------------------------------------------------------

_EULER = np.array([
    (120.0, 60.0, 200.0),
    (35.26, 45.0, 0.0),
    (10.0, 80.0, 330.0),
])

_MTEX_AXES_111 = [
    np.array([
        (-0.925974076, -0.136163064, 0.352181247),
        (-0.212432217, -0.977122973, 0.010161104),
        (-0.059948672, 0.363836936, 0.929531517),
        (0.653593187, -0.477122973, 0.587511373),
    ]),
    np.array([
        (-0.471430058, -0.333297216, 0.816496581),
        (-0.000076614, -0.999999997, 0.000000000),
        (0.471430058, 0.333297216, 0.816496581),
        (0.942783501, -0.333405565, 0.000000000),
    ]),
    np.array([
        (-0.330628542, 0.658116242, 0.676437565),
        (-0.133163120, -0.461765815, 0.876949209),
        (0.671588452, 0.733031709, 0.107858544),
        (0.869053874, -0.386850348, 0.308370188),
    ]),
]

_MTEX_AXES_110 = [
    np.array([
        (-0.697128634, -0.681745682, 0.221888468),
        (-0.603751914, 0.139421204, 0.784885567),
        (0.093376720, 0.821166886, 0.562997099),
        (0.270154818, -0.890540133, 0.365998151),
        (0.363531538, -0.069373246, 0.928995250),
        (0.967283452, -0.208794450, 0.144109682),
    ]),
    np.array([
        (-0.866025401, 0.000066350, 0.500000000),
        (-0.577287712, 0.816540812, 0.000000000),
        (-0.288737689, -0.816474462, 0.500000000),
        (-0.000000000, -0.000000000, 1.000000000),
        (0.288737689, 0.816474462, 0.500000000),
        (0.866025401, -0.000066350, 0.500000000),
    ]),
    np.array([
        (-0.734652443, 0.639908736, 0.225394316),
        (-0.492807680, -0.731661070, 0.470969924),
        (-0.284013230, 0.120239589, 0.951251243),
        (0.208794450, 0.851900659, 0.480281318),
        (0.450639213, -0.519669147, 0.725856926),
        (0.943446893, 0.211991923, 0.254887002),
    ]),
]

_MTEX_AXES_123 = [
    np.array([
        (-0.959394335, 0.210530031, 0.187722179),
        (-0.892578494, -0.436434411, 0.113263576),
        (-0.821992329, 0.184309408, 0.538849379),
        (-0.755176488, -0.462655033, 0.464390776),
        (-0.719883405, -0.152283124, 0.677183678),
        (-0.424870790, -0.851943827, 0.306066214),
        (-0.421100466, 0.415764433, 0.806110621),
        (-0.318991542, 0.079171901, 0.944444920),
        (-0.157610609, 0.673440081, 0.722244663),
        (-0.059272010, -0.930860711, 0.360534554),
        (-0.031522758, 0.957336351, 0.287251505),
        (0.046607238, 0.000255017, 0.998913260),
        (0.168924764, -0.983556973, 0.063875695),
        (0.207988171, 0.594523196, 0.776713003),
        (0.310097095, 0.257930664, 0.915047302),
        (0.334076022, 0.878419466, 0.341719845),
        (0.341619853, -0.699405686, 0.627795796),
        (0.376912936, -0.389033777, 0.840588698),
        (0.562272796, 0.825723204, 0.045060986),
        (0.569816627, -0.752101948, 0.331136937),
        (0.640402792, -0.131358130, 0.756722740),
        (0.833306484, -0.494426301, 0.247270978),
        (0.868599567, -0.184054391, 0.460063880),
        (0.994687417, 0.099841879, 0.025070722),
    ]),
    np.array([
        (-0.981980503, 0.000075234, 0.188982237),
        (-0.872847915, 0.308673574, 0.377964473),
        (-0.763786257, -0.308548184, 0.566946710),
        (-0.763715326, 0.617271913, 0.188982237),
        (-0.654653669, 0.000050156, 0.755928946),
        (-0.545592011, -0.617171601, 0.566946710),
        (-0.327397766, -0.925795019, 0.188982237),
        (-0.327326834, 0.000025078, 0.944911183),
        (-0.327255903, 0.925845175, 0.188982237),
        (-0.218265177, -0.617196679, 0.755928946),
        (-0.109132589, -0.308598340, 0.944911183),
        (-0.000070931, -0.925820097, 0.377964473),
        (0.000070931, 0.925820097, 0.377964473),
        (0.109132589, 0.308598340, 0.944911183),
        (0.218265177, 0.617196679, 0.755928946),
        (0.327255903, -0.925845175, 0.188982237),
        (0.327326834, -0.000025078, 0.944911183),
        (0.327397766, 0.925795019, 0.188982237),
        (0.545592011, 0.617171601, 0.566946710),
        (0.654653669, -0.000050156, 0.755928946),
        (0.763715326, -0.617271913, 0.188982237),
        (0.763786257, 0.308548184, 0.566946710),
        (0.872847915, -0.308673574, 0.377964473),
        (0.981980503, -0.000075234, 0.188982237),
    ]),
    np.array([
        (-0.911934455, 0.403600121, 0.074043856),
        (-0.729116998, -0.633209197, 0.259681567),
        (-0.662691958, 0.529171829, 0.529921262),
        (-0.492366345, 0.332755354, 0.804269393),
        (-0.479874501, -0.507637488, 0.715558973),
        (-0.476428162, 0.805713720, 0.351911363),
        (-0.451444475, -0.875071965, 0.174490523),
        (-0.400957617, -0.185649305, 0.897088248),
        (-0.202201978, -0.749500256, 0.630367929),
        (-0.135776937, 0.412880770, 0.900607625),
        (-0.119838755, 0.885839135, 0.448249595),
        (-0.044368209, -0.105523889, 0.993426480),
        (0.050486858, 0.689422660, 0.722597725),
        (0.154387430, -0.669374841, 0.726706160),
        (0.233304315, -0.347386657, 0.908235436),
        (0.344097564, 0.920518258, 0.185048652),
        (0.514423177, 0.724101783, 0.459396782),
        (0.618323749, -0.634695718, 0.463505217),
        (0.697240633, -0.312707535, 0.645034493),
        (0.725670660, -0.680142011, 0.103966043),
        (0.792095700, 0.482239014, 0.374205738),
        (0.883504428, -0.036165644, 0.467024594),
        (0.899442611, 0.436792721, 0.014666564),
        (0.990851339, -0.081611937, 0.107485419),
    ]),
]



_MTEX = {"111": _MTEX_AXES_111, "110": _MTEX_AXES_110, "123": _MTEX_AXES_123}


def _folded_axes(pf, i, family):
    poles = pf._get_symmetric_poles(family)
    d = (pf.R_stack[i] @ poles.T).T
    return np.where(d[:, 2:3] < 0, -d, d)


@pytest.mark.parametrize("family", ["111", "110", "123"])
@pytest.mark.parametrize("i", [0, 1, 2])
def test_pole_axes_match_mtex(family, i):
    pf = PoleFigure(_EULER, convention="euler_deg")
    got = _folded_axes(pf, i, family)
    want = _MTEX[family][i]
    assert len(got) == len(want)
    # every UPXO axis is an mtex axis and vice versa
    assert (1.0 - np.abs(got @ want.T).max(axis=1)).max() < 1e-8
    assert (1.0 - np.abs(want @ got.T).max(axis=1)).max() < 1e-8


@pytest.mark.parametrize("family, n_axes", [("100", 3), ("110", 6), ("111", 4), ("123", 24), ("210", 12)])
def test_number_of_pole_axes_per_family(family, n_axes):
    pf = PoleFigure(_EULER, convention="euler_deg")
    assert len(pf._get_symmetric_poles(family)) == n_axes


def test_matrix_and_active_quaternion_input_give_the_same_rotation_as_euler_input():
    R = PoleFigure(_EULER, convention="euler_deg").R_stack
    assert np.allclose(PoleFigure(R, convention="matrix").R_stack, R, atol=1e-12)
    q_active = rotation_matrices_to_quats(R)
    assert np.allclose(PoleFigure(q_active, convention="quaternion").R_stack, R, atol=1e-9)


def test_a_defdap_stored_quaternion_must_be_conjugated_before_it_is_used():
    from upxo.pxtal.twinned_simple_3d.steps.steps_visualization_export import sample_frame_quats
    R = PoleFigure(_EULER, convention="euler_deg").R_stack
    q_stored = rotation_matrices_to_quats(np.swapaxes(R, -1, -2))
    direct = PoleFigure(q_stored, convention="quaternion").R_stack
    assert np.allclose(direct, np.swapaxes(R, -1, -2), atol=1e-9)          # the transposed poles
    assert not np.allclose(direct, R, atol=1e-3)
    conj, _ = sample_frame_quats({"gids": np.arange(3), "quats": q_stored}, apply_sample_symmetry=False)
    assert np.allclose(PoleFigure(conj, convention="quaternion").R_stack, R, atol=1e-9)


def test_sample_symmetry_replicates_each_orientation_as_m_times_r():
    group = compute_sample_symmetry_group(True, True, True, True)
    R = PoleFigure(_EULER, convention="euler_deg").R_stack
    rep = PoleFigure(apply_sample_symmetry(_EULER, "euler_deg", group), convention="euler_deg").R_stack
    n = len(_EULER)
    assert len(group) == 4 and len(rep) == 4 * n
    for k, M in enumerate(group):
        assert np.allclose(rep[k * n:(k + 1) * n], np.einsum("ij,njk->nik", M, R), atol=1e-9)


@pytest.mark.parametrize("projection, radius", [
    ("stereographic", lambda t: np.tan(t / 2.0)),
    ("equal_area", lambda t: np.sqrt(2.0) * np.sin(t / 2.0)),
])
def test_projection_radius_follows_the_closed_form(projection, radius):
    pf = PoleFigure(np.array([[0.0, 0.0, 0.0]]), convention="euler_deg")
    t = np.radians([0.0, 20.0, 45.0, 70.0, 90.0])
    vec = np.stack([np.sin(t) * np.cos(0.4), np.sin(t) * np.sin(0.4), np.cos(t)], axis=1)
    X, Y, _ = pf._calculate_projection(vec, projection=projection, hemisphere="both_mapped")
    assert np.allclose(np.hypot(X, Y), radius(t), atol=1e-12)
    assert np.allclose(np.arctan2(Y[1:], X[1:]), 0.4, atol=1e-12)


def test_a_pole_below_the_equator_is_folded_to_the_antipodal_point():
    pf = PoleFigure(np.array([[0.0, 0.0, 0.0]]), convention="euler_deg")
    X, Y, _ = pf._calculate_projection(np.array([[0.3, 0.2, -0.9327379]]), "stereographic", "both_mapped")
    X2, Y2, _ = pf._calculate_projection(np.array([[-0.3, -0.2, 0.9327379]]), "stereographic", "both_mapped")
    assert np.allclose([X, Y], [X2, Y2])


def test_density_is_normalised_and_flat_for_random_orientations():
    from scipy.spatial.transform import Rotation as Rot
    e = Rot.random(3000, random_state=5).as_euler("ZXZ", degrees=True)
    pf = PoleFigure(e, convention="euler_deg")
    _, _, z, mask = pf._compute_mud_grid(pf._get_symmetric_poles("100"), grid_points=101, half_width_deg=7.5)
    v = z[mask]
    assert abs(v.mean() - 1.0) < 0.02
    assert 0.7 < v.min() and v.max() < 1.3


def test_density_of_one_orientation_peaks_on_its_pole_and_has_unit_mean():
    pf = PoleFigure(np.array([[120.0, 60.0, 200.0]]), convention="euler_deg")
    poles = pf._get_symmetric_poles("100")
    Xi, Yi, z, mask = pf._compute_mud_grid(poles, grid_points=161, half_width_deg=7.5)
    assert abs(np.nanmean(z[mask]) - 1.0) < 0.03
    X, Y, _ = pf._calculate_projection(poles, projection="equal_area")
    k = np.nanargmax(z)
    cell = Xi[0, 1] - Xi[0, 0]
    assert np.hypot(X - Xi.ravel()[k], Y - Yi.ravel()[k]).min() <= 3 * cell
