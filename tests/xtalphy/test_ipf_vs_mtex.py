"""
Inverse-pole-figure colours (``upxo.xtalphy.ipf`` and the functions that use it) against mtex.

The colours below were produced once with mtex 1.0.0 (``ipfColorKey`` / ``ipfHSVKey``, mtex's default
key) on a cubic m-3m crystal frame and are frozen here, so the tests need neither mtex nor an EBSD file.
They cover the vertices of the fundamental sector, its centre, points on its edges, interior points
and directions outside the sector (which must take the colour of their equivalent in it).

Run from repo root:
    pytest tests/xtalphy/test_ipf_vs_mtex.py -v
"""

import numpy as np
import pytest

from upxo.viz.ebsdviz import build_ipf_rgb
from upxo.viz.xphy.pole_figure import PoleFigure
from upxo.xtalphy.crystal_orientation import (
    euler_bunge_to_matrix_batch, get_cubic_ops_np, ipf_color, rotation_matrices_to_quats)
from upxo.xtalphy.ipf import (
    CUBIC_SECTOR_VERTICES, cubic_ipf_rgb, ipf_crystal_direction, reduce_to_cubic_sector, sector_mask)

# ---------------------------------------------------------------------------
# Frozen mtex 1.0.0 reference values
# ---------------------------------------------------------------------------

_DIRECTIONS = np.array([
    (0.000000000, 0.000000000, 1.000000000),
    (0.707106781, 0.000000000, 0.707106781),
    (0.577350269, 0.577350269, 0.577350269),
    (0.478625491, 0.215137249, 0.851254136),
    (0.500000002, 0.000000000, 0.866025403),
    (0.299625702, 0.099875234, 0.948814722),
    (0.600570814, 0.450428110, 0.660627895),
    (0.201680957, 0.050420239, 0.978152642),
    (0.538672318, 0.489702107, 0.685582950),
    (0.700003357, 0.100000480, 0.707103391),
    (-0.399500936, 0.799001871, -0.449438552),
    (0.904534034, -0.301511345, 0.301511345),
    (0.100503782, -0.703526471, 0.703526471),
    (0.300010463, 0.300010463, 0.905531581),
])

_MTEX_DIRECTION_COLOURS = np.array([
    (1.000000000, 0.010266727, 0.000000000),
    (0.000000000, 1.000000000, 0.006844484),
    (0.003422242, 0.000000000, 1.000000000),
    (0.900000000, 0.900000000, 0.900000000),
    (0.846363263, 1.000000000, 0.000000000),
    (0.944701990, 0.695134121, 0.547120713),
    (0.212872724, 0.522648654, 0.979914424),
    (0.971206808, 0.507222066, 0.300144155),
    (0.381403096, 0.243290333, 0.976913643),
    (0.026434005, 0.997586316, 0.500401596),
    (0.747046211, 0.251730332, 0.976074657),
    (1.000000000, 0.000000000, 0.887475986),
    (0.000000000, 1.000000000, 0.499937299),
    (1.000000000, 0.000000000, 0.883401936),
])

_EULER = np.array([
    (120.000000, 60.000000, 200.000000),
    (35.260000, 45.000000, 0.000000),
    (10.000000, 80.000000, 330.000000),
    (290.000000, 25.000000, 40.000000),
    (75.000000, 110.000000, 15.000000),
    (200.000000, 150.000000, 250.000000),
])

_MTEX_COLOURS_X = np.array([
    (0.442557555, 0.956294449, 0.917495022),
    (0.794972450, 0.000000000, 1.000000000),
    (0.913032480, 0.908033310, 0.805375730),
    (0.776266961, 0.181897809, 0.982934312),
    (0.960634669, 0.402026054, 0.846369224),
    (0.575824873, 0.941411082, 0.795449294),
])

_MTEX_COLOURS_Y = np.array([
    (0.546568959, 0.456487935, 0.954783819),
    (0.003124850, 0.000000000, 1.000000000),
    (0.963172566, 0.576739835, 0.377976579),
    (0.835993566, 0.923244935, 0.726478913),
    (0.956691177, 0.438883972, 0.835352867),
    (0.171581617, 0.455260585, 0.983932268),
])

_MTEX_COLOURS_Z = np.array([
    (0.816551617, 0.789765532, 0.915096616),
    (0.000000000, 1.000000000, 0.006844484),
    (0.886521396, 0.912094441, 0.812413403),
    (0.975021526, 0.262284729, 0.861810611),
    (0.953827308, 0.465261266, 0.841336230),
    (0.913020969, 0.911884569, 0.805462307),
])


_AXES = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}
_MTEX_COLOURS = {"x": _MTEX_COLOURS_X, "y": _MTEX_COLOURS_Y, "z": _MTEX_COLOURS_Z}


def _stored_quats(euler):
    R = euler_bunge_to_matrix_batch(euler[:, 0], euler[:, 1], euler[:, 2])
    return rotation_matrices_to_quats(np.swapaxes(R, -1, -2)), R


def test_key_colours_match_mtex():
    assert np.allclose(cubic_ipf_rgb(_DIRECTIONS), _MTEX_DIRECTION_COLOURS, atol=1e-9)


def test_the_three_corners_are_red_green_blue_and_the_centre_is_pale():
    rgb = cubic_ipf_rgb(CUBIC_SECTOR_VERTICES)
    assert np.argmax(rgb[0]) == 0 and rgb[0, 0] > 0.9 and rgb[0, 1] < 0.1 and rgb[0, 2] < 0.1   # [001] red
    assert np.argmax(rgb[1]) == 1 and rgb[1, 1] > 0.9 and rgb[1, 0] < 0.1 and rgb[1, 2] < 0.1   # [101] green
    assert np.argmax(rgb[2]) == 2 and rgb[2, 2] > 0.9 and rgb[2, 0] < 0.1 and rgb[2, 1] < 0.1   # [111] blue
    centre = CUBIC_SECTOR_VERTICES.sum(axis=0)
    assert cubic_ipf_rgb(centre / np.linalg.norm(centre)).min() > 0.75                          # near white


def test_colour_is_the_same_for_every_crystal_equivalent_and_for_the_antipode():
    rng = np.random.default_rng(4)
    d = rng.normal(size=(300, 3))
    S = get_cubic_ops_np()
    base = cubic_ipf_rgb(d)
    for k in (0, 5, 11, 17, 23):
        assert np.allclose(cubic_ipf_rgb(d @ S[k].T), base, atol=1e-9)
    assert np.allclose(cubic_ipf_rgb(-d), base, atol=1e-9)


def test_reduction_lands_in_the_fundamental_sector():
    rng = np.random.default_rng(5)
    d = rng.normal(size=(500, 3))
    assert sector_mask(reduce_to_cubic_sector(d)).all()
    assert np.allclose(np.linalg.norm(reduce_to_cubic_sector(d), axis=1), 1.0)


@pytest.mark.parametrize("axis", ["x", "y", "z"])
def test_euler_path_matches_mtex(axis):
    got = np.array([ipf_color(e, _AXES[axis]) for e in _EULER])
    assert np.allclose(got, _MTEX_COLOURS[axis], atol=1e-9)


@pytest.mark.parametrize("axis", ["x", "y", "z"])
def test_defdap_stored_quaternions_give_the_mtex_colours(axis):
    q, _ = _stored_quats(_EULER)
    lfi = (np.arange(len(_EULER)) + 1).reshape(1, -1)
    rgb = build_ipf_rgb(lfi, {i + 1: q[i] for i in range(len(q))}, sample_direction=_AXES[axis])[0]
    assert np.allclose(rgb, _MTEX_COLOURS[axis], atol=1e-6)               # float32 output


@pytest.mark.parametrize("axis", ["x", "y", "z"])
def test_crystal_to_sample_quaternions_give_the_mtex_colours(axis):
    _, R = _stored_quats(_EULER)
    q = rotation_matrices_to_quats(R)
    lfi = (np.arange(len(_EULER)) + 1).reshape(1, -1)
    rgb = build_ipf_rgb(lfi, {i + 1: q[i] for i in range(len(q))}, sample_direction=_AXES[axis],
                        quat_convention="crystal_to_sample")[0]
    assert np.allclose(rgb, _MTEX_COLOURS[axis], atol=1e-6)


def test_a_stored_quaternion_read_as_crystal_to_sample_gives_the_wrong_colours():
    q, _ = _stored_quats(_EULER)
    lfi = (np.arange(len(_EULER)) + 1).reshape(1, -1)
    wrong = build_ipf_rgb(lfi, {i + 1: q[i] for i in range(len(q))}, sample_direction=_AXES["z"],
                          quat_convention="crystal_to_sample")[0]
    assert not np.allclose(wrong, _MTEX_COLOURS["z"], atol=1e-2)


def test_equivalent_orientations_have_one_ipf_colour():
    _, R = _stored_quats(_EULER)
    S = get_cubic_ops_np()
    for k in (3, 9, 20):
        d0 = ipf_crystal_direction(R, _AXES["z"], convention="matrix")
        d1 = ipf_crystal_direction(R @ S[k], _AXES["z"], convention="matrix")
        assert np.allclose(cubic_ipf_rgb(d0), cubic_ipf_rgb(d1), atol=1e-9)


def test_pole_figure_ipf_colours_match_mtex():
    pf = PoleFigure(_EULER, convention="euler_deg")
    for axis in ("x", "y", "z"):
        assert np.allclose(pf._calculate_ipf_colors(_AXES[axis]), _MTEX_COLOURS[axis], atol=1e-9)


def test_unlabelled_pixels_are_grey_and_missing_labels_do_not_break_the_map():
    q, _ = _stored_quats(_EULER)
    lfi = np.array([[0, 1, 2], [-1, 3, 99]])
    rgb = build_ipf_rgb(lfi, {1: q[0], 2: q[1], 3: q[2]}, grey=(0.5, 0.5, 0.5))
    assert rgb.shape == (2, 3, 3)
    for pos in ((0, 0), (1, 0), (1, 2)):
        assert np.allclose(rgb[pos], 0.5)
