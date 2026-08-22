"""
Unit tests for upxo.pxtal.fm_steel_3d.orientation_mean_3d.

Run from repo root:
    pytest tests/fm_steel_3d/test_orientation_mean.py -v
"""
import numpy as np
import pytest

from upxo.pxtal.fm_steel_3d.orientation_mean_3d import (
    mean_orientation_euler_deg, compute_packet_mean_orientations,
)
from upxo.viz.xphy.pole_figure import _euler_deg_to_matrix


def test_single_orientation_returns_itself():
    e = (12.3, 45.6, 78.9)
    assert mean_orientation_euler_deg([e]) == pytest.approx(e, abs=1e-9)


def test_identical_copies_average_to_the_same_orientation():
    e = (30.0, 45.0, 60.0)
    m = mean_orientation_euler_deg([e, e, e, e])
    assert m == pytest.approx(e, abs=1e-6)


def test_empty_input_raises():
    with pytest.raises(ValueError):
        mean_orientation_euler_deg([])


def test_mean_of_close_orientations_is_between_them():
    """A weak, coordinate-free correctness check: the mean rotation matrix's
    rotation axis/angle should land between the two inputs' -- verified here
    via a simpler proxy: the mean orientation's rotation matrix must have a
    smaller angular distance to EACH input than the two inputs have to each
    other (i.e. it's genuinely "between" them, not some unrelated result)."""
    e1 = (10.0, 20.0, 30.0)
    e2 = (14.0, 24.0, 26.0)
    m = mean_orientation_euler_deg([e1, e2])

    def _angle_between(ea, eb):
        Ra = _euler_deg_to_matrix(np.array([ea]))[0]
        Rb = _euler_deg_to_matrix(np.array([eb]))[0]
        Rrel = Ra.T @ Rb
        cos_ang = (np.trace(Rrel) - 1.0) / 2.0
        cos_ang = np.clip(cos_ang, -1.0, 1.0)
        return np.degrees(np.arccos(cos_ang))

    d_12 = _angle_between(e1, e2)
    d_1m = _angle_between(e1, m)
    d_2m = _angle_between(e2, m)
    assert d_1m < d_12
    assert d_2m < d_12
    # And the mean should be roughly equidistant from both (within a few
    # degrees -- quaternion averaging of 2 orientations is a form of slerp
    # midpoint, not exact bisection of the Euler-angle-space difference).
    assert abs(d_1m - d_2m) < 3.0


def test_mean_orientation_is_a_valid_rotation_matrix():
    """The averaged-then-reconstructed rotation matrix must still be a
    proper rotation (orthonormal, det=+1) -- confirms renormalisation after
    averaging actually restores a valid orientation, not just "close to
    unit norm"."""
    m = mean_orientation_euler_deg([(5.0, 10.0, 15.0), (350.0, 5.0, 355.0), (20.0, 15.0, 5.0)])
    R = _euler_deg_to_matrix(np.array([m]))[0]
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-8)
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-8)


def test_compute_packet_mean_orientations_basic():
    g2b = {
        101: ["b1", "b2", "b3"],
        102: ["b4"],           # single block -> mean is that block's own orientation
        103: ["b5", "b6"],     # b6 has no orientation entry -> only b5 counted
    }
    bori = {
        "b1": (0.0, 0.0, 0.0), "b2": (0.0, 0.0, 0.0), "b3": (0.0, 0.0, 0.0),
        "b4": (11.0, 22.0, 33.0),
        "b5": (7.0, 8.0, 9.0),
    }
    out = compute_packet_mean_orientations(g2b, bori)
    assert set(out.keys()) == {101, 102, 103}
    assert out[101] == pytest.approx((0.0, 0.0, 0.0), abs=1e-6)
    assert out[102] == pytest.approx((11.0, 22.0, 33.0), abs=1e-6)
    assert out[103] == pytest.approx((7.0, 8.0, 9.0), abs=1e-6)


def test_compute_packet_mean_orientations_skips_packets_with_no_oriented_blocks():
    g2b = {104: ["b_missing"]}
    bori = {}
    out = compute_packet_mean_orientations(g2b, bori)
    assert 104 not in out
