"""The plane-trace formulas of the wiki theory pages, checked by direct geometry.

What is checked
    * the trace of a plane on the section z = 0 is perpendicular to the plane normal;
    * the apparent width of a lamella equals the distance between its two traces, measured
      directly, and equals t0 / sqrt(1 - n_z^2);
    * the mean apparent width of isotropically oriented lamellae is (pi / 2) t0;
    * the plane of an observed trace is recovered from the four {111} candidates when the
      candidates are separated enough.

Run from the repository root:
    pytest tests/validation_mtex/test_plane_trace_formulas.py -v
"""
import numpy as np
import pytest

from . import _oracle as ref
from . import tolerances as tol

_EZ = np.array([0.0, 0.0, 1.0])


def _unit(v):
    return v / np.linalg.norm(v)


def _random_normals(n, seed):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _point_to_line_distance(p, a, d):
    """Distance from point p to the 3D line through a with direction d."""
    return np.linalg.norm(np.cross(p - a, d)) / np.linalg.norm(d)


@pytest.mark.parametrize("seed", range(10))
def test_trace_is_perpendicular_to_the_plane_normal_and_lies_in_the_section(seed):
    n = _random_normals(1, seed)[0]
    t = np.cross(_EZ, n)
    assert abs(t @ n) < tol.EXACT_ELEMENT
    assert abs(t @ _EZ) < tol.EXACT_ELEMENT


@pytest.mark.parametrize("seed", range(10))
def test_trace_angle_formula(seed):
    n = _random_normals(1, 50 + seed)[0]
    t = np.cross(_EZ, n)
    phi_formula = np.degrees(np.arctan2(n[0], -n[1])) % 180
    phi_direct = np.degrees(np.arctan2(t[1], t[0])) % 180
    assert abs((phi_formula - phi_direct + 90) % 180 - 90) < 1e-9


@pytest.mark.parametrize("seed", range(10))
def test_apparent_width_is_the_measured_distance_between_the_two_traces(seed):
    n = _random_normals(1, 100 + seed)[0]
    t0 = 0.7
    d = _unit(np.cross(_EZ, n))                                # direction of both traces
    # point of each trace in z = 0: along the in-section projection of n
    n_xy = np.array([n[0], n[1], 0.0])
    p0 = np.zeros(3)                                           # trace of the plane n.x = 0
    p1 = (t0 / (n_xy @ n_xy)) * n_xy                           # a point of the trace of n.x = t0
    assert abs(n @ p1 - t0) < tol.EXACT_ELEMENT and abs(p1[2]) < tol.EXACT_ELEMENT
    measured = _point_to_line_distance(p1, p0, d)
    assert measured == pytest.approx(t0 / np.sqrt(1 - n[2] ** 2), abs=1e-9)


def test_apparent_width_limits():
    t0 = 1.0
    perpendicular = _unit(np.array([1.0, 0.3, 0.0]))           # plane normal in the section plane
    assert t0 / np.sqrt(1 - perpendicular[2] ** 2) == pytest.approx(t0)
    near_parallel = _unit(np.array([0.01, 0.0, 1.0]))
    assert t0 / np.sqrt(1 - near_parallel[2] ** 2) > 90


def test_mean_apparent_width_of_isotropic_lamellae_is_pi_over_two():
    n = _random_normals(2_000_000, 20261103)
    w = 1.0 / np.sqrt(1 - n[:, 2] ** 2)
    assert w.mean() == pytest.approx(np.pi / 2, rel=tol.MONTE_CARLO_REL)


def _candidate_traces(R):
    n_s = (R @ ref.FOUR_111.T).T
    return np.degrees(np.arctan2(n_s[:, 0], -n_s[:, 1])) % 180


def _sep(a):
    d = np.abs(a[:, None] - a[None])
    d = np.minimum(d, 180 - d) + np.eye(len(a)) * 1e9
    return d.min()


def test_the_plane_is_recovered_from_a_noisy_trace_when_candidates_are_separated():
    rng = np.random.default_rng(5)
    tested = 0
    for _ in range(400):
        e = rng.uniform([0, 0, 0], [360, 180, 360])
        phi = _candidate_traces(ref.bunge_R(*e)[0])
        if _sep(phi) < tol.TRACE_SEPARATION_DEG:
            continue
        k = int(rng.integers(4))
        obs = phi[k] + rng.normal(scale=tol.TRACE_NOISE_DEG / 3)
        delta = np.abs((phi - obs + 90) % 180 - 90)
        assert int(np.argmin(delta)) == k
        tested += 1
    assert tested > 100


def test_two_candidates_can_be_closer_than_the_noise_so_the_plane_is_ambiguous():
    rng = np.random.default_rng(6)
    close = sum(_sep(_candidate_traces(ref.bunge_R(*rng.uniform([0, 0, 0], [360, 180, 360]))[0])) < 1.0
                for _ in range(1000))
    assert close > 0
