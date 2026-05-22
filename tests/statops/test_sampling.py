import numpy as np
from upxo.statops.sampling import in_limits, bridson_uniform_density


def test_in_limits_inside():
    pts = np.array([[0.5, 0.5]], dtype=np.float32)
    assert in_limits(pts, width=1.0, height=1.0)[0] is np.bool_(True)


def test_in_limits_outside():
    pts = np.array([[1.5, 0.5]], dtype=np.float32)
    assert in_limits(pts, width=1.0, height=1.0)[0] is np.bool_(False)


def test_bridson_returns_points_in_bounds():
    width, height, radius = 1.0, 1.0, 0.1
    pts = bridson_uniform_density(width=width, height=height, radius=radius, k=10)
    assert len(pts) > 0
    assert np.all(pts[:, 0] >= 0) and np.all(pts[:, 0] < width)
    assert np.all(pts[:, 1] >= 0) and np.all(pts[:, 1] < height)
