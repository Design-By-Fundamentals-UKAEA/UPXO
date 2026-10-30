"""Pole figure densities weighted per orientation (for example by grain volume)."""
import numpy as np
import pytest

from upxo.viz.xphy import pole_figure as module
from upxo.viz.xphy.pole_figure import PoleFigure

_EULER = np.array([[0.0, 0.0, 0.0], [30.0, 40.0, 20.0], [100.0, 60.0, 250.0], [220.0, 15.0, 75.0]])


def _field(euler, weights=None, family='100', **kwargs):
    pf = PoleFigure(euler, convention='euler_deg', symmetry='cubic', weights=weights)
    poles = pf._get_symmetric_poles(family)
    return pf._compute_mud_grid(poles, grid_points=40, **kwargs)[2]


def test_no_weights_and_equal_weights_give_the_same_density():
    plain = _field(_EULER)
    for value in (1.0, 7.5):
        np.testing.assert_allclose(_field(_EULER, np.full(len(_EULER), value)), plain, rtol=1e-12, equal_nan=True)


def test_the_density_is_the_weighted_mean_of_the_single_orientation_densities():
    weights = np.array([3.0, 1.0, 0.5, 2.0])
    singles = np.stack([_field(_EULER[i:i + 1]) for i in range(len(_EULER))])
    expected = np.tensordot(weights / weights.sum(), singles, axes=1)
    np.testing.assert_allclose(_field(_EULER, weights), expected, rtol=1e-10, equal_nan=True)


def test_a_zero_weight_removes_the_orientation():
    np.testing.assert_allclose(_field(_EULER, [1.0, 0.0, 0.0, 0.0]), _field(_EULER[:1]), rtol=1e-12, equal_nan=True)
    np.testing.assert_allclose(_field(_EULER, [0.0, 2.0, 0.0, 5.0]),
                               np.tensordot(np.array([2.0, 5.0]) / 7.0, np.stack([_field(_EULER[1:2]), _field(_EULER[3:4])]), axes=1),
                               rtol=1e-10, equal_nan=True)


def test_the_density_difference_uses_the_weights_of_both_sides():
    heavy_first = PoleFigure(_EULER[:2], convention='euler_deg', weights=[10.0, 1.0])
    light_first = PoleFigure(_EULER[:2], convention='euler_deg', weights=[1.0, 10.0])
    diff = heavy_first.compute_density_difference(light_first, pole_family='100', grid_points=30, unit_normalize=False)
    flat = PoleFigure(_EULER[:2], convention='euler_deg').compute_density_difference(
        PoleFigure(_EULER[:2], convention='euler_deg'), pole_family='100', grid_points=30, unit_normalize=False)
    assert np.nanmax(np.abs(diff['zi_diff'])) > 0.5
    assert np.nanmax(np.abs(flat['zi_diff'])) < 1e-12


@pytest.mark.parametrize("weights, message", [([1.0, 2.0], 'one value per orientation'),
                                              ([1.0, -1.0, 1.0, 1.0], 'non-negative'),
                                              ([0.0, 0.0, 0.0, 0.0], 'not all zero'),
                                              ([1.0, np.nan, 1.0, 1.0], 'finite')])
def test_bad_weights_are_refused(weights, message):
    with pytest.raises(ValueError, match=message):
        PoleFigure(_EULER, convention='euler_deg', weights=weights)


def test_the_numba_kernel_gives_the_numpy_answer():
    pytest.importorskip('numba')
    rng = np.random.default_rng(4)
    euler = np.column_stack((rng.uniform(0, 360, 4000), rng.uniform(0, 180, 4000), rng.uniform(0, 360, 4000)))
    weights = rng.uniform(0.1, 5.0, 4000)
    pf = PoleFigure(euler, convention='euler_deg', weights=weights)
    poles = pf._get_symmetric_poles('100')
    with_numba = pf._compute_mud_grid(poles, grid_points=24)[2]                # 24,000 poles: the kernel is used
    saved = module._NUMBA_VMF_KERNEL
    try:
        module._NUMBA_VMF_KERNEL = False                                       # force the numpy path
        without = pf._compute_mud_grid(poles, grid_points=24)[2]
    finally:
        module._NUMBA_VMF_KERNEL = saved
    np.testing.assert_allclose(with_numba, without, rtol=1e-9, equal_nan=True)
