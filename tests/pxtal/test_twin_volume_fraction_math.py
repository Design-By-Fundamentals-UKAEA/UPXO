"""Regression test for the twin volume/volume-fraction dict bug.

``mcgs3_grain_structure`` and ``single_phased`` (in
``mcgs3_temporal_slice.py`` and ``single_phased.py`` respectively) build a
``twin_nvox`` dict keyed by twin feature id -> voxel count, then compute:

    twin_vol_total = sum(twin_nvox)
    tw_vfs = [twvol / pgrainvol for twvol, pgrainvol in zip(twin_nvox, parent_vols)]

Iterating a dict yields its *keys*, not its values, so both lines were
summing/zipping twin feature ids instead of voxel counts -- silently wrong
twin-volume-fraction output with no error raised. The fix appends
``.values()`` in both places. This test does not drive the full grain
structure pipeline (that requires a complete MC run); it reproduces the
exact dict shape and formula in isolation, with non-sequential keys chosen
so the old (keys) and new (values) results provably diverge.
"""
import numpy as np


def _compute_old(twin_nvox, parent_vols):
    """The pre-fix formula: iterating the dict yields its keys."""
    twin_vol_total = sum(twin_nvox)
    tw_vfs = np.array([twvol / pgrainvol
                        for twvol, pgrainvol in zip(twin_nvox, parent_vols)])
    return twin_vol_total, tw_vfs


def _compute_new(twin_nvox, parent_vols):
    """The post-fix formula, matching the current source exactly."""
    twin_vol_total = sum(twin_nvox.values())
    tw_vfs = np.array([twvol / pgrainvol
                        for twvol, pgrainvol in zip(twin_nvox.values(), parent_vols)])
    return twin_vol_total, tw_vfs


def test_fixed_formula_sums_voxel_counts_not_keys():
    # Non-sequential feature ids so summing keys vs. values provably differ.
    twin_nvox = {7: 120, 14: 45, 21: 300}
    parent_vols = np.array([1000.0, 1000.0, 1000.0])

    old_total, old_vfs = _compute_old(twin_nvox, parent_vols)
    new_total, new_vfs = _compute_new(twin_nvox, parent_vols)

    assert old_total == 7 + 14 + 21  # the bug: sum of dict keys
    assert new_total == 120 + 45 + 300  # the fix: sum of voxel counts
    assert new_total != old_total

    np.testing.assert_allclose(new_vfs, [120 / 1000, 45 / 1000, 300 / 1000])
    # The buggy version divides feature ids by parent volume -- nowhere near
    # a physically valid volume fraction.
    assert not np.allclose(old_vfs, new_vfs)
