"""
Unit tests for upxo.interfaces.defdap.ebsd_reader.EBSDReader's
label-connectivity and orientation-averaging helpers:
``split_disconnected_grains`` and ``grain_average_euler_deg``.

Built directly on synthetic ``lfi_ebsd`` / ``quat_ebsd`` arrays (via
``EBSDReader.__new__`` + slot assignment, the same pattern ``crop()`` uses
internally) so these tests need neither DefDAP nor a real ``.ctf`` file --
no EBSD data is shipped with the repository.

Run from repo root:
    pytest tests/interfaces/defdap/test_ebsd_reader.py -v
"""

import math

import numpy as np
import pytest
from scipy import ndimage

from upxo.interfaces.defdap.ebsd_reader import EBSDReader


def _bare_reader(lfi=None, quat=None):
    """Build an EBSDReader with only the given slots populated."""
    rdr = EBSDReader.__new__(EBSDReader)
    if lfi is not None:
        rdr.lfi_ebsd = lfi
    if quat is not None:
        rdr.quat_ebsd = quat
    return rdr


def _quat_z(theta_deg):
    """Unit quaternion [w, x, y, z] for a rotation of theta_deg about Z."""
    half = math.radians(theta_deg) / 2.0
    return np.array([math.cos(half), 0.0, 0.0, math.sin(half)])


# ---------------------------------------------------------------------------
# split_disconnected_grains
# ---------------------------------------------------------------------------

def _two_blob_lfi():
    """Grain 1 split into a 6px and a 4px blob (4-connectivity); grain 2
    is a single untouched 4px blob; 0 is non-indexed background."""
    return np.array([
        [1, 1, 1, 0, 2, 2, 0, 1, 1],
        [1, 1, 1, 0, 2, 2, 0, 1, 1],
    ], dtype=np.int32)


def test_split_disconnected_grains_keeps_largest_piece_original_id():
    rdr = _bare_reader(lfi=_two_blob_lfi())
    split_map = rdr.split_disconnected_grains(connectivity=4)

    assert split_map == {3: 1}
    assert int((rdr.lfi_ebsd == 1).sum()) == 6   # larger blob kept id 1
    assert int((rdr.lfi_ebsd == 3).sum()) == 4   # smaller blob relabelled
    assert int((rdr.lfi_ebsd == 2).sum()) == 4   # untouched grain unchanged
    assert int((rdr.lfi_ebsd == 0).sum()) == 4   # background untouched


def test_split_disconnected_grains_every_id_is_one_component():
    """Invariant check, independent of the cc3d-based implementation:
    every surviving positive label is a single connected component under
    scipy.ndimage.label."""
    rdr = _bare_reader(lfi=_two_blob_lfi())
    rdr.split_disconnected_grains(connectivity=4)

    structure = ndimage.generate_binary_structure(2, 1)  # 4-connectivity
    for gid in np.unique(rdr.lfi_ebsd):
        if gid <= 0:
            continue
        _, n_components = ndimage.label(rdr.lfi_ebsd == gid, structure=structure)
        assert n_components == 1, f"grain {gid} is still disconnected"


def test_split_disconnected_grains_euler_and_quat_untouched():
    lfi = _two_blob_lfi()
    quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (*lfi.shape, 1))
    rdr = _bare_reader(lfi=lfi, quat=quat)
    quat_before = rdr.quat_ebsd.copy()

    rdr.split_disconnected_grains(connectivity=4)

    assert np.array_equal(rdr.quat_ebsd, quat_before)


def test_split_disconnected_grains_noop_when_already_connected():
    rdr = _bare_reader(lfi=np.array([[1, 1, 2], [1, 1, 2]], dtype=np.int32))
    lfi_before = rdr.lfi_ebsd.copy()

    split_map = rdr.split_disconnected_grains(connectivity=4)

    assert split_map == {}
    assert np.array_equal(rdr.lfi_ebsd, lfi_before)


def test_split_disconnected_grains_connectivity_4_vs_8():
    # A single grain touching itself only corner-to-corner: one component
    # under 8-connectivity, two under 4-connectivity.
    diag = np.array([
        [1, 0],
        [0, 1],
    ], dtype=np.int32)

    rdr8 = _bare_reader(lfi=diag.copy())
    assert rdr8.split_disconnected_grains(connectivity=8) == {}

    rdr4 = _bare_reader(lfi=diag.copy())
    split_map = rdr4.split_disconnected_grains(connectivity=4)
    assert len(split_map) == 1


def test_split_disconnected_grains_invalid_connectivity_raises():
    rdr = _bare_reader(lfi=_two_blob_lfi())
    with pytest.raises(ValueError):
        rdr.split_disconnected_grains(connectivity=6)


# ---------------------------------------------------------------------------
# grain_average_euler_deg
# ---------------------------------------------------------------------------

def test_grain_average_euler_deg_identity_and_z_rotation():
    lfi = np.array([[1, 1, 2, 2]], dtype=np.int32)
    quat = np.stack([
        _quat_z(0.0), _quat_z(0.0),     # grain 1: identity
        _quat_z(90.0), _quat_z(90.0),   # grain 2: 90 deg about Z
    ])[None, :, :]
    rdr = _bare_reader(lfi=lfi, quat=quat)

    result = rdr.grain_average_euler_deg()

    phi1, Phi, phi2 = result[1]
    assert phi1 == pytest.approx(0.0, abs=1e-6)
    assert Phi == pytest.approx(0.0, abs=1e-6)
    assert phi2 == pytest.approx(0.0, abs=1e-6)

    phi1, Phi, phi2 = result[2]
    assert phi1 == pytest.approx(90.0, abs=1e-6)
    assert Phi == pytest.approx(0.0, abs=1e-6)
    assert phi2 == pytest.approx(0.0, abs=1e-6)


def test_grain_average_euler_deg_averages_over_pixels():
    lfi = np.array([[1, 1, 1]], dtype=np.int32)
    quat = np.stack([_quat_z(89.0), _quat_z(90.0), _quat_z(91.0)])[None, :, :]
    rdr = _bare_reader(lfi=lfi, quat=quat)

    phi1, Phi, phi2 = rdr.grain_average_euler_deg()[1]

    assert phi1 == pytest.approx(90.0, abs=0.5)
    assert Phi == pytest.approx(0.0, abs=1e-6)
    assert phi2 == pytest.approx(0.0, abs=1e-6)


def test_grain_average_euler_deg_negative_hemisphere_minority_flip():
    """-q represents the same rotation as q. A minority of pixels stored
    in the negative hemisphere (plausible in real EBSD data, since q and
    -q are interchangeable at the storage level) should not perturb the
    averaged result away from the majority's rotation.

    (An exact 50/50 split of q and -q is the one input where a naive
    componentwise mean cancels to the zero quaternion -- a known
    limitation of this non-symmetry-aware average, not exercised here.)
    """
    lfi = np.array([[1, 1, 1]], dtype=np.int32)
    q = _quat_z(37.0)
    quat = np.stack([q, q, -q])[None, :, :]
    rdr = _bare_reader(lfi=lfi, quat=quat)

    phi1, Phi, phi2 = rdr.grain_average_euler_deg()[1]

    assert phi1 == pytest.approx(37.0, abs=1e-6)
    assert Phi == pytest.approx(0.0, abs=1e-6)
    assert phi2 == pytest.approx(0.0, abs=1e-6)


def test_grain_average_euler_deg_rejects_non_positive_labels():
    lfi = np.array([[1, 0]], dtype=np.int32)
    quat = np.stack([_quat_z(0.0), _quat_z(0.0)])[None, :, :]
    rdr = _bare_reader(lfi=lfi, quat=quat)

    with pytest.raises(ValueError):
        rdr.grain_average_euler_deg()
