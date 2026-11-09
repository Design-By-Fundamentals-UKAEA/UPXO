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
import warnings

import numpy as np
import pytest
from scipy import ndimage

from upxo.interfaces.defdap.ebsd_reader import (
    EBSDReader,
    write_subsampled_ctf,
    _ctf_for_defdap,
    _ctf_integer_column_indices,
    _normalize_ctf_integer_row,
)


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


# ---------------------------------------------------------------------------
# write_subsampled_ctf -- a dst_path that collides with src_path would
# silently truncate-and-overwrite the user's original EBSD data (it is
# opened in 'w' mode); these cover the explicit guard against that.
# Pure stdlib text parsing -- no DefDAP needed, same as the module's own
# claim in write_subsampled_ctf's docstring.
# ---------------------------------------------------------------------------

_FAKE_CTF_HEADER = """Channel Text File
Prj\t
Author\t
JobMode\tGrid
XCells\t4
YCells\t2
XStep\t1.0000
YStep\t1.0000
AcqE1\t0.0000
AcqE2\t0.0000
AcqE3\t0.0000
Euler angles refer to Sample Coordinate system (CS0)!\tMag\t100\tCoverage\t100\tDevice\t0\tKV\t20\tTiltAngle\t70\tTiltAxis\t0
Phases\t1
3.520;3.520;3.520\t90.000;90.000;90.000\tNickel\t11\t225
Phase\tX\tY\tBands\tError\tEuler1\tEuler2\tEuler3\tMAD\tBC\tBS
"""


def _write_fake_ctf(path, n_rows=8):
    """A minimal but structurally valid .ctf: the header
    write_subsampled_ctf actually parses (XCells/YCells/XStep/YStep),
    plus n_rows data rows (XCells=4 x YCells=2 = 8 by default)."""
    lines = [_FAKE_CTF_HEADER]
    for i in range(n_rows):
        lines.append(f"1\t{i % 4}\t{i // 4}\t5\t0\t10.0\t20.0\t30.0\t1.0\t100\t50\n")
    path.write_text("".join(lines), encoding="latin-1")


def test_write_subsampled_ctf_rejects_dst_equal_to_src(tmp_path):
    ctf = tmp_path / "scan.ctf"
    _write_fake_ctf(ctf)
    original_bytes = ctf.read_bytes()

    with pytest.raises(ValueError, match="must not be the same file"):
        write_subsampled_ctf(ctf, ctf, stride_x=2, stride_y=2)

    # The whole point of the guard: the original must survive untouched,
    # not just that an exception was raised.
    assert ctf.read_bytes() == original_bytes


def test_write_subsampled_ctf_rejects_dst_equal_to_src_via_different_spelling(tmp_path):
    # Same real file, reached through a path with a redundant ".." segment
    # -- exercises the .resolve()-based comparison, not just literal
    # string equality between the two path arguments.
    ctf = tmp_path / "scan.ctf"
    _write_fake_ctf(ctf)
    original_bytes = ctf.read_bytes()
    roundabout = tmp_path / "sub" / ".." / "scan.ctf"

    with pytest.raises(ValueError, match="must not be the same file"):
        write_subsampled_ctf(ctf, roundabout, stride_x=2, stride_y=2)

    assert ctf.read_bytes() == original_bytes


def test_write_subsampled_ctf_still_works_for_a_genuinely_different_dst(tmp_path):
    ctf = tmp_path / "scan.ctf"
    _write_fake_ctf(ctf)
    dst = tmp_path / "scan_s2.ctf"

    out = write_subsampled_ctf(ctf, dst, stride_x=2, stride_y=2)

    assert out == dst
    assert dst.exists()
    text = dst.read_text(encoding="latin-1")
    assert "XCells\t2" in text
    assert "YCells\t1" in text


# ---------------------------------------------------------------------------
# Symmetry-aware grain mean (mtex's mean): pixels of one grain that are cubic
# symmetric variants of one orientation must average to that orientation, not
# to a quaternion between the variants. A 90 deg turn about Z is a cubic
# symmetry operation, so Z(0) and Z(90) are the same crystal orientation.
# ---------------------------------------------------------------------------

def test_grain_average_euler_deg_averages_symmetric_variants_to_one_orientation():
    lfi = np.array([[1, 1, 1, 1]], dtype=np.int32)
    quat = np.stack([_quat_z(0.0), _quat_z(90.0), _quat_z(0.0), _quat_z(90.0)])[None, :, :]
    rdr = _bare_reader(lfi=lfi, quat=quat)

    phi1, Phi, phi2 = rdr.grain_average_euler_deg()[1]

    # a plain quaternion mean would give 45 deg about Z
    assert phi1 == pytest.approx(0.0, abs=1e-6)
    assert Phi == pytest.approx(0.0, abs=1e-6)
    assert phi2 == pytest.approx(0.0, abs=1e-6)


def test_grain_average_euler_deg_equal_split_of_q_and_minus_q_no_longer_cancels():
    lfi = np.array([[1, 1]], dtype=np.int32)
    q = _quat_z(37.0)
    rdr = _bare_reader(lfi=lfi, quat=np.stack([q, -q])[None, :, :])

    phi1, Phi, phi2 = rdr.grain_average_euler_deg()[1]

    assert phi1 == pytest.approx(37.0, abs=1e-6)


def test_rechar_lfi_fills_a_pixel_with_the_symmetry_aware_grain_mean():
    pytest.importorskip("cc3d")
    lfi = np.array([[1, 1, 1, 1, 0]], dtype=np.int32)
    quat = np.stack([_quat_z(0.0), _quat_z(90.0), _quat_z(0.0), _quat_z(90.0),
                     _quat_z(33.0)])[None, :, :]
    euler = np.zeros((1, 5, 3))
    euler[0, 4] = [np.radians(33.0), 0.0, 0.0]
    rdr = _bare_reader(lfi=lfi, quat=quat)
    rdr.euler_ebsd = euler

    rdr.rechar_lfi(connectivity=4)

    assert rdr.lfi_ebsd.tolist() == [[1, 1, 1, 1, 1]]
    filled_q = rdr.quat_ebsd[0, 4]
    assert abs(filled_q @ _quat_z(0.0)) == pytest.approx(1.0, abs=1e-9)    # identity, not 45 deg
    assert rdr.euler_ebsd[0, 4, 0] % (2 * np.pi) == pytest.approx(0.0, abs=1e-9)
    assert rdr.euler_ebsd[0, 4, 1] == pytest.approx(0.0, abs=1e-9)


def test_rechar_lfi_keeps_filled_euler_consistent_with_the_filled_quaternion():
    pytest.importorskip("cc3d")
    lfi = np.array([[1, 1, 1, 0]], dtype=np.int32)
    quat = np.stack([_quat_z(20.0), _quat_z(22.0), _quat_z(24.0), _quat_z(0.0)])[None, :, :]
    euler = np.zeros((1, 4, 3))
    for i, a in enumerate((20.0, 22.0, 24.0)):
        euler[0, i] = [np.radians(a), 0.0, 0.0]
    rdr = _bare_reader(lfi=lfi, quat=quat)
    rdr.euler_ebsd = euler

    rdr.rechar_lfi(connectivity=4)

    assert math.degrees(rdr.euler_ebsd[0, 3, 0]) == pytest.approx(22.0, abs=1e-4)
    assert abs(rdr.quat_ebsd[0, 3] @ _quat_z(22.0)) == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# CTF integer-column normalisation (BC/Phase/Bands/Error/BS written as float
# strings, e.g. "0.0"). numpy's np.loadtxt tolerated this into an integer
# dtype through 2.2.x with a DeprecationWarning, and raises ValueError from
# numpy 2.4 onward -- real CTF files have been observed writing BC as "0.0".
# Pure stdlib text parsing; no DefDAP or numpy-version dependence needed to
# exercise these.
# ---------------------------------------------------------------------------

_FAKE_CTF_HEADER_FLOAT_BC = _FAKE_CTF_HEADER


def _write_fake_ctf_float_bc(path, n_rows=8):
    """Like _write_fake_ctf, but BC is written as a float string ("0.0"),
    reproducing the real-file defect that trips numpy's integer-dtype
    np.loadtxt parsing from numpy 2.4 onward."""
    lines = [_FAKE_CTF_HEADER_FLOAT_BC]
    for i in range(n_rows):
        lines.append(f"1\t{i % 4}\t{i // 4}\t5\t0\t10.0\t20.0\t30.0\t1.0\t0.0\t50\n")
    path.write_text("".join(lines), encoding="latin-1")


def test_ctf_integer_column_indices_finds_bc_and_friends():
    col_header = "Phase\tX\tY\tBands\tError\tEuler1\tEuler2\tEuler3\tMAD\tBC\tBS\n"
    indices = _ctf_integer_column_indices(col_header)
    names = col_header.strip('\n').split('\t')
    assert [names[i] for i in indices] == ["Phase", "Bands", "Error", "BC", "BS"]


def test_normalize_ctf_integer_row_rewrites_float_string_bc():
    indices = _ctf_integer_column_indices(
        "Phase\tX\tY\tBands\tError\tEuler1\tEuler2\tEuler3\tMAD\tBC\tBS\n")
    row = "1\t0\t0\t5\t0\t10.0\t20.0\t30.0\t1.0\t0.0\t50\n"

    out = _normalize_ctf_integer_row(row, indices)

    parts = out.rstrip("\n").split("\t")
    assert parts[9] == "0"          # BC: "0.0" -> "0"
    assert parts[5] == "10.0"       # a non-integer column (Euler1) untouched
    assert out.endswith("\n")


def test_normalize_ctf_integer_row_noop_without_indices():
    row = "1\t0\t0\t5\t0\t10.0\t20.0\t30.0\t1.0\t0.0\t50\n"
    assert _normalize_ctf_integer_row(row, []) == row


def test_normalize_ctf_integer_row_leaves_unparseable_value_as_is():
    indices = _ctf_integer_column_indices("Phase\tBC\n")
    row = "1\tNaN\n"
    out = _normalize_ctf_integer_row(row, indices)
    assert out == row


def test_write_subsampled_ctf_normalizes_float_string_bc(tmp_path):
    ctf = tmp_path / "scan.ctf"
    _write_fake_ctf_float_bc(ctf)
    dst = tmp_path / "scan_s2.ctf"

    write_subsampled_ctf(ctf, dst, stride_x=2, stride_y=2)

    text = dst.read_text(encoding="latin-1")
    data_lines = text.splitlines()[len(_FAKE_CTF_HEADER_FLOAT_BC.splitlines()):]
    assert data_lines, "expected at least one sub-sampled data row"
    for line in data_lines:
        bc = line.split("\t")[9]
        assert bc == "0", f"BC column not normalised: {bc!r}"


def test_ctf_for_defdap_normalizes_and_cleans_up(tmp_path):
    ctf = tmp_path / "scan.ctf"
    _write_fake_ctf_float_bc(ctf)

    with _ctf_for_defdap(ctf, 'OxfordText') as load_path:
        assert load_path != ctf        # a distinct, normalised temp copy was made
        assert load_path.exists()
        text = load_path.read_text(encoding="latin-1")
        for line in text.splitlines()[len(_FAKE_CTF_HEADER_FLOAT_BC.splitlines()):]:
            bc = line.split("\t")[9]
            assert bc == "0"
        # the source file itself is never touched
        assert "0.0\t50" in ctf.read_text(encoding="latin-1")

    assert not load_path.exists()      # removed on context exit


def test_ctf_for_defdap_passthrough_for_non_oxfordtext(tmp_path):
    crc = tmp_path / "scan.crc"
    crc.write_bytes(b"not actually parsed by this helper")

    with _ctf_for_defdap(crc, 'OxfordBinary') as load_path:
        assert load_path == crc        # unchanged, no temp file involved


def test_ctf_for_defdap_passthrough_when_no_column_header_found(tmp_path):
    # No line starting with the "Phase" token at all -- _ctf_for_defdap can't
    # locate the column layout, so it must leave the file for DefDAP's own
    # loader to handle (and fail on, if it's genuinely malformed) rather than
    # raising or silently producing an empty normalised copy.
    ctf = tmp_path / "scan.ctf"
    ctf.write_text("Channel Text File\nnot a recognised header\n1\t0\t0\t10.0\n",
                    encoding="latin-1")

    with _ctf_for_defdap(ctf, 'OxfordText') as load_path:
        assert load_path == ctf


def test_ebsd_reader_load_actually_loads_a_float_formatted_bc_file(tmp_path):
    """End-to-end check of the real defect (not just the helpers that fix it): without
    _ctf_for_defdap wired into EBSDReader.load, this raises ValueError on numpy>=2.4
    ('could not convert string '0.0' to uint8', the exact error from the source report).
    Promoted to an error here so a numpy version that still only warns can't mask a
    regression in the wiring between load() and _ctf_for_defdap.
    """
    pytest.importorskip('defdap')
    ctf = tmp_path / "scan.ctf"
    _write_fake_ctf_float_bc(ctf, n_rows=8)

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        rdr = EBSDReader.load(str(ctf))

    assert rdr.shape == (2, 4)                 # n_rows=8 over the header's XCells=4, YCells=2
    assert ctf.read_text(encoding="latin-1").splitlines()[-1].split("\t")[9] == "0.0"
