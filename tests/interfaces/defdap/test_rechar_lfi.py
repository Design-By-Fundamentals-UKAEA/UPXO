"""
``EBSDReader.rechar_lfi``: the fill of every non-positive pixel with the label of the largest
neighbouring grain.

``_oracle_fill`` below is the algorithm the method used before it was vectorised (one pass over the
whole map per unknown region). The tests hold the method to the oracle's label map and counts on
random maps, 4- and 8-connectivity, ties between equal grains and gaps in the grain ids, and check
that a map with many unknown regions is filled in bounded time.

Run from repo root:
    pytest tests/interfaces/defdap/test_rechar_lfi.py -v
"""

import time

import numpy as np
import pytest

cc3d = pytest.importorskip("cc3d")

from upxo.gsdataops.gid_ops import find_neighs2d
from upxo.interfaces.defdap.ebsd_reader import EBSDReader


def _reader(lfi, seed=0):
    lfi = np.asarray(lfi, dtype=np.int32)
    ny, nx = lfi.shape
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(ny, nx, 4))
    q /= np.linalg.norm(q, axis=2, keepdims=True)
    q[..., 0] = np.abs(q[..., 0])
    rdr = EBSDReader.__new__(EBSDReader)
    rdr.lfi_ebsd = np.array(lfi, dtype=np.int32)
    rdr.quat_ebsd = q
    rdr.euler_ebsd = rng.uniform(0.0, 2.0 * np.pi, (ny, nx, 3))
    return rdr


def _oracle_fill(lfi, connectivity):
    """The per-region loop: largest neighbouring grain, lowest id on a tie."""
    lfi = np.asarray(lfi, dtype=np.int32)
    unknown = lfi <= 0
    n_valid = int(lfi.max())
    cc = cc3d.connected_components(unknown.astype(np.uint8), connectivity=connectivity)
    n_cc = int(cc.max())
    aug = lfi.copy()
    aug[unknown] = (n_valid + cc[unknown]).astype(np.int32)
    neigh = find_neighs2d(aug, conn=connectivity)
    ids, counts = np.unique(lfi[~unknown], return_counts=True)
    size = dict(zip(ids.tolist(), counts.tolist()))
    filled = lfi.copy()
    assigned = {}
    for cc_id in range(1, n_cc + 1):
        nbs = [nb for nb in neigh.get(n_valid + cc_id, []) if 1 <= nb <= n_valid]
        if not nbs:
            continue
        best = max(nbs, key=lambda g: size.get(g, 0))
        filled[cc == cc_id] = best
        assigned[cc_id] = best
    n_pixels = int(sum((cc == c).sum() for c in assigned))
    return filled, len(assigned), n_cc, n_pixels


def _blocky_map(seed, ny=60, nx=70, block=7, n_grains=25, holes=90, id_gaps=False):
    rng = np.random.default_rng(seed)
    coarse = rng.integers(1, n_grains + 1, size=(ny // block + 1, nx // block + 1))
    lfi = np.kron(coarse, np.ones((block, block), dtype=int))[:ny, :nx]
    if id_gaps:
        lfi = lfi * 3 + 2
    ys, xs = rng.integers(0, ny, holes), rng.integers(0, nx, holes)
    codes = rng.choice([0, -1, -2], size=holes)
    lfi[ys, xs] = codes
    lfi[10:14, 20:30] = 0                                   # one larger region
    lfi[40:43, 5:9] = -1
    return lfi.astype(np.int32)


@pytest.mark.parametrize("connectivity", [4, 8])
@pytest.mark.parametrize("seed", range(6))
def test_label_map_equals_the_per_region_loop(seed, connectivity):
    lfi = _blocky_map(seed)
    expected, *_ = _oracle_fill(lfi, connectivity)
    rdr = _reader(lfi, seed)
    rdr.rechar_lfi(connectivity=connectivity)
    assert np.array_equal(rdr.lfi_ebsd, expected)
    assert rdr.lfi_ebsd.dtype == np.int32
    assert (rdr.lfi_ebsd > 0).all()


@pytest.mark.parametrize("seed", range(3))
def test_label_map_equals_the_loop_when_grain_ids_have_gaps(seed):
    lfi = _blocky_map(seed, id_gaps=True)
    expected, *_ = _oracle_fill(lfi, 4)
    rdr = _reader(lfi, seed)
    rdr.rechar_lfi(connectivity=4)
    assert np.array_equal(rdr.lfi_ebsd, expected)


def test_an_unknown_region_takes_the_largest_neighbouring_grain():
    rdr = _reader([[1, 0, 2, 2, 2]])
    rdr.rechar_lfi(connectivity=4)
    assert rdr.lfi_ebsd.tolist() == [[1, 2, 2, 2, 2]]


def test_equal_neighbours_resolve_to_the_lowest_grain_id():
    rdr = _reader([[3, 3, 0, 1, 1]])
    rdr.rechar_lfi(connectivity=4)
    assert rdr.lfi_ebsd.tolist() == [[3, 3, 1, 1, 1]]


def test_a_region_is_filled_whole_with_one_grain():
    lfi = np.array([[1, 1, 1, 1],
                    [1, 0, 0, 2],
                    [1, 0, 0, 2],
                    [2, 2, 2, 2]], dtype=np.int32)
    rdr = _reader(lfi)
    rdr.rechar_lfi(connectivity=4)
    assert rdr.lfi_ebsd[1:3, 1:3].tolist() == [[1, 1], [1, 1]]         # grains 1 and 2 both hold 6 pixels


def test_reported_counts_equal_the_loop(capsys):
    lfi = _blocky_map(2)
    _, n_assigned, n_cc, n_pixels = _oracle_fill(lfi, 4)
    rdr = _reader(lfi)
    rdr.rechar_lfi(connectivity=4)
    out = capsys.readouterr().out
    assert f"{n_cc} connected components found" in out
    assert f"{n_assigned}/{n_cc} CCs assigned ({n_pixels} pixels filled)" in out


def test_filled_pixels_take_the_orientation_of_their_assigned_grain():
    lfi = _blocky_map(1)
    expected, *_ = _oracle_fill(lfi, 4)
    rdr = _reader(lfi, 1)
    q0 = rdr.quat_ebsd.copy()
    rdr.rechar_lfi(connectivity=4)
    known = lfi > 0
    assert np.array_equal(rdr.quat_ebsd[known], q0[known])                # untouched where indexed
    filled = ~known
    for gid in np.unique(expected[filled])[:5]:
        sel = filled & (expected == gid)
        assert np.allclose(rdr.quat_ebsd[sel], rdr.quat_ebsd[sel][0])    # one orientation per grain
        assert abs(np.linalg.norm(rdr.quat_ebsd[sel][0]) - 1.0) < 1e-9


def test_a_map_with_nothing_indexed_is_left_unchanged():
    lfi = np.array([[0, -1, 0], [-2, 0, 0]], dtype=np.int32)
    rdr = _reader(lfi)
    q0 = rdr.quat_ebsd.copy()
    rdr.rechar_lfi(connectivity=4)
    assert np.array_equal(rdr.lfi_ebsd, lfi)
    assert np.array_equal(rdr.quat_ebsd, q0)


def test_many_unknown_regions_are_filled_in_bounded_time():
    rng = np.random.default_rng(5)
    ny = nx = 1500
    coarse = rng.integers(1, 400, size=(ny // 30 + 1, nx // 30 + 1))
    lfi = np.kron(coarse, np.ones((30, 30), dtype=int))[:ny, :nx].astype(np.int32)
    lfi[rng.integers(2, ny - 2, 8000), rng.integers(2, nx - 2, 8000)] = 0
    rdr = _reader(lfi)
    t0 = time.perf_counter()
    rdr.rechar_lfi(connectivity=4)
    assert time.perf_counter() - t0 < 20.0
    assert (rdr.lfi_ebsd > 0).all()
