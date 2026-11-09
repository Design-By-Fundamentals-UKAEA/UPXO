"""Tests for upxo._writer_data.synthetic_ctf -- the mcgs_v2-backed synthetic .ctf writer.

No real or restricted EBSD data is involved: every value comes from a small mcgs_v2 grain
structure with a random cubic orientation per grain. These tests are deliberately slower
than the rest of tests/interfaces/defdap/ (they run a real, if small, Monte Carlo simulation).
This module is independent of defdap -- it writes CTF text, nothing more -- and is tested
alongside EBSDReader here only to confirm the two round-trip together.
"""
import numpy as np
import pytest

from upxo.interfaces.defdap.ebsd_reader import EBSDReader
from upxo._writer_data.synthetic_ctf import (
    make_synthetic_grain_structure,
    write_synthetic_ctf,
)


def test_make_synthetic_grain_structure_shape_and_labels():
    lgi, gids = make_synthetic_grain_structure(shape=(10, 14), mcsteps=4, rng_seed=0)

    assert lgi.shape == (10, 14)
    assert len(gids) >= 1
    assert set(np.unique(lgi)) == set(gids)


def test_write_synthetic_ctf_is_a_valid_ctf_header(tmp_path):
    path = write_synthetic_ctf(tmp_path / "synth.ctf", shape=(8, 12), mcsteps=4, rng_seed=0)

    text = path.read_text(encoding="latin-1")
    assert text.startswith("Channel Text File")
    assert "XCells\t12" in text
    assert "YCells\t8" in text
    assert "Phase\tX\tY\tBands\tError\tEuler1\tEuler2\tEuler3\tMAD\tBC\tBS" in text


def test_write_synthetic_ctf_round_trips_through_ebsd_reader(tmp_path):
    pytest.importorskip('defdap')
    shape = (10, 15)
    path = write_synthetic_ctf(tmp_path / "synth.ctf", shape=shape, mcsteps=5, rng_seed=2)

    rdr = EBSDReader.load(str(path))

    assert rdr.shape == shape
    assert rdr.euler_ebsd.shape == (*shape, 3)
    assert rdr.quat_ebsd.shape == (*shape, 4)


def test_write_synthetic_ctf_detect_grains_finds_more_than_one_grain(tmp_path):
    pytest.importorskip('defdap')
    path = write_synthetic_ctf(tmp_path / "synth.ctf", shape=(12, 16), mcsteps=6, rng_seed=3)

    rdr = EBSDReader.load(str(path))
    rdr.detect_grains(min_grain_size=1, misori_tol=5)

    assert len(set(rdr.lfi_ebsd.ravel().tolist()) - {0}) > 1


