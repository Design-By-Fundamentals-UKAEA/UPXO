"""
``_reduce_to_reference``: for every quaternion, the cubic-symmetry variant closest to a reference,
on the reference's hemisphere. ``_oracle`` is the form used before the 24 variants were no longer
built per pixel; the two must give the same quaternions.

Run from repo root:
    pytest tests/xtalphy/test_reduce_to_reference.py -v
"""

import numpy as np
import pytest

from upxo.xtalphy.crystal_orientation import (
    _quat_mul_batch,
    _reduce_to_reference,
    cubic_symmetry_quats,
    grain_avg_quats,
)


def _oracle(q, ref):
    S = cubic_symmetry_quats()
    eq = _quat_mul_batch(S[None, :, :], q[:, None, :])
    dots = np.einsum("ckf,cf->ck", eq, ref)
    best = np.argmax(np.abs(dots), axis=1)
    rows = np.arange(len(q))
    return eq[rows, best] * np.where(dots[rows, best] < 0, -1.0, 1.0)[:, None]


def _random_unit(n, seed):
    q = np.random.default_rng(seed).normal(size=(n, 4))
    return q / np.linalg.norm(q, axis=1, keepdims=True)


@pytest.mark.parametrize("seed", range(4))
def test_matches_the_per_variant_form(seed):
    q, ref = _random_unit(5000, seed), _random_unit(5000, seed + 100)
    assert np.allclose(_reduce_to_reference(q, ref), _oracle(q, ref), atol=1e-12)


def test_matches_across_chunk_boundaries():
    q, ref = _random_unit(1003, 7), _random_unit(1003, 8)
    assert np.allclose(_reduce_to_reference(q, ref, chunk=100), _oracle(q, ref), atol=1e-12)


def test_result_is_a_symmetric_variant_on_the_reference_hemisphere():
    q, ref = _random_unit(2000, 3), _random_unit(2000, 4)
    out = _reduce_to_reference(q, ref)
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0)
    assert (np.einsum("ij,ij->i", out, ref) >= 0).all()


def test_a_pixel_already_nearest_the_reference_is_unchanged():
    q = _random_unit(500, 9)
    assert np.allclose(_reduce_to_reference(q, q), q, atol=1e-12)


def test_grain_mean_of_variants_of_one_orientation_is_that_orientation():
    S = cubic_symmetry_quats()
    base = _random_unit(1, 11)[0]
    variants = np.array([_quat_mul_batch(S[k], base) for k in range(24)])
    lfi = np.ones((1, 24), dtype=int)
    _, mean = grain_avg_quats(lfi, variants[None, :, :], symmetry=True)
    equivalents = np.array([_quat_mul_batch(S[k], base) for k in range(24)])
    assert np.abs(equivalents @ mean[0]).max() == pytest.approx(1.0, abs=1e-9)
