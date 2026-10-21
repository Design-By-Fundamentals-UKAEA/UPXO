"""expand_grain_quats_to_voxels: lookup-table indexing (O(voxels)), not a
per-grain boolean-mask loop (O(grains x voxels) -- see base_3d.py's
calculate_lfi for the sibling fix to the same class of problem). No
existing test touched this function before; these lock its documented
contract in place."""
import numpy as np
import pytest

from upxo.xtalphy.crystal_orientation import expand_grain_quats_to_voxels


def _voronoi_lgi(n, n_grains, seed):
    from scipy.spatial import cKDTree
    rng = np.random.default_rng(seed)
    seeds = rng.random((n_grains, 3)) * n
    zz, yy, xx = np.mgrid[0:n, 0:n, 0:n]
    pts = np.c_[zz.ravel(), yy.ravel(), xx.ravel()]
    _, idx = cKDTree(seeds).query(pts)
    return (idx.reshape(n, n, n) + 1).astype(np.int32)


def _random_quats(gids, seed):
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(len(gids), 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return {int(g): q[i] for i, g in enumerate(gids)}


def test_every_voxel_gets_its_grains_quaternion():
    lgi = np.array([[1, 1, 2], [2, 3, 3]])
    quats = {1: np.array([1.0, 0, 0, 0]), 2: np.array([0.0, 1.0, 0, 0]),
             3: np.array([0.0, 0, 1.0, 0])}
    field = expand_grain_quats_to_voxels(lgi, quats)
    assert field.shape == (2, 3, 4)
    assert np.array_equal(field[0, 0], quats[1])
    assert np.array_equal(field[0, 2], quats[2])
    assert np.array_equal(field[1, 2], quats[3])


def test_a_label_missing_from_the_dict_reads_back_as_zero():
    lgi = np.array([[1, 2, 2]])
    quats = {1: np.array([1.0, 0, 0, 0])}   # grain 2 has no orientation
    field = expand_grain_quats_to_voxels(lgi, quats)
    assert np.array_equal(field[0, 0], quats[1])
    assert np.array_equal(field[0, 1], [0, 0, 0, 0])
    assert np.array_equal(field[0, 2], [0, 0, 0, 0])


def test_background_label_zero_is_left_as_zero_by_default():
    lgi = np.array([[0, 1], [1, 0]])
    quats = {1: np.array([0.5, 0.5, 0.5, 0.5])}
    field = expand_grain_quats_to_voxels(lgi, quats)
    assert np.array_equal(field[0, 0], [0, 0, 0, 0])
    assert np.array_equal(field[1, 1], [0, 0, 0, 0])
    assert np.array_equal(field[0, 1], quats[1])


def test_label_zero_with_an_explicit_quaternion_is_honoured():
    # 0 is conventionally "background", but the function has no special
    # case for it -- if the caller DOES give it an orientation, it's used,
    # exactly like the old per-grain loop (quat_field[lgi == gid] = q) did.
    lgi = np.array([[0, 1]])
    quats = {0: np.array([0.1, 0.2, 0.3, 0.4]), 1: np.array([1.0, 0, 0, 0])}
    field = expand_grain_quats_to_voxels(lgi, quats)
    assert np.array_equal(field[0, 0], quats[0])


def test_empty_quats_dict_gives_an_all_zero_field():
    lgi = np.array([[1, 2], [3, 4]])
    field = expand_grain_quats_to_voxels(lgi, {})
    assert field.shape == (2, 2, 4)
    assert not field.any()


def test_empty_lgi_does_not_crash():
    lgi = np.zeros((0, 5), dtype=np.int32)
    field = expand_grain_quats_to_voxels(lgi, {1: np.array([1.0, 0, 0, 0])})
    assert field.shape == (0, 5, 4)


def test_negative_labels_are_left_as_zero():
    lgi = np.array([[-1, 1]])
    quats = {1: np.array([1.0, 0, 0, 0])}
    field = expand_grain_quats_to_voxels(lgi, quats)
    assert np.array_equal(field[0, 0], [0, 0, 0, 0])
    assert np.array_equal(field[0, 1], quats[1])


def test_works_on_a_2d_field_not_just_3d():
    lgi = np.array([[1, 2, 1], [2, 1, 2]])
    quats = {1: np.array([1.0, 0, 0, 0]), 2: np.array([0.0, 1.0, 0, 0])}
    field = expand_grain_quats_to_voxels(lgi, quats)
    assert field.shape == (2, 3, 4)
    for (i, j), g in np.ndenumerate(lgi):
        assert np.array_equal(field[i, j], quats[int(g)])


def test_return_dtype_and_shape():
    lgi = np.array([[1]], dtype=np.int32)
    field = expand_grain_quats_to_voxels(lgi, {1: np.array([1.0, 0, 0, 0])})
    assert field.dtype == np.float64
    assert field.shape == lgi.shape + (4,)


@pytest.mark.parametrize("n,n_grains,seed", [(16, 20, 1), (16, 20, 2), (10, 5, 3)])
def test_matches_a_direct_per_grain_reference_implementation(n, n_grains, seed):
    """A plain, obviously-correct (if slow) reference -- the same shape of
    computation the old production code did -- on realistic label fields,
    as an independent cross-check beyond the hand-built cases above."""
    lgi = _voronoi_lgi(n, n_grains, seed)
    gids = np.unique(lgi[lgi > 0])
    quats = _random_quats(gids, seed + 100)

    reference = np.zeros(lgi.shape + (4,), dtype=np.float64)
    for gid, q in quats.items():
        reference[lgi == gid] = q

    field = expand_grain_quats_to_voxels(lgi, quats)
    assert np.array_equal(field, reference)
