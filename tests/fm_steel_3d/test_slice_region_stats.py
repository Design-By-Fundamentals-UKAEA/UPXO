"""Region properties of PAGs and packets on slices."""
import numpy as np
import pytest

from upxo.pxtal.fm_steel_3d import slice_region_stats as srs


# ── label maps ────────────────────────────────────────────────────────────────

def test_pag_label_grid_relabels_grains_by_their_pag():
    lgi = np.array([[[1, 2], [3, 4]], [[0, 5], [6, 1]]], dtype=np.int32)
    grid = srs.pag_label_grid(lgi, {10: [1, 2], 20: [3, 6]})
    np.testing.assert_array_equal(grid, [[[10, 10], [20, 0]], [[0, 0], [20, 10]]])
    assert grid.dtype == np.int32 and grid.shape == lgi.shape


def test_pag_label_grid_ignores_grain_ids_outside_the_grid_labels():
    lgi = np.array([[[1, 2]]], dtype=np.int32)
    grid = srs.pag_label_grid(lgi, {7: [1, 99, 0, -3]})
    np.testing.assert_array_equal(grid, [[[7, 0]]])


# ── neighbours ────────────────────────────────────────────────────────────────

def test_neighbour_counts_of_a_strip_wrap_around():
    strip = np.array([[1, 1, 2, 2, 3, 3]])
    assert srs.periodic_neighbour_counts(strip) == {1: 2, 2: 2, 3: 2}      # 3 touches 1 across the edge


def test_void_is_not_a_neighbour():
    image = np.zeros((4, 4), dtype=int)
    image[0, 0] = 1
    image[2, 2] = 2
    assert srs.periodic_neighbour_counts(image) == {1: 0, 2: 0}


def test_blocks_see_the_blocks_they_share_an_edge_with():
    image = np.array([[1, 1, 2, 2],
                      [1, 1, 2, 2],
                      [3, 3, 4, 4],
                      [3, 3, 4, 4]])
    assert srs.periodic_neighbour_counts(image) == {1: 2, 2: 2, 3: 2, 4: 2}
    corner = image.copy()
    corner[3, 3] = 5          # a single pixel in the corner of block 4
    counts = srs.periodic_neighbour_counts(corner)
    assert counts[5] == 3     # block 4 directly; blocks 2 and 3 across the bottom and right edges


# ── properties of one slice ───────────────────────────────────────────────────

def _rectangles():
    image = np.zeros((10, 12), dtype=int)
    image[0:4, 0:4] = 1           # 4 x 4 square
    image[5:7, 0:8] = 2           # 2 x 8 rectangle
    image[9, 10] = 3              # one pixel
    return image


def test_area_perimeter_and_solidity_of_rectangles():
    values = srs.region_property_values(_rectangles(), 0.5, ['area', 'solidity', 'perimeter'])
    assert values['area'] == [16 * 0.25, 16 * 0.25, 1 * 0.25]
    assert values['solidity'] == pytest.approx([1.0, 1.0, 1.0])
    assert values['perimeter'][0] == pytest.approx(0.5 * 4 * 4 * 0.75, rel=0.3)   # staircase estimate, scaled
    twice = srs.region_property_values(_rectangles(), 1.0, ['perimeter'])['perimeter']
    assert values['perimeter'] == pytest.approx([p / 2 for p in twice])


def test_aspect_ratio():
    values = srs.region_property_values(_rectangles(), 1.0, ['aspect_ratio'])['aspect_ratio']
    assert values[0] == pytest.approx(1.0, abs=0.05)         # square
    assert values[1] > 3.0                                    # 2 x 8
    assert values[2] == 1.0                                   # one pixel: minor axis is 0


def test_neighbours_property_uses_the_wrapped_counts():
    image = np.array([[1, 1, 2, 2, 3, 3]] * 2)
    assert srs.region_property_values(image, 1.0, ['n_neighbours'])['n_neighbours'] == [2.0, 2.0, 2.0]


def test_only_the_requested_properties_are_returned_and_empty_images_are_fine():
    assert list(srs.region_property_values(_rectangles(), 1.0, ['area'])) == ['area']
    empty = srs.region_property_values(np.zeros((5, 5), dtype=int), 1.0, ['area', 'perimeter'])
    assert empty == {'area': [], 'perimeter': []}


def test_unknown_property_is_rejected():
    with pytest.raises(ValueError, match='unknown properties'):
        srs.region_property_values(_rectangles(), 1.0, ['area', 'shininess'])


# ── distributions over slices ────────────────────────────────────────────────

def _structure():
    """Four column grains in a 12 x 10 x 8 grid, grouped into two PAGs; one void slab."""
    x, y = np.meshgrid(np.arange(12), np.arange(10), indexing='ij')
    columns = (1 + (x >= 6) + 2 * (y >= 5)).astype(np.int32)
    lgi = np.repeat(columns[:, :, None], 8, axis=2)
    lgi[:, :, 7] = 0
    return lgi, {5: [1, 2], 9: [3, 4]}


@pytest.mark.parametrize('plane, size', [('xy', 8), ('yz', 12), ('xz', 10)])
def test_slice_positions_and_progress(plane, size):
    lgi, clusters = _structure()
    calls = []
    srs.pag_slice_distributions(lgi, clusters, 1.0, plane, 4, ['pag'], ['area'],
                                progress=lambda *a: calls.append(a))
    expected = np.linspace(1, size - 2, 4, endpoint=True, dtype=int)
    assert [(c[0], c[1], c[2], c[3]) for c in calls] == [(plane, int(i), k, 4) for k, i in enumerate(expected, 1)]


def test_region_counts_follow_the_labels_on_the_slices():
    lgi, clusters = _structure()
    n = 3
    result = srs.pag_slice_distributions(lgi, clusters, 1.0, 'xy', n, ['pag', 'packet'], ['area'])
    # slices z = 1, 3, 6 (all inside the grains): two PAGs and four grains on each
    assert len(result['pag']['area']) == 2 * n
    assert len(result['packet']['area']) == 4 * n
    assert sorted(result['packet']['area'][:4]) == [30.0] * 4         # 6 x 5 voxels each
    assert sorted(result['pag']['area'][:2]) == [60.0, 60.0]


def test_a_slice_through_void_contributes_nothing():
    lgi, clusters = _structure()
    lgi[:, :, 1] = 0                      # 'xy' slices are taken at z = 1 and z = 6
    result = srs.pag_slice_distributions(lgi, clusters, 1.0, 'xy', 2, ['pag', 'packet'], ['area'])
    assert len(result['pag']['area']) == 2
    assert len(result['packet']['area']) == 4


def test_groups_and_properties_are_what_was_asked_for():
    lgi, clusters = _structure()
    only_packets = srs.pag_slice_distributions(lgi, clusters, 1.0, 'xz', 2, ['packet'], ['area', 'solidity'])
    assert list(only_packets) == ['packet'] and list(only_packets['packet']) == ['area', 'solidity']
    both = srs.pag_slice_distributions(lgi, clusters, 1.0, 'xz', 2, ['packet', 'pag'], ['area'])
    assert list(both) == ['packet', 'pag']
    assert both['packet'] == srs.pag_slice_distributions(lgi, clusters, 1.0, 'xz', 2, ['packet'], ['area'])['packet']


def test_voxel_size_scales_areas():
    lgi, clusters = _structure()
    unit = srs.pag_slice_distributions(lgi, clusters, 1.0, 'xy', 2, ['pag'], ['area'])['pag']['area']
    scaled = srs.pag_slice_distributions(lgi, clusters, 3.0, 'xy', 2, ['pag'], ['area'])['pag']['area']
    assert scaled == pytest.approx([9 * a for a in unit])


def test_bad_arguments():
    lgi, clusters = _structure()
    with pytest.raises(ValueError, match='plane must be one of'):
        srs.pag_slice_distributions(lgi, clusters, 1.0, 'ab', 2, ['pag'], ['area'])
    with pytest.raises(ValueError, match='unknown groups'):
        srs.pag_slice_distributions(lgi, clusters, 1.0, 'xy', 2, ['block'], ['area'])
    with pytest.raises(ValueError, match='unknown properties'):
        srs.pag_slice_distributions(lgi, clusters, 1.0, 'xy', 2, ['pag'], ['colour'])
