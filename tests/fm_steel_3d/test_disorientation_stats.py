"""Disorientation statistics: sampled pairs and slice histograms."""
import numpy as np
import pytest

from upxo.pxtal.fm_steel_3d import disorientation_stats as ds
from upxo.pxtal.fm_steel_3d.orientation_assigner_3d import OrientationAssigner3D
from upxo.xtalphy.crystal_orientation import euler_bunge_to_matrix

from .hierarchy_state import build_hierarchy


class RecordingAssigner:
    """Returns the pair of orientations it was asked about."""

    def disorientation_deg(self, a, b):
        return (a, b)


@pytest.fixture(scope='module')
def hierarchy():
    return build_hierarchy(seed=0)


def _owner(mapping):
    """Orientation -> key, for mappings of key -> orientation triple."""
    return {tuple(v): k for k, v in mapping.items()}


# ── helpers ───────────────────────────────────────────────────────────────────

def test_parent_block_id():
    assert ds.parent_block_id('SB_3_1_2_1') == '3_1_2'
    assert ds.parent_block_id('SB_20_1_11_4') == '20_1_11'


def test_reference_angles_are_ascending_and_inside_the_histogram_range():
    angles = ds.KS_CHARACTERISTIC_ANGLES
    assert list(angles) == sorted(angles)
    assert 0 < angles[0] and angles[-1] <= ds.DEFAULT_ANGLE_BINS[-1]


# ── sampled pairs ─────────────────────────────────────────────────────────────

def test_adjacent_pairs_are_taken_once_whichever_side_lists_them():
    orientations = {1: 'a', 2: 'b', 3: 'c'}
    neighbours = {1: [2, 3], 2: [1], 3: [1, 2]}
    pairs = ds.adjacent_pag_disorientations(orientations, neighbours, RecordingAssigner())
    assert len(pairs) == 3
    assert {frozenset(p) for p in pairs} == {frozenset('ab'), frozenset('ac'), frozenset('bc')}


def test_adjacent_pairs_skip_pags_without_an_orientation():
    pairs = ds.adjacent_pag_disorientations({1: 'a', 2: 'b'}, {1: [2, 9], 2: [1], 9: [1]}, RecordingAssigner())
    assert pairs == [('a', 'b')]


def test_random_pag_pairs(hierarchy):
    _, fm_blocks, _ = hierarchy
    pairs = ds.random_pag_disorientations(fm_blocks.pag_orientations, 25, np.random.default_rng(5),
                                          RecordingAssigner())
    assert len(pairs) == 25
    assert all(a != b for a, b in pairs)
    again = ds.random_pag_disorientations(fm_blocks.pag_orientations, 25, np.random.default_rng(5),
                                          RecordingAssigner())
    assert again == pairs


def test_random_pag_pairs_need_two_pags():
    assert ds.random_pag_disorientations({1: (0, 0, 0)}, 10, np.random.default_rng(0)) == []
    assert ds.random_pag_disorientations({}, 10, np.random.default_rng(0)) == []


def test_subblock_pairs_lie_within_or_across_blocks(hierarchy):
    _, _, fm_subblocks = hierarchy
    owner = _owner(fm_subblocks.subblock_orientations)
    sampled = ds.subblock_disorientations(
        fm_subblocks.subblock_orientations, fm_subblocks.block_to_subblocks_map,
        30, np.random.default_rng(1), assigner=RecordingAssigner())
    assert len(sampled['within']) == 30
    assert 0 < len(sampled['across']) <= 30
    for a, b in sampled['within']:
        assert owner[a] != owner[b]
        assert ds.parent_block_id(owner[a]) == ds.parent_block_id(owner[b])
    for a, b in sampled['across']:
        assert ds.parent_block_id(owner[a]) != ds.parent_block_id(owner[b])


def test_subblock_pairs_can_be_limited_to_one_kind(hierarchy):
    _, _, fm_subblocks = hierarchy
    only_across = ds.subblock_disorientations(
        fm_subblocks.subblock_orientations, fm_subblocks.block_to_subblocks_map,
        5, np.random.default_rng(1), within=False, assigner=RecordingAssigner())
    assert list(only_across) == ['across']


def test_no_within_pairs_when_every_block_has_one_subblock():
    orientations = {'SB_1_1_1_1': (0, 0, 0), 'SB_1_1_2_1': (1, 0, 0), 'SB_1_1_3_1': (2, 0, 0)}
    blocks = {'1_1_1': ['SB_1_1_1_1'], '1_1_2': ['SB_1_1_2_1'], '1_1_3': ['SB_1_1_3_1']}
    sampled = ds.subblock_disorientations(orientations, blocks, 10, np.random.default_rng(0),
                                          assigner=RecordingAssigner())
    assert sampled['within'] == [] and len(sampled['across']) == 10


def test_hierarchical_selection_and_reproducibility(hierarchy):
    _, fm_blocks, fm_subblocks = hierarchy
    everything = ds.hierarchical_disorientations(fm_blocks, ds.SAMPLED_TYPES, 20, 7, fm_subblocks)
    assert set(everything) == set(ds.SAMPLED_TYPES)
    assert len(everything['pag_adj']) == 4                  # four neighbouring PAG pairs in the fixture
    assert len(everything['pag_rand']) == 20
    distribution = fm_blocks.compute_misorientation_distribution(20, 7)
    assert everything['blk_within'] == distribution['within_pag']
    assert everything['blk_across'] == distribution['across_pag']

    again = ds.hierarchical_disorientations(fm_blocks, ds.SAMPLED_TYPES, 20, 7, fm_subblocks)
    assert again == everything

    subset = ds.hierarchical_disorientations(fm_blocks, {'pag_rand', 'sb_across'}, 20, 7, fm_subblocks)
    assert set(subset) == {'pag_rand', 'sb_across'}
    assert subset['pag_rand'] == everything['pag_rand']      # drawn first from the same stream


def test_hierarchical_skips_subblock_types_without_subblocks(hierarchy):
    _, fm_blocks, _ = hierarchy
    out = ds.hierarchical_disorientations(fm_blocks, ds.SAMPLED_TYPES, 10, 0)
    assert set(out) == {'blk_within', 'blk_across', 'pag_adj', 'pag_rand'}


def test_hierarchical_without_neighbour_data(hierarchy):
    _, fm_blocks, _ = hierarchy
    bare = type('Bare', (), {})()
    bare.pag_orientations = fm_blocks.pag_orientations
    assert ds.hierarchical_disorientations(bare, {'pag_adj'}, 10, 0) == {'pag_adj': []}


def test_unknown_types_are_rejected(hierarchy):
    pag_id_map, fm_blocks, _ = hierarchy
    with pytest.raises(ValueError, match='unknown disorientation types'):
        ds.hierarchical_disorientations(fm_blocks, {'nope'}, 10, 0)
    with pytest.raises(ValueError, match='unknown disorientation types'):
        ds.slice_disorientation_histograms(pag_id_map, fm_blocks, {'pag_rand'}, (1, 1, 1))


# ── slice histograms, against a voxel-by-voxel definition ────────────────────

BINS = ds.DEFAULT_ANGLE_BINS


def _slice_indices(n_voxels, n_slices):
    return np.linspace(1, n_voxels - 2, max(1, n_slices), endpoint=True).astype(int)


def _slices(grid, n_slices):
    out = {'x': [], 'y': [], 'z': []}
    for i in _slice_indices(grid.shape[0], n_slices[0]):
        out['x'].append(grid[i, :, :])
    for j in _slice_indices(grid.shape[1], n_slices[1]):
        out['y'].append(grid[:, j, :])
    for k in _slice_indices(grid.shape[2], n_slices[2]):
        out['z'].append(grid[:, :, k])
    return out


def _oracle(grid_by_voxel, n_slices, pair_angle):
    """Histograms over slices: every 4-connected pair of different positive labels."""
    result = {}
    for axis, planes in _slices(grid_by_voxel, n_slices).items():
        histograms = []
        for plane in planes:
            angles = []
            rows, cols = plane.shape
            for r in range(rows):
                for c in range(cols):
                    for dr, dc in ((1, 0), (0, 1)):
                        r2, c2 = r + dr, c + dc
                        if r2 < rows and c2 < cols:
                            a, b = plane[r, c], plane[r2, c2]
                            if a != b and a > 0 and b > 0:
                                angle = pair_angle(a, b)
                                if angle is not None:
                                    angles.append(angle)
            if angles:
                with np.errstate(divide='ignore', invalid='ignore'):     # angles all beyond the last bin
                    histograms.append(np.nan_to_num(np.histogram(angles, bins=BINS, density=True)[0]))
        result[axis] = histograms
    return result


def _label_grid(shape, groups):
    grid = np.zeros(shape, dtype=np.int32)
    for label, voxels in enumerate(groups.values(), start=1):
        grid[tuple(voxels.T)] = label
    return grid


def _rotation_angle(orientation_a, orientation_b):
    ra, rb = euler_bunge_to_matrix(*orientation_a), euler_bunge_to_matrix(*orientation_b)
    return float(np.degrees(np.arccos(np.clip((np.trace(ra.T @ rb) - 1) / 2, -1, 1))))


def _assert_same(result, expected):
    assert set(result) == set(expected)
    for axis in expected:
        assert len(result[axis]) == len(expected[axis]), axis
        for got, want in zip(result[axis], expected[axis]):
            np.testing.assert_allclose(got, want, atol=1e-9, rtol=0)


N_SLICES = (3, 4, 2)


@pytest.mark.parametrize('within', [True, False])
def test_block_histograms(hierarchy, within):
    pag_id_map, fm_blocks, _ = hierarchy
    names = list(fm_blocks.all_blocks)
    grid = _label_grid(pag_id_map.shape, fm_blocks.all_blocks)
    pag_of = {i + 1: int(pag_id_map[tuple(fm_blocks.all_blocks[n][0])]) for i, n in enumerate(names)}

    def pair_angle(a, b):
        if (pag_of[a] == pag_of[b]) != within:
            return None
        return _rotation_angle(fm_blocks.block_orientations[names[a - 1]],
                               fm_blocks.block_orientations[names[b - 1]])

    key = 'blk_within' if within else 'blk_across'
    result = ds.slice_disorientation_histograms(pag_id_map, fm_blocks, {key}, N_SLICES)
    assert list(result) == [key]
    expected = _oracle(grid, N_SLICES, pair_angle)
    assert sum(len(v) for v in expected.values()) > 0
    _assert_same(result[key], expected)


@pytest.mark.parametrize('within', [True, False])
def test_subblock_histograms(hierarchy, within):
    pag_id_map, fm_blocks, fm_subblocks = hierarchy
    names = list(fm_subblocks.all_subblocks)
    grid = _label_grid(pag_id_map.shape, fm_subblocks.all_subblocks)

    def pair_angle(a, b):
        same_block = ds.parent_block_id(names[a - 1]) == ds.parent_block_id(names[b - 1])
        if same_block != within:
            return None
        return _rotation_angle(fm_subblocks.subblock_orientations[names[a - 1]],
                               fm_subblocks.subblock_orientations[names[b - 1]])

    key = 'sb_within' if within else 'sb_across'
    result = ds.slice_disorientation_histograms(pag_id_map, fm_blocks, {key}, N_SLICES, fm_subblocks)
    expected = _oracle(grid, N_SLICES, pair_angle)
    assert sum(len(v) for v in expected.values()) > 0
    _assert_same(result[key], expected)


def test_pag_histograms_use_the_cubic_disorientation_of_each_pair(hierarchy):
    pag_id_map, fm_blocks, _ = hierarchy
    assigner = OrientationAssigner3D()

    def pair_angle(a, b):
        # single precision, as the pair table is kept
        return float(np.float32(assigner.disorientation_deg(fm_blocks.pag_orientations[a],
                                                            fm_blocks.pag_orientations[b])))

    result = ds.slice_disorientation_histograms(pag_id_map, fm_blocks, {'pag_adj'}, N_SLICES)
    expected = _oracle(pag_id_map, N_SLICES, pair_angle)
    assert sum(len(v) for v in expected.values()) > 0
    _assert_same(result['pag_adj'], expected)


def test_pags_without_an_orientation_are_left_out(hierarchy):
    pag_id_map, fm_blocks, _ = hierarchy
    reduced = type('Reduced', (), {})()
    reduced.pag_orientations = {k: v for k, v in fm_blocks.pag_orientations.items() if k != 7}
    full = ds.slice_disorientation_histograms(pag_id_map, fm_blocks, {'pag_adj'}, N_SLICES)['pag_adj']
    result = ds.slice_disorientation_histograms(pag_id_map, reduced, {'pag_adj'}, N_SLICES)['pag_adj']
    assert sum(len(v) for v in result.values()) <= sum(len(v) for v in full.values())

    def pair_angle(a, b):
        if 7 in (a, b):
            return None
        return float(np.float32(OrientationAssigner3D().disorientation_deg(
            reduced.pag_orientations[a], reduced.pag_orientations[b])))

    _assert_same(result, _oracle(pag_id_map, N_SLICES, pair_angle))


def test_selection_decides_which_types_are_computed(hierarchy):
    pag_id_map, fm_blocks, fm_subblocks = hierarchy
    everything = ds.slice_disorientation_histograms(
        pag_id_map, fm_blocks, ds.SLICE_TYPES, (2, 2, 2), fm_subblocks)
    assert list(everything) == ['blk_within', 'blk_across', 'pag_adj', 'sb_within', 'sb_across']
    block_mode = ds.slice_disorientation_histograms(pag_id_map, fm_blocks, ds.SLICE_TYPES, (2, 2, 2))
    assert list(block_mode) == ['blk_within', 'blk_across', 'pag_adj']


def test_histograms_are_densities_over_the_bins(hierarchy):
    pag_id_map, fm_blocks, _ = hierarchy
    result = ds.slice_disorientation_histograms(pag_id_map, fm_blocks, {'blk_across'}, N_SLICES)
    widths = np.diff(ds.DEFAULT_ANGLE_BINS)
    for histograms in result['blk_across'].values():
        for h in histograms:
            assert h.shape == (len(widths),)
            assert (h * widths).sum() == pytest.approx(1.0)


def test_custom_bins(hierarchy):
    pag_id_map, fm_blocks, _ = hierarchy
    bins = np.linspace(0, 65, 14)
    result = ds.slice_disorientation_histograms(pag_id_map, fm_blocks, {'blk_across'}, (2, 2, 2), bins=bins)
    assert all(h.shape == (13,) for hs in result['blk_across'].values() for h in hs)


def test_slices_without_boundaries_are_dropped():
    # one block only: no boundary anywhere
    pag_id_map = np.ones((6, 6, 6), dtype=int)
    voxels = np.argwhere(pag_id_map == 1)
    fm_blocks = type('One', (), {})()
    fm_blocks.all_blocks = {'1_1_1': voxels}
    fm_blocks.block_orientations = {'1_1_1': (10., 20., 30.)}
    fm_blocks.pag_orientations = {1: (10., 20., 30.)}
    result = ds.slice_disorientation_histograms(pag_id_map, fm_blocks, set(ds.SLICE_TYPES[:3]), (2, 2, 2))
    assert all(hs == [] for type_result in result.values() for hs in type_result.values())
