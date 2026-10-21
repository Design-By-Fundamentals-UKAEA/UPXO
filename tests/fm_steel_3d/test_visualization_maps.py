"""Label maps of the hierarchy for visualisation."""
import colorsys

import numpy as np
import pytest

from upxo.pxtal.fm_steel_3d import visualization_maps as vm

from .hierarchy_state import build_pipeline_states


@pytest.fixture()
def states():
    return build_pipeline_states(seed=0)


def _build(states, subblock=False, **kwargs):
    fm_pag, fm_blocks, fm_subblocks = states
    return vm.build_visualization_data(
        fm_pag, fm_blocks, fm_blocks, fm_subblocks if subblock else fm_blocks,
        subblock_mode=subblock, **kwargs)


# ── random label order ───────────────────────────────────────────────────────

def _groups():
    return {'a': np.array([[0, 0, 0], [0, 0, 1]]), 'b': np.array([[1, 0, 0]]), 'c': np.array([[2, 2, 2], [2, 2, 1]])}


def test_every_group_gets_its_own_label_from_a_permutation():
    template = np.zeros((3, 3, 3), dtype=np.int32)
    grid, labels = vm.label_groups_in_random_order(template, _groups(), seed=5)
    assert sorted(labels) == [1, 2, 3]
    for label, voxels in zip(labels, _groups().values()):
        assert (grid[tuple(voxels.T)] == label).all()
    assert np.count_nonzero(grid) == 5
    assert grid.dtype == template.dtype


def test_label_order_is_fixed_by_the_seed():
    template = np.zeros((3, 3, 3), dtype=np.int64)
    first = vm.label_groups_in_random_order(template, _groups(), seed=5)
    again = vm.label_groups_in_random_order(template, _groups(), seed=5)
    other = [vm.label_groups_in_random_order(template, _groups(), seed=s)[1] for s in range(8)]
    assert first[1] == again[1] and np.array_equal(first[0], again[0])
    assert len({tuple(o) for o in other}) > 1


# ── sub-block colours ─────────────────────────────────────────────────────────

def test_subblock_colours_share_a_hue_per_block():
    block_to_sub = {'B1': ['s1', 's2', 's3'], 'B2': ['s4'], 'B3': ['s5', 's6']}
    labels = {i + 1: f's{i + 1}' for i in range(6)}
    colours = vm.subblock_family_colours(labels, block_to_sub, 6, seed=3)
    assert colours.shape == (8, 3) and colours.dtype == np.float32
    assert (colours[0] == 0).all() and (colours[7] == 0).all()
    hsv = {i: colorsys.rgb_to_hsv(*colours[i]) for i in range(1, 7)}
    for family in ([1, 2, 3], [5, 6]):
        assert len({round(hsv[i][0], 4) for i in family}) == 1
    assert [round(hsv[i][2], 3) for i in (1, 2, 3)] == [0.30, 0.60, 0.90]
    assert [round(hsv[i][2], 3) for i in (5, 6)] == [0.30, 0.90]
    assert round(hsv[4][2], 3) == 0.60                                      # a single sub-block
    assert all(round(hsv[i][1], 2) == 0.82 for i in range(1, 7))
    hues = sorted(round(hsv[i][0], 4) for i in (1, 4, 5))
    assert hues == pytest.approx([0.0, 1 / 3, 2 / 3], abs=1e-3)             # one hue per block, evenly spread


def test_unknown_subblocks_stay_black():
    colours = vm.subblock_family_colours({1: 'orphan', 2: 's1'}, {'B1': ['s1']}, 2, seed=0)
    assert (colours[1] == 0).all() and colours[2].any()


# ── packet indices ───────────────────────────────────────────────────────────

def test_packet_index_of_a_block_comes_from_its_grain():
    template = np.zeros((2, 2, 2), dtype=np.int32)
    blocks = {'1_1_7': np.array([[0, 0, 0], [0, 0, 1]]), '2_1_8': np.array([[1, 1, 1]]),
              '2_1_9': np.empty((0, 3), dtype=int)}
    grid = vm.packet_index_grid(template, blocks, {7: 2, 8: 0})
    assert grid[0, 0, 0] == grid[0, 0, 1] == 3
    assert grid[1, 1, 1] == 1
    assert np.count_nonzero(grid) == 3


def test_a_grain_without_a_plane_index_gets_the_first_packet():
    grid = vm.packet_index_grid(np.zeros((1, 1, 1), dtype=int), {'1_1_5': np.array([[0, 0, 0]])}, {})
    assert grid[0, 0, 0] == 1


# ── whole preparation ─────────────────────────────────────────────────────────

def test_maps_in_block_mode(states):
    fm_pag, fm_blocks, _ = states
    data = _build(states, seed=11)
    base = fm_pag._parent
    assert data['grain_id_map'] is not base.lgi
    np.testing.assert_array_equal(data['grain_id_map'], base.lgi)

    # PAG of every voxel: the PAG its grain belongs to
    pag_of_grain = {g: pag for pag, gs in fm_pag.clusters_dict.items() for g in gs}
    expected = np.zeros_like(base.lgi)
    for g, pag in pag_of_grain.items():
        expected[base.lgi == g] = pag
    np.testing.assert_array_equal(data['pag_id_map'], expected)
    assert data['pag_id_map'].dtype == base.lgi.dtype

    n = len(fm_blocks.all_blocks)
    assert data['n_blocks'] == n
    for key in ('block_idx_map', 'block_fid_map'):
        assert sorted(np.unique(data[key][data[key] > 0])) == list(range(1, n + 1))
        assert ((data[key] > 0) == (base.lgi > 0)).all()
    assert not np.array_equal(data['block_idx_map'], data['block_fid_map'])    # different seeds
    for vox in fm_blocks.all_blocks.values():                                  # one label per block
        assert len(np.unique(data['block_fid_map'][tuple(vox.T)])) == 1

    assert data['subblock_fid_map'] is None and data['subblock_fid_map_rs'] is None
    assert data['fid_to_rgb'] is None and data['fid_to_sb_id'] == {} and data['packet_id_map_rs'] is None
    assert data['fm_state'] is fm_blocks
    assert data['isolated_grain_orientations'] == {4: (1.0, 2.0, 3.0)}
    assert data['isolated_grain_orientations'] is not fm_pag.isolated_grain_orientations


def test_maps_in_subblock_mode(states):
    _, fm_blocks, fm_subblocks = states
    data = _build(states, subblock=True, seed=3, family_colours=True)
    n = len(fm_subblocks.all_subblocks)
    assert sorted(np.unique(data['subblock_fid_map'][data['subblock_fid_map'] > 0])) == list(range(1, n + 1))
    assert sorted(data['fid_to_sb_id']) == list(range(1, n + 1))
    assert sorted(data['fid_to_sb_id'].values()) == sorted(fm_subblocks.all_subblocks)
    for label, sb_id in data['fid_to_sb_id'].items():
        voxels = fm_subblocks.all_subblocks[sb_id]
        assert (data['subblock_fid_map'][tuple(voxels.T)] == label).all()
    assert data['fid_to_rgb'].shape == (n + 2, 3)
    assert data['packet_id_map_rs'].shape == data['grain_id_map_rs'].shape
    assert data['packet_id_map_rs'].max() <= 3
    assert data['fm_state'] is fm_subblocks


def test_colours_only_when_asked_for(states):
    assert _build(states, subblock=True)['fid_to_rgb'] is None


def test_rescaling_and_voxel_size(states):
    base_shape = states[0]._parent.lgi.shape
    data = _build(states, rescale_factor=0.5, domain_length=70.0)
    expected = tuple(int(round(n * 0.5)) for n in base_shape)
    for key in ('grain_id_map_rs', 'pag_id_map_rs', 'block_fid_map_rs'):
        assert data[key].shape == expected
    assert data['grain_id_map'].shape == base_shape                    # full-resolution maps stay
    assert data['voxel_size_rs'] == pytest.approx(70.0 / expected[0])


def test_unit_rescale_copies_the_maps(states):
    data = _build(states, rescale_factor=1.0)
    np.testing.assert_array_equal(data['grain_id_map_rs'], data['grain_id_map'])
    assert data['grain_id_map_rs'] is not data['grain_id_map']


def test_the_seed_decides_the_block_ids(states):
    a, b, c = (_build(states, seed=s)['block_fid_map'] for s in (1, 1, 2))
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_progress_messages(states):
    messages = []
    _build(states, subblock=True, log=messages.append)
    assert messages[0] == 'Starting vis-data preparation...'
    assert any(m.startswith('Mode: subblock') for m in messages)
    assert any(m.startswith('Rescaled grid shape:') for m in messages)
    quiet = []
    _build(states, log=quiet.append)
    assert 'Sub-block map: skipped (block mode)' in quiet


def test_subblock_mode_needs_subblock_data(states):
    fm_pag, fm_blocks, _ = states
    with pytest.raises(RuntimeError, match='Sub-block data not found'):
        vm.build_visualization_data(fm_pag, fm_blocks, fm_blocks, None, subblock_mode=True)
