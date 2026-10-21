"""Lookups over the PAG / packet / block / sub-block hierarchy."""
from types import SimpleNamespace

import numpy as np
import pytest

from upxo.pxtal.fm_steel_3d import hierarchy_queries as hq


def _voxels(*rows):
    return np.array(rows, dtype=int).reshape(-1, 3)


@pytest.fixture()
def state():
    return SimpleNamespace(
        clusters_dict={1: [10, 11], 2: [20]},
        grain_to_pag_id={10: 1, 11: 1, 20: 2},
        grain_to_local_pkt_idx={10: 1, 11: 2, 20: 1},
        grain_to_blocks_map={10: ['b10a', 'b10b'], 11: ['b11'], 20: ['b20'], 30: ['b30']},   # 30 has no PAG
        block_to_subblocks_map={'b10a': ['s1', 's2'], 'b11': ['s3'], 'b30': ['s9']},
        grain_locs={10: _voxels((0, 0, 0), (1, 0, 0)), 11: _voxels((2, 0, 0)), 20: _voxels((3, 1, 1), (3, 1, 2)),
                    99: _voxels((5, 5, 5))},
        all_blocks={'b10a': _voxels((0, 0, 0)), 'b10b': _voxels((1, 0, 0)), 'b11': _voxels((2, 0, 0))},
        all_subblocks={'s1': _voxels((0, 0, 0))},
        pag_orientations={1: (1., 2., 3.), 2: (4., 5., 6.)},
        block_orientations={'b10a': (7., 8., 9.), 'b10b': (1., 1., 1.), 'b11': (2., 2., 2.)},
        subblock_orientations={'s1': (3., 3., 3.)},
        grain_orientations={10: (0., 1., 2.)},
        retained_austenite_pag_ids={2}, isolated_grains={99},
    )


# ── PAG of each item ─────────────────────────────────────────────────────────

def test_pag_of_items(state):
    g2pag, blk, sb = hq.pag_of_items(state)
    assert g2pag == {10: 1, 11: 1, 20: 2}
    assert blk == {'b10a': 1, 'b10b': 1, 'b11': 1, 'b20': 2}          # b30's grain has no PAG
    assert sb == {'s1': 1, 's2': 1, 's3': 1}                            # s9's block has none


def test_pag_of_items_without_hierarchy():
    assert hq.pag_of_items(SimpleNamespace()) == ({}, {}, {})


# ── phase ─────────────────────────────────────────────────────────────────────

def test_phase_of_pags_blocks_subblocks_and_grains(state):
    assert hq.phase_of_items(state, 'pag', [1, 2]) == {1: 'martensite', 2: 'retained_austenite'}
    assert hq.phase_of_items(state, 'block', ['b10a', 'b20', 'b30', 'unknown']) == {
        'b10a': 'martensite', 'b20': 'retained_austenite', 'b30': 'martensite', 'unknown': 'martensite'}
    assert hq.phase_of_items(state, 'subblock', ['s1', 's9']) == {'s1': 'martensite', 's9': 'martensite'}
    assert hq.phase_of_items(state, 'grain', [10, 20, 99, 77]) == {
        10: 'martensite', 20: 'retained_austenite', 99: 'retained_austenite', 77: 'martensite'}


# ── voxels, orientations ─────────────────────────────────────────────────────

def test_voxels_of_one_item(state):
    np.testing.assert_array_equal(hq.voxels_for_level(state, 'pag', 1), _voxels((0, 0, 0), (1, 0, 0), (2, 0, 0)))
    np.testing.assert_array_equal(hq.voxels_for_level(state, 'block', 'b11'), _voxels((2, 0, 0)))
    np.testing.assert_array_equal(hq.voxels_for_level(state, 'subblock', 's1'), _voxels((0, 0, 0)))
    np.testing.assert_array_equal(hq.voxels_for_level(state, 'grain', 20), _voxels((3, 1, 1), (3, 1, 2)))


@pytest.mark.parametrize('level, item', [('pag', 7), ('block', 'x'), ('subblock', 'x'), ('grain', 5)])
def test_voxels_of_an_unknown_item_are_empty(state, level, item):
    assert hq.voxels_for_level(state, level, item).shape == (0, 3)


def test_voxels_of_a_state_without_the_level():
    assert hq.voxels_for_level(SimpleNamespace(), 'block', 'b').shape == (0, 3)


def test_orientations_for_level(state):
    assert hq.orientations_for_level(state, 'pag') is state.pag_orientations
    assert hq.orientations_for_level(state, 'block') is state.block_orientations
    assert hq.orientations_for_level(state, 'subblock') is state.subblock_orientations
    assert hq.orientations_for_level(state, 'grain') is state.grain_orientations
    assert hq.orientations_for_level(state, 'phase') == {}
    assert hq.orientations_for_level(SimpleNamespace(), 'pag') == {}


# ── ids with orientations ─────────────────────────────────────────────────────

def test_all_ids_with_their_angles(state):
    ids, eulers = hq.collect_ids_and_orientations(state, 'block', state.block_orientations)
    assert ids == ['b10a', 'b10b', 'b11']
    np.testing.assert_array_equal(eulers, [[7, 8, 9], [1, 1, 1], [2, 2, 2]])
    assert eulers.dtype == np.float64


def test_scope_restricts_the_ids(state):
    ids, eulers = hq.collect_ids_and_orientations(state, 'block', state.block_orientations,
                                                   scope_ids={'b11', 'b10a', 'nope'})
    assert ids == ['b10a', 'b11'] and eulers.shape == (2, 3)


def test_slice_restricts_the_ids_to_items_with_a_voxel_in_it(state):
    ids, _ = hq.collect_ids_and_orientations(state, 'block', state.block_orientations, 'X', 1)
    assert ids == ['b10b']
    ids, _ = hq.collect_ids_and_orientations(state, 'pag', state.pag_orientations, 'Z', 2)
    assert ids == [2]                                                   # PAG 2: grain 20 reaches z = 2
    ids, _ = hq.collect_ids_and_orientations(state, 'pag', state.pag_orientations, 'Y', 0)
    assert ids == [1]


def test_a_negative_slice_index_or_missing_axis_does_not_restrict(state):
    every = hq.collect_ids_and_orientations(state, 'block', state.block_orientations)[0]
    assert hq.collect_ids_and_orientations(state, 'block', state.block_orientations, 'X', -1)[0] == every
    assert hq.collect_ids_and_orientations(state, 'block', state.block_orientations, None, 2)[0] == every
    assert hq.collect_ids_and_orientations(state, 'block', state.block_orientations, 'X', None)[0] == every


def test_nothing_selected_gives_an_empty_array(state):
    ids, eulers = hq.collect_ids_and_orientations(state, 'block', state.block_orientations, scope_ids=[])
    assert ids == [] and eulers.shape == (0, 3)


# ── relationships ─────────────────────────────────────────────────────────────

def test_packets_of_pags(state):
    assert hq.packet_ids_for_pags(state, [1]) == {10, 11}
    assert hq.packet_ids_for_pags(state, [1, 2, 3]) == {10, 11, 20}


def test_pags_of_blocks_and_subblocks(state):
    assert hq.pag_ids_for_blocks(state, ['b10a', 'b20', 'b30', 'zz']) == {1, 2}
    assert hq.pag_ids_for_subblocks(state, ['s1', 's3', 's9']) == {1}


def test_packets_of_blocks_and_subblocks(state):
    assert hq.packet_ids_for_blocks(state, ['b10a', 'b10b', 'b30']) == {10, 30}
    assert hq.block_ids_for_subblocks(state, ['s1', 's2', 's3', 'zz']) == {'b10a', 'b11'}
    assert hq.packet_ids_for_subblocks(state, ['s1', 's3']) == {10, 11}


def test_relationships_on_a_state_without_hierarchy():
    empty = SimpleNamespace()
    assert hq.packet_ids_for_pags(empty, [1]) == set()
    assert hq.pag_ids_for_blocks(empty, ['b']) == set()
    assert hq.packet_ids_for_blocks(empty, ['b']) == set()
    assert hq.block_ids_for_subblocks(empty, ['s']) == set()
    assert hq.packet_ids_for_subblocks(empty, ['s']) == set()


def test_packet_from_its_ordinal_within_a_pag(state):
    assert hq.resolve_packet_grain_id(state, 1, 1) == 10
    assert hq.resolve_packet_grain_id(state, 1, 2) == 11
    assert hq.resolve_packet_grain_id(state, 2, 1) == 20
    assert hq.resolve_packet_grain_id(state, 1, 3) is None
    assert hq.resolve_packet_grain_id(state, 9, 1) is None
    assert hq.resolve_packet_grain_id(SimpleNamespace(), 1, 1) is None
