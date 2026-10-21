"""Colour composites of the FM steel hierarchy."""
import colorsys

import matplotlib
import numpy as np
import pytest

from upxo.pxtal.fm_steel_3d import hierarchy_colouring as hc


def _hsv(rgb):
    return colorsys.rgb_to_hsv(*[float(c) for c in rgb])


# ── hues and lightness ───────────────────────────────────────────────────────

def test_packet_hues_come_from_the_colour_map():
    cmap = matplotlib.colormaps['Set1']
    hues = hc.packet_hues('Set1')
    assert len(hues) == hc.N_HABIT_PLANES == 4
    for i, hue in enumerate(hues):
        assert hue == pytest.approx(_hsv(cmap(i / 3.0)[:3])[0])
    assert hc.packet_hues(cmap) == hues                       # a colour map object works as well


@pytest.mark.parametrize('position, count, expected', [
    (0, 1, 0.60), (0, 2, 0.35), (1, 2, 0.90), (0, 3, 0.35), (1, 3, 0.625), (2, 3, 0.90), (0, 0, 0.60),
])
def test_lightness_by_position(position, count, expected):
    assert hc.order_lightness(position, count) == pytest.approx(expected)


# ── PAGs and isolated grains ─────────────────────────────────────────────────

def _grid():
    grain = np.zeros((4, 1, 1), dtype=int)
    grain[:, 0, 0] = [1, 2, 3, 0]                  # three grains and a void voxel
    pag = np.zeros_like(grain)
    pag[:, 0, 0] = [1, 2, 0, 0]                    # grain 3 is in no PAG
    return grain, pag


def test_pags_and_isolated_grains_are_coloured_from_their_maps():
    grain, pag = _grid()
    rgb = hc.base_composite(grain, pag, 'viridis', 'Greys')
    viridis, greys = matplotlib.colormaps['viridis'], matplotlib.colormaps['Greys']
    assert rgb.shape == (4, 1, 1, 3) and rgb.dtype == np.float32
    np.testing.assert_allclose(rgb[0, 0, 0], viridis(0.0)[:3], atol=1e-6)        # PAG 1 of 2
    np.testing.assert_allclose(rgb[1, 0, 0], viridis(1.0)[:3], atol=1e-6)        # PAG 2 of 2
    np.testing.assert_allclose(rgb[2, 0, 0], greys((3 - 1) / max(1, 3 - 1))[:3], atol=1e-6)   # last of 3 grains
    assert (rgb[3, 0, 0] == 0).all()                                              # void


def test_a_single_pag_takes_the_start_of_the_colour_map():
    grain = np.ones((2, 2, 2), dtype=int)
    rgb = hc.base_composite(grain, grain, 'viridis', 'Greys')
    np.testing.assert_allclose(rgb[0, 0, 0], matplotlib.colormaps['viridis'](0.0)[:3], atol=1e-6)


def test_without_isolated_grains_or_pags_everything_is_black():
    zero = np.zeros((2, 2, 2), dtype=int)
    assert (hc.base_composite(zero, zero, 'viridis', 'Greys') == 0).all()


# ── blocks ────────────────────────────────────────────────────────────────────

def _blocks():
    return {
        '1_1_5': np.array([[0, 0, 0]]),
        '1_1_5_b': np.array([[1, 0, 0]]),
        '2_1_6': np.array([[2, 0, 0]]),
        '2_1_9': np.empty((0, 3), dtype=int),
    }


def test_blocks_take_the_hue_of_their_packet_with_one_lightness():
    rgb = np.zeros((3, 1, 1, 3), dtype=np.float32)
    hues = [0.0, 0.25, 0.5, 0.75]
    hc.paint_blocks(rgb, _blocks(), {5: 1, 6: 6}, hues)               # plane 6 wraps to 2
    assert _hsv(rgb[0, 0, 0]) == pytest.approx((0.25, 0.85, 0.60), abs=1e-5)
    assert _hsv(rgb[1, 0, 0]) == pytest.approx((0.25, 0.85, 0.60), abs=1e-5)
    assert _hsv(rgb[2, 0, 0]) == pytest.approx((0.5, 0.85, 0.60), abs=1e-5)


def test_lightness_follows_the_blocks_of_the_grain_when_they_are_given():
    rgb = np.zeros((3, 1, 1, 3), dtype=np.float32)
    hc.paint_blocks(rgb, _blocks(), {5: 0, 6: 0}, [0.1, 0.2, 0.3, 0.4], {5: ['1_1_5', '1_1_5_b'], 6: ['2_1_6']})
    assert _hsv(rgb[0, 0, 0])[2] == pytest.approx(0.35, abs=1e-5)
    assert _hsv(rgb[1, 0, 0])[2] == pytest.approx(0.90, abs=1e-5)
    assert _hsv(rgb[2, 0, 0])[2] == pytest.approx(0.60, abs=1e-5)    # the only block of its grain


def test_a_grain_without_a_plane_index_uses_the_first_hue():
    rgb = np.zeros((3, 1, 1, 3), dtype=np.float32)
    hc.paint_blocks(rgb, _blocks(), {}, [0.4, 0.2, 0.3, 0.1])
    assert _hsv(rgb[0, 0, 0])[0] == pytest.approx(0.4, abs=1e-5)


# ── sub-blocks ────────────────────────────────────────────────────────────────

def _subblocks():
    return ({'SB_1_1_5_1': np.array([[0, 0, 0]]), 'SB_1_1_5_2': np.array([[1, 0, 0]]),
             'SB_orphan': np.array([[2, 0, 0]])},
            {'1_1_5': ['SB_1_1_5_1', 'SB_1_1_5_2']})


def test_subblocks_are_graded_within_their_block():
    subblocks, mapping = _subblocks()
    rgb = np.zeros((3, 1, 1, 3), dtype=np.float32)
    hc.paint_subblocks(rgb, subblocks, mapping, {5: 3}, [0.1, 0.2, 0.3, 0.7])
    assert _hsv(rgb[0, 0, 0]) == pytest.approx((0.7, 0.85, 0.35), abs=1e-5)
    assert _hsv(rgb[1, 0, 0]) == pytest.approx((0.7, 0.85, 0.90), abs=1e-5)
    assert (rgb[2, 0, 0] == 0).all()                                  # in no block: untouched


def test_the_colour_table_agrees_with_painting():
    subblocks, mapping = _subblocks()
    hues = [0.1, 0.2, 0.3, 0.7]
    table = hc.subblock_hue_table({1: 'SB_1_1_5_1', 4: 'SB_1_1_5_2', 2: 'SB_orphan'}, mapping, {5: 3}, hues)
    rgb = np.zeros((3, 1, 1, 3), dtype=np.float32)
    hc.paint_subblocks(rgb, subblocks, mapping, {5: 3}, hues)
    assert table.shape == (6, 3) and table.dtype == np.float32           # largest label 4, plus 2
    np.testing.assert_allclose(table[1], rgb[0, 0, 0], atol=1e-6)
    np.testing.assert_allclose(table[4], rgb[1, 0, 0], atol=1e-6)
    assert (table[2] == 0).all() and (table[0] == 0).all() and (table[3] == 0).all()


def test_the_colour_table_without_labels():
    assert hc.subblock_hue_table({}, {}, {}, [0.1, 0.2, 0.3, 0.4]).shape == (2, 3)


# ── composites ────────────────────────────────────────────────────────────────

def _structure():
    grain = np.zeros((4, 1, 1), dtype=int)
    grain[:, 0, 0] = [5, 5, 6, 0]
    pag = np.zeros_like(grain)
    pag[:, 0, 0] = [1, 1, 2, 0]
    blocks = {'1_1_5': np.array([[0, 0, 0]]), '1_1_5_b': np.array([[1, 0, 0]])}      # voxel 2 has no block
    return grain, pag, blocks


def test_block_level_composite():
    grain, pag, blocks = _structure()
    rgb = hc.block_level_composite(grain, pag, blocks, {5: 0}, {5: ['1_1_5', '1_1_5_b']}, 'viridis', 'Greys', 'tab10')
    hue = hc.packet_hues('tab10')[0]
    assert _hsv(rgb[0, 0, 0]) == pytest.approx((hue, 0.85, 0.35), abs=1e-5)
    assert _hsv(rgb[1, 0, 0]) == pytest.approx((hue, 0.85, 0.90), abs=1e-5)
    np.testing.assert_allclose(rgb[2, 0, 0], matplotlib.colormaps['viridis'](1.0)[:3], atol=1e-6)   # PAG only
    assert (rgb[3, 0, 0] == 0).all()


def test_subblock_level_composite_paints_subblocks_over_blocks():
    grain, pag, blocks = _structure()
    subblocks = {'SB_1_1_5_1': np.array([[0, 0, 0]])}
    rgb = hc.subblock_level_composite(grain, pag, blocks, subblocks, {'1_1_5': ['SB_1_1_5_1', 'SB_1_1_5_2']},
                                      {5: 2}, 'viridis', 'Greys', 'tab10')
    hue = hc.packet_hues('tab10')[2]
    assert _hsv(rgb[0, 0, 0]) == pytest.approx((hue, 0.85, 0.35), abs=1e-5)      # sub-block wins
    assert _hsv(rgb[1, 0, 0]) == pytest.approx((hue, 0.85, 0.60), abs=1e-5)      # block, no sub-block


# ── flattening ────────────────────────────────────────────────────────────────

def test_flatten_follows_fortran_order():
    rgb = np.arange(2 * 3 * 4 * 3, dtype=np.float32).reshape(2, 3, 4, 3)
    flat = hc.flatten_rgb(rgb)
    assert flat.shape == (24, 3)
    for index, (i, j, k) in enumerate(np.ndindex(4, 3, 2)):
        np.testing.assert_array_equal(flat[index], rgb[k, j, i])
