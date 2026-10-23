"""Boundary and junction overlays, the subsampled smoothed structure, texture component assignment and map."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from upxo.viz.ebsd_maps import euler_map_figure, texture_component_map_figure
from upxo.viz.ebsd_overlays import junction_points, overlay_grain_boundaries, overlay_junctions
from upxo.viz.ebsd_smoothing import smooth_structure
from upxo.xtalphy.crystal_orientation import (
    assign_grains_to_texture_components, euler_bunge_to_matrix, rotation_matrices_to_quats)


def _four_grains(n=20):
    lfi = np.ones((n, n), int)
    lfi[:, n // 2:] = 2
    lfi[n // 2:, :] = 3
    lfi[n // 2:, n // 2:] = 4
    return lfi


def test_four_corner_junction_is_found_as_quadruple():
    t, q = junction_points(_four_grains())
    assert len(t) == 0 and q.tolist() == [[9.5, 9.5]]


def test_triple_junction_is_found():
    lfi = np.ones((10, 10), int)
    lfi[:, 5:] = 2
    lfi[5:, :5] = 3
    t, q = junction_points(lfi)
    assert len(q) == 0 and t.tolist() == [[4.5, 4.5]]


def test_no_junctions_in_two_grains():
    lfi = np.ones((10, 10), int)
    lfi[:, 5:] = 2
    t, q = junction_points(lfi)
    assert len(t) == 0 and len(q) == 0


def test_overlay_counts_and_modes():
    fig, ax = plt.subplots()
    lfi = _four_grains()
    ax.imshow(lfi)
    assert overlay_grain_boundaries(ax, lfi) == 'pixel'
    assert overlay_junctions(ax, lfi, triple=True, quadruple=False) == 0
    assert overlay_junctions(ax, lfi, marker='square', size=30, edgecolor='red', facecolor='white') == 1
    plt.close(fig)


def test_smoothed_overlay_uses_the_polygons():
    lfi = _four_grains(30)
    s = smooth_structure(lfi, "A", (1, (1,)), subsample=1)
    fig, ax = plt.subplots()
    assert overlay_grain_boundaries(ax, lfi, polygons=s['polygons'], linewidth=1.5) == 'smoothed'
    plt.close(fig)


@pytest.mark.parametrize("technique, params", [("A", (1, (1,))), ("B", (3,))])
def test_subsampled_smoothing_is_back_in_full_resolution_coordinates(technique, params):
    lfi = np.kron(_four_grains(10), np.ones((4, 4), int))                    # 40 x 40
    s = smooth_structure(lfi, technique, params, subsample=2)
    assert s['subsample'] == 2 and s['grains'].shape == lfi.shape
    xmax = max(p.bounds[2] for p in s['polygons'].values())
    assert xmax > 30                                                       # not the 20-pixel subsampled extent


def test_smoothing_rejects_a_bad_subsample():
    with pytest.raises(ValueError):
        smooth_structure(_four_grains(), "A", (1, (1,)), subsample=0)


def _g_quat(euler):
    return rotation_matrices_to_quats(euler_bunge_to_matrix(*euler).T[None])[0]


def test_grains_go_to_the_nearest_component_with_a_cutoff():
    comps = [(0.0, 0.0, 0.0), (90.0, 35.0, 45.0)]
    quats = np.array([_g_quat((3.0, 2.0, 1.0)), _g_quat((92.0, 33.0, 46.0)), _g_quat((40.0, 70.0, 10.0))])
    idx, ang = assign_grains_to_texture_components(quats, comps, max_angle_deg=15.0)
    assert idx.tolist() == [0, 1, -1]
    assert ang[0] < 15 and ang[1] < 15 and ang[2] > 15


def test_a_cubic_equivalent_of_a_component_is_assigned_to_it():
    from upxo.xtalphy.crystal_orientation import get_cubic_ops_np
    g = euler_bunge_to_matrix(90.0, 35.0, 45.0).T
    q = rotation_matrices_to_quats((get_cubic_ops_np()[7] @ g)[None])
    idx, ang = assign_grains_to_texture_components(q, [(90.0, 35.0, 45.0)])
    assert idx[0] == 0 and ang[0] < 1e-6


def test_texture_component_map_has_outside_legend_and_scale_bar():
    lfi = _four_grains()
    fig = texture_component_map_figure(lfi, [1, 2, 3, 4], [0, 1, -1, 0], ["Cube", "Goss"], step_um=0.5)
    ax = fig.axes[0]
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert labels == ["Cube", "Goss", "Other"]
    assert any('\u00b5m' in t.get_text() for t in ax.texts)
    fig.canvas.draw()
    assert ax.get_legend().get_window_extent().x0 >= ax.get_window_extent().x1 - 1


def test_euler_map_options():
    lfi = _four_grains()
    euler = np.zeros((*lfi.shape, 3)) + 0.5
    fig = euler_map_figure(lfi, euler, 0, step_um=0.5, title="my title", show_scale_bar=True,
                           font_sizes={'title': 15}, xlim=(0, 5), ylim=(0, 4))
    ax = fig.axes[0]
    assert ax.get_title() == "my title" and ax.title.get_fontsize() == 15
    assert ax.get_xlim() == (0, 5) and ax.get_ylim() == (4, 0)
    assert any('\u00b5m' in t.get_text() for t in ax.texts)
    hidden = euler_map_figure(lfi, euler, 0, show_axes=False)
    assert not hidden.axes[0].axison
