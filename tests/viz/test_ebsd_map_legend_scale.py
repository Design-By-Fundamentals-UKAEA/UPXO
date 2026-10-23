"""Legend outside the map and a length scale on the CSL and parent/twin maps; pole-figure colour bar options."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from upxo.viz.ebsdviz import (
    add_map_scale_bar, plot_combined_parent_twin_map, plot_csl_grain_map, plot_parent_twin_map)
from upxo.viz.xphy.pole_figure import PoleFigure

LFI = np.kron(np.arange(1, 17).reshape(4, 4), np.ones((20, 30), int))
CSL = {'S3': {'grains_all': np.array([1, 2, 5]), 'n_pairs': 4, 'n_grains': 3}}
PI = {'S3': {'pure_parents': np.array([1]), 'pure_twins': np.array([2]), 'intermediates': np.array([5]),
             'n_pure_parents': 1, 'n_pure_twins': 1, 'n_intermediates': 1}}


def _legend_is_outside(fig, ax):
    fig.canvas.draw()
    leg = ax.get_legend().get_window_extent()
    box = ax.get_window_extent()
    return leg.x0 >= box.x1 - 1


@pytest.mark.parametrize("make", [
    lambda: plot_csl_grain_map(LFI, CSL, figsize=(6, 4), step_size=0.5),
    lambda: plot_combined_parent_twin_map(LFI, PI, figsize=(6, 4), step_size=0.5),
])
def test_legend_is_outside_the_map_and_scale_bar_is_drawn(make):
    fig, ax = make()
    assert _legend_is_outside(fig, ax)
    assert any('µm' in t.get_text() for t in ax.texts)
    plt.close(fig)


def test_parent_twin_map_legend_outside_each_subplot():
    fig, axes = plot_parent_twin_map(LFI, PI, figsize=(6, 4), step_size=0.5)
    assert _legend_is_outside(fig, axes[0, 0])
    plt.close(fig)


def test_no_scale_bar_without_a_step_size():
    fig, ax = plot_csl_grain_map(LFI, CSL, figsize=(6, 4))
    assert not ax.texts
    plt.close(fig)


def test_scale_bar_length_is_a_round_value_near_a_fifth_of_the_width():
    fig, ax = plt.subplots()
    ax.imshow(np.zeros((100, 200)))
    length = add_map_scale_bar(ax, 0.5, (100, 200))          # width 100 um -> target 20
    assert length == 20
    plt.close(fig)


def _pf():
    q = np.random.default_rng(0).normal(size=(40, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return PoleFigure(q, convention='quaternion')


@pytest.mark.parametrize("orientation", ["vertical", "horizontal"])
def test_pole_figure_colourbar_title_and_orientation(orientation):
    fig, ax = plt.subplots()
    _pf().plot_density(pole_family='111', ax=ax, grid_points=30,
                       colorbar_label="my bar", colorbar_orientation=orientation)
    cax = fig.axes[-1]
    label = cax.get_ylabel() if orientation == "vertical" else cax.get_xlabel()
    assert label == "my bar"
    w, h = cax.get_position().width, cax.get_position().height
    assert (h > w) == (orientation == "vertical")
    plt.close(fig)


def test_pole_figure_colourbar_default_label_unchanged():
    fig, ax = plt.subplots()
    _pf().plot_density(pole_family='111', ax=ax, grid_points=30)
    assert fig.axes[-1].get_ylabel() == 'MUD (Multiples of Uniform Distribution)'
    plt.close(fig)
