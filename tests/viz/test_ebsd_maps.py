import numpy as np
import pytest

from upxo._sup import progress
from upxo.xtalphy.gb_disorientation_map import (
    compute_gb_disorientation, pair_disorientation_deg)
from upxo.viz import ebsd_maps


def _rot_z(deg):
    half = np.radians(deg) / 2
    return np.array([np.cos(half), 0.0, 0.0, np.sin(half)])


def _three_grain_map(n=24):
    lfi = np.ones((n, n), dtype=np.int32)
    lfi[:, n // 2:] = 2
    lfi[n // 2:, : n // 2] = 3
    quat = np.zeros((n, n, 4))
    quat[lfi == 1] = _rot_z(0)
    quat[lfi == 2] = _rot_z(30)
    quat[lfi == 3] = _rot_z(15)
    return lfi, quat


def test_two_grains_give_one_shared_line_with_the_right_angle():
    lfi = np.ones((20, 20), dtype=np.int32)
    lfi[:, 10:] = 2
    quat = np.zeros((20, 20, 4))
    quat[lfi == 1] = _rot_z(0)
    quat[lfi == 2] = _rot_z(30)
    result = compute_gb_disorientation(lfi, quat)
    shared = ~np.isnan(result.angles)
    assert shared.sum() == 1
    assert result.angles[shared][0] == pytest.approx(30.0, abs=1e-6)
    line = result.segments[int(np.flatnonzero(shared)[0])]
    assert np.allclose(line[:, 0], 9.5, atol=0.3)          # between columns 9 and 10
    assert line[:, 1].min() == pytest.approx(-0.5, abs=0.3)
    assert line[:, 1].max() == pytest.approx(19.5, abs=0.3)
    assert np.isnan(result.angles).sum() >= 2               # map-edge lines carry no angle


def test_each_pair_in_a_three_grain_map_has_its_own_angle():
    lfi, quat = _three_grain_map()
    result = compute_gb_disorientation(lfi, quat)
    by_pair = {}
    for pair, angle in zip(map(tuple, result.pairs), result.angles):
        if pair[1] > 0:
            by_pair.setdefault(pair, set()).add(round(float(angle), 4))
    # grain ids after relabelling follow the connected pieces; angles are 15, 15 and 30
    assert sorted(a for v in by_pair.values() for a in v) == [15.0, 15.0, 30.0]
    assert result.n_grains == 3


def test_a_grain_in_two_disconnected_pieces_becomes_two_grains():
    lfi = np.ones((10, 21), dtype=np.int32)
    lfi[:, 8:13] = 2                                         # grain 1 is left and right of grain 2
    quat = np.zeros((10, 21, 4))
    quat[lfi == 1] = _rot_z(0)
    quat[lfi == 2] = _rot_z(20)
    result = compute_gb_disorientation(lfi, quat)
    assert result.n_grains == 3
    assert sorted(a for a in result.angles if not np.isnan(a)) == pytest.approx([20.0, 20.0])


def test_progress_reaches_every_stage_and_ends_complete():
    lfi, quat = _three_grain_map()
    sink = progress.ListSink()
    with progress.progress_scope(sink):
        compute_gb_disorientation(lfi, quat)
    stages = {e.stage for e in sink.events}
    assert {"Preparing grains", "Geometrifying the grain map", "Smoothing boundaries",
            "Disorientation"} <= stages
    fractions = [e.fraction for e in sink.events if e.fraction is not None]
    assert fractions == sorted(fractions) and fractions[-1] == pytest.approx(1.0)


def test_unlabelled_pixels_and_bad_arguments_are_rejected():
    lfi, quat = _three_grain_map()
    bad = lfi.copy()
    bad[0, 0] = 0
    with pytest.raises(ValueError):
        compute_gb_disorientation(bad, quat)
    with pytest.raises(ValueError):
        compute_gb_disorientation(lfi, quat[:, :-1])
    with pytest.raises(ValueError):
        compute_gb_disorientation(lfi, quat, smooth_passes=3)


def test_pair_disorientation_uses_cubic_symmetry_and_chunks():
    from upxo.xtalphy.crystal_orientation import quat_to_R_batch
    quats = np.stack([_rot_z(0), _rot_z(80), _rot_z(45)])
    R = quat_to_R_batch(quats)
    a = np.array([0, 0, 1] * 700)
    b = np.array([1, 2, 2] * 700)
    out = pair_disorientation_deg(R, a, b, chunk=500)
    assert out[:3] == pytest.approx([10.0, 45.0, 35.0], abs=1e-6)     # 80 deg is 10 deg from a 90-deg symmetry
    assert out.shape == (2100,)


def test_figures_are_built_without_showing_anything():
    lfi, quat = _three_grain_map()
    euler = np.zeros((*lfi.shape, 3))
    euler[..., 0] = np.radians(lfi * 40.0)
    fig = ebsd_maps.euler_map_figure(lfi, euler, 0, step_um=0.5)
    assert fig.axes[0].get_title() == "Bunge phi1 (deg)"
    ipf = ebsd_maps.ipf_map_figure(lfi, quat, (0, 0, 1), "Z (ND)", step_um=0.5)
    assert ipf.axes[0].get_title() == "IPF map, sample direction Z (ND)"
    assert "Grain-boundary pixels" in ebsd_maps.gb_pixel_map_figure(lfi).axes[0].get_title()

    result = compute_gb_disorientation(lfi, quat)
    fig = ebsd_maps.gb_disorientation_figure(result, cmap="plasma", step_um=0.5, vmin=0, vmax=40)
    ax = fig.axes[0]
    assert ax.get_ylim()[0] > ax.get_ylim()[1]                   # y points down
    assert any(getattr(c, "get_array", lambda: None)() is not None and len(c.get_array())
               for c in ax.collections)
    assert fig.axes[1].get_ylabel() == "Disorientation angle (deg)"


def _twin_layout():
    """A | T1 | T2 | B over one long grain C. Original labels: A=10 T1=20 T2=30 B=40 C=50."""
    lfi = np.full((24, 24), 50, dtype=np.int32)
    lfi[:12, 0:6], lfi[:12, 6:12], lfi[:12, 12:18], lfi[:12, 18:24] = 10, 20, 30, 40
    quat = np.zeros((24, 24, 4))
    for gid, deg in ((10, 0), (20, 20), (30, 45), (40, 10), (50, 30)):
        quat[lfi == gid] = _rot_z(deg)
    return lfi, quat


def _line_collections(ax):
    return [c for c in ax.collections if c.get_array() is not None]


def test_boundaries_are_classified_by_twin_status():
    from upxo.xtalphy.gb_disorientation_map import (
        classify_boundaries, EDGE, GRAIN_GRAIN, TWIN_GRAIN, TWIN_TWIN)
    lfi, quat = _twin_layout()
    result = compute_gb_disorientation(lfi, quat)
    kinds = classify_boundaries(result, twin_ids={20, 30})
    counts = {k: int((kinds == k).sum()) for k in (EDGE, GRAIN_GRAIN, TWIN_GRAIN, TWIN_TWIN)}
    assert counts[TWIN_TWIN] == 1 and counts[GRAIN_GRAIN] == 2 and counts[TWIN_GRAIN] == 4
    assert counts[EDGE] >= 4
    # a relabelled piece of a twin grain still counts as a twin
    assert set(result.original_ids[1:].tolist()) == {10, 20, 30, 40, 50}


def test_line_styles_follow_boundary_type_with_one_line_width():
    lfi, quat = _twin_layout()
    result = compute_gb_disorientation(lfi, quat)
    fig = ebsd_maps.gb_disorientation_figure(result, twin_ids={20, 30}, style_by_type=True,
                                             linewidth=1.7)
    collections = _line_collections(fig.axes[0])
    assert len(collections) == 3
    styles = {c.get_linestyle()[0][1] is None or tuple(c.get_linestyle()[0][1]) for c in collections}
    assert len(styles) == 3                                   # solid, dashed and dotted differ
    assert {tuple(c.get_linewidth()) for c in collections} == {(1.7,)}
    assert sorted(len(c.get_segments()) for c in collections) == [1, 2, 4]
    assert [t.get_text() for t in fig.legends[0].get_texts()] == [
        "twin - twin", "grain - grain", "twin - grain"]
    plain = ebsd_maps.gb_disorientation_figure(result)
    assert len(_line_collections(plain.axes[0])) == 1 and not plain.legends


def test_only_in_range_drops_lines_outside_the_range():
    lfi, quat = _three_grain_map()                             # boundary angles 15, 15, 30
    result = compute_gb_disorientation(lfi, quat)

    def drawn(**kw):
        fig = ebsd_maps.gb_disorientation_figure(result, **kw)
        return sum(len(c.get_segments()) for c in _line_collections(fig.axes[0]))

    assert drawn(vmin=20, vmax=40) == 3                         # default: out-of-range lines stay, clamped
    assert drawn(vmin=20, vmax=40, only_in_range=True) == 1
    assert drawn(vmin=10, vmax=20, only_in_range=True) == 2
    assert drawn(vmin=40, vmax=60, only_in_range=True) == 0     # nothing left, no crash
    assert drawn(only_in_range=True) == 3                       # auto range keeps everything


@pytest.mark.parametrize("where", ["right", "left", "top", "bottom"])
def test_colour_bar_goes_where_asked_with_its_text(where):
    lfi, quat = _three_grain_map()
    result = compute_gb_disorientation(lfi, quat)
    fig = ebsd_maps.gb_disorientation_figure(
        result, colorbar_location=where, colorbar_label="Angle", colorbar_label_size=13,
        colorbar_tick_size=8, colorbar_length=0.6, colorbar_pad=0.1)
    fig.canvas.draw()
    plot, bar = fig.axes[0].get_position(), fig.axes[1].get_position()
    if where == "right":
        assert bar.x0 >= plot.x1 - 1e-6
    elif where == "left":
        assert bar.x1 <= plot.x0 + 1e-6
    elif where == "top":
        assert bar.y0 >= plot.y1 - 1e-6
    else:
        assert bar.y1 <= plot.y0 + 1e-6
    axis = fig.axes[1].xaxis if where in ("top", "bottom") else fig.axes[1].yaxis
    assert axis.label.get_text() == "Angle" and axis.label.get_fontsize() == 13
    assert axis.get_ticklabels()[0].get_fontsize() == 8
    vertical = where in ("left", "right")
    extent = bar.height if vertical else bar.width
    assert extent < (plot.height if vertical else plot.width)    # shortened by the length setting


def _overlaps(renderer, transform, a, b):
    bb_a = a.get_window_extent(renderer).transformed(transform)
    bb_b = b.get_window_extent(renderer).transformed(transform)
    return not (bb_a.x1 <= bb_b.x0 or bb_b.x1 <= bb_a.x0 or bb_a.y1 <= bb_b.y0 or bb_b.y1 <= bb_a.y0)


@pytest.mark.parametrize("where", ["right", "left", "top", "bottom"])
def test_legend_never_overlaps_the_colour_bar_or_the_title(where):
    lfi, quat = _twin_layout()
    result = compute_gb_disorientation(lfi, quat)
    fig = ebsd_maps.gb_disorientation_figure(result, twin_ids={20, 30}, style_by_type=True,
                                             colorbar_location=where)
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    renderer = canvas.get_renderer()
    legend = fig.legends[0]
    bar_ax = fig.axes[1]
    tf = fig.transFigure.inverted()
    assert not _overlaps(renderer, tf, legend, bar_ax)
    assert not _overlaps(renderer, tf, legend, fig._suptitle)
    leg_bb = legend.get_window_extent(renderer).transformed(tf)
    assert -0.01 <= leg_bb.x0 and leg_bb.x1 <= 1.01 and -0.01 <= leg_bb.y0 and leg_bb.y1 <= 1.01


def test_bad_colour_bar_location_is_rejected():
    lfi, quat = _three_grain_map()
    with pytest.raises(ValueError):
        ebsd_maps.gb_disorientation_figure(compute_gb_disorientation(lfi, quat),
                                           colorbar_location="middle")


# ── Pixellated (subsampling) / Smoothed (polygon-fill) display modes ────

def test_euler_map_figure_rejects_bad_angle_index():
    lfi, quat = _three_grain_map()
    euler = np.zeros((*lfi.shape, 3))
    with pytest.raises(ValueError, match="angle_index"):
        ebsd_maps.euler_map_figure(lfi, euler, 3)


def test_pixel_stride_shrinks_the_rendered_array_same_physical_extent():
    lfi, quat = _three_grain_map(n=24)
    euler = np.zeros((*lfi.shape, 3))
    euler[..., 0] = np.radians(lfi * 40.0)
    full = ebsd_maps.euler_map_figure(lfi, euler, 0, step_um=1.0, pixel_stride=1)
    strided = ebsd_maps.euler_map_figure(lfi, euler, 0, step_um=1.0, pixel_stride=2)
    full_img = full.axes[0].images[0]
    strided_img = strided.axes[0].images[0]
    assert strided_img.get_array().shape[0] < full_img.get_array().shape[0]
    # same physical extent despite fewer pixels
    assert full_img.get_extent() == pytest.approx(strided_img.get_extent(), abs=1.0)


def test_smoothed_euler_map_fills_polygons_with_per_grain_mean():
    from upxo.viz.ebsd_smoothing import smooth_technique_a
    lfi, quat = _three_grain_map()
    euler = np.zeros((*lfi.shape, 3))
    euler[..., 0] = np.radians(lfi * 40.0)   # grain 1=0deg, 2=80deg, 3=120deg
    polygons, grains, _gb_rings = smooth_technique_a(lfi, smooth_passes=1, max_smooth_levels=(3,))
    fig = ebsd_maps.euler_map_figure(lfi, euler, 0, polygons=polygons, grains=grains)
    ax = fig.axes[0]
    # no raster image in smoothed mode, but real filled patches instead
    assert not ax.images
    assert len(ax.patches) == len(polygons)


def test_smoothed_ipf_map_reuses_the_raster_colour_per_grain():
    from upxo.viz.ebsd_smoothing import smooth_technique_b
    lfi, quat = _three_grain_map()
    polygons, grains, _gb_ridges = smooth_technique_b(lfi, seeds_per_grain=10, rng_seed=0)
    fig = ebsd_maps.ipf_map_figure(lfi, quat, (0, 0, 1), "Z (ND)",
                                   polygons=polygons, smooth_grains=grains)
    ax = fig.axes[0]
    assert not ax.images
    assert len(ax.patches) == len(polygons)


def test_smoothed_gb_pixel_map_colours_by_grain_id():
    from upxo.viz.ebsd_smoothing import smooth_technique_a
    lfi, quat = _three_grain_map()
    polygons, grains, _gb_rings = smooth_technique_a(lfi, smooth_passes=1, max_smooth_levels=(3,))
    fig = ebsd_maps.gb_pixel_map_figure(lfi, polygons=polygons)
    ax = fig.axes[0]
    assert not ax.images
    assert len(ax.patches) == len(polygons)
    assert "smoothed" in ax.get_title().lower()


def test_polygon_fill_overlay_handles_a_hole_without_filling_it_solid():
    from shapely.geometry import Polygon
    from upxo.viz.ebsd_maps import _polygon_fill_overlay
    import matplotlib.pyplot as plt
    outer = [(0, 0), (10, 0), (10, 10), (0, 10)]
    hole = [(4, 4), (6, 4), (6, 6), (4, 6)]
    donut = Polygon(outer, [hole])
    fig, ax = plt.subplots()
    _polygon_fill_overlay(ax, {1: donut}, {1: (1.0, 0.0, 0.0, 1.0)})
    assert len(ax.patches) == 1
    path = ax.patches[0].get_path()
    # exterior (4 verts + close) + interior (4 verts + close) = 10 vertices
    assert len(path.vertices) == 10
    plt.close(fig)
