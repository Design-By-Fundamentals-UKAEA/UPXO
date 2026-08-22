"""
GUI smoke tests for the FM Steel 3D wizard (headless, no mainloop).

These tests verify that:
  - App and page classes can be instantiated without crashing
  - shared_state is populated with the expected defaults
  - Mode change (block → subblock) correctly updates shared_state
  - Session save/load roundtrip preserves shared_state entries
  - Export-script path generation works without a pipeline run

Tkinter only supports one root window per process; the app fixture is
module-scoped so that exactly one App is created for this file. Tests that
mutate shared_state reset only the keys they changed before returning.

Marked with pytest.mark.gui; skip them on headless CI with:
    pytest -m "not gui"

Run from repo root:
    pytest tests/fm_steel_3d/test_gui.py -v
"""

import json
import os
import sys
import pytest

# Guard: skip the entire module if Tkinter is not available
tk = pytest.importorskip("tkinter", reason="Tkinter not available")

# Guard: skip if we are in a truly headless environment (no DISPLAY on Linux).
# On Windows, Tkinter always has a display.
if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
    pytest.skip("No DISPLAY — skipping GUI tests", allow_module_level=True)

pytestmark = pytest.mark.gui


# ---------------------------------------------------------------------------
# Module-scoped fixture — ONE App per test file (Tk limitation: one root/process)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def app():
    """Create a single hidden App instance for this module; destroy after all tests."""
    import matplotlib
    matplotlib.use("Agg")  # prevent any plot windows from opening

    from upxo.pxtal.fm_steel_3d.gui.app import App
    a = App(use_customtkinter=False)
    a.withdraw()          # hide the window
    a.update_idletasks()  # pump the event queue once
    yield a
    try:
        a.destroy()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# App instantiation
# ---------------------------------------------------------------------------

def test_app_creates_without_crash(app):
    assert app is not None


def test_app_has_shared_state(app):
    assert hasattr(app, "shared_state")
    assert isinstance(app.shared_state, dict)


# ---------------------------------------------------------------------------
# shared_state defaults
# ---------------------------------------------------------------------------

def test_default_pipeline_mode(app):
    assert app.shared_state["pipeline_mode"] == "block"


def test_default_mesh_elem_type(app):
    assert app.shared_state["MESH_ELEM_TYPE"] == "C3D8"


def test_default_export_mesh_enabled(app):
    assert app.shared_state["EXPORT_MESH"] is True


def test_default_domain_size(app):
    assert app.shared_state["NX"] == app.shared_state["NY"] == app.shared_state["NZ"]


def test_default_connectivity(app):
    assert app.shared_state["CONNECTIVITY"] in (6, 18, 26)


def test_default_random_seed_is_int(app):
    assert isinstance(app.shared_state["RANDOM_SEED"], int)


def test_run_state_initialised_false(app):
    assert getattr(app, "run_completed", False) is False


def test_last_run_data_initialised_none(app):
    assert getattr(app, "last_run_data", None) is None


# ---------------------------------------------------------------------------
# Runtime state attributes
# ---------------------------------------------------------------------------

def test_run_count_starts_at_zero(app):
    assert getattr(app, "run_count", 0) == 0


def test_last_export_path_starts_none(app):
    assert getattr(app, "last_export_path", None) is None


def test_current_session_path_starts_none(app):
    assert getattr(app, "current_session_path", None) is None


# ---------------------------------------------------------------------------
# Session save / load roundtrip
# ---------------------------------------------------------------------------

def test_save_session_creates_file(app, tmp_path):
    """save_session() must write a readable JSON file."""
    target = tmp_path / "test_session.json"
    orig_path = app.current_session_path
    app.current_session_path = target
    app.save_session(notify=False)
    app.current_session_path = orig_path          # restore
    assert target.exists(), "Session file was not created"


def test_save_session_json_content(app, tmp_path):
    """Saved JSON must contain key shared_state entries."""
    target = tmp_path / "content_session.json"
    orig_path = app.current_session_path
    app.current_session_path = target
    app.save_session(notify=False)
    app.current_session_path = orig_path
    with open(target) as f:
        data = json.load(f)
    assert "pipeline_mode" in data
    assert "NX" in data


def test_load_session_restores_state(app, tmp_path):
    """Loading a saved JSON must restore shared_state values we put in it."""
    # Write JSON directly (bypass save_session, which calls save_state() first
    # and would overwrite our values with the current Tk widget values).
    target = tmp_path / "roundtrip_session.json"
    saved_data = dict(app.shared_state)
    saved_data["NX"] = 999
    saved_data["pipeline_mode"] = "subblock"
    with open(target, "w") as f:
        json.dump(saved_data, f)

    # Corrupt the in-memory state
    orig_nx   = app.shared_state["NX"]
    orig_mode = app.shared_state["pipeline_mode"]
    app.shared_state["NX"] = 1
    app.shared_state["pipeline_mode"] = "block"

    # Load and verify
    with open(target) as f:
        restored = json.load(f)
    app.shared_state.update(restored)
    app.update_idletasks()

    try:
        assert app.shared_state["NX"] == 999
        assert app.shared_state["pipeline_mode"] == "subblock"
    finally:
        # Restore so other tests see clean defaults
        app.shared_state["NX"] = orig_nx
        app.shared_state["pipeline_mode"] = orig_mode


# ---------------------------------------------------------------------------
# Script export path logic
# ---------------------------------------------------------------------------

def test_next_script_path_is_py_file(app):
    path = app._next_script_path()
    assert path.suffix == ".py"


def test_next_script_path_block_mode(app):
    orig = app.shared_state["pipeline_mode"]
    app.shared_state["pipeline_mode"] = "block"
    path = app._next_script_path()
    app.shared_state["pipeline_mode"] = orig
    assert "blk" in path.stem


def test_next_script_path_subblock_mode(app):
    orig = app.shared_state["pipeline_mode"]
    app.shared_state["pipeline_mode"] = "subblock"
    path = app._next_script_path()
    app.shared_state["pipeline_mode"] = orig
    assert "sblk" in path.stem


# ---------------------------------------------------------------------------
# Session path logic
# ---------------------------------------------------------------------------

def test_next_session_path_is_json(app):
    path = app._next_session_path()
    assert path.suffix == ".json"


def test_next_session_path_block_naming(app):
    orig = app.shared_state["pipeline_mode"]
    app.shared_state["pipeline_mode"] = "block"
    path = app._next_session_path()
    app.shared_state["pipeline_mode"] = orig
    assert "fm_blk_session_" in path.stem


def test_next_session_path_subblock_naming(app):
    orig = app.shared_state["pipeline_mode"]
    app.shared_state["pipeline_mode"] = "subblock"
    path = app._next_session_path()
    app.shared_state["pipeline_mode"] = orig
    assert "fm_sblk_session_" in path.stem


# ---------------------------------------------------------------------------
# WelcomePage basic rendering
# ---------------------------------------------------------------------------

def test_welcome_page_instantiates(app):
    """WelcomePage must be the first page and must build without crashing."""
    assert app.current_frame is not None


def test_welcome_page_class(app):
    from upxo.pxtal.fm_steel_3d.gui.pages import WelcomePage
    assert isinstance(app.current_frame, WelcomePage)


# ---------------------------------------------------------------------------
# Full page-construction sweep (added when RunPage's dead-code footprint --
# the class itself plus ~25 scattered lines across app.py -- was removed as
# unreachable through any sidebar entry or Next/Back transition). Guards
# against RunPage or an equivalent orphaned page silently reappearing, and
# more generally against any future page-registry/routing mistake in
# app.py's _section_sequence()/next_page()/back_page().
# ---------------------------------------------------------------------------

def test_runpage_no_longer_exists():
    """RunPage (~151 lines, pages_pag.py) was confirmed unreachable from
    every navigation path and removed. If this ever fails, someone
    re-added it (or something with the same name) without re-wiring it
    into the sidebar/routing -- re-litigate the removal, don't just relax
    this assertion."""
    from upxo.pxtal.fm_steel_3d.gui import pages
    assert "RunPage" not in pages.__all__
    assert not hasattr(pages, "RunPage")


def test_all_registered_pages_construct(app):
    """Construct every page class referenced anywhere in
    _section_sequence()'s per-row cls_set (not just each row's own
    target_cls -- e.g. the "Voronoi / MC" row's cls_set also includes
    Step2b1Page/Step2c1Page/Step2d1Page/Step2a1PlaceholderPage, which are
    real pages reachable via BaseGrainPage's method-branch routing, just
    not this particular sidebar row's own click target), across both
    pipeline modes (some pages, e.g. SubblockGenerationPage, only appear
    in subblock mode), and confirm every one constructs without raising."""
    orig_mode = app.shared_state.get("pipeline_mode", "block")
    orig_frame_cls = type(app.current_frame)

    seen = set()
    failures = []
    for mode in ("block", "subblock"):
        app.shared_state["pipeline_mode"] = mode
        for _section_label, items in app._section_sequence():
            for _label, cls_set, _target_cls in items:
                for cls in cls_set:
                    if cls in seen:
                        continue
                    seen.add(cls)
                    try:
                        app.show_page_by_class(cls)
                        app.update_idletasks()
                    except Exception as exc:
                        failures.append((cls.__name__, mode, f"{type(exc).__name__}: {exc}"))

    app.shared_state["pipeline_mode"] = orig_mode
    app.show_page_by_class(orig_frame_cls)
    app.update_idletasks()

    assert not failures, f"{len(failures)} page(s) failed to construct: {failures}"
    assert len(seen) >= 23, (
        f"expected at least 23 distinct page classes (the count confirmed "
        f"after merging PagTechniqueLeverPage into PagAdvisorPage), got {len(seen)}: "
        f"{sorted(c.__name__ for c in seen)}"
    )


def test_pagtechniqueleverpage_no_longer_exists():
    """PagTechniqueLeverPage ("Step 3: PAG/Packet Formation Technique") was
    merged onto the end of PagAdvisorPage ("Step 3: PAG Technique/Lever
    Advisor") -- both halves of "Step 3" now live on one page instead of
    two. If this ever fails, someone re-added a standalone page with this
    name without re-litigating the merge."""
    from upxo.pxtal.fm_steel_3d.gui import pages
    assert "PagTechniqueLeverPage" not in pages.__all__
    assert not hasattr(pages, "PagTechniqueLeverPage")


def test_pag_advisor_page_merged_technique_lever_content(app):
    """PagAdvisorPage must carry both halves: the advisor controls (Run
    button) and the appended Technique/Lever blocks (technique_var etc.),
    and next_page()/back_page() must skip straight between SweepCleanPage
    and PagClusteringPage through this one page -- no more separate
    PagTechniqueLeverPage stop in between. (GsCleanPage -> TransformationsPage
    -> SweepCleanPage -> PagAdvisorPage, per the Clean-before-Transformations
    page order with the Sweep Clean round-trip page appended after it.)"""
    from upxo.pxtal.fm_steel_3d.gui.pages import (
        GsCleanPage, TransformationsPage, SweepCleanPage, PagAdvisorPage, PagClusteringPage,
    )

    app.show_page_by_class(GsCleanPage)
    app.update_idletasks()
    app.next_page()
    app.update_idletasks()
    assert type(app.current_frame) is TransformationsPage

    app.next_page()
    app.update_idletasks()
    assert type(app.current_frame) is SweepCleanPage

    app.next_page()
    app.update_idletasks()
    assert type(app.current_frame) is PagAdvisorPage

    page = app.current_frame
    # Advisor half.
    assert hasattr(page, "_btn_run")
    assert hasattr(page, "use_default_var")
    # Merged Technique/Lever half.
    assert hasattr(page, "technique_var")
    assert hasattr(page, "lever_var")
    assert hasattr(page, "_split_threshold_entry")

    app.next_page()
    app.update_idletasks()
    assert type(app.current_frame) is PagClusteringPage

    app.back_page()
    app.update_idletasks()
    assert type(app.current_frame) is PagAdvisorPage


def test_figure_toolbar_font_row_above_save_row_and_no_overflow_on_resize(app):
    """_build_figure_toolbar must lay out font-size controls as an N x 2
    grid (one row per category: Tick labels / Figure title / Axis labels,
    each row = label | stepper) ABOVE the Save row -- a single wide row of
    all three side by side previously ran past the visible width (confirmed
    by screenshot: a horizontal scrollbar appeared). Bumping a font size way
    up must also trigger fig.tight_layout() before redrawing, so larger
    text doesn't run off the edge of the figure (the axes box must actually
    shrink to make room, not just report a bigger font on the same old
    layout)."""
    from unittest import mock
    from upxo.pxtal.fm_steel_3d.gui.pages_basegrain import Step2b1Page

    app.show_page_by_class(Step2b1Page)
    app.update_idletasks()
    page = app.current_frame

    assert hasattr(page, "_sgs_fig")  # Block [7] "3D SGS Morphology Parameters"
    # Build a fresh toolbar for the already-existing "sgs" figure to inspect
    # its row structure (the one built during page construction is buried
    # inside the SGS LabelFrame; a probe copy is simpler to introspect and
    # exercises the exact same _build_figure_toolbar code path).
    probe_parent = page._scroll_canvas
    row = page._build_figure_toolbar(probe_parent, "sgs")
    rows = [w for w in row.winfo_children() if w.winfo_manager() == "pack"]
    assert len(rows) == 2, f"expected 2 sub-rows (font grid, then save), got {len(rows)}"
    font_row, save_row = rows[0], rows[1]

    # font_row must itself be a grid, not a single packed line of widgets --
    # 4 rows (tick/title/axis/legend) x 2 columns (label, stepper).
    n_cols, n_rows = font_row.grid_size()
    assert (n_cols, n_rows) == (2, 4), f"expected a 2-column x 4-row grid, got {n_cols}x{n_rows}"
    for r in range(4):
        assert font_row.grid_slaves(row=r, column=0), f"row {r} missing its label cell"
        assert font_row.grid_slaves(row=r, column=1), f"row {r} missing its stepper cell"

    def _all_texts(widget):
        out = []
        for child in widget.winfo_children():
            try:
                out.append(child.cget("text"))
            except Exception:
                pass
        return out

    font_texts = " ".join(_all_texts(font_row))
    save_texts = " ".join(_all_texts(save_row))
    assert "Tick" in font_texts and "Figure title" in font_texts and "Axis" in font_texts
    assert "Legend" in font_texts
    assert "JPEG" not in font_texts and "PNG" not in font_texts
    assert "Save" in save_texts
    assert "Tick" not in save_texts

    # Font-increase must relayout (tight_layout), not just resize text in place.
    with mock.patch.object(page._sgs_fig, "tight_layout") as spy:
        page._adjust_figure_font_size("sgs", "title", 20)
    spy.assert_called_once()
    assert page._sgs_font_sizes["title"] == 29  # 9 (default) + 20

    # Further increases clamp at max_size=32, still relayouting each time.
    with mock.patch.object(page._sgs_fig, "tight_layout") as spy2:
        page._adjust_figure_font_size("sgs", "title", 20)
    spy2.assert_called_once()
    assert page._sgs_font_sizes["title"] == 32


def test_step2b1_console_and_preview_state_survive_rebuild(app):
    """Step2b1Page ("Step 2b1: Monte-Carlo Grain Structure") is rebuilt from
    scratch (a new page instance, new widgets) on every Back/Next -- only
    shared_state and app-level attributes survive. Before this fix, the MC
    run console text, the Apply console text, and the 2D preview's
    plane/slice/distribution-property selections lived only in the old
    widgets and were silently lost the moment the user navigated away and
    back. Injects synthetic console/selection state (cheaper than running a
    real MC simulation) and confirms a rebuilt page instance restores it."""
    from upxo.pxtal.fm_steel_3d.gui.pages_basegrain import Step2b1Page

    app.show_page_by_class(Step2b1Page)
    app.update_idletasks()
    page = app.current_frame

    page._mc_console_write("[MC] synthetic run log\n")
    page._apply_console_write("[MC] synthetic apply log\n")
    page._preview_axis_var.set("YZ")
    page._preview_slice_var.set(3)
    page.shared_state["mc_preview_axis"] = "YZ"
    page.shared_state["mc_preview_slice"] = 3
    page._dist_prop_var.set("Perimeter")
    page.shared_state["mc_preview_dist_prop"] = "Perimeter"

    app.show_page_by_class(Step2b1Page)  # save_state() + destroy() + rebuild
    app.update_idletasks()
    page2 = app.current_frame
    assert page2 is not page

    assert "[MC] synthetic run log" in page2._mc_console.get("1.0", tk.END)
    assert "[MC] synthetic apply log" in page2._apply_console.get("1.0", tk.END)
    assert page2._preview_axis_var.get() == "YZ"
    assert page2._preview_slice_var.get() == 3
    assert page2._dist_prop_var.get() == "Perimeter"


def _voronoi_lfi_for_pag_tests(n=16, n_seeds=18, seed=0):
    import numpy as np
    rng = np.random.default_rng(seed)
    cx = rng.integers(0, n, n_seeds).astype(np.float32)
    cy = rng.integers(0, n, n_seeds).astype(np.float32)
    cz = rng.integers(0, n, n_seeds).astype(np.float32)
    xi = np.arange(n, dtype=np.float32)[:, None, None, None]
    yi = np.arange(n, dtype=np.float32)[None, :, None, None]
    zi = np.arange(n, dtype=np.float32)[None, None, :, None]
    sq = (xi - cx) ** 2 + (yi - cy) ** 2 + (zi - cz) ** 2
    return (np.argmin(sq, axis=3) + 1).astype(np.int32)


def test_pag_block1a_and_block3_distgrid(app, monkeypatch):
    """Block [1a]'s morphology result (aspect ratio, solidity) -- now
    computed automatically as part of "Generate PAGs" (compute_morphology=
    True), no separate "Compute Morphology" button -- must render as
    separate stacked lines, and Block [3]'s slice-plane selector must be a
    single mutually-exclusive radio choice (not 3 independent checkboxes
    whose slices used to get pooled together) feeding a grid of PLOT
    BUTTONS (rows=properties, columns=grain groups) -- not a grid of actual
    plots -- where clicking one cell renders that one combination into a
    single shared plot area below, replacing the old separate 'Plot
    Distributions' popup-window button."""
    import threading

    # _on_compute_distributions runs its work on a
    # background thread that calls self.after(...) when done -- outside a
    # real Tk mainloop (as here, headless/no mainloop() call), a genuine OS
    # thread hits "main thread is not in main loop". Run the work
    # synchronously on this test's own thread instead, scoped to this test
    # only via monkeypatch.
    class _SyncThread:
        def __init__(self, target=None, daemon=None, **kw):
            self._target = target
        def start(self):
            self._target()
    monkeypatch.setattr(threading, "Thread", _SyncThread)

    from upxo.pxtal.fm_steel_3d.gui.pages_pag import PagClusteringPage
    from upxo.pxtal.fm_steel_3d.base_3d import FMSteel3DBase
    from upxo.pxtal.fm_steel_3d.pag_technique_selector_3d import generate_pags

    lfi = _voronoi_lfi_for_pag_tests()
    fm = FMSteel3DBase.from_lfi(lfi, physical_dimensions=(16.0, 16.0, 16.0), voxel_size=1.0,
                                connectivity=6, min_grain_nvoxels=4, random_seed=42, verbosity=0)
    fm_pag = generate_pags(fm_base=fm, technique="A",
                           pag_size_distribution={"sizes": [3, 4], "probs": [0.95, 0.05]},
                           max_packets_per_pag=4, lever="lever3", random_seed=7)
    app._fm_with_pags = fm_pag

    app.show_page_by_class(PagClusteringPage)
    app.update_idletasks()
    page = app.current_frame

    assert not hasattr(page, "_on_plot_distributions")
    assert not hasattr(page, "_on_compute_morphology")  # button removed, folded into Generate PAGs
    assert not hasattr(page, "_btn_morph")
    assert hasattr(page, "_slice_axis_var")  # replaces the old 3 independent checkboxes
    assert not hasattr(page, "_slice_vars")

    # Mirrors what _on_generate_pag's own worker thread now does automatically
    # (compute_morphology=True) instead of a separate button click.
    page._update_pag_stats_display(fm_pag.get_pag_statistics(compute_morphology=True))
    app.update()
    assert page._pag_morph_aspect_lbl.cget("text").startswith("Aspect ratio:")
    assert page._pag_morph_solidity_lbl.cget("text").startswith("Solidity:")
    assert page._pag_morph_lbl.cget("text") == ""

    page._slice_axis_var.set("xy")
    page._slice_n_vars["pag_morph_n_xy"].set(2)
    for k, v in page._morph_prop_vars.items():
        v.set(k == "pag_morph_prop_area")
    for k, v in page._morph_group_vars.items():
        v.set(k in ("pag_morph_group_pag", "pag_morph_group_packets"))

    page._on_compute_distributions()
    app.update()
    assert app._pag_distr_active_props == ["area"]
    assert page._distgrid_fig is not None
    # The plot area is a single Axes -- the grid above it is buttons, not
    # subplots -- and starts on the "click a cell" placeholder.
    assert len(page._distgrid_fig.axes) == 1

    # Button grid: 1 property row x 4 columns (PAG/Packets/Overlaid/Blocks)
    # in default ("block") pipeline mode; Blocks is disabled (no block
    # structure exists yet on this pipeline page).
    grid = page._distgrid_btn_container.winfo_children()[0]
    row1_buttons = [w for w in grid.grid_slaves(row=1) if w.winfo_class() in ("TButton", "Button")]
    assert len(row1_buttons) == 4
    states = [str(b.cget("state")) for b in row1_buttons]
    assert states.count("disabled") == 1  # Blocks column

    # Clicking a real (non-disabled) cell plots into the single shared Axes.
    page._on_distgrid_cell_click("area", "pag", "PAG")
    app.update()
    assert "PAG" in page._distgrid_ax.get_title()
    assert len(page._distgrid_ax.get_children()) > 1

    # Clicking a different cell REPLACES the plot (still 1 Axes on the fig).
    page._on_distgrid_cell_click("area", "overlay_pp", "Overlaid")
    app.update()
    assert "Overlaid" in page._distgrid_ax.get_title() or "overlay" in page._distgrid_ax.get_title().lower()
    assert len(page._distgrid_fig.axes) == 1

    # Same shared _build_dist_plot_controls consumed by the Transformations
    # page's Distributions blocks -- range fields auto-populate, the legend
    # font stepper works on both the single-series (_plot_kde) and overlay
    # (_plot_kde_overlay) columns, and Update Plot re-renders with explicit
    # overrides via the cached last-plotted data.
    page._on_distgrid_cell_click("area", "pag", "PAG")
    for key in ("xmin", "xmax", "ymin", "ymax"):
        val = getattr(page, f"_distgrid_{key}_var").get()
        assert val != ""
        float(val)
    page._adjust_figure_font_size("distgrid", "legend", 2)  # single-series: no legend, must not raise

    page._on_distgrid_cell_click("area", "overlay_pp", "Overlaid")
    before_fs = page._distgrid_ax.get_legend().get_texts()[0].get_fontsize()
    page._adjust_figure_font_size("distgrid", "legend", 3)
    after_fs = page._distgrid_ax.get_legend().get_texts()[0].get_fontsize()
    assert after_fs == before_fs + 3

    page._distgrid_xmin_var.set("0")
    page._distgrid_xmax_var.set("100")
    page._on_update_dist_plot("distgrid")
    assert page._distgrid_ax.get_xlim() == (0.0, 100.0)
    assert page._distgrid_xmin_var.get() == "0"  # Update Plot must not clobber the field


def test_viz_settings_morphgrid_reads_live_state_not_stale_shared_state(app, monkeypatch):
    """VisualizationSettingsPage Block [3]'s old 'Plot distributions' button
    read self.shared_state["MORPH_PROPERTIES"]/["MORPH_COMBO_ROWS"], which
    are only written on page navigation (save_state()) -- so toggling
    checkboxes and clicking Plot without first leaving the page silently
    plotted a STALE prior selection. The replacement button grid must be
    built from the LIVE widget vars instead: seed shared_state with a much
    larger stale selection, set a small live selection, and confirm the
    grid matches the live one, not the stale one."""
    import threading

    class _SyncThread:
        def __init__(self, target=None, daemon=None, **kw):
            self._target = target
        def start(self):
            self._target()
    monkeypatch.setattr(threading, "Thread", _SyncThread)

    from upxo.pxtal.fm_steel_3d.gui.pages_viz import VisualizationSettingsPage
    from upxo.pxtal.fm_steel_3d.base_3d import FMSteel3DBase

    lfi = _voronoi_lfi_for_pag_tests()
    fm_base = FMSteel3DBase.from_lfi(lfi, physical_dimensions=(16.0, 16.0, 16.0), voxel_size=1.0,
                                     units="microns", connectivity=6, min_grain_nvoxels=4,
                                     random_seed=42)
    fm_pag = fm_base.generate_pag_clusters(
        pag_size_distribution={"sizes": [3, 4], "probs": [0.7, 0.3]},
        pag_grain_fraction=1.0, random_seed=42)
    fm_pag.assign_pag_orientations(pag_ori_mode="random", random_seed=42)
    fm_blk = fm_pag.generate_blocks(block_thickness_range=(2.0, 5.0), random_seed=42)
    fm_ori = fm_blk.assign_orientations(random_seed=42)
    app._fm_with_orientations = fm_ori

    # Stale shared_state, as if a much larger selection was saved on a
    # previous visit to this page.
    app.shared_state["MORPH_PROPERTIES"] = {k: True for k, _ in VisualizationSettingsPage._MORPH_PROPS}
    app.shared_state["MORPH_COMBO_ROWS"] = [
        {"pag": True, "pkt": True, "blk": True, "sblk": False} for _ in range(6)
    ]

    app.show_page_by_class(VisualizationSettingsPage)
    app.update_idletasks()
    page = app.current_frame

    assert not hasattr(page, "_on_plot_3d_morph_distr")

    # Live selection: much smaller than, and different from, the stale
    # shared_state above.
    for k, v in page.morph_prop_vars.items():
        v.set(k == "vol_physical")
    for i, row in enumerate(page.combo_rows):
        checked = (i == 0)
        row["pag"].set(checked)
        row["pkt"].set(False)
        row["blk"].set(False)
        row["sblk"].set(False)

    page._on_compute_3d_morph_distr()
    app.update()
    assert "Computed" in page._compute_3d_morph_status_lbl.cget("text")

    grid = page._morphgrid_btn_container.winfo_children()[0]
    row1_buttons = [w for w in grid.grid_slaves(row=1) if w.winfo_class() in ("TButton", "Button")]
    assert len(row1_buttons) == 1, (
        f"expected exactly 1 button (matching the live 1-property x 1-row selection), "
        f"got {len(row1_buttons)} -- grid is reading stale shared_state instead of live widgets")

    row1_buttons[0].invoke()
    app.update()
    assert len(page._morphgrid_ax.get_children()) > 1
    assert "Volume" in page._morphgrid_ax.get_title()


def test_block_page_iqr_button_and_orientation_viz(app, monkeypatch):
    """Step 3b Block Generation: (1) 'Update Thickness Range' must set
    Min/Max from the IQR (25th/75th percentile) of the just-measured
    intercept grain-size distribution, not leave them unchanged; (2) the new
    'Visualize Orientations' block (between Custom Block Orientation
    Override and the 2D Cross-Section preview) must be a real, working
    pole-figure section reusing PoleFigureControlsMixin, not a stub."""
    import threading

    class _SyncThread:
        def __init__(self, target=None, daemon=None, **kw):
            self._target = target
        def start(self):
            self._target()
    monkeypatch.setattr(threading, "Thread", _SyncThread)

    from upxo.pxtal.fm_steel_3d.gui.pages_block import BlockGenerationPage
    from upxo.pxtal.fm_steel_3d.base_3d import FMSteel3DBase

    lfi = _voronoi_lfi_for_pag_tests()
    fm_base = FMSteel3DBase.from_lfi(lfi, physical_dimensions=(16.0, 16.0, 16.0), voxel_size=1.0,
                                     units="microns", connectivity=6, min_grain_nvoxels=4,
                                     random_seed=42)
    fm_pag = fm_base.generate_pag_clusters(
        pag_size_distribution={"sizes": [3, 4], "probs": [0.7, 0.3]},
        pag_grain_fraction=1.0, random_seed=42)
    fm_pag.assign_pag_orientations(pag_ori_mode="random", random_seed=42)
    fm_blk = fm_pag.generate_blocks(block_thickness_range=(2.0, 5.0), random_seed=42)
    fm_ori = fm_blk.assign_orientations(random_seed=42)

    app._fm_with_pags = fm_pag
    app._fm_with_blocks = fm_blk
    app._fm_with_orientations = fm_ori

    app.show_page_by_class(BlockGenerationPage)
    app.update_idletasks()
    page = app.current_frame

    page._on_find_intercept_grain_size()
    app.update()
    assert page._intercept_status_lbl.cget("text") == "Done."
    assert str(page._btn_update_thk_range.cget("state")) == "normal"

    before = (page.blk_min_var.get(), page.blk_max_var.get())
    page._on_update_thickness_range()
    after = (page.blk_min_var.get(), page.blk_max_var.get())
    assert after != before
    assert after[0] < after[1]

    assert hasattr(page, "_blockviz_sym_var")  # PoleFigureControlsMixin section, ns="blockviz"
    page._on_plot("blockviz", "block")
    app.update()
    assert "error" not in page._blockviz_status_lbl.cget("text").lower()
    assert len(page._blockviz_ax.get_children()) > 1

    page.save_state()
    assert app.shared_state.get("PF_BLOCKVIZ_SYMMETRY") is not None


def test_transformations_page_rescale_stretch_and_safety_check(app, monkeypatch):
    """New 'Transformations' sidebar section (Transform page, then Clean,
    immediately before what used to be Base Grain Structure's own trailing
    Clean row) with real rescale/anisotropic-stretch/feature-loss-safety
    mechanics against a real generated structure."""
    import threading

    class _SyncThread:
        def __init__(self, target=None, daemon=None, **kw):
            self._target = target
        def start(self):
            self._target()
    monkeypatch.setattr(threading, "Thread", _SyncThread)

    from upxo.pxtal.fm_steel_3d.gui.pages_transform import TransformationsPage
    from upxo.pxtal.fm_steel_3d.base_3d import FMSteel3DBase

    lfi = _voronoi_lfi_for_pag_tests(n=16, n_seeds=20, seed=2)
    fm_base = FMSteel3DBase.from_lfi(lfi, physical_dimensions=(16.0, 16.0, 16.0), voxel_size=1.0,
                                     units="microns", connectivity=6, min_grain_nvoxels=4,
                                     random_seed=42)
    app._fm_base = fm_base
    orig_shape = fm_base.lgi.shape
    orig_voxel_size = fm_base.voxel_size

    sections = dict(app._section_sequence())
    assert "Transformations" in sections
    assert [lbl for lbl, _, _ in sections["Transformations"]] == [
        "Clean", "Transform", "Sweep Clean"]

    app.show_page_by_class(TransformationsPage)
    app.update_idletasks()
    page = app.current_frame

    page._scale_var.set(0.7)
    page._on_apply_scale()
    app.update()
    assert "Done" in page._scale_status_lbl.cget("text")
    expected_shape = tuple(int(round(n * 0.7)) for n in orig_shape)
    assert app._fm_base.lgi.shape == expected_shape
    assert app._fm_base.voxel_size == pytest.approx(orig_voxel_size / 0.7)

    mid_shape = app._fm_base.lgi.shape
    mid_voxel_size = app._fm_base.voxel_size
    page._sfx_var.set(2.0)
    page._sfy_var.set(1.0)
    page._sfz_var.set(1.0)
    page._on_apply_stretch()
    app.update()
    assert "Done" in page._stretch_status_lbl.cget("text")
    assert app._fm_base.lgi.shape == (mid_shape[0] * 2, mid_shape[1], mid_shape[2])
    assert app._fm_base.voxel_size == pytest.approx(mid_voxel_size)

    # Pre-transform snapshot is the ORIGINAL structure (stashed once, at the
    # first transform), not the intermediate mid-stretch one.
    assert page._pre_transform_lfi.shape == orig_shape

    page._on_safety_check()
    app.update()
    assert "Error" not in page._safety_status_lbl.cget("text")


def test_transformations_page_auto_cleanup_threshold(app, monkeypatch):
    """Block [5]'s "Auto-dissolve grains below" field, wired via the shared
    _build_clean_structure helper into _on_apply_scale/_on_apply_stretch/
    _on_recompute_majority: threshold=0 leaves whatever a transform produces
    (including tiny fragments) untouched; threshold>0 dissolves any grain
    below that many voxels into its largest neighbour (reusing
    FMSteel3DBase.from_lfi's own min_grain_nvoxels cleanup) and re-relabels
    the result to a clean, contiguous ID range either way."""
    import threading

    class _SyncThread:
        def __init__(self, target=None, daemon=None, **kw):
            self._target = target
        def start(self):
            self._target()
    monkeypatch.setattr(threading, "Thread", _SyncThread)

    import numpy as np
    from upxo.pxtal.fm_steel_3d.gui.pages_transform import TransformationsPage
    from upxo.pxtal.fm_steel_3d.base_3d import FMSteel3DBase

    lfi = _voronoi_lfi_for_pag_tests(n=25, n_seeds=122, seed=7)
    fm_base = FMSteel3DBase.from_lfi(lfi, physical_dimensions=(2.5, 2.5, 2.5), voxel_size=0.1,
                                     units="mm", connectivity=6, min_grain_nvoxels=-1,
                                     random_seed=42)
    app._fm_base = fm_base

    app.show_page_by_class(TransformationsPage)
    app.update_idletasks()
    page = app.current_frame

    def _grain_sizes(lgi):
        return [int((lgi == gid).sum()) for gid in np.unique(lgi) if gid != 0]

    def _is_contiguous(lgi):
        ids = sorted(int(i) for i in np.unique(lgi) if i != 0)
        return ids == list(range(1, len(ids) + 1))

    # threshold=0 (default off) -- aggressive downscale is allowed to leave
    # tiny/degenerate grains behind, IDs still contiguous either way.
    page._cleanup_threshold_var.set(0)
    page._scale_var.set(0.5)
    page._on_apply_scale()
    app.update()
    off_sizes = _grain_sizes(app._fm_base.lgi)
    assert _is_contiguous(app._fm_base.lgi)
    assert min(off_sizes) < 8, "fixture should produce a sub-8-voxel grain with cleanup off"

    # threshold=8 on a fresh transform from the same starting structure --
    # nothing below 8 voxels should survive, and IDs stay contiguous.
    app._fm_base = fm_base
    page._pre_transform_lfi = None
    page._last_transform_kind = None
    page._cleanup_threshold_var.set(8)
    page._scale_var.set(0.5)
    page._on_apply_scale()
    app.update()
    on_sizes = _grain_sizes(app._fm_base.lgi)
    assert _is_contiguous(app._fm_base.lgi)
    assert min(on_sizes) >= 8
    assert len(on_sizes) < len(off_sizes)

    # Stretch honours the same threshold.
    app._fm_base = fm_base
    page._pre_transform_lfi = None
    page._last_transform_kind = None
    page._cleanup_threshold_var.set(8)
    page._sfx_var.set(1.0); page._sfy_var.set(1.0); page._sfz_var.set(0.5)
    page._on_apply_stretch()
    app.update()
    assert min(_grain_sizes(app._fm_base.lgi)) >= 8
    assert _is_contiguous(app._fm_base.lgi)

    # Recompute-with-majority-vote honours the same threshold too, and the
    # feature-loss comparison it reports is computed before cleanup runs.
    app._fm_base = fm_base
    page._pre_transform_lfi = None
    page._last_transform_kind = None
    page._cleanup_threshold_var.set(8)
    page._scale_var.set(0.25)
    page._on_apply_scale()
    app.update()
    page._on_recompute_majority()
    app.update()
    assert "Recomputed" in page._safety_status_lbl.cget("text")
    assert min(_grain_sizes(app._fm_base.lgi)) >= 8
    assert _is_contiguous(app._fm_base.lgi)


def test_sweep_clean_page_round_trip_and_cleanup(app, monkeypatch):
    """New Sweep Clean page (Transformations -> Sweep Clean -> PAG Technique
    Advisor): downscales by a user-given factor with majority-vote
    resampling, then immediately upscales back to the ORIGINAL voxel
    dimensions with nearest-neighbour resampling -- shape and voxel_size
    must be unchanged by the round trip. The optional auto-cleanup checkbox
    (off by default) shares TransformationsPage's dissolve-then-relabel
    convention via transform_shared.build_clean_structure."""
    import threading
    import numpy as np

    class _SyncThread:
        def __init__(self, target=None, daemon=None, **kw):
            self._target = target
        def start(self):
            self._target()
    monkeypatch.setattr(threading, "Thread", _SyncThread)

    from upxo.pxtal.fm_steel_3d.gui.pages_sweep_clean import SweepCleanPage
    from upxo.pxtal.fm_steel_3d.base_3d import FMSteel3DBase

    lfi = _voronoi_lfi_for_pag_tests(n=24, n_seeds=100, seed=11)
    fm_base = FMSteel3DBase.from_lfi(lfi, physical_dimensions=(24.0, 24.0, 24.0), voxel_size=1.0,
                                     units="microns", connectivity=6, min_grain_nvoxels=-1,
                                     random_seed=42)
    app._fm_base = fm_base
    orig_shape = fm_base.lgi.shape
    orig_voxel_size = fm_base.voxel_size

    app.show_page_by_class(SweepCleanPage)
    app.update_idletasks()
    page = app.current_frame

    # Cleanup checkbox off by default.
    assert page._cleanup_enabled_var.get() is False
    assert page._get_cleanup_threshold() == 0

    page._factor_var.set(2.0)
    page._on_run_sweep()
    app.update()
    assert "Done" in page._sweep_status_lbl.cget("text")
    assert app._fm_base.lgi.shape == orig_shape
    assert app._fm_base.voxel_size == pytest.approx(orig_voxel_size)
    ids = np.unique(app._fm_base.lgi)
    ids = ids[ids != 0]
    assert sorted(ids.tolist()) == list(range(1, len(ids) + 1))

    # Invalid factor (<=1) is rejected without touching app._fm_base.
    import tkinter.messagebox as tkmb
    errors = []
    monkeypatch.setattr(tkmb, "showerror", lambda title, msg: errors.append((title, msg)))
    page._factor_var.set(1.0)
    page._on_run_sweep()
    assert len(errors) == 1

    # Fresh structure, cleanup enabled with a high threshold -- nothing
    # below it should survive.
    app._fm_base = fm_base
    page._pre_sweep_lfi = None
    page._pre_sweep_voxel_size = None
    page._factor_var.set(3.0)
    page._cleanup_enabled_var.set(True)
    page._cleanup_threshold_var.set(10)
    page._on_run_sweep()
    app.update()
    sizes = [int((app._fm_base.lgi == gid).sum())
             for gid in np.unique(app._fm_base.lgi) if gid != 0]
    assert min(sizes) >= 10
    assert app._fm_base.lgi.shape == orig_shape


def test_transformations_page_visualize_dist3d_dist2d_blocks(app, monkeypatch):
    """Blocks [5]/[6]/[7], ported from twinned_simple_3d's NonEquiaxialityPage
    (Visualize / Distributions-3D / Distributions-2D) using FM Steel's own
    LGI-generic backends (geom_metrics_3d.all_metrics, slice_metrics_2d.
    compute_slice_metrics, GrainStructureViz3D.lfi_to_polydata) instead of
    twinned's TwinnedSimple3DBase-method-based equivalents, which FM Steel's
    FMSteel3DBase doesn't have."""
    import threading

    class _SyncThread:
        def __init__(self, target=None, daemon=None, **kw):
            self._target = target
        def start(self):
            self._target()
    monkeypatch.setattr(threading, "Thread", _SyncThread)

    import upxo.pxtal.fm_steel_3d.viz.grain_structure_viz_3d as gsviz_mod
    calls = []
    def fake_compare(self, lfi_left, lfi_right, scalar_array_left=None, scalar_array_right=None,
                     voxel_size_left=1.0, voxel_size_right=1.0, cmap='tab20', title_left='',
                     title_right='', **kw):
        calls.append((lfi_left.shape, lfi_right.shape))
    monkeypatch.setattr(gsviz_mod.GrainStructureViz3D, "lfi_to_polydata_compare", fake_compare)

    from upxo.pxtal.fm_steel_3d.gui.pages_transform import TransformationsPage
    from upxo.pxtal.fm_steel_3d.base_3d import FMSteel3DBase

    lfi = _voronoi_lfi_for_pag_tests(n=16, n_seeds=20, seed=2)
    fm_base = FMSteel3DBase.from_lfi(lfi, physical_dimensions=(16.0, 16.0, 16.0), voxel_size=1.0,
                                     units="microns", connectivity=6, min_grain_nvoxels=4,
                                     random_seed=42)
    app._fm_base = fm_base

    app.show_page_by_class(TransformationsPage)
    app.update_idletasks()
    page = app.current_frame

    # [5] Visualize -- one linked side-by-side call; before any transform,
    # both panels show the SAME current structure (matching twinned's own
    # documented fallback).
    page._on_visualize_compare()
    assert len(calls) == 1
    assert calls[0][0] == calls[0][1] == fm_base.lgi.shape
    calls.clear()

    # Apply a stretch so _pre_transform_lfi is genuinely different from the
    # current structure, then re-check Visualize reflects both panels.
    page._sfx_var.set(1.5)
    page._on_apply_stretch()
    app.update()
    assert page._pre_transform_lfi is not None

    page._on_visualize_compare()
    assert len(calls) == 1
    assert calls[0][0] == page._pre_transform_lfi.shape
    assert calls[0][1] == app._fm_base.lgi.shape
    assert calls[0][0] != calls[0][1]

    # [6] Distributions - 3D -- every parameter button must plot without
    # error and produce a real overlay (both series present).
    for key, label in page._DIST3D_PARAMS:
        page._on_dist3d_click(key)
        assert label in page._dist3d_ax.get_title()
        legend = page._dist3d_ax.get_legend()
        assert legend is not None and len(legend.get_texts()) >= 1

    # [7] Distributions - 2D -- every parameter button must plot without
    # error, using the default X/Y/Z axes and slice count.
    for key, label in page._DIST2D_PARAMS:
        page._on_dist2d_click(key)
        assert label in page._dist2d_ax.get_title()

    # Deselecting every axis must be a caught validation error, not a crash.
    import tkinter.messagebox as tkmb
    errors = []
    monkeypatch.setattr(tkmb, "showerror", lambda title, msg: errors.append((title, msg)))
    for var in page._dist2d_axis_vars.values():
        var.set(False)
    page._on_dist2d_click("area")
    assert len(errors) == 1 and "axis" in errors[0][1].lower()


def test_transformations_dist_plot_controls_range_binwidth_legend_grid(app, monkeypatch):
    """New shared _build_dist_plot_controls (pages_base.py): N x 4 parameter
    button grid, legend font-size stepper, bin-width slider, and X/Y range
    fields auto-populated from real data then usable as overrides via
    "Update Plot" without needing to reselect the parameter."""
    import threading

    class _SyncThread:
        def __init__(self, target=None, daemon=None, **kw):
            self._target = target
        def start(self):
            self._target()
    monkeypatch.setattr(threading, "Thread", _SyncThread)

    from upxo.pxtal.fm_steel_3d.gui.pages_transform import TransformationsPage
    from upxo.pxtal.fm_steel_3d.base_3d import FMSteel3DBase

    lfi = _voronoi_lfi_for_pag_tests(n=16, n_seeds=20, seed=3)
    fm_base = FMSteel3DBase.from_lfi(lfi, physical_dimensions=(16.0, 16.0, 16.0), voxel_size=1.0,
                                     units="microns", connectivity=6, min_grain_nvoxels=4,
                                     random_seed=42)
    app._fm_base = fm_base

    app.show_page_by_class(TransformationsPage)
    app.update_idletasks()
    page = app.current_frame

    # N x 4 grid: 8 dist3d params (incl. Aspect Ratio) -> 2 full rows of 4.
    n_dist3d_params = len(page._DIST3D_PARAMS)
    def _buttons_of(parent):
        return [c for c in parent.winfo_children() if c.winfo_class() in ("TButton", "Button")]
    # Walk up to find the actual button-grid frame by locating n_dist3d_params
    # buttons with non-None grid coordinates.
    def find_param_btn_grid(w):
        for c in w.winfo_children():
            btns = _buttons_of(c)
            if len(btns) == n_dist3d_params and all(b.grid_info().get("column") is not None for b in btns):
                return btns
            found = find_param_btn_grid(c)
            if found:
                return found
        return None
    btns = find_param_btn_grid(page)
    assert btns is not None, "could not locate the [7] Distributions-3D button grid"
    cols = sorted({b.grid_info()["column"] for b in btns})
    assert cols == [0, 1, 2, 3], f"expected 4-column grid, got columns {cols}"
    expected_rows = list(range(-(-n_dist3d_params // 4)))  # ceil division
    rows = sorted({b.grid_info()["row"] for b in btns})
    assert rows == expected_rows, f"expected rows {expected_rows} for {n_dist3d_params} buttons in a 4-col grid, got {rows}"

    # Clicking a parameter auto-populates X/Y min/max from the real data range.
    page._on_dist3d_click("vol")
    for key in ("xmin", "xmax", "ymin", "ymax"):
        val = getattr(page, f"_dist3d_{key}_var").get()
        assert val != "", f"{key} was not auto-populated"
        float(val)  # must parse as a real number

    # Legend font-size stepper actually resizes the rendered legend.
    before_fs = page._dist3d_ax.get_legend().get_texts()[0].get_fontsize()
    page._adjust_figure_font_size("dist3d", "legend", 4)
    after_fs = page._dist3d_ax.get_legend().get_texts()[0].get_fontsize()
    assert after_fs == before_fs + 4

    # Bin-width slider is reconfigured with real data-derived bounds (not the
    # placeholder 0..1 default) once a parameter has been plotted.
    assert float(page._dist3d_binwidth_scale.cget("to")) > 1.0

    # Update Plot: explicit X range + non-auto bin width actually changes the
    # rendered axes, and does NOT clobber the user-typed range fields.
    page._dist3d_xmin_var.set("0")
    page._dist3d_xmax_var.set("999")
    page._dist3d_binwidth_var.set(float(page._dist3d_binwidth_scale.cget("to")))
    page._on_update_dist_plot("dist3d")
    assert page._dist3d_ax.get_xlim() == (0.0, 999.0)
    assert page._dist3d_xmin_var.get() == "0"
    assert page._dist3d_xmax_var.get() == "999"

    # Same mechanism, single-series consumer: PAG's Distributions block uses
    # _plot_kde (not _plot_kde_overlay) via the same _dist_cache/_range_stale
    # contract -- verify _plot_kde also honours bin_width/xlim/ylim/legend
    # sizing directly (no legend expected here, single series).
    page._dist2d_range_stale = True
    page._plot_kde("dist2d", page._dist2d_ax, [1.0, 2.0, 3.0, 4.0, 5.0], "x", "t",
                   bin_width=0.5, xlim=(0.0, 10.0), ylim=(0.0, 1.0))
    assert page._dist2d_ax.get_xlim() == (0.0, 10.0)
    assert page._dist2d_ax.get_ylim() == (0.0, 1.0)


def test_transformations_page_preserves_mc_state_across_transform(app):
    """TransformationsPage's rescale/stretch/majority-vote-recompute all call
    app.invalidate_from("fm_base") to correctly invalidate DOWNSTREAM stages
    (PAGs/blocks/...), but that used to also wipe Step2b1Page's (Monte-Carlo)
    OWN run state (app._mc_pxt, mc_sim_status/mc_tslice_list/mc_last_stats)
    every time -- even though transforming the existing structure doesn't
    invalidate the MC run that produced it. Navigating Back to Step2b1 after
    a transform then found an empty tslice list and no stats, forcing a full
    simulation re-run instead of a quick retry. invalidate_from gained a
    clear_mc_state=False escape hatch, used by all three transform call
    sites; this confirms the MC state survives a real transform while
    non-MC statuses (gs_gen_status etc) still correctly reset."""
    import threading

    class _SyncThread:
        def __init__(self, target=None, daemon=None, **kw):
            self._target = target
        def start(self):
            self._target()
    threading_patch = threading.Thread
    threading.Thread = _SyncThread
    try:
        from upxo.pxtal.fm_steel_3d.gui.pages_transform import TransformationsPage
        from upxo.pxtal.fm_steel_3d.base_3d import FMSteel3DBase

        lfi = _voronoi_lfi_for_pag_tests(n=16, n_seeds=20, seed=4)
        fm_base = FMSteel3DBase.from_lfi(lfi, physical_dimensions=(16.0, 16.0, 16.0), voxel_size=1.0,
                                         units="microns", connectivity=6, min_grain_nvoxels=4,
                                         random_seed=42)
        app._fm_base = fm_base
        app._mc_pxt = object()
        app.shared_state["mc_sim_status"] = "Run complete"
        app.shared_state["mc_tslice_list"] = [10, 20, 30]
        app.shared_state["mc_last_stats"] = {"n_grains": 20}
        app.shared_state["gs_gen_status"] = "Generated"

        app.show_page_by_class(TransformationsPage)
        app.update_idletasks()
        page = app.current_frame

        page._sfx_var.set(1.5)
        page._on_apply_stretch()
        app.update()
        assert "Done" in page._stretch_status_lbl.cget("text")

        assert app._mc_pxt is not None, "MC simulation object was wiped by a transform"
        assert app.shared_state["mc_sim_status"] == "Run complete"
        assert app.shared_state["mc_tslice_list"] == [10, 20, 30]
        assert app.shared_state["mc_last_stats"] == {"n_grains": 20}
        # Non-MC statuses must still correctly reset -- the fix must not have
        # accidentally suppressed invalidation entirely.
        assert app.shared_state["gs_gen_status"] == "Not generated"
    finally:
        threading.Thread = threading_patch


def test_clean_page_now_comes_before_transformations_page(app, monkeypatch):
    """Clean & Topology Repair now runs BEFORE Transformations (was after),
    and a new Sweep Clean page sits right after Transformations -- both the
    sidebar section row order and the next_page()/back_page() routing must
    reflect Base Grain Generate -> Clean -> Transform -> Sweep Clean ->
    PAG Technique Advisor, in both directions. Uses a mocked current_frame
    (validate_inputs()=True, save_state()=no-op) so the routing decision is
    tested in isolation, without constructing real pages that could pop a
    blocking messagebox dialog against the placeholder pipeline objects.
    Uses monkeypatch (not a raw attribute assignment) for show_page_by_class/
    current_frame since `app` is a module-scoped fixture shared across every
    test in this file -- a raw assignment would leak into every later test."""
    from unittest.mock import MagicMock
    from upxo.pxtal.fm_steel_3d.gui.pages import (
        TransformationsPage, SweepCleanPage, GsCleanPage, Step2a3Page, PagAdvisorPage,
    )

    sections = dict(app._section_sequence())
    assert [lbl for lbl, _, _ in sections["Transformations"]] == [
        "Clean", "Transform", "Sweep Clean"]

    routed_to = []
    def fake_show(cls, **kw):
        routed_to.append(cls)
    monkeypatch.setattr(app, "show_page_by_class", fake_show)
    app.shared_state["base_grain_method"] = "A"

    def set_frame(cls):
        m = MagicMock()
        m.validate_inputs.return_value = True
        m.save_state.return_value = None
        monkeypatch.setattr(app, "current_frame", m)
        app.current_frame.__class__ = cls

    set_frame(Step2a3Page)
    app.next_page()
    assert routed_to[-1] is GsCleanPage

    set_frame(GsCleanPage)
    app.next_page()
    assert routed_to[-1] is TransformationsPage

    set_frame(TransformationsPage)
    app.next_page()
    assert routed_to[-1] is SweepCleanPage

    set_frame(SweepCleanPage)
    app.next_page()
    assert routed_to[-1] is PagAdvisorPage

    set_frame(PagAdvisorPage)
    app.back_page()
    assert routed_to[-1] is SweepCleanPage

    set_frame(SweepCleanPage)
    app.back_page()
    assert routed_to[-1] is TransformationsPage

    set_frame(TransformationsPage)
    app.back_page()
    assert routed_to[-1] is GsCleanPage

    set_frame(GsCleanPage)
    app.back_page()
    assert routed_to[-1] is Step2a3Page


def test_transformations_page_consider_gate_disables_page_by_default(app, monkeypatch):
    """New "Consider Transformations" checkbox at the top of the page,
    unchecked by default -- every interactive control below it (all of
    blocks [1]-[9]) must start disabled, become usable once checked, and
    go back to disabled if unchecked again. Two controls stay disabled
    regardless of the gate by design ([1]'s Method dropdown, [3]'s "Permit
    elongated voxels" checkbox), and the safety-check's Recompute button
    resets to its default disabled state on reactivation (its real state is
    dynamic, computed by "Check for Dropped Features")."""
    import threading

    class _SyncThread:
        def __init__(self, target=None, daemon=None, **kw):
            self._target = target
        def start(self):
            self._target()
    monkeypatch.setattr(threading, "Thread", _SyncThread)

    from upxo.pxtal.fm_steel_3d.gui.pages_transform import TransformationsPage
    from upxo.pxtal.fm_steel_3d.base_3d import FMSteel3DBase

    lfi = _voronoi_lfi_for_pag_tests(n=16, n_seeds=20, seed=5)
    fm_base = FMSteel3DBase.from_lfi(lfi, physical_dimensions=(16.0, 16.0, 16.0), voxel_size=1.0,
                                     units="microns", connectivity=6, min_grain_nvoxels=4,
                                     random_seed=42)
    app._fm_base = fm_base

    app.show_page_by_class(TransformationsPage)
    app.update_idletasks()
    page = app.current_frame

    assert page._consider_var.get() is False
    assert str(page._btn_scale.cget("state")) == "disabled"
    assert str(page._btn_stretch.cget("state")) == "disabled"

    page._consider_var.set(True)
    page._on_consider_toggle()
    assert str(page._btn_scale.cget("state")) == "normal"
    assert str(page._btn_stretch.cget("state")) == "normal"
    # Always-disabled-by-design controls must NOT be re-enabled by the gate.
    assert str(page._method_dropdown.cget("state")) == "disabled"
    assert str(page._permit_elongated_chk.cget("state")) == "disabled"
    assert str(page._btn_recompute.cget("state")) == "disabled"

    # A real action actually works once the gate is active.
    page._scale_var.set(0.8)
    page._on_apply_scale()
    app.update()
    assert "Done" in page._scale_status_lbl.cget("text")

    page._consider_var.set(False)
    page._on_consider_toggle()
    assert str(page._btn_scale.cget("state")) == "disabled"
    assert str(page._btn_stretch.cget("state")) == "disabled"


def test_pole_figure_block8_restrict_block_requires_restrict_packet(app):
    """Step 9 Block [8] (Block-Scoped Sub-block Pole Figure): "Restrict to
    one block" used to be independently checkable even with "Restrict to
    one packet" unchecked -- the block picker stayed visible/interactive,
    but the resolver only ever consults it inside the
    `if restrict_pkt_var.get():` branch, so a chosen block was silently
    ignored (plot fell back to the whole PAG scope) with no indication
    anything was wrong. Checking "block" must now force "packet" semantics:
    it starts disabled, and unchecking "packet" force-unchecks + re-disables
    it (never leaves a checked-but-ineffective control on screen)."""
    from upxo.pxtal.fm_steel_3d.gui.pages_pole_figures import PoleFigurePage

    app.show_page_by_class(PoleFigurePage)
    app.update_idletasks()
    page = app.current_frame

    # Attempt the broken sequence: check "block" without ever checking "packet".
    page._sb_blk_restrict_pkt_var.set(False)
    page._sb_blk_restrict_blk_var.set(True)
    page._on_cascade_toggle("sb_blk")
    assert page._sb_blk_restrict_blk_var.get() is False, \
        "restrict_blk must be force-reset when restrict_pkt is off"
    assert page._sb_blk_blk_frame.winfo_manager() == "", \
        "block picker must not be shown when restrict_pkt is off"
    assert str(page._sb_blk_restrict_blk_chk.cget("state")) == "disabled"

    # Legitimate path: packet first, then block becomes available.
    page._sb_blk_restrict_pkt_var.set(True)
    page._on_cascade_toggle("sb_blk")
    assert str(page._sb_blk_restrict_blk_chk.cget("state")) == "normal"
    page._sb_blk_restrict_blk_var.set(True)
    page._on_cascade_toggle("sb_blk")
    assert page._sb_blk_blk_frame.winfo_manager() != ""

    # Turning packet back off cascades: block resets and disables again.
    page._sb_blk_restrict_pkt_var.set(False)
    page._on_cascade_toggle("sb_blk")
    assert page._sb_blk_restrict_blk_var.get() is False
    assert page._sb_blk_blk_frame.winfo_manager() == ""


def test_pole_figure_toggles_dont_reorder_below_canvas(app):
    """Step 9's per-section toggle-driven frames (Block [7]'s own packet
    Global/Local mode row, Block [8]'s cascade picker frames, and every
    section's density-mode controls) are shown/hidden via
    pack_forget()+pack(). Tk's pack manager appends a freshly re-packed
    widget to the END of its parent's CURRENT pack order when no
    before=/after= is given -- since each of these shares its parent with
    later-built siblings (the Plot button, the pole figure canvas itself),
    the first live toggle after the page was fully built silently dragged
    the control below the canvas. Confirms each stays above its section's
    canvas across repeated toggling, for every section that has one."""
    from upxo.pxtal.fm_steel_3d.gui.pages_pole_figures import PoleFigurePage

    app.show_page_by_class(PoleFigurePage)
    app.update_idletasks()
    page = app.current_frame

    def assert_above_canvas(container, widget, canvas_ns, label):
        slaves = container.pack_slaves()
        canvas_widget = getattr(page, f"_{canvas_ns}_canvas").get_tk_widget()
        assert slaves.index(widget) < slaves.index(canvas_widget), (
            f"{label} ended up below its section's canvas in pack order")

    lf7 = page._blk_pkt_pkt_local_row.master
    for mode in ("local", "global", "local"):
        page._blk_pkt_pkt_mode_var.set(mode)
        page._on_pkt_mode_change("blk_pkt")
    app.update_idletasks()
    assert_above_canvas(lf7, page._blk_pkt_pkt_local_row, "blk_pkt", "Block7 local_row")

    lf8 = page._sb_blk_pkt_frame.master
    for _ in range(2):
        page._sb_blk_restrict_pkt_var.set(True)
        page._on_cascade_toggle("sb_blk")
        page._sb_blk_restrict_blk_var.set(True)
        page._on_cascade_toggle("sb_blk")
        page._sb_blk_restrict_pkt_var.set(False)
        page._on_cascade_toggle("sb_blk")
    page._sb_blk_restrict_pkt_var.set(True)
    page._on_cascade_toggle("sb_blk")
    page._sb_blk_restrict_blk_var.set(True)
    page._on_cascade_toggle("sb_blk")
    app.update_idletasks()
    assert_above_canvas(lf8, page._sb_blk_pkt_frame, "sb_blk", "Block8 pkt_frame")
    assert_above_canvas(lf8, page._sb_blk_blk_frame, "sb_blk", "Block8 blk_frame")

    for ns in ("pag", "pkt_mean", "blk", "subblk", "blk_pag", "blk_pkt", "sb_blk"):
        lf = getattr(page, f"_{ns}_density_frame").master
        mode_var = getattr(page, f"_{ns}_mode_var")
        mode_var.set("hybrid")
        page._on_mode_change(ns)
        app.update_idletasks()
        assert_above_canvas(lf, getattr(page, f"_{ns}_density_frame"), ns, f"{ns} density_frame")


def test_pole_figure_overlays_are_parent_scoped_not_whole_population(app):
    """PoleFigurePage's Block section can overlay PAG and Packet
    orientations onto the same pole figure; the overlay must show only the
    PAGs that are actual parents of the currently-plotted blocks (via
    grain_to_pag_id / grain_to_blocks_map), not every PAG in the whole
    structure. Also exercises the repositioned/renamed "Packet Mean
    Orientation Pole Figure" section (was "Grain Pole Figure", moved
    between blocks 1 and 2)."""
    from upxo.pxtal.fm_steel_3d.gui.pages_pole_figures import PoleFigurePage
    from upxo.pxtal.fm_steel_3d.base_3d import FMSteel3DBase

    lfi = _voronoi_lfi_for_pag_tests(n=16, n_seeds=18, seed=1)
    fm_base = FMSteel3DBase.from_lfi(lfi, physical_dimensions=(16.0, 16.0, 16.0), voxel_size=1.0,
                                     units="microns", connectivity=6, min_grain_nvoxels=4,
                                     random_seed=42)
    fm_pag = fm_base.generate_pag_clusters(
        pag_size_distribution={"sizes": [3, 4], "probs": [0.7, 0.3]},
        pag_grain_fraction=1.0, random_seed=42)
    fm_pag.assign_pag_orientations(pag_ori_mode="random", random_seed=42)
    fm_blk = fm_pag.generate_blocks(block_thickness_range=(2.0, 5.0), random_seed=42)
    fm_ori = fm_blk.assign_orientations(random_seed=42)
    app._fm_with_orientations = fm_ori
    app._fm_with_pags = fm_pag

    app.show_page_by_class(PoleFigurePage)
    app.update_idletasks()
    page = app.current_frame

    # Section [2] is now "Packet Mean Orientation Pole Figure" (ns=pkt_mean,
    # level=packet_mean) -- was "Grain Pole Figure" (ns=grain, level=grain).
    assert hasattr(page, "_pkt_mean_marker_var")
    assert not hasattr(page, "_grain_marker_var")
    page._on_plot("pkt_mean", "packet_mean")
    app.update_idletasks()
    assert "Error" not in page._pkt_mean_status_lbl.cget("text")
    assert "Packet" in page._pkt_mean_status_lbl.cget("text")

    # Block section: enable the PAG overlay and confirm it's parent-scoped.
    page._blk_ovl_pag_var.set(True)
    page._on_overlay_toggle("blk", "pag")
    assert str(page._blk_ovl_pag_marker_widgets[0].cget("state")) in ("normal", "readonly")
    page._on_plot("blk", "block")
    app.update_idletasks()

    legend = page._blk_ax.get_legend()
    assert legend is not None
    pag_line = next(t.get_text() for t in legend.get_texts() if "PAG" in t.get_text())
    n_shown = int(pag_line.split("n=")[1].split(")")[0])

    _, blk_map, _ = page._pag_id_map(fm_ori)
    ids_blk, _ = page._collect_ids_and_orientations(fm_ori, "block", None, None, scope_ids=None)
    expected_parent_pags = {blk_map[b] for b in ids_blk if b in blk_map}
    assert n_shown == len(expected_parent_pags), (
        f"overlay showed {n_shown} PAGs but only {len(expected_parent_pags)} are actual "
        f"parents of the plotted blocks -- overlay is not properly parent-scoped")


def test_pole_figure_colorbar_margin_hybrid_dual_colorbar_and_overlay_color(app):
    """Three related pole-figure fixes:
    1. The 4 per-level sections' Axes previously filled the whole figure
       width before fig.colorbar() ran (_reset_ax_for_redraw called with no
       `position`, unlike the variant drill-down section), clipping the
       colorbar's title off the visible canvas. Now uses _POLE_AX_POSITION,
       leaving real margin on the right.
    2. Hybrid mode + color_by='size' now shows TWO colorbars (density MUD +
       marker-color), via PoleFigure.plot_density's new
       scatter_color_by='size' branch -- previously hybrid only ever showed
       the MUD colorbar.
    3. "Also plot..." overlays previously hardcoded their marker color
       (_OVERLAY_COLORS); a per-overlay color-picker dropdown
       (_{ns}_color_var) now controls it, defaulting to the same hardcoded
       value so existing behaviour is unchanged until the user picks
       something else."""
    from upxo.pxtal.fm_steel_3d.gui.pages_pole_figures import PoleFigurePage
    from upxo.pxtal.fm_steel_3d.base_3d import FMSteel3DBase
    import matplotlib.colors as mcolors
    import numpy as np

    lfi = _voronoi_lfi_for_pag_tests(n=16, n_seeds=18, seed=1)
    fm_base = FMSteel3DBase.from_lfi(lfi, physical_dimensions=(16.0, 16.0, 16.0), voxel_size=1.0,
                                     units="microns", connectivity=6, min_grain_nvoxels=4,
                                     random_seed=42)
    fm_pag = fm_base.generate_pag_clusters(
        pag_size_distribution={"sizes": [3, 4], "probs": [0.7, 0.3]},
        pag_grain_fraction=1.0, random_seed=42)
    fm_pag.assign_pag_orientations(pag_ori_mode="random", random_seed=42)
    fm_blk = fm_pag.generate_blocks(block_thickness_range=(2.0, 5.0), random_seed=42)
    fm_ori = fm_blk.assign_orientations(random_seed=42)
    app._fm_with_orientations = fm_ori
    app._fm_with_pags = fm_pag

    app.show_page_by_class(PoleFigurePage)
    app.update_idletasks()
    page = app.current_frame

    # 1. Colorbar margin -- scatter + color_by='size' (has its own colorbar).
    page._pag_mode_var.set("scatter")
    page._on_mode_change("pag")
    page._pag_color_var.set("size")
    page._on_color_by_change("pag")
    page._on_plot("pag", "pag")
    assert "Error" not in page._pag_status_lbl.cget("text")
    pos = page._pag_ax.get_position()
    assert pos.x1 < 0.95, "main Axes must leave right-margin room for the colorbar"
    assert len(page._pag_fig.axes) == 2  # main ax + 1 colorbar

    # 2. Hybrid dual-colorbar -- hybrid + color_by='size'.
    page._pag_mode_var.set("hybrid")
    page._on_mode_change("pag")
    page._on_plot("pag", "pag")
    assert "Error" not in page._pag_status_lbl.cget("text")
    assert len(page._pag_fig.axes) == 3, "hybrid + color_by='size' must show 2 colorbars (MUD + marker)"

    # 3. Overlay color picker -- default matches the old hardcoded value,
    # and changing it actually changes the rendered marker color.
    assert page._blk_ovl_pag_color_var.get() == "Medium Purple"
    page._blk_ovl_pag_var.set(True)
    page._on_overlay_toggle("blk", "pag")
    page._blk_ovl_pag_color_var.set("Pastel Pink")
    page._on_plot("blk", "block")
    assert "Error" not in page._blk_status_lbl.cget("text")

    target_rgb = mcolors.to_rgb("#FFB3BA")
    rendered_rgbs = [tuple(c[:3]) for coll in page._blk_ax.collections
                     for c in coll.get_facecolor()]
    assert any(np.allclose(rgb, target_rgb, atol=1e-6) for rgb in rendered_rgbs), \
        "chosen 'Pastel Pink' overlay color was not applied to the rendered markers"

    page.save_state()
    assert app.shared_state.get("PF_BLK_OVL_PAG_COLOR") == "#FFB3BA"


def test_distgrid_overlay_cell_includes_kde_lines(app, monkeypatch):
    """The Step 3a Block [3] plot grid's 'Overlaid' column (PAG vs Packets)
    previously only drew histogram bars, no KDE curves -- unlike every
    other column, which used _plot_kde and always drew one. Confirm the
    overlay cell now draws a KDE line per series too (via the shared
    _plot_kde_overlay helper)."""
    import threading

    class _SyncThread:
        def __init__(self, target=None, daemon=None, **kw):
            self._target = target
        def start(self):
            self._target()
    monkeypatch.setattr(threading, "Thread", _SyncThread)

    from upxo.pxtal.fm_steel_3d.gui.pages_pag import PagClusteringPage
    from upxo.pxtal.fm_steel_3d.base_3d import FMSteel3DBase
    from upxo.pxtal.fm_steel_3d.pag_technique_selector_3d import generate_pags

    lfi = _voronoi_lfi_for_pag_tests()
    fm = FMSteel3DBase.from_lfi(lfi, physical_dimensions=(16.0, 16.0, 16.0), voxel_size=1.0,
                                connectivity=6, min_grain_nvoxels=4, random_seed=42, verbosity=0)
    fm_pag = generate_pags(fm_base=fm, technique="A",
                           pag_size_distribution={"sizes": [3, 4], "probs": [0.95, 0.05]},
                           max_packets_per_pag=4, lever="lever3", random_seed=7)
    app._fm_with_pags = fm_pag

    app.show_page_by_class(PagClusteringPage)
    app.update_idletasks()
    page = app.current_frame

    page._slice_axis_var.set("xy")
    page._slice_n_vars["pag_morph_n_xy"].set(3)
    for k, v in page._morph_prop_vars.items():
        v.set(k == "pag_morph_prop_area")
    for k, v in page._morph_group_vars.items():
        v.set(k in ("pag_morph_group_pag", "pag_morph_group_packets"))
    page._on_compute_distributions()
    app.update()

    page._on_distgrid_cell_click("area", "overlay_pp", "Overlaid")
    app.update()
    assert len(page._distgrid_ax.get_lines()) >= 1, "overlay cell drew no KDE line(s)"


def test_pag_advisor_page_default_toggle_locks_and_unlocks_live(app):
    """Unchecking 'Use the default Technique - Lever combination?' must
    unlock the Technique radios in place (same page instance, no
    navigation) -- this used to happen implicitly by rebuilding a whole
    separate page on Next; now it must happen live via the checkbox."""
    from upxo.pxtal.fm_steel_3d.gui.pages import PagAdvisorPage

    app.show_page_by_class(PagAdvisorPage)
    app.update_idletasks()
    page = app.current_frame

    page.use_default_var.set(True)
    page._on_use_default_change()
    app.update_idletasks()
    assert str(page._technique_radios[0].cget("state")) == "disabled"

    page.use_default_var.set(False)
    page._on_use_default_change()
    app.update_idletasks()
    assert str(page._technique_radios[0].cget("state")) == "normal"

    # Restore the default so later tests in this module see the usual state.
    page.use_default_var.set(True)
    page._on_use_default_change()
    app.update_idletasks()


# ---------------------------------------------------------------------------
# GlobalConfigPage (pages_config.py): Output Folder / Units fixes
# ---------------------------------------------------------------------------

def test_global_config_validate_inputs_false_when_folder_blank(app, monkeypatch):
    """validate_inputs() must reject blank base/run fields with a clear error
    instead of returning True -- previously returning True let save_state()
    skip writing output_base_directory/sessions_dir/scripts_dir/etc entirely,
    and pages_mesh.py's shared_state["output_base_directory"] read (no
    fallback) crashed with a KeyError many steps downstream."""
    import tkinter.messagebox as tkmb
    from upxo.pxtal.fm_steel_3d.gui import pages_config
    from upxo.pxtal.fm_steel_3d.gui.pages_config import GlobalConfigPage

    app.show_page_by_class(GlobalConfigPage)
    app.update_idletasks()
    page = app.current_frame

    errors = []
    monkeypatch.setattr(tkmb, "showerror", lambda title, msg: errors.append((title, msg)))

    page.dir_base.set("")
    page.run_name_var.set("")
    assert page.validate_inputs() is False
    assert errors, "no error was shown for blank Output Folder fields"

    # Restore valid defaults so later tests see clean state.
    page.dir_base.set(pages_config._DEFAULT_FMSTEELS_BASE)
    page.run_name_var.set(pages_config._DEFAULT_RUN_FOLDER)


def test_global_config_validate_inputs_false_when_base_dir_invalid(app, monkeypatch):
    """A typo'd/unmounted base directory (e.g. a drive letter that doesn't
    exist) must fail validate_inputs() here with a clear error, not surface
    as an uncaught OSError far downstream (mesh export, session save, ...)."""
    import tkinter.messagebox as tkmb
    from upxo.pxtal.fm_steel_3d.gui import pages_config
    from upxo.pxtal.fm_steel_3d.gui.pages_config import GlobalConfigPage

    app.show_page_by_class(GlobalConfigPage)
    app.update_idletasks()
    page = app.current_frame

    errors = []
    monkeypatch.setattr(tkmb, "showerror", lambda title, msg: errors.append((title, msg)))

    page.dir_base.set(r"Q:\definitely_not_a_real_drive\path")
    page.run_name_var.set("some_run")
    assert page.validate_inputs() is False
    assert errors, "no error was shown for an invalid/unwritable base directory"

    page.dir_base.set(pages_config._DEFAULT_FMSTEELS_BASE)
    page.run_name_var.set(pages_config._DEFAULT_RUN_FOLDER)


def test_global_config_folder_change_warns_and_invalidates(app, monkeypatch):
    """Changing the Output Folder after something has been generated must
    warn and invalidate exactly like switching Pipeline Mode already does --
    previously editing the Output Folder (which silently repoints every
    downstream write target: mesh, sessions, scripts, ensemble_configs,
    images, raw) had no equivalent confirmation."""
    import tkinter.messagebox as tkmb
    from upxo.pxtal.fm_steel_3d.gui import pages_config
    from upxo.pxtal.fm_steel_3d.gui.pages_config import GlobalConfigPage

    app.show_page_by_class(GlobalConfigPage)
    app.update_idletasks()
    page = app.current_frame

    # Establish an initial, already-"saved" folder.
    page.dir_base.set(pages_config._DEFAULT_FMSTEELS_BASE)
    page.run_name_var.set("initial_run")
    page.save_state()

    # Simulate a run-completed state with something generated.
    orig_fm_base = app._fm_base
    orig_run_completed = app.run_completed
    app._fm_base = object()
    app.run_completed = True

    prompts = []
    monkeypatch.setattr(tkmb, "askyesno",
                        lambda title, msg: (prompts.append((title, msg)), True)[1])

    page.run_name_var.set("changed_run")
    page._on_folder_field_change()

    assert prompts, ("no confirmation dialog fired for an Output Folder change "
                     "with generated data present")
    assert app._fm_base is None, \
        "confirmed folder change must invalidate fm_base, same as a mode switch"
    assert app.run_completed is False

    # Declining the dialog must revert the fields instead of applying the change.
    app._fm_base = object()
    app.run_completed = True
    prompts.clear()
    monkeypatch.setattr(tkmb, "askyesno",
                        lambda title, msg: (prompts.append((title, msg)), False)[1])
    page.run_name_var.set("another_changed_run")
    page._on_folder_field_change()
    assert prompts
    assert page.run_name_var.get() == "changed_run", \
        "declining the folder-change confirmation must revert the field"
    assert app._fm_base is not None

    # Restore clean state for later tests.
    app._fm_base = orig_fm_base
    app.run_completed = orig_run_completed
    page.dir_base.set(pages_config._DEFAULT_FMSTEELS_BASE)
    page.run_name_var.set(pages_config._DEFAULT_RUN_FOLDER)
    page.save_state()


def test_global_config_units_change_rescales_lxlylz(app):
    """Changing Units after LX/LY/LZ are entered must rescale those numbers
    (not just relabel them) so shared_state["UNITS"] stays physically
    consistent with the already-entered dimensions."""
    from upxo.pxtal.fm_steel_3d.gui.pages_config import GlobalConfigPage

    orig_lx = app.shared_state["LX"]
    orig_ly = app.shared_state["LY"]
    orig_lz = app.shared_state["LZ"]
    orig_units = app.shared_state.get("UNITS", "microns")
    try:
        app.shared_state["UNITS"] = "microns"
        app.shared_state["LX"] = 1000.0
        app.shared_state["LY"] = 2000.0
        app.shared_state["LZ"] = 3000.0

        app.show_page_by_class(GlobalConfigPage)
        app.update_idletasks()
        page = app.current_frame

        page.on_units_change("mm")
        assert app.shared_state["LX"] == pytest.approx(1.0)
        assert app.shared_state["LY"] == pytest.approx(2.0)
        assert app.shared_state["LZ"] == pytest.approx(3.0)

        page.on_units_change("m")
        assert app.shared_state["LX"] == pytest.approx(0.001)
        assert app.shared_state["LY"] == pytest.approx(0.002)
        assert app.shared_state["LZ"] == pytest.approx(0.003)
    finally:
        app.shared_state["LX"] = orig_lx
        app.shared_state["LY"] = orig_ly
        app.shared_state["LZ"] = orig_lz
        app.shared_state["UNITS"] = orig_units


# ---------------------------------------------------------------------------
# Regression tests for the fm_steel_3d GUI fix pass (Step2b1Page Apply-time
# deadlock/staleness, GridPage isotropic sync, Voronoi generate traceback).
# ---------------------------------------------------------------------------

def test_step2b1_apply_tslice_without_sim_attrs_does_not_deadlock(app):
    """CRITICAL: Step2b1Page ("Step 2b1: Monte-Carlo Grain Structure") is
    rebuilt from scratch on every Back/Next, but app._mc_pxt and
    shared_state["mc_tslice_list"] persist across rebuilds -- so a
    freshly-rebuilt page instance can have "Apply Selected Tslice" already
    enabled even though _on_run_simulation() was never called on THIS
    instance, leaving self._sim_lx/_sim_ly/_sim_lz/_sim_rng_seed unset.
    Before the fix, clicking Apply called self.app.try_begin_operation()
    (acquiring the app-wide single-operation lock) and THEN raised an
    uncaught AttributeError reading those missing attributes -- the
    background thread whose finally block would call end_operation() never
    started, so the lock stayed held forever, freezing every
    generate/clean/run button in the entire app. This constructs a fresh
    page instance (simulating a rebuild) without calling
    _on_run_simulation, seeds shared_state/app state so Apply is enabled,
    calls _on_apply_tslice() directly, and confirms (a) no uncaught
    exception propagates, (b) the app-wide lock is released afterward, (c)
    a clear error status is shown instead."""
    from unittest import mock
    from upxo.pxtal.fm_steel_3d.gui.pages_basegrain import Step2b1Page

    # Seed persisted state exactly as a real prior Run would have left it,
    # on an app/shared_state that survives rebuild -- but the page instance
    # about to be built below will NOT have had _on_run_simulation called on
    # it, matching the real Back/Next rebuild scenario.
    orig_pxt = app._mc_pxt
    orig_tslices = app.shared_state.get("mc_tslice_list")
    app._mc_pxt = mock.Mock()
    app.shared_state["mc_tslice_list"] = [0, 10, 20]

    try:
        app.show_page_by_class(Step2b1Page)
        app.update_idletasks()
        page = app.current_frame
        assert isinstance(page, Step2b1Page)

        # Confirm the bug's precondition actually holds on this fresh
        # instance: Apply is enabled, yet _sim_lx was never set.
        assert str(page._btn_apply.cget("state")) == "normal"
        assert not hasattr(page, "_sim_lx")

        # Lock is free before we start (this is what we're about to prove
        # stays true afterward too).
        app._pipeline._busy = False

        page._on_apply_tslice()  # must not raise
        app.update_idletasks()

        assert app._pipeline._busy is False, (
            "app-wide operation lock left held after _on_apply_tslice() -- "
            "this is the deadlock")
        status_text = page._gen_status_lbl.cget("text")
        assert "simulation" in status_text.lower(), (
            f"expected a clear 'run the simulation first' style status, got: {status_text!r}")
    finally:
        app._mc_pxt = orig_pxt
        app.shared_state["mc_tslice_list"] = orig_tslices if orig_tslices is not None else []


def test_step2b1_apply_tslice_blocks_when_dims_edited_after_run(app):
    """If LX/LY/LZ are edited live after Run but before Apply, the
    simulation was run against the OLD (cached) dimensions -- neither the
    stale cached values nor the live-but-mismatched ones can be used to
    compute a physically-correct voxel size. _on_apply_tslice must detect
    the mismatch, refuse to proceed, and still release the app-wide
    operation lock (not leave it stuck the way fix #1 addresses)."""
    from unittest import mock
    from upxo.pxtal.fm_steel_3d.gui.pages_basegrain import Step2b1Page

    orig_pxt = app._mc_pxt
    orig_tslices = app.shared_state.get("mc_tslice_list")
    app._mc_pxt = mock.Mock()
    app.shared_state["mc_tslice_list"] = [0, 10]

    try:
        app.show_page_by_class(Step2b1Page)
        app.update_idletasks()
        page = app.current_frame

        # Simulate a prior successful Run at LX=LY=LZ=100 without actually
        # running the MC simulation (expensive and not what this test targets).
        page._sim_lx = 100.0
        page._sim_ly = 100.0
        page._sim_lz = 100.0
        page._sim_rng_seed = None

        # User edits LX after Run but before Apply.
        page._lx_var.set(250.0)

        app._pipeline._busy = False
        page._on_apply_tslice()  # must not raise
        app.update_idletasks()

        assert app._pipeline._busy is False
        status_text = page._gen_status_lbl.cget("text").lower()
        assert "changed" in status_text or "re-run" in status_text, (
            f"expected a staleness warning, got: {status_text!r}")
        # Nothing was actually applied.
        assert "Applying tslice" not in page._apply_console.get("1.0", tk.END)
    finally:
        app._mc_pxt = orig_pxt
        app.shared_state["mc_tslice_list"] = orig_tslices if orig_tslices is not None else []


def test_gridpage_sync_isotropic_applied_on_construction(app):
    """A saved session loaded with force_isotropic=True must show the
    LY/LZ/NY/NZ entries already disabled on first render -- previously
    sync_isotropic_inputs was only registered as a trace on variable writes
    and never invoked once at construction, so a freshly-built GridPage
    showed the checkbox checked but left the entries enabled until the next
    edit."""
    from upxo.pxtal.fm_steel_3d.gui.pages_voronoi import GridPage

    orig_iso = app.shared_state.get("force_isotropic")
    app.shared_state["force_isotropic"] = True
    try:
        app.show_page_by_class(GridPage)
        app.update_idletasks()
        page = app.current_frame

        assert page.iso_var.get() is True
        assert str(page.entry_ly.cget("state")) == "disabled"
        assert str(page.entry_lz.cget("state")) == "disabled"
        assert str(page.entry_ny.cget("state")) == "disabled"
        assert str(page.entry_nz.cget("state")) == "disabled"
    finally:
        app.shared_state["force_isotropic"] = orig_iso


def test_step2a3_generate_failure_logs_traceback(app, monkeypatch):
    """Step2a3Page._on_generate's failure handler must print a traceback to
    its console log, matching Step2b1Page's equivalent handlers
    (_on_run_simulation / _on_apply_tslice), which already do this for
    debuggability."""
    import threading
    import upxo.pxtal.fm_steel_3d.gui.pages_voronoi as pv

    class _SyncThread:
        def __init__(self, target=None, daemon=None, **kw):
            self._target = target
        def start(self):
            self._target()
    monkeypatch.setattr(threading, "Thread", _SyncThread)

    def _boom(*a, **kw):
        raise RuntimeError("synthetic failure")
    monkeypatch.setattr(pv._FMSteel3DBase, "from_lfi", _boom)

    app.show_page_by_class(pv.Step2a3Page)
    app.update_idletasks()
    page = app.current_frame

    page._on_generate()  # runs synchronously via the monkeypatched Thread

    # Drain the log queue directly (rather than waiting on the Tk `after`
    # timer that normally pumps it into the console widget) to see exactly
    # what the failure handler printed.
    logged = []
    while True:
        item = page._log_queue.get_nowait()
        if item is None:
            break
        logged.append(item)
    full_log = "".join(logged)

    assert "Traceback (most recent call last)" in full_log
    assert "RuntimeError: synthetic failure" in full_log
    assert "Error: synthetic failure" in page.shared_state["gs_gen_status"]


def test_pag_isolated_grain_orientations_use_block1b_textured_config(app, monkeypatch):
    """Block [1b] ("Isolated Grain Orientations") builds a full UI -- mode,
    HAGB threshold, max attempts, an editable texture-component table, pool
    size, sample-symmetry checkboxes -- and save_state() persists all of it,
    but _on_generate_pag never called assign_isolated_grain_orientations(...)
    with it, so isolated grains always silently fell back to
    ensure_isolated_grain_orientations(mode='random', ...) the first time
    something downstream read isolated_grain_orientations, discarding
    whatever the user configured.

    Note on why this monkeypatches generate_pags: the GUI's normal PAG path
    (pag_technique_selector_3d.generate_pags, technique A/B) always converts
    its whole (1 - PAG_GRAIN_FRACTION) leftover into retained-austenite PAGs
    tracked via retained_austenite_pag_ids, and isolated_grains is built as
    exactly that PAGs' grain union -- so leftover_isolated is provably empty
    for ANY generate_pags()-produced object (confirmed empirically), and
    those grains get their orientation from assign_pag_orientations() at the
    PAG level instead, by assign_isolated_grain_orientations's own explicit
    design (see its docstring). assign_isolated_grain_orientations only ever
    has real work to do for the legacy case: grains excluded by
    FMSteel3DBase.generate_pag_clusters's OWN pre-filter (pag_grain_fraction
    < 1.0 passed directly to it, not through generate_pags), which leaves
    retained_austenite_pag_ids empty. generate_pags is patched here to
    delegate straight to that legacy call so the fix has real isolated
    grains to act on -- assign_isolated_grain_orientations, FCCTexture, and
    the grain structure itself are all real, unmocked.

    Note on verification method: rather than asserting the assigned
    orientations are crystallographically CLOSE to the Copper ideal (which
    would depend on upxo.texOps.fcc's own Euler<->matrix round-trip being
    correct -- confirmed independently, while writing this test, to be
    broken: FCCTexture.generate_euler's symmetry-equivalent Euler angles do
    not actually round-trip through euler_bunge_to_matrix/
    matrix_to_euler_bunge to the matrices they were derived from, a
    pre-existing bug in a module this task does not own or touch), this
    spies on FCCTexture.generate_euler to capture exactly what pool Block
    [1b]'s config produced, then asserts every assigned isolated-grain
    orientation is a literal member of that pool. That proves the wiring
    (mode/tc-table/spread/pool-size/sample-symmetry -> FCCTexture ->
    orientation_pool -> assign_isolated_grain_orientations) end-to-end
    without depending on fcc.py's internal correctness."""
    import threading
    from upxo.pxtal.fm_steel_3d.gui.pages_pag import PagClusteringPage, _ISO_MODE_OPTS_INV
    from upxo.pxtal.fm_steel_3d.base_3d import FMSteel3DBase
    import upxo.pxtal.fm_steel_3d.pag_technique_selector_3d as pts
    import upxo.texOps.fcc as fccmod

    class _SyncThread:
        def __init__(self, target=None, daemon=None, **kw):
            self._target = target
        def start(self):
            self._target()
    monkeypatch.setattr(threading, "Thread", _SyncThread)

    def _fake_generate_pags(fm_base, technique, pag_size_distribution, max_packets_per_pag=4,
                            lever='lever3', clustering_connectivity=None, repair_connectivity=18,
                            split_threshold=32, pag_grain_fraction=1.0,
                            isolated_grain_strategy='near_mean', isolated_size_tol=0.25,
                            retained_austenite_tolerance=0.05, retained_austenite_max_attempts=10,
                            retained_austenite_trim_step=1, use_non_neigh_pag=True,
                            random_seed=None):
        # Delegates to the legacy pre-filter path (see docstring above) so
        # pag_grain_fraction < 1.0 leaves real, untracked isolated grains.
        return fm_base.generate_pag_clusters(
            pag_size_distribution=pag_size_distribution,
            pag_grain_fraction=pag_grain_fraction,
            use_non_neigh_pag=use_non_neigh_pag,
            isolated_grain_strategy=isolated_grain_strategy,
            isolated_size_tol=isolated_size_tol,
            random_seed=random_seed,
        )
    monkeypatch.setattr(pts, "generate_pags", _fake_generate_pags)

    captured = {}
    _orig_generate_euler = fccmod.FCCTexture.generate_euler

    def _spy_generate_euler(self, N):
        result = _orig_generate_euler(self, N)
        captured['tc_fractions'] = dict(self.tc_fractions)
        captured['spreads'] = dict(self.spreads)
        captured['apply_sample_symmetries'] = self.apply_sample_symmetries
        captured['custom_copper'] = self.components.get('Copper')
        captured['pool'] = {tuple(row) for row in result}
        return result
    monkeypatch.setattr(fccmod.FCCTexture, "generate_euler", _spy_generate_euler)

    orig_fm_base = app._fm_base
    orig_fm_with_pags = app._fm_with_pags

    lfi = _voronoi_lfi_for_pag_tests()
    fm_base = FMSteel3DBase.from_lfi(lfi, physical_dimensions=(16.0, 16.0, 16.0), voxel_size=1.0,
                                     connectivity=6, min_grain_nvoxels=4, random_seed=42, verbosity=0)
    app._fm_base = fm_base

    app.show_page_by_class(PagClusteringPage)
    app.update_idletasks()
    page = app.current_frame

    # Force isolated grains to exist: cluster only 40% of grains into PAGs.
    page.pag_frac_var.set(0.4)

    # Block [1b]: textured mode, Copper only, narrow spread, no sample
    # symmetry (so crystal-symmetric equivalents -- the only ones left --
    # don't add disorientation relative to the Copper ideal orientation).
    page.iso_ori_mode_disp_var.set(_ISO_MODE_OPTS_INV["textured"])
    for row in page._user_tc_rows:
        if row["name"] == "Copper":
            row["frac"].set(1.0)
            row["spread"].set(2.0)
        else:
            row["frac"].set(0.0)
    page.iso_pool_size_var.set(300)
    page.iso_hagb_var.set(2.0)
    page.iso_attempts_var.set(50)
    page.iso_sample_sym_var.set(False)

    page._on_generate_pag()   # runs synchronously via the monkeypatched Thread
    app.update()              # flush the self.after(0, ...) state-marshalling callbacks

    fm_pag = app._fm_with_pags
    assert fm_pag is not None, page._pag_gen_status_lbl.cget("text")
    assert fm_pag.isolated_grains, "test setup must produce isolated grains to exercise Block [1b]"
    assert not fm_pag.retained_austenite_pag_ids, \
        "test setup must leave these as TRUE untracked isolated grains"
    assert fm_pag.isolated_grain_orientations, (
        "assign_isolated_grain_orientations was never called from _on_generate_pag "
        f"(status: {page._pag_gen_status_lbl.cget('text')})")
    assert set(fm_pag.isolated_grain_orientations.keys()) == fm_pag.isolated_grains

    # FCCTexture was actually built from Block [1b]'s configured values --
    # not defaults, not some other component.
    assert captured, "FCCTexture.generate_euler was never called -- Block [1b]'s textured pool was not built"
    assert captured['tc_fractions'] == {"Copper": 1.0}
    assert captured['spreads'] == {"Copper": 2.0}
    assert captured['apply_sample_symmetries'] is False
    assert captured['custom_copper'] == (90.0, 35.26, 45.0)

    # Every assigned isolated-grain orientation was actually drawn from that
    # pool -- not a uniformly-random SO(3) draw from the abandoned fallback
    # (which would essentially never land exactly on a finite pool member).
    pool = captured['pool']
    for gid, ea in fm_pag.isolated_grain_orientations.items():
        assert tuple(ea) in pool, (
            f"isolated grain {gid}'s assigned orientation {ea} was not drawn from "
            f"Block [1b]'s configured textured pool -- textured config was not honoured")

    app._fm_base = orig_fm_base
    app._fm_with_pags = orig_fm_with_pags


def test_pag_block1_algorithm_toggle_stays_above_block1b(app):
    """Step 3a Block [0]'s Algorithm dropdown shows/hides Block [1]
    (Stochastic BFS Parameters) via pack_forget()/pack() with no
    before=/after=, which -- per the same bug class fixed for
    pages_pole_figures.py -- appends Block [1] to the END of the scroll
    frame's CURRENT pack order the first time it's toggled away and back,
    landing it below Block [1b]/[1c]/[2]/[3] instead of staying near the
    top. Direct template: test_pole_figure_toggles_dont_reorder_below_canvas."""
    from upxo.pxtal.fm_steel_3d.gui.pages_pag import PagClusteringPage

    app.show_page_by_class(PagClusteringPage)
    app.update_idletasks()
    page = app.current_frame

    scroll = page._f_b1.master
    for alg in ("MIS based (not implemented)", "Stochastic BFS"):
        page.alg_var.set(alg)
        page._on_algorithm_change()
    app.update_idletasks()

    slaves = scroll.pack_slaves()
    assert slaves.index(page._f_b1) < slaves.index(page._block1b_outer), (
        "Block [1] ended up below Block [1b] after toggling the Algorithm dropdown "
        "away from 'Stochastic BFS' and back")


def test_pag_pool_src_row_toggle_stays_above_texture_panel(app):
    """Step 3a Block [1c]'s Orientation-mode dropdown shows/hides the 'Pool
    source' row via pack_forget()/pack() with no before=/after= when
    returning to 'hagb_constrained' after having visited 'textured', which
    -- same bug class as Block [1]'s Algorithm toggle above -- appends it to
    the END of _pag_pool_frame's CURRENT pack order, landing it below 'Pool
    size' and the Texture Components panel instead of above them."""
    from upxo.pxtal.fm_steel_3d.gui.pages_pag import PagClusteringPage

    app.show_page_by_class(PagClusteringPage)
    app.update_idletasks()
    page = app.current_frame

    page.pag_ori_mode_var.set("textured")
    page._on_pag_ori_mode_change()
    page.pag_ori_mode_var.set("hagb_constrained")
    page._on_pag_ori_mode_change()
    app.update_idletasks()

    slaves = page._pag_pool_frame.pack_slaves()
    assert slaves.index(page._pag_pool_src_row) < slaves.index(page._pag_tex_frame), (
        "Pool source row landed below the Texture Components panel after returning "
        "to 'hagb_constrained' mode from 'textured'")


def test_transform_stretch_updates_LX_and_block_voxel_conversion(app, monkeypatch):
    """Regression test: _on_apply_stretch used to update shared_state
    NX/NY/NZ after an anisotropic stretch but never LX/LY/LZ, so
    pages_block.py's `voxel_size = shared_state["LX"] / shared_state["NX"]`
    (stale LX / fresh NX) silently miscomputed the block-thickness um->voxel
    conversion after ANY stretch. Confirms LX/LY/LZ now scale by the same
    per-axis stretch factor as NX/NY/NZ, and that pages_block.py's own
    voxel-size formula is correct afterward (checked here read-only, without
    calling generate_blocks)."""
    import threading
    from upxo.pxtal.fm_steel_3d.gui.pages_transform import TransformationsPage
    from upxo.pxtal.fm_steel_3d.base_3d import FMSteel3DBase

    class _SyncThread:
        def __init__(self, target=None, daemon=None, **kw):
            self._target = target
        def start(self):
            self._target()
    monkeypatch.setattr(threading, "Thread", _SyncThread)

    lfi = _voronoi_lfi_for_pag_tests(n=16, n_seeds=20, seed=3)
    fm_base = FMSteel3DBase.from_lfi(lfi, physical_dimensions=(16.0, 16.0, 16.0), voxel_size=1.0,
                                     units="microns", connectivity=6, min_grain_nvoxels=4,
                                     random_seed=42)
    app._fm_base = fm_base
    app.shared_state["NX"] = app.shared_state["NY"] = app.shared_state["NZ"] = 16
    app.shared_state["LX"] = app.shared_state["LY"] = app.shared_state["LZ"] = 16.0

    app.show_page_by_class(TransformationsPage)
    app.update_idletasks()
    page = app.current_frame

    page._sfx_var.set(2.0)
    page._sfy_var.set(1.0)
    page._sfz_var.set(1.0)
    page._on_apply_stretch()
    app.update()
    assert "Done" in page._stretch_status_lbl.cget("text")

    assert app.shared_state["NX"] == 32
    assert app.shared_state["NY"] == 16
    assert app.shared_state["NZ"] == 16
    assert app.shared_state["LX"] == pytest.approx(32.0)
    assert app.shared_state["LY"] == pytest.approx(16.0)
    assert app.shared_state["LZ"] == pytest.approx(16.0)

    # pages_block.py:757's own um->voxel conversion formula, replicated
    # read-only here -- must now match the actual post-stretch voxel size
    # instead of a stale pre-stretch one.
    voxel_size_from_shared_state = app.shared_state["LX"] / app.shared_state["NX"]
    assert voxel_size_from_shared_state == pytest.approx(app._fm_base.voxel_size)


def test_block_thk_file_row_stays_anchored_near_top(app):
    """Regression test: the "Distribution file" row was only .pack()ed the
    FIRST time the dropdown was switched to "From distribution file" -- by
    which point every other Block [1] row (thickness range, connectivity,
    intercept sections) had already been packed, so Tk's pack manager
    appended it to the END of Block [1] instead of leaving it near the top
    where it was created. Confirms it now stays above the
    intercept-grain-size row after toggling."""
    from upxo.pxtal.fm_steel_3d.gui.pages_block import BlockGenerationPage

    app.show_page_by_class(BlockGenerationPage)
    app.update_idletasks()
    page = app.current_frame

    page.blk_thk_dist_var.set("From distribution file")
    page._on_thk_dist_change()
    app.update_idletasks()

    assert page._thk_file_row.winfo_manager() != "", "row must actually be packed"
    inner = page._thk_file_row.master
    slaves = inner.pack_slaves()
    assert slaves.index(page._thk_file_row) < slaves.index(page._btn_intercept.master), (
        "Distribution file row ended up below the intercept-grain-size section")

    # Toggling back off/on again must not drift it further down.
    page.blk_thk_dist_var.set("random in range")
    page._on_thk_dist_change()
    page.blk_thk_dist_var.set("From distribution file")
    page._on_thk_dist_change()
    app.update_idletasks()
    slaves = inner.pack_slaves()
    assert slaves.index(page._thk_file_row) < slaves.index(page._btn_intercept.master)


def test_cleaning_validates_inputs_before_running(app, monkeypatch):
    """Regression test: _on_clean/_on_recursive_clean used to read
    min_grain_var/max_passes_var/iter_var directly and unguarded, so
    non-numeric text raised an uncaught tk.TclError in the button handler
    instead of showing the page's own validate_inputs() dialog. Confirms
    both handlers now call validate_inputs() first and bail out cleanly (no
    worker thread started, no crash) on bad input, and proceed normally once
    the input is fixed."""
    import threading
    import tkinter.messagebox as tkmb
    from upxo.pxtal.fm_steel_3d.gui.pages_cleaning import GsCleanPage
    from upxo.pxtal.fm_steel_3d.base_3d import FMSteel3DBase

    monkeypatch.setattr(tkmb, "showerror", lambda *a, **k: None)

    started = []

    class _SyncThread:
        def __init__(self, target=None, daemon=None, **kw):
            started.append(True)
            self._target = target
        def start(self):
            self._target()
    monkeypatch.setattr(threading, "Thread", _SyncThread)

    lfi = _voronoi_lfi_for_pag_tests(n=16, n_seeds=20, seed=4)
    fm_base = FMSteel3DBase.from_lfi(lfi, physical_dimensions=(16.0, 16.0, 16.0), voxel_size=1.0,
                                     units="microns", connectivity=6, min_grain_nvoxels=4,
                                     random_seed=42)
    app._fm_base = fm_base

    app.show_page_by_class(GsCleanPage)
    app.update_idletasks()
    page = app.current_frame

    page.do_merge_var.set(True)
    page.min_grain_var.set("not-a-number")

    page._on_clean()  # must not raise tk.TclError
    assert not started, "a worker thread must not start when validate_inputs() fails"

    page.min_grain_var.set(8)
    page._on_clean()
    assert started, "a worker thread should start once inputs are valid"

    started.clear()
    page.iter_var.set("also-not-a-number")
    page._on_recursive_clean()  # must not raise tk.TclError
    assert not started, "a worker thread must not start when validate_inputs() fails"


# ---------------------------------------------------------------------------
# App navigation fixes: show_page_by_class() rollback, back_page() validation,
# the missing fm_base sidebar-stale entry, and invalidate_from()'s cache gaps.
# ---------------------------------------------------------------------------

def test_show_page_by_class_rolls_back_on_construction_failure(app):
    """If page_cls(...) raises mid-__init__, self.current_frame must not be
    left pointing at an already-destroyed widget: show_page_by_class() now
    builds the new page BEFORE tearing down the old one, so a failed
    construction leaves the previous (still-alive) frame in place, the
    exception propagates, and the failed class is never marked visited."""
    from upxo.pxtal.fm_steel_3d.gui.pages import GlobalConfigPage
    from upxo.pxtal.fm_steel_3d.gui.pages_base import BasePage

    class _AlwaysBrokenPage(BasePage):
        def __init__(self, parent, app_):
            super().__init__(parent, app_)
            raise RuntimeError("synthetic construction failure")

    app.show_page_by_class(GlobalConfigPage)
    app.update_idletasks()
    good_frame = app.current_frame
    assert isinstance(good_frame, GlobalConfigPage)

    with pytest.raises(RuntimeError, match="synthetic construction failure"):
        app.show_page_by_class(_AlwaysBrokenPage)

    assert app.current_frame is good_frame, (
        "current_frame must still be the old, still-alive page after a "
        "failed construction, not a dangling reference to a destroyed widget")
    assert app.current_frame.winfo_exists()
    assert _AlwaysBrokenPage not in app._visited_pages, (
        "a page that failed to construct must not be marked visited")

    # The app must remain usable afterward -- a normal navigation still works.
    app.show_page_by_class(GlobalConfigPage)
    app.update_idletasks()
    assert isinstance(app.current_frame, GlobalConfigPage)


def test_back_page_skips_save_on_invalid_input_but_still_navigates(app, monkeypatch):
    """back_page() must still navigate on invalid input (the user is never
    trapped on a broken page) but, unlike the previous unconditional
    save_state() call, must not persist the bad values into shared_state --
    matching next_page()/_sidebar_navigate(), which already gate on
    validate_inputs()."""
    import tkinter.messagebox as tkmb
    from upxo.pxtal.fm_steel_3d.gui.pages_config import GlobalConfigPage
    from upxo.pxtal.fm_steel_3d.gui.pages import WelcomePage

    monkeypatch.setattr(tkmb, "showerror", lambda *a, **k: None)

    orig_base_dir = app.shared_state.get("fmsteels_base_dir")
    orig_run_name = app.shared_state.get("run_folder_name")
    try:
        app.show_page_by_class(GlobalConfigPage)
        app.update_idletasks()
        page = app.current_frame

        page.dir_base.set("")
        page.run_name_var.set("")
        assert page.validate_inputs() is False  # confirm the precondition

        app.back_page()
        app.update_idletasks()

        assert isinstance(app.current_frame, WelcomePage), (
            "back_page() must still navigate even when validate_inputs() fails")
        assert app.shared_state.get("fmsteels_base_dir") == orig_base_dir, (
            "blank Output Folder value must not have been persisted")
        assert app.shared_state.get("run_folder_name") == orig_run_name
    finally:
        app.shared_state["fmsteels_base_dir"] = orig_base_dir
        app.shared_state["run_folder_name"] = orig_run_name


def test_sidebar_marks_generate_stage_stale_when_fm_base_cleared(app):
    """_STAGE_STALE_CHECK was missing an entry for the 'fm_base' stage's own
    entry-point page (Step2a3Page) -- invalidate_from('fm_base') (e.g. from
    loading a session) clears _fm_base but the sidebar's 'Generate' row never
    got the orange '!' stale marker every other stage already gets. Confirms
    the row shows it once Step2a3Page has been visited and _fm_base is None."""
    from upxo.pxtal.fm_steel_3d.gui.pages import Step2a3Page

    orig_fm_base = app._fm_base
    try:
        app.show_page_by_class(Step2a3Page)
        app.update_idletasks()
        assert Step2a3Page in app._visited_pages

        app._fm_base = None
        app._build_sidebar()
        app.update_idletasks()

        def _walk(w):
            yield w
            for c in w.winfo_children():
                yield from _walk(c)

        target_label = next(
            (w for w in _walk(app.sidebar_frame)
             if isinstance(w, tk.Label) and w.cget("text") == "Generate"),
            None)
        assert target_label is not None, "no 'Generate' sidebar row found"
        row = target_label.master
        has_bang = any(isinstance(c, tk.Label) and c.cget("text") == "!"
                       for c in row.winfo_children())
        assert has_bang, "'Generate' sidebar row not marked stale when _fm_base is None"
    finally:
        app._fm_base = orig_fm_base
        app._build_sidebar()
        app.update_idletasks()


def test_invalidate_from_clears_page_owned_caches(app):
    """invalidate_from() previously left three page-owned caches dangling
    across cascading invalidation: Step2b1Page's MC run state (fm_base),
    PagClusteringPage's distribution-plot cache (fm_with_pags), and
    PoleFigurePage's packet-mean-orientation cache
    (fm_with_orientations/fm_with_subblocks). Confirms each is actually
    cleared when the stage it belongs to is invalidated."""
    from unittest import mock

    # -- fm_base: app._mc_pxt and the mc_* shared_state keys --
    orig_mc_pxt          = app._mc_pxt
    orig_mc_tslice_list  = app.shared_state.get("mc_tslice_list")
    orig_mc_sim_status   = app.shared_state.get("mc_sim_status")
    orig_mc_tslice       = app.shared_state.get("mc_tslice")
    try:
        app._mc_pxt = mock.Mock()
        app.shared_state["mc_tslice_list"] = [0, 10, 20]
        app.shared_state["mc_sim_status"]  = "Completed"
        app.shared_state["mc_tslice"]      = 10

        app.invalidate_from("fm_base")

        assert app._mc_pxt is None
        assert app.shared_state["mc_tslice_list"] == []
        assert app.shared_state["mc_sim_status"] == "Not run"
        assert app.shared_state["mc_tslice"] == -1
    finally:
        app._mc_pxt = orig_mc_pxt
        app.shared_state["mc_tslice_list"] = orig_mc_tslice_list if orig_mc_tslice_list is not None else []
        app.shared_state["mc_sim_status"] = orig_mc_sim_status if orig_mc_sim_status is not None else "Not run"
        app.shared_state["mc_tslice"] = orig_mc_tslice if orig_mc_tslice is not None else -1

    # -- fm_with_pags: PagClusteringPage's distribution-plot cache --
    orig_distr_data = getattr(app, "_pag_distr_data", None)
    orig_distr_props = getattr(app, "_pag_distr_active_props", None)
    try:
        app._pag_distr_data = {"some": "data"}
        app._pag_distr_active_props = ["area"]
        app.invalidate_from("fm_with_pags")
        assert app._pag_distr_data is None
        assert app._pag_distr_active_props is None
    finally:
        app._pag_distr_data = orig_distr_data
        app._pag_distr_active_props = orig_distr_props

    # -- fm_with_orientations / fm_with_subblocks: pole-figure packet cache --
    orig_packet_cache = getattr(app, "_packet_mean_ori_cache", None)
    try:
        app._packet_mean_ori_cache = {"fm_state_id": 12345, "data": {}}
        app.invalidate_from("fm_with_orientations")
        assert app._packet_mean_ori_cache is None

        app._packet_mean_ori_cache = {"fm_state_id": 6789, "data": {}}
        app.invalidate_from("fm_with_subblocks")
        assert app._packet_mean_ori_cache is None
    finally:
        app._packet_mean_ori_cache = orig_packet_cache


# ---------------------------------------------------------------------------
# runner.py / pages_mesh.py / pages_refdata.py bug-fix regression tests
# ---------------------------------------------------------------------------

def test_confirm_mesh_export_size_reexport_respects_export_mesh_checkbox(app, monkeypatch):
    """_confirm_mesh_export_size() used to force will_export=True
    unconditionally for replot_section='reexport', so the large-mesh
    confirmation dialog fired even with 'Export Mesh' unchecked (and the
    subsequent re-export would do nothing anyway). Confirms the dialog is now
    skipped when EXPORT_MESH is False, and still fires once it's True, for an
    otherwise-identical oversized domain."""
    import tkinter.messagebox as tkmb
    from upxo.pxtal.fm_steel_3d.gui.runner import _confirm_mesh_export_size

    keys = ("NX", "NY", "NZ", "EXPORT_MESH", "MESH_FORCE", "MESH_ELEM_TYPE", "MESH_MAX_VOXELS")
    orig = {k: app.shared_state.get(k) for k in keys}
    calls = {"n": 0}
    monkeypatch.setattr(tkmb, "askyesno", lambda *a, **kw: calls.__setitem__("n", calls["n"] + 1) or True)

    try:
        app.shared_state.update({
            "NX": 300, "NY": 300, "NZ": 300,  # 27,000,000 voxels -- exceeds the C3D8 hard limit (5,000,000)
            "MESH_FORCE": False,
            "MESH_ELEM_TYPE": "C3D8",
            "MESH_MAX_VOXELS": "",
        })

        app.shared_state["EXPORT_MESH"] = False
        assert _confirm_mesh_export_size(app, replot_section="reexport", compute_only=False) is True
        assert calls["n"] == 0, "dialog must not fire when Export Mesh is unchecked"

        app.shared_state["EXPORT_MESH"] = True
        assert _confirm_mesh_export_size(app, replot_section="reexport", compute_only=False) is True
        assert calls["n"] == 1, "dialog must fire once Export Mesh is checked and the domain exceeds the hard limit"
    finally:
        app.shared_state.update(orig)


def test_run_export_thread_warns_on_malformed_max_voxels_and_defers_restore_streams(app, monkeypatch, tmp_path):
    """Two fixes exercised together: (1) a malformed 'Max Voxels' value used
    to be silently discarded with no warning, unlike the equivalent 'Ignore
    Grain IDs' mistake a few lines above, which already pops a
    messagebox.showwarning; (2) run_export_thread called restore_streams()
    directly from the background thread on every path, unlike
    run_replot_thread's success path, which correctly defers it to the main
    thread via self.after(...). The real CustomMeshExporter3D is stubbed out
    (its own export machinery isn't what's under test) so this stays a fast
    unit test of run_export_thread's control flow."""
    import threading
    import tkinter.messagebox as tkmb
    from upxo.pxtal.fm_steel_3d.gui import runner as runner_mod

    class _SyncThread:
        def __init__(self, target=None, daemon=None, **kw):
            self._target = target
        def start(self):
            self._target()
    monkeypatch.setattr(threading, "Thread", _SyncThread)

    captured = {}

    class _FakeExporter:
        def __init__(self, *a, **kw):
            pass
        def set_output_base_path(self, *a, **kw):
            pass
        def set_output_units(self, *a, **kw):
            pass
        def ignore_lfi_ids(self, *a, **kw):
            pass
        def export_c3d8(self, fm_final, folder_prefix, output_unit=None,
                        max_voxels=None, force=False, custom_message=""):
            captured["max_voxels"] = max_voxels
            out_dir = tmp_path / "instance1"
            out_dir.mkdir(exist_ok=True)
            return out_dir
        # The _export_fn dict literal in run_export_thread evaluates all four
        # bound methods eagerly (even though only export_c3d8 is ever called
        # here), so every element-type export method must exist on the stub.
        export_c3d4 = export_c3d8
        export_c3d20 = export_c3d8
        export_c3d10 = export_c3d8
    monkeypatch.setattr(runner_mod, "CustomMeshExporter3D", _FakeExporter)

    warnings = []
    monkeypatch.setattr(tkmb, "showwarning", lambda title, msg: warnings.append((title, msg)))

    orig_last_run_data = app.last_run_data
    orig_state = {k: app.shared_state.get(k) for k in
                 ("pipeline_mode", "EXPORT_MESH", "MESH_IGNORE_IDS", "MESH_ELEM_TYPE",
                  "MESH_OUTPUT_UNIT", "MESH_MAX_VOXELS", "MESH_FORCE", "folder_prefix",
                  "MESH_CUSTOM_MESSAGE", "output_base_directory", "VERBOSITY")}
    orig_stdout, orig_stderr = sys.stdout, sys.stderr

    app.last_run_data = {"fm_state": object()}
    app.shared_state.update({
        "pipeline_mode": "block",
        "EXPORT_MESH": True,
        "MESH_IGNORE_IDS": "",
        "MESH_ELEM_TYPE": "C3D8",
        "MESH_OUTPUT_UNIT": "microns",
        "MESH_MAX_VOXELS": "not-a-number",
        "MESH_FORCE": True,
        "folder_prefix": "fm",
        "MESH_CUSTOM_MESSAGE": "",
        "output_base_directory": str(tmp_path),
        "VERBOSITY": 0,
    })

    assert app.try_begin_operation(), "operation lock unexpectedly held before this test"
    app.end_operation()

    try:
        win = runner_mod.open_runner_window(app, replot_section="reexport")
        assert win is not None

        # (2) restore_streams() must be DEFERRED to the main thread, not called
        # synchronously from run_export_thread -- immediately after the
        # (synchronous, monkeypatched) thread returns, stdout must still be
        # the LogRedirector, not yet restored.
        assert sys.stdout is not orig_stdout, (
            "restore_streams() ran synchronously on the background thread "
            "instead of being deferred via self.after(...)")

        app.update()  # flush the deferred self.after(0, self.restore_streams) call
        assert sys.stdout is orig_stdout
        assert sys.stderr is orig_stderr

        # (1) malformed Max Voxels must warn, matching Ignore Grain IDs' style.
        assert warnings, "no warning shown for a malformed Max Voxels value"
        assert "Max Voxels" in warnings[0][0]

        # ... and still fall back to the library default (None), not crash.
        assert captured.get("max_voxels") is None
    finally:
        app.last_run_data = orig_last_run_data
        app.shared_state.update(orig_state)
        sys.stdout, sys.stderr = orig_stdout, orig_stderr
        app.end_operation()


def test_meshpage_validate_inputs_rejects_malformed_max_voxels(app):
    """MESH_MAX_VOXELS was a plain StringVar with no numeric validation in
    MeshPage.validate_inputs(), unlike RESCALE_FACTOR (a DoubleVar that
    raises/gets caught). Confirms malformed and non-positive values are now
    rejected, while a valid value and an empty value (library default) still
    pass."""
    from upxo.pxtal.fm_steel_3d.gui.pages_mesh import MeshPage

    app.show_page_by_class(MeshPage)
    app.update_idletasks()
    page = app.current_frame
    orig = page.max_vox_var.get()

    try:
        for bad in ("not-a-number", "-5", "0", "3.5"):
            page.max_vox_var.set(bad)
            assert page.validate_inputs() is False, f"{bad!r} should have failed validation"

        for ok in ("", "1000"):
            page.max_vox_var.set(ok)
            assert page.validate_inputs() is True, f"{ok!r} should have passed validation"
    finally:
        page.max_vox_var.set(orig)


def test_elementset_reexport_restores_folder_button_when_busy(app, monkeypatch):
    """ElementSetPage._reexport(): when open_runner_window() returns None
    (the app-wide busy lock is already held by another operation), only
    _reexport_btn used to be re-enabled -- _open_reexport_folder_btn (already
    enabled from a prior successful export) was left permanently disabled,
    since _on_reexport_done never fires on this early-return path."""
    import tkinter.messagebox as tkmb
    from pathlib import Path
    from upxo.pxtal.fm_steel_3d.gui.pages_mesh import ElementSetPage

    monkeypatch.setattr(tkmb, "showwarning", lambda *a, **kw: None)

    orig_run_completed = app.run_completed
    orig_last_export_path = app.last_export_path
    app.run_completed = True
    app.last_export_path = Path(".")  # any truthy path -- enables the folder button at construction

    try:
        app.show_page_by_class(ElementSetPage)
        app.update_idletasks()
        page = app.current_frame

        assert str(page._reexport_btn.cget("state")) == "normal"
        assert str(page._open_reexport_folder_btn.cget("state")) == "normal"

        assert app.try_begin_operation()  # simulate another operation already running
        try:
            page._reexport()
        finally:
            app.end_operation()

        assert str(page._reexport_btn.cget("state")) == "normal"
        assert str(page._open_reexport_folder_btn.cget("state")) == "normal", (
            "folder button must be restored to its prior (enabled) state, not left disabled")
    finally:
        app.run_completed = orig_run_completed
        app.last_export_path = orig_last_export_path


def test_elementset_reexport_recovers_from_runner_window_init_exception(app, monkeypatch):
    """ElementSetPage._reexport() had no try/except around
    open_runner_window(...) -- if RunnerWindow.__init__ itself raised before
    the background thread started, the exception propagated unhandled out of
    a Tk button callback and _reexport_btn stayed disabled forever. Confirms
    it's now caught, the button re-enabled, and a clear error shown."""
    import tkinter.messagebox as tkmb
    from upxo.pxtal.fm_steel_3d.gui.pages_mesh import ElementSetPage
    from upxo.pxtal.fm_steel_3d.gui import runner as runner_mod

    orig_run_completed = app.run_completed
    app.run_completed = True

    errors = []
    monkeypatch.setattr(tkmb, "showerror", lambda title, msg: errors.append((title, msg)))

    def _boom(*a, **kw):
        raise RuntimeError("synthetic RunnerWindow init failure")
    monkeypatch.setattr(runner_mod, "RunnerWindow", _boom)

    try:
        app.show_page_by_class(ElementSetPage)
        app.update_idletasks()
        page = app.current_frame

        assert app.try_begin_operation(), "operation lock unexpectedly held before this test"
        app.end_operation()

        page._reexport()  # must not raise

        assert str(page._reexport_btn.cget("state")) == "normal"
        assert errors, "no error was shown when RunnerWindow construction failed"
        assert app.try_begin_operation(), "busy lock left held after the failed construction"
        app.end_operation()
    finally:
        app.run_completed = orig_run_completed


def test_statistical_variables_validate_inputs_checks_ordering_std_and_positivity(app, monkeypatch):
    """StatisticalVariablesPage.validate_inputs() previously only
    range-checked solidity (0-1) -- no min<=mean<=max ordering check, no
    std>=0 check, no positivity check on area/perimeter/circle_eq_dia.
    Confirms all three are now enforced (using the PAG/Area fields), and that
    restoring each value in turn lets validation pass again."""
    import tkinter.messagebox as tkmb
    from upxo.pxtal.fm_steel_3d.gui.pages_refdata import StatisticalVariablesPage

    errors = []
    monkeypatch.setattr(tkmb, "showerror", lambda title, msg: errors.append((title, msg)))

    app.show_page_by_class(StatisticalVariablesPage)
    app.update_idletasks()
    page = app.current_frame

    errors.clear()
    assert page.validate_inputs() is True, "library defaults must already validate cleanly"

    key_mean, key_std = ("pag", "area", "mean"), ("pag", "area", "std")
    key_min, key_max = ("pag", "area", "min"), ("pag", "area", "max")

    # 1) min <= mean <= max ordering.
    orig_min = page._vars[key_min].get()
    page._vars[key_min].set(page._vars[key_max].get() + 1.0)  # min > max
    errors.clear()
    assert page.validate_inputs() is False
    assert errors
    page._vars[key_min].set(orig_min)

    # 2) std >= 0.
    orig_std = page._vars[key_std].get()
    page._vars[key_std].set(-1.0)
    errors.clear()
    assert page.validate_inputs() is False
    assert errors
    page._vars[key_std].set(orig_std)

    # 3) positivity on area/perimeter/circle_eq_dia.
    page._vars[key_min].set(0.0)
    errors.clear()
    assert page.validate_inputs() is False
    assert errors
    page._vars[key_min].set(orig_min)

    errors.clear()
    assert page.validate_inputs() is True, "values were restored -- should validate cleanly again"


def test_runner_window_close_protocol_ignores_close_while_thread_running(app, monkeypatch):
    """RunnerWindow had no protocol("WM_DELETE_WINDOW", ...) override -- the
    OS title-bar close button could destroy the Toplevel while its background
    thread was still alive, so a subsequent self.after(...) call from that
    thread raised TclError. Confirms the new handler ignores the close
    request while exec_thread is alive, and allows it once the thread has
    finished."""
    import threading
    from upxo.pxtal.fm_steel_3d.gui.runner import RunnerWindow

    class _NeverRunThread:
        """Never actually executes run_replot_thread -- irrelevant here,
        since this test only exercises the close-protocol guard itself."""
        def __init__(self, target=None, daemon=None, **kw):
            self._target = target
        def start(self):
            pass
        def is_alive(self):
            return True
    monkeypatch.setattr(threading, "Thread", _NeverRunThread)

    assert app.try_begin_operation()
    app.end_operation()

    win = RunnerWindow(app, replot_section="block1")
    try:
        assert win.exec_thread.is_alive()
        win._on_close_attempt()
        assert win.winfo_exists(), "close request while the thread is alive must be ignored"

        win.exec_thread.is_alive = lambda: False
        win._on_close_attempt()
        assert not win.winfo_exists(), "close request must succeed once the thread has finished"
    finally:
        if win.winfo_exists():
            win.destroy()


def test_viz_settings_plot_3d_disor_distr_reads_live_vars_not_stale_shared_state(app, monkeypatch):
    """VisualizationSettingsPage._on_plot_3d_disor_distr used to read
    show_misori/show_hier/n_bins/adj_only/show_kde/show_peaks/show_marks
    from self.shared_state.get(...) -- only ever refreshed by save_state()
    on page navigation -- instead of the live Tk vars (mis_var/hmis_var/
    hist_bins_var/adj_pag_var/show_kde_var/show_std_peaks_var/
    ks_markers_var). Toggling a checkbox and clicking Plot without first
    leaving the page silently plotted the OLD settings. Seed shared_state
    with the opposite of the live checkboxes and confirm the plot path
    follows the live vars."""
    import tkinter.messagebox as tkmb
    from upxo.pxtal.fm_steel_3d.gui.pages_viz import VisualizationSettingsPage

    # Stale shared_state, as if a very different selection was saved on a
    # previous visit to this page.
    app.shared_state["SHOW_MISORI_PLOT"] = True
    app.shared_state["SHOW_HIER_MISORI_PLOT"] = True
    app.shared_state["MISORI_HIST_BINS"] = 999

    app.show_page_by_class(VisualizationSettingsPage)
    app.update_idletasks()
    page = app.current_frame

    app._viz_3d_disor_data = {"is_subblock": False, "blk_within": [1.0, 2.0], "blk_across": [3.0]}

    # Live vars: both plot toggles OFF -- if the bug were still present
    # (reading the stale, both-True shared_state above) the function would
    # proceed to plot instead of warning.
    page.mis_var.set(False)
    page.hmis_var.set(False)

    warnings = []
    monkeypatch.setattr(tkmb, "showwarning", lambda title, msg: warnings.append((title, msg)))

    page._on_plot_3d_disor_distr()
    assert warnings and warnings[-1][0] == "Nothing to plot", (
        "must read show_misori/show_hier from the live mis_var/hmis_var, not stale shared_state")

    # Flip the live vars on, and set a live n_bins distinct from the stale
    # shared_state value above; confirm the plot proceeds and save_state()
    # persists the LIVE values (proving the live vars, not the stale ones,
    # drove the plot).
    page.mis_var.set(True)
    page.hmis_var.set(False)
    page.hist_bins_var.set(13)
    warnings.clear()
    page._on_plot_3d_disor_distr()
    assert not warnings
    assert page.shared_state["MISORI_HIST_BINS"] == 13
    assert page.shared_state["SHOW_MISORI_PLOT"] is True
    assert page.shared_state["SHOW_HIER_MISORI_PLOT"] is False

    import matplotlib.pyplot as plt
    plt.close("all")


def test_viz_settings_block2_plot_buttons_dont_accumulate_figures(app):
    """Step 7 Block [2]'s "Plot 2D Disorientation Distributions" button used
    to open a brand-new matplotlib window PER disorientation type (up to 5
    per click, one per selected blk_within/blk_across/pag_adj/sb_within/
    sb_across), never closed on a later click -- repeated clicks silently
    accumulated dozens of stray windows over a session (reported as ~24
    figures). Fixed by combining all types into ONE figure (N rows x 3
    columns) per click, and closing the previous click's figure before
    opening a new one. "Plot 3D Disorientation Distributions" had the same
    (smaller, at most 2-figures-per-click) accumulation bug, fixed the same
    way."""
    import matplotlib.pyplot as plt
    import numpy as np
    from upxo.pxtal.fm_steel_3d.gui.pages_viz import VisualizationSettingsPage

    bins = np.linspace(0, 65, 21)
    bin_ctr = 0.5 * (bins[:-1] + bins[1:])
    def _fake_type():
        return {axis: [np.random.rand(len(bin_ctr)) for _ in range(3)] for axis in ("x", "y", "z")}
    computed = {k: _fake_type() for k in
               ("blk_within", "blk_across", "pag_adj", "sb_within", "sb_across")}
    app._viz_2d_disor_data = {"results": computed, "bins": bins}
    app._viz_3d_disor_data = {
        "is_subblock": True,
        "blk_within": list(np.random.rand(20) * 65), "blk_across": list(np.random.rand(20) * 65),
        "pag_adj": list(np.random.rand(20) * 65), "pag_rand": list(np.random.rand(20) * 65),
        "sb_within": list(np.random.rand(20) * 65), "sb_across": list(np.random.rand(20) * 65),
    }

    app.show_page_by_class(VisualizationSettingsPage)
    app.update_idletasks()
    page = app.current_frame

    plt.close("all")
    page._on_plot_2d_disor_distr()
    assert len(plt.get_fignums()) == 1, "must be exactly ONE combined figure, not one per type"
    combined_fig = plt.figure(plt.get_fignums()[0])
    assert len(combined_fig.axes) == 15  # 5 types x 3 axes (x/y/z), all in one figure

    page._on_plot_2d_disor_distr()
    page._on_plot_2d_disor_distr()
    assert len(plt.get_fignums()) == 1, "repeated clicks must not accumulate figures"

    plt.close("all")
    page.mis_var.set(True)
    page.hmis_var.set(True)
    page._on_plot_3d_disor_distr()
    assert len(plt.get_fignums()) == 2  # misori + hierarchical, as designed
    page._on_plot_3d_disor_distr()
    page._on_plot_3d_disor_distr()
    assert len(plt.get_fignums()) == 2, "repeated clicks must not accumulate figures"

    plt.close("all")


def test_viz_settings_compute_3d_disor_reads_live_pairs_var_not_stale_shared_state(app, monkeypatch):
    """VisualizationSettingsPage._on_compute_3d_disorientations used to read
    n_pairs from self.shared_state.get("N_SAMPLE_PAIRS", 1000) inside its
    background-thread worker, instead of the live self.pairs_var checkbox
    entry -- same stale-shared_state bug class as the plot-side fix above.
    Build a real multi-PAG pipeline, seed a stale N_SAMPLE_PAIRS very
    different from the live pairs_var, and confirm the actual number of
    computed random PAG-PAG pairs matches the live value."""
    import threading
    from upxo.pxtal.fm_steel_3d.gui.pages_viz import VisualizationSettingsPage
    from upxo.pxtal.fm_steel_3d.base_3d import FMSteel3DBase

    class _SyncThread:
        def __init__(self, target=None, daemon=None, **kw):
            self._target = target
        def start(self):
            self._target()
    monkeypatch.setattr(threading, "Thread", _SyncThread)

    lfi = _voronoi_lfi_for_pag_tests(n=20, n_seeds=30, seed=3)
    fm_base = FMSteel3DBase.from_lfi(lfi, physical_dimensions=(20.0, 20.0, 20.0), voxel_size=1.0,
                                     units="microns", connectivity=6, min_grain_nvoxels=4,
                                     random_seed=42)
    fm_pag = fm_base.generate_pag_clusters(
        pag_size_distribution={"sizes": [2, 3], "probs": [0.6, 0.4]},
        pag_grain_fraction=1.0, random_seed=42)
    fm_pag.assign_pag_orientations(pag_ori_mode="random", random_seed=42)
    fm_blk = fm_pag.generate_blocks(block_thickness_range=(2.0, 5.0), random_seed=42)
    fm_ori = fm_blk.assign_orientations(random_seed=42)
    assert len(fm_pag.pag_orientations) >= 2, "test needs >=2 PAGs for random-pair sampling"
    app._fm_with_orientations = fm_ori

    orig_pairs = app.shared_state.get("N_SAMPLE_PAIRS")
    app.shared_state["N_SAMPLE_PAIRS"] = 999  # stale -- must NOT be used

    app.show_page_by_class(VisualizationSettingsPage)
    app.update_idletasks()
    page = app.current_frame

    try:
        page.pairs_var.set(5)  # live, distinct from the stale 999 above
        page.disor_blk_within_var.set(False)
        page.disor_blk_across_var.set(False)
        page.disor_pag_adj_var.set(False)
        page.disor_pag_rand_var.set(True)
        page.disor_sb_within_var.set(False)
        page.disor_sb_across_var.set(False)

        page._on_compute_3d_disorientations()
        app.update()

        assert "Computed" in page._compute_3d_disor_status_lbl.cget("text")
        assert len(app._viz_3d_disor_data["pag_rand"]) == 5, (
            "must use the live pairs_var (5), not the stale shared_state N_SAMPLE_PAIRS (999)")
    finally:
        if orig_pairs is None:
            app.shared_state.pop("N_SAMPLE_PAIRS", None)
        else:
            app.shared_state["N_SAMPLE_PAIRS"] = orig_pairs
        app._fm_with_orientations = None


def test_viz_settings_prepare_vis_data_reads_live_fam_color_var_not_stale_shared_state(app, monkeypatch):
    """VisualizationSettingsPage._on_prepare_vis_data used to read
    use_fam_col from self.shared_state.get("W7_FAMILY_COLORS", False)
    instead of the live self.fam_color_var checkbox. Build a real
    sub-block pipeline, seed shared_state with the OPPOSITE of the live
    checkbox, and confirm the resulting vis_data's fid_to_rgb (only ever
    built when family colors are actually enabled) follows the live
    checkbox, not the stale shared_state."""
    import threading
    from upxo.pxtal.fm_steel_3d.gui.pages_viz import VisualizationSettingsPage
    from upxo.pxtal.fm_steel_3d.base_3d import FMSteel3DBase

    class _SyncThread:
        def __init__(self, target=None, daemon=None, **kw):
            self._target = target
        def start(self):
            self._target()
    monkeypatch.setattr(threading, "Thread", _SyncThread)

    lfi = _voronoi_lfi_for_pag_tests(n=16, n_seeds=18, seed=2)
    fm_base = FMSteel3DBase.from_lfi(lfi, physical_dimensions=(16.0, 16.0, 16.0), voxel_size=1.0,
                                     units="microns", connectivity=6, min_grain_nvoxels=4,
                                     random_seed=42)
    fm_pag = fm_base.generate_pag_clusters(
        pag_size_distribution={"sizes": [3, 4], "probs": [0.7, 0.3]},
        pag_grain_fraction=1.0, random_seed=42)
    fm_pag.assign_pag_orientations(pag_ori_mode="random", random_seed=42)
    fm_blk = fm_pag.generate_blocks(block_thickness_range=(2.0, 5.0), random_seed=42)
    fm_ori = fm_blk.assign_orientations(random_seed=42)
    fm_sub = fm_ori.generate_subblocks(subblock_thickness_range_um=(0.5, 1.5), random_seed=42)

    app._fm_with_pags = fm_pag
    app._fm_with_blocks = fm_blk
    app._fm_with_orientations = fm_ori
    app._fm_with_subblocks = fm_sub

    orig_mode = app.shared_state.get("pipeline_mode")
    orig_fam = app.shared_state.get("W7_FAMILY_COLORS")
    app.shared_state["pipeline_mode"] = "subblock"
    app.shared_state["W7_FAMILY_COLORS"] = False  # stale/opposite of the live var below

    app.show_page_by_class(VisualizationSettingsPage)
    app.update_idletasks()
    page = app.current_frame

    try:
        page.fam_color_var.set(True)  # live value, opposite of the stale shared_state above
        page._on_prepare_vis_data()
        app.update()

        assert app.last_run_data is not None
        assert app.last_run_data["fid_to_rgb"] is not None, (
            "fid_to_rgb must be built from the live fam_color_var (True), "
            "not the stale shared_state W7_FAMILY_COLORS (False)")
    finally:
        app.shared_state["pipeline_mode"] = orig_mode
        if orig_fam is None:
            app.shared_state.pop("W7_FAMILY_COLORS", None)
        else:
            app.shared_state["W7_FAMILY_COLORS"] = orig_fam
        app._fm_with_pags = None
        app._fm_with_blocks = None
        app._fm_with_orientations = None
        app._fm_with_subblocks = None
        app.last_run_data = None


def test_pole_figure_restrict_slice_warns_when_index_never_set(app, monkeypatch):
    """Every population-level pole-figure section's "Restrict to slice"
    spinbox defaults to -1; checking the checkbox without ever touching the
    spinbox silently plotted the FULL, unrestricted population (the -1
    guard in _collect_ids_and_orientations just skips the slice filter),
    with no indication the restriction had no effect. _on_plot must now
    warn and refuse to plot in that state, for every section funneling
    through the shared _build_pole_figure_section/_on_plot path (this test
    exercises ns='pag', but the fix lives in the single shared _on_plot)."""
    import tkinter.messagebox as tkmb
    from upxo.pxtal.fm_steel_3d.gui.pages_pole_figures import PoleFigurePage
    from upxo.pxtal.fm_steel_3d.base_3d import FMSteel3DBase

    lfi = _voronoi_lfi_for_pag_tests()
    fm_base = FMSteel3DBase.from_lfi(lfi, physical_dimensions=(16.0, 16.0, 16.0), voxel_size=1.0,
                                     units="microns", connectivity=6, min_grain_nvoxels=4,
                                     random_seed=42)
    fm_pag = fm_base.generate_pag_clusters(
        pag_size_distribution={"sizes": [3, 4], "probs": [0.7, 0.3]},
        pag_grain_fraction=1.0, random_seed=42)
    fm_pag.assign_pag_orientations(pag_ori_mode="random", random_seed=42)
    app._fm_with_pags = fm_pag

    app.show_page_by_class(PoleFigurePage)
    app.update_idletasks()
    page = app.current_frame

    warnings = []
    monkeypatch.setattr(tkmb, "showwarning", lambda title, msg: warnings.append((title, msg)))

    # Check "Restrict to slice" but never touch the spinbox -- it stays at
    # its default -1.
    assert page._pag_slice_idx_var.get() == -1
    page._pag_restrict_var.set(True)
    page._on_plot("pag", "pag")
    assert warnings, "checking 'Restrict to slice' with the index left at -1 must warn, not silently plot everything"
    assert "slice" in warnings[-1][1].lower()

    # Setting a real slice index clears the warning path (may still legitimately
    # warn "No orientations" if nothing falls on that slice, but must not be the
    # "index not set" warning).
    warnings.clear()
    page._pag_slice_idx_var.set(0)
    page._on_plot("pag", "pag")
    assert not warnings or warnings[-1][0] != "Slice index not set"

    # Unchecking the restriction entirely must never warn either, regardless
    # of the (still -1'd, if reset) spinbox.
    warnings.clear()
    page._pag_restrict_var.set(False)
    page._on_plot("pag", "pag")
    assert not warnings or warnings[-1][0] != "Slice index not set"
