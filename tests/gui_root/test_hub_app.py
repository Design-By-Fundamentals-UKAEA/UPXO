"""
GUI smoke tests for the UPXO hub (upxo.gui.root), headless (no mainloop).
Mirrors tests/fm_steel_3d/test_gui.py's established convention: a
module-scoped fixture builds the real App and withdraws it, tests are
marked pytest.mark.gui.

Run from repo root:
    pytest tests/gui_root/test_hub_app.py -v
"""
import os
import sys
from unittest import mock

import pytest

tk = pytest.importorskip("tkinter", reason="Tkinter not available")

if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
    pytest.skip("No DISPLAY — skipping GUI tests", allow_module_level=True)

pytestmark = pytest.mark.gui


@pytest.fixture(scope="module")
def app():
    import matplotlib
    matplotlib.use("Agg")  # prevent any plot windows from opening
    from upxo.gui.root.app import App
    a = App(use_customtkinter=False)
    a.withdraw()
    a.update_idletasks()
    yield a
    try:
        a.destroy()
    except Exception:
        pass


def _collect_button_texts(widget, out):
    for child in widget.winfo_children():
        try:
            if hasattr(child, "cget"):
                cls = child.winfo_class()
                if "Button" in cls:
                    out.append(child.cget("text"))
        except Exception:
            pass
        _collect_button_texts(child, out)
    return out


def test_app_creates_without_crash(app):
    assert app is not None


def test_welcome_page_has_no_back_or_home_button(app):
    from upxo.gui.root.pages_welcome import WelcomePage
    app.show_page_by_class(WelcomePage)
    app.update_idletasks()
    texts = _collect_button_texts(app.current_frame, [])
    assert "< Back" not in texts
    assert "HOME" not in texts
    # WelcomePage IS a navigation page (special-cased), so it still gets Next.
    assert any(t in ("Next >", "Finish") for t in texts)


def test_ops_welcome_page_has_navigation_chrome(app):
    from upxo.gui.GSImageOperations3D.pages import OpsWelcomePage
    app.show_page_by_class(OpsWelcomePage)
    app.update_idletasks()
    texts = _collect_button_texts(app.current_frame, [])
    assert any(t in ("Next >", "Finish") for t in texts), (
        "OpsWelcomePage should get Next/Finish chrome via the registry-driven "
        "is_navigation_page check (it's a registered in_process pipeline page)")
    # "< Back" is nested inside the same is_navigation_page gate as Next in
    # pages_base.py -- confirms the fix cascades to both, not just Next.
    assert "< Back" in texts
    assert "HOME" in texts


def test_xtal_welcome_page_has_navigation_chrome(app):
    from upxo.gui.CrystallographicAnalysis import XtalWelcomePage
    app.show_page_by_class(XtalWelcomePage)
    app.update_idletasks()
    texts = _collect_button_texts(app.current_frame, [])
    assert any(t in ("Next >", "Finish") for t in texts), (
        "XtalWelcomePage should get Next/Finish chrome via the registry-driven "
        "is_navigation_page check (it's a registered in_process pipeline page)")
    assert "< Back" in texts
    assert "HOME" in texts


def test_plain_unregistered_page_has_no_next_or_back_button(app):
    """A BasePage subclass that is NOT the hub's WelcomePage and NOT
    registered in the pipeline registry must get no Next/Finish/Back chrome
    -- confirms the registry-driven check doesn't over-grant navigation
    chrome to arbitrary pages, matching the old string-prefix check's
    behavior for anything that didn't start with "Ops"/"Xtal". "< Back" is
    nested inside the same is_navigation_page gate as Next, so it must be
    absent here too. HOME is independently gated (only excluded on
    WelcomePage itself) and stays present.

    Reuses the module-scoped `app` fixture rather than constructing a
    second App() -- Tkinter only supports one root window per process (see
    tests/fm_steel_3d/test_gui.py's own docstring on this exact point);
    a second root here previously crashed with a stale PhotoImage TclError
    from pages_welcome.py's image widgets."""
    from upxo.gui.root.pages_base import BasePage

    class _PlainUnregisteredPage(BasePage):
        def __init__(self, parent, app):
            super().__init__(parent, app)
            self.add_navigation_bar(self)

    app.show_page_by_class(_PlainUnregisteredPage)
    app.update_idletasks()
    texts = _collect_button_texts(app.current_frame, [])
    assert "Next >" not in texts
    assert "Finish" not in texts
    assert "< Back" not in texts
    assert "HOME" in texts


# ---------------------------------------------------------------------------
# Phase 4: registry-driven navigation (app.py rewrite) + CatalogPage
# ---------------------------------------------------------------------------

def test_next_from_welcome_reaches_catalog(app):
    from upxo.gui.root.pages_welcome import WelcomePage
    from upxo.gui.root.pages_catalog import CatalogPage
    app.show_page_by_class(WelcomePage)
    app.update_idletasks()
    app.next_page()
    app.update_idletasks()
    assert type(app.current_frame) is CatalogPage


def test_catalog_exposes_17_cards_across_9_categories(app):
    from upxo.gui.root.pages_catalog import CatalogPage
    app.show_page_by_class(CatalogPage)
    app.update_idletasks()
    assert len(app.current_frame._cards) == 17
    assert len(app.current_frame._category_sections) == 9


def _walk_family_and_verify_bounds(app, entry_key, stop_before_last=False):
    """Walk next_page() through an in_process pipeline family's own page
    order, confirming each transition, then confirm falling off either end
    lands on CatalogPage.

    stop_before_last: skip constructing (and navigating Next past) the
    family's final page. Used only for crystallographic_analysis, whose
    last page is XtalPoleFigurePage -- CrystallographicAnalysis/
    pages_polefigure.py (out of scope, untouched) unconditionally calls
    matplotlib.use("TkAgg") at module import time, which permanently
    overrides this fixture's matplotlib.use("Agg") for the rest of the
    process and makes constructing that page try to open a real TkAgg
    figure window; in this dev environment that fails with a Tcl/init.tcl
    error unrelated to anything in this redesign. next_page()/back_page()
    are fully generic (no per-family special-casing, driven purely by
    entry.flattened_pages() + index arithmetic), so the "falls off the
    last page" edge is still exercised in full by
    test_walk_gs_image_ops_3d_family_and_back; skipping it here avoids an
    environment-specific failure in untouched code without leaving that
    code path unverified anywhere."""
    from upxo.gui.root.pages_catalog import CatalogPage
    from upxo.gui.root.pipelines import get_registry

    entry = get_registry().get(entry_key)
    pages = entry.flattened_pages()
    assert len(pages) > 0

    # OpsImportPage.validate_inputs() (GSImageOperations3D/pages.py, out of
    # scope, left untouched) legitimately blocks Next until shared_state
    # ["lfi_3d"] is set -- satisfy that real gate with dummy data so this
    # walk can proceed past it, rather than weakening the assertion.
    import numpy as np
    app.shared_state["lfi_3d"] = np.zeros((2, 2, 2), dtype=int)

    walk_pages = pages[:-1] if stop_before_last else pages

    app.show_page_by_class(walk_pages[0])
    app.update_idletasks()
    assert type(app.current_frame) is walk_pages[0]

    for i in range(1, len(walk_pages)):
        app.next_page()
        app.update_idletasks()
        assert type(app.current_frame) is walk_pages[i], (
            f"{entry_key}: expected page index {i} ({walk_pages[i].__name__}) after "
            f"next_page(), got {type(app.current_frame).__name__}")

    if not stop_before_last:
        # One more Next from the family's last page falls off the end -> Catalog.
        app.next_page()
        app.update_idletasks()
        assert type(app.current_frame) is CatalogPage, (
            f"{entry_key}: Next from the last page should land on CatalogPage")

    # Back from the family's own first page also falls off the start -> Catalog.
    app.show_page_by_class(pages[0])
    app.update_idletasks()
    app.back_page()
    app.update_idletasks()
    assert type(app.current_frame) is CatalogPage, (
        f"{entry_key}: Back from the first page should land on CatalogPage")


def test_walk_gs_image_ops_3d_family_and_back(app):
    _walk_family_and_verify_bounds(app, "gs_image_ops_3d")


def test_walk_crystallographic_analysis_family_and_back(app):
    _walk_family_and_verify_bounds(app, "crystallographic_analysis", stop_before_last=True)


def test_launch_pipeline_records_and_clears_running_process(app):
    from upxo.gui.root.pipelines import get_registry

    entry = get_registry().get("fm_steel_3d")
    fake_proc = mock.Mock()
    fake_proc.poll.return_value = None  # still running

    with mock.patch("subprocess.Popen", return_value=fake_proc) as mock_popen:
        app.launch_pipeline(entry)

    assert entry.key in app.running_processes
    assert app.running_processes[entry.key] is fake_proc
    mock_popen.assert_called_once()
    (called_argv,), _ = mock_popen.call_args
    assert called_argv[0] == sys.executable
    assert called_argv[1].endswith(entry.script_path.replace("/", os.sep))

    # A second launch_pipeline call while "running" must be a no-op (matches
    # the pre-existing "already running" guard).
    with mock.patch("subprocess.Popen") as mock_popen_2:
        app.launch_pipeline(entry)
    mock_popen_2.assert_not_called()

    # Simulate the subprocess finishing, then let the poller clear it.
    fake_proc.poll.return_value = 0
    app._check_running_processes()
    assert entry.key not in app.running_processes


def test_search_resets_scroll_position_to_top(app):
    """Filtering can shrink the visible content a lot (e.g. the 11-card
    default view down to a couple of cards). If the canvas was scrolled
    down beforehand, its viewport must snap back to the top so the
    (correctly filtered) results are immediately visible, rather than
    leaving the viewport pointing past where the shorter content now ends.

    Asserts that _on_search_changed() actually calls yview_moveto(0) on the
    scroll canvas, via a spy, rather than inspecting the resulting yview()
    state directly -- confirmed by a manual before/after check that a
    withdrawn/headless window never computes a real scrollregion here, so
    checking the resulting yview state passes trivially whether or not the
    reset call is even present (verified this test would NOT have caught
    the reset being accidentally removed, before switching to this spy-based
    approach)."""
    from upxo.gui.root.pages_catalog import CatalogPage

    app.show_page_by_class(CatalogPage)
    app.update_idletasks()
    cat_page = app.current_frame

    with mock.patch.object(cat_page._scroll_canvas, "yview_moveto") as spy:
        cat_page._search_var.set("monte")
        cat_page._on_search_changed()
        app.update_idletasks()

    spy.assert_called_once_with(0)
