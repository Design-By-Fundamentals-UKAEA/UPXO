"""
Integration tests for the centralized reporting module (upxo.reporting) as
wired into the FM Steel 3D GUI (Reporting-3):

  - App owns a ReportSession (app.report), opened at __init__ and repointed
    whenever GlobalConfigPage.save_state() changes the run's output folder.
  - BasePage.add_report_button() wires a real "Append to Report" button
    (here: VisualizationSettingsPage's block-level IPF slice viewer) through
    to app.report, saving to disk after every click.
  - ReportPage lists/removes entries and generates report.html on demand.

A separate module-scoped App from test_gui.py's (Tkinter tolerates multiple
sequential roots in one process; pytest runs test modules one at a time so
the two never coexist). Every test that writes report files repoints
shared_state["reports_dir"] at tmp_path first and restores the app-level
report session afterward, so nothing lands in the real data/ tree.

Marked with pytest.mark.gui; skip on headless CI with: pytest -m "not gui"
"""
import os
import sys

import pytest

tk = pytest.importorskip("tkinter", reason="Tkinter not available")

if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
    pytest.skip("No DISPLAY — skipping GUI tests", allow_module_level=True)

pytestmark = pytest.mark.gui


@pytest.fixture(scope="module")
def app():
    import matplotlib
    matplotlib.use("Agg")

    from upxo.pxtal.fm_steel_3d.gui.app import App
    a = App(use_customtkinter=False)
    a.withdraw()
    a.update_idletasks()
    yield a
    try:
        a.destroy()
    except Exception:
        pass


def _find_button_by_text(widget, text):
    """Depth-first search for a plain ttk/tk Button with this exact label.

    Multiple "Append to Report" buttons legitimately coexist on one page now
    (one per plot section, e.g. VisualizationSettingsPage has both blk_ipf's
    and morphgrid's) -- this returns the FIRST match in tree order, which is
    only safe when the caller knows there's exactly one match on the page
    under test. For a page with more than one, use _find_button_near()
    instead to disambiguate by a nearby sibling button's text.
    """
    for c in widget.winfo_children():
        if c.winfo_class() in ("TButton", "Button") and c.cget("text") == text:
            return c
        found = _find_button_by_text(c, text)
        if found is not None:
            return found
    return None


def _find_button_near(widget, anchor_text, target_text):
    """Find the button labeled `anchor_text`, then return the button labeled
    `target_text` among ITS PARENT's children -- disambiguates two buttons
    sharing the same label on one page (e.g. "Append to Report" appearing
    once per plot section) by anchoring to a label unique to the section."""
    anchor = _find_button_by_text(widget, anchor_text)
    assert anchor is not None, f"could not locate anchor button {anchor_text!r}"
    parent = anchor.master
    for c in parent.winfo_children():
        if c.winfo_class() in ("TButton", "Button") and c.cget("text") == target_text:
            return c
    return None


class TestAppReportSession:
    def test_app_has_report_session(self, app):
        from upxo.reporting import ReportSession
        assert isinstance(app.report, ReportSession)
        assert app.report.pipeline_name == "FM Steel 3D"

    def test_global_config_save_state_sets_and_reopens_reports_dir(self, app, tmp_path):
        from upxo.pxtal.fm_steel_3d.gui.pages import GlobalConfigPage

        orig_reports_dir = app.shared_state.get("reports_dir")
        orig_report = app.report
        try:
            app.show_page_by_class(GlobalConfigPage)
            app.update_idletasks()
            page = app.current_frame

            page.dir_base.set(str(tmp_path))
            page.run_name_var.set("my_run")
            page.save_state()

            expected = str(tmp_path / "my_run" / "reports")
            assert app.shared_state["reports_dir"] == expected
            assert app.report.reports_dir == tmp_path / "my_run" / "reports"
        finally:
            app.shared_state["reports_dir"] = orig_reports_dir
            app.report = orig_report


class TestIpfViewerAppendToReport:
    def test_append_to_report_button_adds_image_entry(self, app, tmp_path, monkeypatch):
        import tkinter.messagebox as tkmb
        from upxo.pxtal.fm_steel_3d.gui.pages import VisualizationSettingsPage

        # add_report_button()'s success path pops a real messagebox.showinfo,
        # which blocks waiting for a click if left unpatched -- see module
        # docstring's rationale for isolating this fixture from test_gui.py.
        monkeypatch.setattr(tkmb, "showinfo", lambda *a, **k: None)
        monkeypatch.setattr(tkmb, "showwarning", lambda *a, **k: None)

        orig_reports_dir = app.shared_state.get("reports_dir")
        orig_report = app.report
        try:
            app.shared_state["reports_dir"] = str(tmp_path / "reports")
            app._open_report_session()

            app.show_page_by_class(VisualizationSettingsPage)
            app.update_idletasks()
            page = app.current_frame

            btn = _find_button_near(page, "Save Photo", "Append to Report")
            assert btn is not None, "could not locate the blk_ipf 'Append to Report' button"
            btn.invoke()

            entries = [e for e in app.report.entries if e.kind == "image"]
            assert len(entries) == 1
            entry = entries[0]
            assert entry.title.startswith("IPF-")
            assert entry.section == "VisualizationSettingsPage"
            assert entry.source_page == "VisualizationSettingsPage"
            assert entry.params["Plane"] in ("XY", "XZ", "YZ")

            asset_path = app.report.reports_dir / entry.image_path
            assert asset_path.exists()
            # add_report_button() must persist immediately, not require a
            # separate explicit save() from the caller.
            assert (app.report.reports_dir / "report_data.json").exists()
        finally:
            app.shared_state["reports_dir"] = orig_reports_dir
            app.report = orig_report


class TestSharedFigureToolbarAppendToReport:
    """_build_figure_toolbar() (pages_base.py) gained a generic "Append to
    Report" button used by every page that calls it (distgrid/sgs/preview/
    dist3d/dist2d/morphgrid, ...) -- exercised here via PagClusteringPage's
    "distgrid" plot, whose placeholder figure exists immediately on
    construction (no pipeline run needed), same as blk_ipf's."""

    def test_distgrid_toolbar_button_adds_image_entry(self, app, tmp_path, monkeypatch):
        import tkinter.messagebox as tkmb
        from upxo.pxtal.fm_steel_3d.gui.pages import PagClusteringPage

        monkeypatch.setattr(tkmb, "showinfo", lambda *a, **k: None)
        monkeypatch.setattr(tkmb, "showwarning", lambda *a, **k: None)

        orig_reports_dir = app.shared_state.get("reports_dir")
        orig_report = app.report
        try:
            app.shared_state["reports_dir"] = str(tmp_path / "reports")
            app._open_report_session()

            app.show_page_by_class(PagClusteringPage)
            app.update_idletasks()
            page = app.current_frame

            btn = _find_button_by_text(page, "Append to Report")
            assert btn is not None, "could not locate distgrid's 'Append to Report' button"
            btn.invoke()

            entries = [e for e in app.report.entries if e.kind == "image"]
            assert len(entries) == 1
            assert entries[0].title == "Distgrid"
            assert entries[0].source_page == "PagClusteringPage"
            assert (app.report.reports_dir / entries[0].image_path).exists()
        finally:
            app.shared_state["reports_dir"] = orig_reports_dir
            app.report = orig_report


class TestReportPage:
    def test_lists_entries_and_generates_html(self, app, tmp_path, monkeypatch):
        import tkinter.messagebox as tkmb
        from upxo.pxtal.fm_steel_3d.gui.pages import ReportPage

        monkeypatch.setattr(tkmb, "showinfo", lambda *a, **k: None)

        orig_reports_dir = app.shared_state.get("reports_dir")
        orig_report = app.report
        try:
            app.shared_state["reports_dir"] = str(tmp_path / "reports")
            app._open_report_session()
            app.report.add_text("hello world", "A Note", section="Testing")

            app.show_page_by_class(ReportPage)
            app.update_idletasks()
            page = app.current_frame

            labels_text = []
            def _collect(w):
                if w.winfo_class() in ("TLabel", "Label"):
                    try:
                        labels_text.append(w.cget("text"))
                    except tk.TclError:
                        pass
                for c in w.winfo_children():
                    _collect(c)
            _collect(page._entries_container)
            assert any("A Note" in t for t in labels_text)
            assert any("Testing" in t for t in labels_text)

            page._on_generate()
            assert (app.report.reports_dir / "report.html").exists()
        finally:
            app.shared_state["reports_dir"] = orig_reports_dir
            app.report = orig_report

    def test_remove_entry_updates_list_and_disk(self, app, tmp_path):
        from upxo.pxtal.fm_steel_3d.gui.pages import ReportPage

        orig_reports_dir = app.shared_state.get("reports_dir")
        orig_report = app.report
        try:
            app.shared_state["reports_dir"] = str(tmp_path / "reports")
            app._open_report_session()
            entry = app.report.add_text("body", "Removable")
            app.report.save()

            app.show_page_by_class(ReportPage)
            app.update_idletasks()
            page = app.current_frame

            page._on_remove(entry.id)

            assert app.report.remove_entry(entry.id) is False  # already gone
            import json
            payload = json.loads((app.report.reports_dir / "report_data.json").read_text())
            assert payload["entries"] == []
        finally:
            app.shared_state["reports_dir"] = orig_reports_dir
            app.report = orig_report
