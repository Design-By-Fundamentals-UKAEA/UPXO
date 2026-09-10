"""
Integration tests for the centralized reporting module (upxo.reporting) as
wired into the Twinned FCC 3D GUI (Reporting-4):

  - App owns a ReportSession (app.report), opened at __init__ and repointed
    whenever GlobalConfigPage.save_state() changes GLOBAL_OUTPUT_DIR.
  - BasePage.add_report_button() wires a real "Append to Report" button
    (here: PreTwinRepresentativenessPage's pass-rate chart) through to
    app.report, saving to disk after every click.
  - ReportBuilderPage lists/removes entries and generates report.html on
    demand -- distinct from SummaryReportPage's own auto-computed Markdown
    report, which this module does not touch.

There is no pre-existing GUI test suite for this pipeline (unlike FM
Steel's tests/fm_steel_3d/test_gui.py) -- this file only covers the new
reporting wiring, not the rest of the wizard.

A separate module-scoped App from FM Steel's test fixtures (Tkinter
tolerates multiple sequential roots in one process; pytest runs test
modules one at a time so the two never coexist). Every test that writes
report files repoints GLOBAL_OUTPUT_DIR at tmp_path first and restores the
app-level report session afterward, so nothing lands in the real data/ tree.

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

    from upxo.pxtal.twinned_simple_3d.gui.app import App
    a = App(use_customtkinter=False)
    a.withdraw()
    a.update_idletasks()
    yield a
    try:
        a.destroy()
    except Exception:
        pass


def _find_button_by_text(widget, text):
    """Depth-first search for a plain ttk/tk Button with this exact label."""
    for c in widget.winfo_children():
        if c.winfo_class() in ("TButton", "Button") and c.cget("text") == text:
            return c
        found = _find_button_by_text(c, text)
        if found is not None:
            return found
    return None


def _make_fig():
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    fig = Figure(figsize=(4, 3))
    ax = fig.add_subplot(111)
    ax.bar(["X", "Y", "Z"], [80, 60, 90])
    return fig


class TestAppReportSession:
    def test_app_has_report_session(self, app):
        from upxo.reporting import ReportSession
        assert isinstance(app.report, ReportSession)
        assert app.report.pipeline_name == "Twinned FCC 3D"

    def test_global_config_save_state_repoints_report_session(self, app, tmp_path):
        from upxo.pxtal.twinned_simple_3d.gui.pages_global import GlobalConfigPage

        orig_output_dir = app.shared_state.get("GLOBAL_OUTPUT_DIR")
        orig_report = app.report
        try:
            app.show_page_by_class(GlobalConfigPage)
            app.update_idletasks()
            page = app.current_frame

            page.output_dir_var.set(str(tmp_path))
            page.save_state()

            assert app.shared_state["GLOBAL_OUTPUT_DIR"] == str(tmp_path)
            assert app.report.reports_dir == tmp_path / "TwinnedFCC" / "Reports"
        finally:
            app.shared_state["GLOBAL_OUTPUT_DIR"] = orig_output_dir
            app.report = orig_report


class TestPreTwinValidationAppendToReport:
    def test_append_to_report_button_adds_image_entry(self, app, tmp_path, monkeypatch):
        import tkinter.messagebox as tkmb
        from upxo.pxtal.twinned_simple_3d.gui.pages_reprvalidation import (
            PreTwinRepresentativenessPage,
        )

        # add_report_button()'s success path pops a real messagebox.showinfo,
        # which blocks waiting for a click if left unpatched.
        monkeypatch.setattr(tkmb, "showinfo", lambda *a, **k: None)
        monkeypatch.setattr(tkmb, "showwarning", lambda *a, **k: None)

        orig_output_dir = app.shared_state.get("GLOBAL_OUTPUT_DIR")
        orig_report = app.report
        try:
            app.shared_state["GLOBAL_OUTPUT_DIR"] = str(tmp_path)
            app._open_report_session()

            app.show_page_by_class(PreTwinRepresentativenessPage)
            app.update_idletasks()
            page = app.current_frame

            # Bypasses the real validator computation (needs a full
            # pipeline run) -- only the report-button wiring is under test.
            page.pass_rate_fig = _make_fig()

            btn = _find_button_by_text(page, "Append to Report")
            assert btn is not None, "could not locate PreTwinRepresentativenessPage's 'Append to Report' button"
            btn.invoke()

            entries = [e for e in app.report.entries if e.kind == "image"]
            assert len(entries) == 1
            entry = entries[0]
            assert entry.title.startswith("Pre-Twin Representativeness")
            assert entry.section == "PreTwinRepresentativenessPage"
            assert "Wasserstein Threshold" in entry.params

            asset_path = app.report.reports_dir / entry.image_path
            assert asset_path.exists()
            assert (app.report.reports_dir / "report_data.json").exists()
        finally:
            app.shared_state["GLOBAL_OUTPUT_DIR"] = orig_output_dir
            app.report = orig_report

    def test_append_button_warns_when_nothing_plotted_yet(self, app, tmp_path, monkeypatch):
        import tkinter.messagebox as tkmb
        from upxo.pxtal.twinned_simple_3d.gui.pages_reprvalidation import (
            PreTwinRepresentativenessPage,
        )

        warnings = []
        monkeypatch.setattr(tkmb, "showwarning", lambda title, msg: warnings.append((title, msg)))
        monkeypatch.setattr(tkmb, "showinfo", lambda *a, **k: None)

        orig_output_dir = app.shared_state.get("GLOBAL_OUTPUT_DIR")
        orig_report = app.report
        try:
            app.shared_state["GLOBAL_OUTPUT_DIR"] = str(tmp_path)
            app._open_report_session()

            app.show_page_by_class(PreTwinRepresentativenessPage)
            app.update_idletasks()
            page = app.current_frame
            assert page.pass_rate_fig is None  # nothing computed yet

            btn = _find_button_by_text(page, "Append to Report")
            btn.invoke()

            assert warnings, "clicking with no plot yet should warn, not silently append"
            assert not [e for e in app.report.entries if e.kind == "image"]
        finally:
            app.shared_state["GLOBAL_OUTPUT_DIR"] = orig_output_dir
            app.report = orig_report


class TestReportBuilderPage:
    def test_lists_entries_and_generates_html(self, app, tmp_path, monkeypatch):
        import tkinter.messagebox as tkmb
        from upxo.pxtal.twinned_simple_3d.gui.pages_report_builder import ReportBuilderPage

        monkeypatch.setattr(tkmb, "showinfo", lambda *a, **k: None)

        orig_output_dir = app.shared_state.get("GLOBAL_OUTPUT_DIR")
        orig_report = app.report
        try:
            app.shared_state["GLOBAL_OUTPUT_DIR"] = str(tmp_path)
            app._open_report_session()
            app.report.add_text("hello world", "A Note", section="Testing")

            app.show_page_by_class(ReportBuilderPage)
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
            app.shared_state["GLOBAL_OUTPUT_DIR"] = orig_output_dir
            app.report = orig_report

    def test_remove_entry_updates_list_and_disk(self, app, tmp_path):
        from upxo.pxtal.twinned_simple_3d.gui.pages_report_builder import ReportBuilderPage

        orig_output_dir = app.shared_state.get("GLOBAL_OUTPUT_DIR")
        orig_report = app.report
        try:
            app.shared_state["GLOBAL_OUTPUT_DIR"] = str(tmp_path)
            app._open_report_session()
            entry = app.report.add_text("body", "Removable")
            app.report.save()

            app.show_page_by_class(ReportBuilderPage)
            app.update_idletasks()
            page = app.current_frame

            page._on_remove(entry.id)

            assert app.report.remove_entry(entry.id) is False  # already gone
            import json
            payload = json.loads((app.report.reports_dir / "report_data.json").read_text())
            assert payload["entries"] == []
        finally:
            app.shared_state["GLOBAL_OUTPUT_DIR"] = orig_output_dir
            app.report = orig_report


class TestNavigationOrder:
    def test_report_builder_sits_between_export3_and_summary_report(self, app):
        from upxo.pxtal.twinned_simple_3d.gui.pages_export import Export3Page, SummaryReportPage
        from upxo.pxtal.twinned_simple_3d.gui.pages_report_builder import ReportBuilderPage

        app.show_page_by_class(Export3Page)
        app.update_idletasks()
        app.next_page()
        app.update_idletasks()
        assert type(app.current_frame) is ReportBuilderPage

        app.next_page()
        app.update_idletasks()
        assert type(app.current_frame) is SummaryReportPage

        app.back_page()
        app.update_idletasks()
        assert type(app.current_frame) is ReportBuilderPage

        app.back_page()
        app.update_idletasks()
        assert type(app.current_frame) is Export3Page
