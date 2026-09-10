"""Regression tests for upxo.reporting.render_html.

Covers: default output path, section wrapping, escaping of user-supplied
text (titles/captions/table cells/free text), and correct relative image
src references. No tkinter/GUI dependency.
"""
from upxo.reporting import ReportSession, render_html


class TestRenderHtml:
    def test_default_output_path(self, tmp_path):
        session = ReportSession(tmp_path)
        session.add_text("hello", "Note")
        out = render_html(session)
        assert out == tmp_path / "report.html"
        assert out.exists()

    def test_custom_output_path(self, tmp_path):
        session = ReportSession(tmp_path)
        session.add_text("hello", "Note")
        out_path = tmp_path / "nested" / "custom.html"
        out = render_html(session, out_path=out_path)
        assert out == out_path
        assert out.exists()

    def test_sections_wrapped_and_closed(self, tmp_path):
        session = ReportSession(tmp_path)
        session.add_text("a", "A", section="First")
        session.add_text("b", "B", section="Second")
        html_text = render_html(session).read_text(encoding="utf-8")

        assert html_text.count("<section>") == 2
        assert html_text.count("</section>") == 2
        assert "First" in html_text and "Second" in html_text

    def test_html_is_escaped(self, tmp_path):
        session = ReportSession(tmp_path)
        session.add_text("<script>alert(1)</script>", "Bad <title>")
        html_text = render_html(session).read_text(encoding="utf-8")

        assert "<script>alert(1)</script>" not in html_text
        assert "&lt;script&gt;" in html_text
        assert "&lt;title&gt;" in html_text

    def test_table_cells_escaped(self, tmp_path):
        session = ReportSession(tmp_path)
        session.add_table({"col": ["<b>x</b>"]}, "T")
        html_text = render_html(session).read_text(encoding="utf-8")
        assert "<b>x</b>" not in html_text
        assert "&lt;b&gt;x&lt;/b&gt;" in html_text

    def test_image_src_is_relative_and_asset_exists(self, tmp_path):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        session = ReportSession(tmp_path)
        fig, ax = plt.subplots()
        ax.plot([0, 1], [1, 0])
        entry = session.add_image(fig, "Plot")

        html_text = render_html(session).read_text(encoding="utf-8")
        assert f'src="{entry.image_path}"' in html_text
        assert (session.reports_dir / entry.image_path).exists()

    def test_params_table_rendered_under_image(self, tmp_path):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        session = ReportSession(tmp_path)
        fig, ax = plt.subplots()
        session.add_image(fig, "Plot", params={"N_SEEDS": 20})
        html_text = render_html(session).read_text(encoding="utf-8")
        assert "N_SEEDS" in html_text
        assert "20" in html_text

    def test_run_title_and_pipeline_name_in_header(self, tmp_path):
        session = ReportSession(tmp_path, run_title="My Run", pipeline_name="FM Steel")
        html_text = render_html(session).read_text(encoding="utf-8")
        assert "My Run" in html_text
        assert "FM Steel" in html_text

    def test_source_page_shown_for_context(self, tmp_path):
        session = ReportSession(tmp_path)
        session.add_text("body", "Sgs", source_page="Step2b1Page")
        html_text = render_html(session).read_text(encoding="utf-8")
        assert "Step2b1Page" in html_text

    def test_no_source_page_does_not_crash_or_leave_stray_separator(self, tmp_path):
        session = ReportSession(tmp_path)
        session.add_text("body", "Note")  # no source_page given
        html_text = render_html(session).read_text(encoding="utf-8")
        assert "&middot;" not in html_text
