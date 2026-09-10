"""Regression tests for upxo.reporting.ReportSession.

Covers:
  Appending                  -- images (Figure and path input), tables
                                 (DataFrame/dict/records), text, params
  Section grouping            -- implicit header insertion on section change
  Entry management             -- remove_entry, clear
  Persistence                  -- save()/load()/open() round-trips, including
                                 numpy dtypes that plain json.dumps can't handle

No tkinter/GUI dependency -- this module is exercised standalone.
"""
import json

import numpy as np
import pandas as pd
import pytest

from upxo.reporting import ReportSession


def _make_fig():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    return fig


class TestAddImage:
    def test_from_figure_writes_png_into_assets(self, tmp_path):
        session = ReportSession(tmp_path, run_title="Test Run")
        fig = _make_fig()
        entry = session.add_image(fig, "Slice XY")

        assert entry.image_path.startswith("assets/")
        assert (session.reports_dir / entry.image_path).exists()
        assert (session.reports_dir / entry.image_path).suffix == ".png"

    def test_from_existing_path_copies_file(self, tmp_path):
        src = tmp_path / "external.png"
        src.write_bytes(b"\x89PNG\r\n\x1a\nfake")
        session = ReportSession(tmp_path / "run")
        entry = session.add_image(src, "External Photo")

        dest = session.reports_dir / entry.image_path
        assert dest.exists()
        assert dest.read_bytes() == src.read_bytes()

    def test_params_snapshot_attached_to_image(self, tmp_path):
        session = ReportSession(tmp_path)
        fig = _make_fig()
        entry = session.add_image(fig, "Slice", params={"N_SEEDS": 20})
        assert entry.params == {"N_SEEDS": 20}


class TestAddTable:
    def test_dataframe_input(self, tmp_path):
        session = ReportSession(tmp_path)
        df = pd.DataFrame({"grain_id": [1, 2, 3], "area": [1.5, 2.5, 3.5]})
        entry = session.add_table(df, "Grain Areas")

        assert entry.columns == ["grain_id", "area"]
        assert entry.rows == [[1, 1.5], [2, 2.5], [3, 3.5]]
        # ndarray.tolist() must already have downcast numpy scalars.
        assert all(isinstance(v, (int, float)) for row in entry.rows for v in row)

    def test_dict_of_lists_input(self, tmp_path):
        session = ReportSession(tmp_path)
        entry = session.add_table({"a": [1, 2], "b": [3, 4]}, "T")
        assert entry.columns == ["a", "b"]
        assert entry.rows == [[1, 3], [2, 4]]

    def test_list_of_records_input(self, tmp_path):
        session = ReportSession(tmp_path)
        data = [{"a": 1, "b": 3}, {"a": 2, "b": 4}]
        entry = session.add_table(data, "T")
        assert entry.columns == ["a", "b"]
        assert entry.rows == [[1, 3], [2, 4]]

    def test_unsupported_type_raises(self, tmp_path):
        session = ReportSession(tmp_path)
        with pytest.raises(TypeError):
            session.add_table(object(), "Bad")


class TestSections:
    def test_section_header_inserted_on_change(self, tmp_path):
        session = ReportSession(tmp_path)
        session.add_text("hello", "Note", section="Voronoi")
        session.add_text("world", "Note 2", section="Voronoi")
        session.add_text("!", "Note 3", section="PAG")

        kinds = [e.kind for e in session.entries]
        assert kinds == ["section", "text", "text", "section", "text"]
        assert session.entries[0].title == "Voronoi"
        assert session.entries[3].title == "PAG"

    def test_no_section_passed_inherits_current(self, tmp_path):
        session = ReportSession(tmp_path)
        session.add_section("Voronoi")
        session.add_text("hello", "Note")  # no section= -> stays "Voronoi"
        assert session.entries[-1].section == "Voronoi"


class TestEntryManagement:
    def test_remove_entry(self, tmp_path):
        session = ReportSession(tmp_path)
        e1 = session.add_text("a", "A")
        session.add_text("b", "B")
        assert session.remove_entry(e1.id) is True
        assert len(session) == 1
        assert session.remove_entry(9999) is False

    def test_clear(self, tmp_path):
        session = ReportSession(tmp_path)
        session.add_section("S")
        session.add_text("a", "A")
        session.clear()
        assert len(session) == 0
        assert session.current_section is None


class TestPersistence:
    def test_save_produces_valid_json(self, tmp_path):
        session = ReportSession(tmp_path, run_title="R", pipeline_name="FM Steel")
        session.add_text("hello", "Note", section="S1")
        path = session.save()
        assert path == tmp_path / "report_data.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["run_title"] == "R"
        assert payload["pipeline_name"] == "FM Steel"
        assert [e["kind"] for e in payload["entries"]] == ["section", "text"]

    def test_save_handles_numpy_scalars(self, tmp_path):
        session = ReportSession(tmp_path)
        session.add_params_snapshot(
            {"N_SEEDS": np.int64(20), "temp": np.float64(1.5), "ok": np.bool_(True)},
            keys=["N_SEEDS", "temp", "ok"],
        )
        path = session.save()  # must not raise TypeError
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["entries"][0]["params"] == {"N_SEEDS": 20, "temp": 1.5, "ok": True}

    def test_load_round_trips_entries(self, tmp_path):
        session = ReportSession(tmp_path, run_title="R", pipeline_name="Twinned")
        session.add_section("Voronoi")
        session.add_table({"a": [1, 2]}, "T", caption="cap")
        session.add_text("body text", "Note")
        session.save()

        reloaded = ReportSession.load(tmp_path)
        assert reloaded.run_title == "R"
        assert reloaded.pipeline_name == "Twinned"
        assert len(reloaded) == len(session)
        assert [e.kind for e in reloaded.entries] == [e.kind for e in session.entries]
        assert reloaded.entries[1].rows == [[1], [2]]

        # A freshly-loaded session must continue id numbering, not restart at 1
        # (a restart would let a new entry silently collide with an old one).
        new_entry = reloaded.add_text("more", "Another")
        assert new_entry.id not in {e.id for e in session.entries}

    def test_open_creates_fresh_session_when_no_json_exists(self, tmp_path):
        session = ReportSession.open(tmp_path / "run", run_title="Fresh")
        assert len(session) == 0
        assert session.run_title == "Fresh"

    def test_open_loads_existing_session(self, tmp_path):
        run_dir = tmp_path / "run"
        first = ReportSession(run_dir, run_title="Original")
        first.add_text("a", "A")
        first.save()

        reopened = ReportSession.open(run_dir, run_title="Ignored")
        assert reopened.run_title == "Original"
        assert len(reopened) == 1
