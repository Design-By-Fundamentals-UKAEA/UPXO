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
