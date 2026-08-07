"""
Tests for upxo.gui.root.pipelines -- the concrete 17-pipeline catalog.
Confirms every subprocess entry's script_path resolves to a real file,
every in_process entry's pages are real BasePage subclasses, and nothing
silently failed to register.
"""
import warnings

import pytest

pytest.importorskip("upxo.gui", reason="upxo.gui is a local-only package, not tracked/shipped")

from upxo.gui.root.pages_base import _REPO_ROOT
from upxo.gui.root.pipelines import build_registry

_EXPECTED_CATEGORIES = [
    "Generation — Monte Carlo 3D",
    "Generation — Monte Carlo 2D",
    "Generation — Geometric Tessellation",
    "Generation — Cellular Automata",
    "Generation — Custom",
    "Image Operations",
    "Crystallographic Analysis",
    "Analysis & Comparison",
    "Experimental Operations",
]


def test_build_registry_raises_no_warnings():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        build_registry()
    registration_warnings = [w for w in caught if "pipeline registration" in str(w.message)]
    assert registration_warnings == [], (
        f"{len(registration_warnings)} pipeline registration(s) failed: "
        f"{[str(w.message) for w in registration_warnings]}")


def test_entry_count_is_17():
    registry = build_registry()
    assert len(registry.all()) == 17


def test_categories_match_expected_order():
    registry = build_registry()
    assert registry.categories() == _EXPECTED_CATEGORIES


def test_all_keys_unique():
    registry = build_registry()
    keys = [e.key for e in registry.all()]
    assert len(keys) == len(set(keys))


def test_subprocess_entries_script_paths_exist_on_disk():
    registry = build_registry()
    subprocess_entries = [e for e in registry.all() if e.kind == "subprocess"]
    assert len(subprocess_entries) == 3
    for entry in subprocess_entries:
        resolved = _REPO_ROOT / entry.script_path
        assert resolved.is_file(), f"{entry.key}: script_path does not exist: {resolved}"


def test_ofhc_and_cucrzr_share_the_same_script_path():
    registry = build_registry()
    assert registry.get("ofhc_cu").script_path == registry.get("cucrzr").script_path


def test_planned_entries_have_no_script_path_or_sections():
    registry = build_registry()
    planned = [e for e in registry.all() if e.kind == "planned"]
    assert len(planned) == 6
    for entry in planned:
        assert entry.script_path is None
        assert entry.sections is None


def test_in_process_entries_flattened_pages_are_classes():
    registry = build_registry()
    in_process = [e for e in registry.all() if e.kind == "in_process"]
    assert len(in_process) == 8
    for entry in in_process:
        pages = entry.flattened_pages()
        assert len(pages) > 0
        for page_cls in pages:
            assert isinstance(page_cls, type)


def test_gs_image_ops_3d_has_8_pages():
    registry = build_registry()
    assert len(registry.get("gs_image_ops_3d").flattened_pages()) == 8


def test_crystallographic_analysis_has_7_pages():
    registry = build_registry()
    assert len(registry.get("crystallographic_analysis").flattened_pages()) == 7


@pytest.mark.parametrize("key", [
    "gs_comparisons", "grain_growth_kinetics", "visualization",
    "convert_data", "ebsd", "tension_test",
])
def test_known_gap_aliases_share_gs_image_ops_3d_pages(key):
    registry = build_registry()
    assert registry.get(key).flattened_pages() == registry.get("gs_image_ops_3d").flattened_pages()


def test_get_registry_returns_same_cached_instance():
    from upxo.gui.root.pipelines import get_registry
    assert get_registry() is get_registry()
