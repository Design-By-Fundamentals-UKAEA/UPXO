"""
Unit tests for upxo.gui.root.pipeline_registry -- pure Python, no Tk, no
dependency on any real pipeline package. Uses synthetic fake page classes.
"""
import pytest

from upxo.gui.root.pipeline_registry import (
    PipelineEntry,
    PipelineRegistry,
    SidebarItem,
    SidebarSection,
)


# ---------------------------------------------------------------------------
# Synthetic fake page classes standing in for real BasePage subclasses.
# ---------------------------------------------------------------------------
class FakeWelcome:
    pass


class FakeImport:
    pass


class FakeClean:
    pass


class FakeOtherFamilyWelcome:
    pass


def _sections_a():
    return (
        SidebarSection("Start", (
            SidebarItem("Welcome", frozenset({FakeWelcome}), FakeWelcome),
        )),
        SidebarSection("Ops", (
            SidebarItem("Import", frozenset({FakeImport}), FakeImport),
            SidebarItem("Clean", frozenset({FakeClean}), FakeClean),
        )),
    )


def _sections_b():
    return (
        SidebarSection("Start", (
            SidebarItem("Welcome", frozenset({FakeOtherFamilyWelcome}), FakeOtherFamilyWelcome),
        )),
    )


# ---------------------------------------------------------------------------
# PipelineEntry.__post_init__ validation
# ---------------------------------------------------------------------------

def test_subprocess_entry_valid():
    e = PipelineEntry(key="a", title="A", description="d", category="c",
                       kind="subprocess", script_path="a/b.py")
    assert e.script_path == "a/b.py"
    assert e.sections is None


def test_in_process_entry_valid():
    e = PipelineEntry(key="a", title="A", description="d", category="c",
                       kind="in_process", sections=_sections_a())
    assert e.sections is not None
    assert e.script_path is None


def test_planned_entry_valid():
    e = PipelineEntry(key="a", title="A", description="d", category="c", kind="planned")
    assert e.script_path is None
    assert e.sections is None


@pytest.mark.parametrize("field,value", [("key", ""), ("title", ""), ("category", "")])
def test_empty_required_string_raises(field, value):
    kwargs = dict(key="a", title="A", description="d", category="c", kind="planned")
    kwargs[field] = value
    with pytest.raises(ValueError):
        PipelineEntry(**kwargs)


def test_invalid_kind_raises():
    with pytest.raises(ValueError, match="kind must be one of"):
        PipelineEntry(key="a", title="A", description="d", category="c", kind="bogus")


def test_subprocess_without_script_path_raises():
    with pytest.raises(ValueError, match="requires script_path"):
        PipelineEntry(key="a", title="A", description="d", category="c", kind="subprocess")


def test_subprocess_with_sections_raises():
    with pytest.raises(ValueError, match="must not set sections"):
        PipelineEntry(key="a", title="A", description="d", category="c",
                       kind="subprocess", script_path="x.py", sections=_sections_a())


def test_in_process_without_sections_raises():
    with pytest.raises(ValueError, match="requires a non-empty sections"):
        PipelineEntry(key="a", title="A", description="d", category="c", kind="in_process")


def test_in_process_with_empty_sections_raises():
    with pytest.raises(ValueError, match="requires a non-empty sections"):
        PipelineEntry(key="a", title="A", description="d", category="c",
                       kind="in_process", sections=())


def test_in_process_with_script_path_raises():
    with pytest.raises(ValueError, match="must not set script_path"):
        PipelineEntry(key="a", title="A", description="d", category="c",
                       kind="in_process", sections=_sections_a(), script_path="x.py")


def test_planned_with_script_path_raises():
    with pytest.raises(ValueError, match="must not set"):
        PipelineEntry(key="a", title="A", description="d", category="c",
                       kind="planned", script_path="x.py")


def test_planned_with_sections_raises():
    with pytest.raises(ValueError, match="must not set"):
        PipelineEntry(key="a", title="A", description="d", category="c",
                       kind="planned", sections=_sections_a())


def test_entry_is_frozen():
    e = PipelineEntry(key="a", title="A", description="d", category="c", kind="planned")
    with pytest.raises(Exception):
        e.title = "changed"


# ---------------------------------------------------------------------------
# PipelineEntry.flattened_pages()
# ---------------------------------------------------------------------------

def test_flattened_pages_in_process():
    e = PipelineEntry(key="a", title="A", description="d", category="c",
                       kind="in_process", sections=_sections_a())
    assert e.flattened_pages() == (FakeWelcome, FakeImport, FakeClean)


def test_flattened_pages_empty_for_subprocess():
    e = PipelineEntry(key="a", title="A", description="d", category="c",
                       kind="subprocess", script_path="x.py")
    assert e.flattened_pages() == ()


def test_flattened_pages_empty_for_planned():
    e = PipelineEntry(key="a", title="A", description="d", category="c", kind="planned")
    assert e.flattened_pages() == ()


# ---------------------------------------------------------------------------
# PipelineRegistry
# ---------------------------------------------------------------------------

def _reg_with_two_entries():
    reg = PipelineRegistry()
    reg.register(PipelineEntry(key="fam_a", title="Family A", description="d",
                                category="Cat 1", kind="in_process", sections=_sections_a()))
    reg.register(PipelineEntry(key="fam_b", title="Family B", description="d",
                                category="Cat 2", kind="in_process", sections=_sections_b()))
    return reg


def test_register_and_get():
    reg = _reg_with_two_entries()
    assert reg.get("fam_a").title == "Family A"
    assert reg.get("fam_b").title == "Family B"


def test_get_missing_key_raises_with_known_keys_listed():
    reg = _reg_with_two_entries()
    with pytest.raises(KeyError, match="fam_a"):
        reg.get("nope")


def test_register_duplicate_key_raises():
    reg = _reg_with_two_entries()
    with pytest.raises(ValueError, match="Duplicate pipeline key"):
        reg.register(PipelineEntry(key="fam_a", title="Dup", description="d",
                                    category="Cat 1", kind="planned"))


def test_all_returns_insertion_order():
    reg = _reg_with_two_entries()
    assert [e.key for e in reg.all()] == ["fam_a", "fam_b"]


def test_categories_first_seen_order_and_unique():
    reg = PipelineRegistry()
    reg.register(PipelineEntry(key="a", title="A", description="d", category="Z", kind="planned"))
    reg.register(PipelineEntry(key="b", title="B", description="d", category="Y", kind="planned"))
    reg.register(PipelineEntry(key="c", title="C", description="d", category="Z", kind="planned"))
    assert reg.categories() == ["Z", "Y"]


def test_by_category_groups_correctly():
    reg = PipelineRegistry()
    reg.register(PipelineEntry(key="a", title="A", description="d", category="X", kind="planned"))
    reg.register(PipelineEntry(key="b", title="B", description="d", category="X", kind="planned"))
    reg.register(PipelineEntry(key="c", title="C", description="d", category="Y", kind="planned"))
    grouped = reg.by_category()
    assert [e.key for e in grouped["X"]] == ["a", "b"]
    assert [e.key for e in grouped["Y"]] == ["c"]


def test_pipeline_for_page_resolves_to_owning_entry():
    reg = _reg_with_two_entries()
    assert reg.pipeline_for_page(FakeImport).key == "fam_a"
    assert reg.pipeline_for_page(FakeClean).key == "fam_a"
    assert reg.pipeline_for_page(FakeOtherFamilyWelcome).key == "fam_b"


def test_pipeline_for_page_unknown_class_returns_none():
    reg = _reg_with_two_entries()

    class NotRegistered:
        pass

    assert reg.pipeline_for_page(NotRegistered) is None


def test_pipeline_for_page_ignores_subprocess_and_planned_entries():
    reg = PipelineRegistry()
    reg.register(PipelineEntry(key="sub", title="Sub", description="d",
                                category="C", kind="subprocess", script_path="x.py"))
    reg.register(PipelineEntry(key="planned", title="Planned", description="d",
                                category="C", kind="planned"))
    assert reg.pipeline_for_page(FakeWelcome) is None


def test_pipeline_for_page_cache_invalidated_by_new_registration():
    reg = PipelineRegistry()
    reg.register(PipelineEntry(key="fam_a", title="Family A", description="d",
                                category="Cat 1", kind="in_process", sections=_sections_a()))
    assert reg.pipeline_for_page(FakeOtherFamilyWelcome) is None  # builds/caches the lookup
    reg.register(PipelineEntry(key="fam_b", title="Family B", description="d",
                                category="Cat 2", kind="in_process", sections=_sections_b()))
    # Must reflect the newly-registered entry, not a stale cached miss.
    assert reg.pipeline_for_page(FakeOtherFamilyWelcome).key == "fam_b"
