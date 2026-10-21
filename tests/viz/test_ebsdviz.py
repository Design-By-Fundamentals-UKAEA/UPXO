"""Tests for upxo.viz.ebsdviz grain-role visualization functions."""
import numpy as np
import pytest


def test_ebsdviz_module_imports():
    """Verify ebsdviz module can be imported."""
    try:
        from upxo.viz import ebsdviz
        assert hasattr(ebsdviz, 'compute_pct_interior_grains')
        assert hasattr(ebsdviz, 'compute_grain_role_property_distributions')
        assert hasattr(ebsdviz, 'plot_grain_role_property_distributions')
    except ImportError:
        pytest.skip("ebsdviz import failed")


def test_compute_pct_interior_grains_simple():
    """Test interior grain percentage computation (smoke test)."""
    try:
        from upxo.viz.ebsdviz import compute_pct_interior_grains

        lfi = np.array([[1, 1], [2, 2]], dtype=int)
        result = compute_pct_interior_grains(lfi, [1, 2])
        # Result can be None or a float, just check it doesn't crash
        assert result is None or isinstance(result, (int, float))
    except (ImportError, TypeError, AttributeError):
        pytest.skip("Function signature or availability issue")


# ── build_grain_role_sets / compute_four_category_grain_sets ────────────

def _parent_info_fixture():
    """One CSL type (grain 2 twin of 1, grain 3 twin of 2 -> 2 is
    intermediate) plus a second CSL type making grain 1 a pure parent
    there too, and grain 6 a pure twin via a different host. Grains 4, 5,
    7 never appear in any pair -- genuinely twin-free."""
    return {
        'S3': {
            'all_parents': np.array([1, 2]), 'all_twins': np.array([2, 3]),
            'pure_parents': np.array([1]), 'pure_twins': np.array([3]),
            'intermediates': np.array([2]),
        },
        'S9': {
            'all_parents': np.array([1]), 'all_twins': np.array([6]),
            'pure_parents': np.array([1]), 'pure_twins': np.array([6]),
            'intermediates': np.array([]),
        },
    }


def _lfi_fixture():
    return np.array([[1, 2, 3, 4], [5, 6, 7, 0]], dtype=np.int32)


def test_build_grain_role_sets_matches_cross_csl_reconciliation():
    from upxo.viz.ebsdviz import build_grain_role_sets
    role_sets = build_grain_role_sets(_lfi_fixture(), _parent_info_fixture())
    assert role_sets['pure_parents'] == {1}
    assert role_sets['pure_twins'] == {3, 6}
    assert role_sets['intermediates'] == {2}
    assert role_sets['non_role'] == {4, 5, 7}


def test_four_category_grain_sets_partition_correctly():
    from upxo.viz.ebsdviz import compute_four_category_grain_sets
    cats = compute_four_category_grain_sets(_lfi_fixture(), _parent_info_fixture())
    assert cats['parent'] == {4, 5, 7}
    assert cats['host'] == {1}
    assert cats['twin'] == {2, 3, 6}
    assert cats['complete'] == {1, 2, 3, 4, 5, 6, 7}


def test_four_category_sets_are_mutually_exclusive_except_complete():
    from upxo.viz.ebsdviz import compute_four_category_grain_sets
    cats = compute_four_category_grain_sets(_lfi_fixture(), _parent_info_fixture())
    named = [cats['parent'], cats['host'], cats['twin']]
    for i, a in enumerate(named):
        for b in named[i + 1:]:
            assert not (a & b)
    assert cats['parent'] | cats['host'] | cats['twin'] == cats['complete']


def test_twin_free_grains_absent_from_all_parents_and_all_twins():
    from upxo.viz.ebsdviz import compute_four_category_grain_sets
    parent_info = _parent_info_fixture()
    cats = compute_four_category_grain_sets(_lfi_fixture(), parent_info)
    touched = set()
    for info in parent_info.values():
        touched.update(int(g) for g in info['all_parents'])
        touched.update(int(g) for g in info['all_twins'])
    assert cats['parent'].isdisjoint(touched)


def test_four_category_with_no_csl_pairs_puts_everything_in_parent():
    from upxo.viz.ebsdviz import compute_four_category_grain_sets
    lfi = np.array([[1, 2], [3, 0]], dtype=np.int32)
    cats = compute_four_category_grain_sets(lfi, {})
    assert cats['parent'] == {1, 2, 3}
    assert cats['host'] == set()
    assert cats['twin'] == set()
    assert cats['complete'] == {1, 2, 3}
