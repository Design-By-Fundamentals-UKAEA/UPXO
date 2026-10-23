"""The 'full' group of the grain-role property distributions: every indexed grain."""
import numpy as np

from upxo.viz.ebsdviz import compute_grain_role_property_distributions


def _inputs():
    lfi = np.array([[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]])
    parent_info = {'S3': {
        'pure_parents': np.array([1]), 'pure_twins': np.array([2]),
        'intermediates': np.array([], dtype=int)}}
    prop = {g: {'area': float(g)} for g in (1, 2, 3, 4)}
    neigh = {1: np.array([2, 3]), 2: np.array([1, 4]), 3: np.array([1, 4]), 4: np.array([2, 3])}
    return lfi, parent_info, prop, neigh


def test_full_group_holds_every_indexed_grain():
    lfi, parent_info, prop, neigh = _inputs()
    out = compute_grain_role_property_distributions(
        lfi, parent_info, prop, neigh, selected_props=['area'],
        selected_groups=['full', 'pure_parents', 'non_role'])
    assert sorted(out['area']['full'].tolist()) == [1.0, 2.0, 3.0, 4.0]
    assert out['area']['pure_parents'].tolist() == [1.0]
    assert sorted(out['area']['non_role'].tolist()) == [3.0, 4.0]


def test_default_groups_do_not_include_full():
    lfi, parent_info, prop, neigh = _inputs()
    out = compute_grain_role_property_distributions(lfi, parent_info, prop, neigh, selected_props=['area'])
    assert 'full' not in out['area']
