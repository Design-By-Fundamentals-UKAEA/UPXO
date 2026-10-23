"""
assign_pag_orientations(pag_ori_mode='hagb_constrained', ...) only reported a
failed HAGB draw through a RuntimeWarning, with the fallback count buried in
its message text. These tests check the count is also exposed as
FMSteel3DWithPAGs.hagb_fallback_count, set from
OrientationAssigner3D.assign_pag_orientations_with_hagb's new
last_hagb_fallback_count attribute, with no change to the orientation
assignment itself.
"""
import pytest

from upxo.pxtal.fm_steel_3d import FMSteel3DWithPAGs


def test_hagb_fallback_count_is_zero_when_threshold_is_easily_met():
    fm_pag = FMSteel3DWithPAGs(
        parent=None, clusters_dict={1: [10], 2: [20]}, neigh_clid={1: [2], 2: [1]})
    fm_pag.assign_pag_orientations(
        pag_ori_mode='hagb_constrained',
        pag_ori_params={'hagb_threshold': 15.0, 'max_attempts': 200},
        random_seed=0)
    assert fm_pag.hagb_fallback_count == 0


def test_hagb_fallback_count_is_reported_when_threshold_is_impossible():
    # 65 deg exceeds the cubic disorientation ceiling (~62.8 deg), so every
    # PAG with an already-assigned neighbour is guaranteed to fall back.
    fm_pag = FMSteel3DWithPAGs(
        parent=None,
        clusters_dict={1: [10], 2: [20], 3: [30]},
        neigh_clid={1: [2], 2: [1, 3], 3: [2]})
    with pytest.warns(RuntimeWarning):
        fm_pag.assign_pag_orientations(
            pag_ori_mode='hagb_constrained',
            pag_ori_params={'hagb_threshold': 65.0, 'max_attempts': 5},
            random_seed=0)
    assert fm_pag.hagb_fallback_count is not None
    assert fm_pag.hagb_fallback_count >= 1


def test_hagb_fallback_count_stays_none_for_modes_without_a_hagb_constraint():
    fm_pag = FMSteel3DWithPAGs(parent=None, clusters_dict={1: [10]}, neigh_clid={1: []})
    fm_pag.assign_pag_orientations(pag_ori_mode='random', random_seed=0)
    assert fm_pag.hagb_fallback_count is None
