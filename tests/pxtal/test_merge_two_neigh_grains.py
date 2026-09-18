"""Regression tests for the ``merge_two_neigh_grains`` adjacency check.

``mcgs3_grain_structure``, ``mcgs2_grain_structure`` and ``polyxtal2d`` each
carried an independent copy of the same bug: the post-check ``if
any((check_for_neigh, ...))`` guard was always true whenever
``check_for_neigh=True`` (its own first argument), so a failed adjacency
check was silently overridden and the merge happened anyway.
"""
import types

import pytest

from upxo.pxtal.mcgs3_temporal_slice import mcgs3_grain_structure
from upxo.pxtal.mcgs2_temporal_slice import mcgs2_grain_structure
from upxo.pxtal.pxtal_ori_map_2d import polyxtal2d

_CLASSES = [mcgs3_grain_structure, mcgs2_grain_structure, polyxtal2d]


class _MergeStub:
    """Exposes only what ``merge_two_neigh_grains`` touches on ``self``.

    ``check_for_neigh`` is bound from the real class under test so each
    class's own adjacency-lookup implementation is exercised as-is.
    """

    def __init__(self, cls, neigh_gid):
        self.neigh_gid = neigh_gid
        self.merge_calls = []
        self.check_for_neigh = types.MethodType(cls.check_for_neigh, self)

    def _merge_two_grains_(self, parent_gid, other_gid, print_msg=False):
        self.merge_calls.append((parent_gid, other_gid))
        return True


@pytest.mark.parametrize("cls", _CLASSES)
def test_merge_rejects_non_neighbours(cls):
    stub = _MergeStub(cls, neigh_gid={1: [2, 3], 2: [1]})
    merge_success = cls.merge_two_neigh_grains(stub, parent_gid=1, other_gid=5,
                                                check_for_neigh=True)
    assert merge_success is False
    assert stub.merge_calls == []


@pytest.mark.parametrize("cls", _CLASSES)
def test_merge_accepts_neighbours(cls):
    stub = _MergeStub(cls, neigh_gid={1: [2, 3], 2: [1]})
    merge_success = cls.merge_two_neigh_grains(stub, parent_gid=1, other_gid=2,
                                                check_for_neigh=True)
    assert merge_success is True
    assert stub.merge_calls == [(1, 2)]


@pytest.mark.parametrize("cls", _CLASSES)
def test_merge_bypasses_check_when_disabled(cls):
    stub = _MergeStub(cls, neigh_gid={1: [2, 3]})
    merge_success = cls.merge_two_neigh_grains(stub, parent_gid=1, other_gid=99,
                                                check_for_neigh=False)
    assert merge_success is True
    assert stub.merge_calls == [(1, 99)]
