"""
Cubic-symmetry conventions in ``upxo.xtalphy.texops.tops``, checked against the
disorientation that matches mtex (see test_cubic_disorientation_vs_mtex.py for the
frozen mtex values).

* ``tops.fcc_symmetrise_ori`` must return crystal-equivalent orientations of its input.
* ``tops.cubic_misorientation_V3`` must reproduce mtex's disorientation angle.
* ``tops.synth_fcc`` must sample around the requested texture component.
* ``tops.synth_fcc_quats`` must return quaternions in the convention DefDAP stores and
  the EBSD grain means use (the quaternion of the Bunge g matrix), because they are
  mixed with those in one pool.

Run from repo root:
    pytest tests/xtalphy/test_texops_cubic_conventions.py -v
"""

import contextlib
import io

import numpy as np
import pytest

from upxo.xtalphy.crystal_orientation import cubic_misorientation
from upxo.xtalphy.texops import tops

from .test_cubic_disorientation_vs_mtex import (
    _PAIR_A, _PAIR_ANGLE_DEG, _PAIR_B, _quat_disorientation_deg, _stored_quat)

_BASES = [(35.26, 45.0, 0.0), (120.0, 60.0, 200.0), (10.0, 80.0, 330.0)]
_COMPONENTS = {"copper": (90.0, 35.0, 45.0), "brass": (35.0, 45.0, 0.0),
               "s": (59.0, 37.0, 63.0), "goss": (90.0, 90.0, 45.0)}


def _quiet(fn, *args, **kwargs):
    """Call ``fn`` with the console progress text of tops suppressed."""
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


@pytest.mark.parametrize("bea", _BASES)
def test_fcc_symmetrise_ori_returns_the_24_crystal_equivalents(bea):
    eq = tops().fcc_symmetrise_ori(np.array(bea), dtype=np.float64)
    assert eq.shape == (24, 3)
    for e in eq:
        assert cubic_misorientation(bea, e)[0] < 1e-4


def test_cubic_misorientation_v3_matches_mtex():
    t = tops()
    got = np.array([float(t.cubic_misorientation_V3(a, b)) for a, b in zip(_PAIR_A, _PAIR_B)])
    assert np.allclose(got, _PAIR_ANGLE_DEG, atol=1e-4), \
        f"max deviation from mtex {np.abs(got - _PAIR_ANGLE_DEG).max():.4f} deg"


@pytest.mark.parametrize("name", sorted(_COMPONENTS))
def test_synth_fcc_samples_around_the_requested_component(name):
    np.random.seed(7)
    tex = _quiet(tops.synth_fcc, N=240, tc_info={name: [1.0, list(_COMPONENTS[name])]},
                 n_tex_instances=1, n_sampling_instances=1)
    stacks = tex.tex['tex_instance.1']['sampling_instances']['ossi.1']['tc_ori_stacks']
    ea = np.vstack(list(stacks.values())).astype(float)
    d = np.array([cubic_misorientation(_COMPONENTS[name], e)[0] for e in ea])
    assert len(ea) == 240
    assert np.mean(d <= 15.0) >= 0.95, f"only {np.mean(d <= 15.0):.0%} within 15 deg"
    assert np.median(d) < 8.0


@pytest.mark.parametrize("name", sorted(_COMPONENTS))
def test_synth_fcc_quats_are_stored_convention_quaternions_of_the_component(name):
    np.random.seed(11)
    q = _quiet(tops.synth_fcc_quats, N=120, tc_info={name: [1.0, list(_COMPONENTS[name])]})
    assert q.shape == (120, 4)
    assert np.allclose(np.linalg.norm(q, axis=1), 1.0)
    assert (q[:, 0] >= 0).all()
    target = _stored_quat(_COMPONENTS[name])
    d = np.array([_quat_disorientation_deg(qi, target) for qi in q])
    assert np.mean(d <= 15.0) >= 0.95, f"only {np.mean(d <= 15.0):.0%} within 15 deg"


def test_synth_fcc_quats_default_recipe_has_the_default_components():
    # copper, brass, goss and rotated cube by default (see tops.synth_fcc)
    np.random.seed(3)
    q = _quiet(tops.synth_fcc_quats, N=200)
    comps = [(90.0, 35.0, 45.0), (35.0, 45.0, 0.0), (0.0, 45.0, 0.0), (45.0, 0.0, 0.0)]
    near = np.zeros(len(q), bool)
    for c in comps:
        t = _stored_quat(c)
        near |= np.array([_quat_disorientation_deg(qi, t) <= 15.0 for qi in q])
    assert near.mean() >= 0.95
