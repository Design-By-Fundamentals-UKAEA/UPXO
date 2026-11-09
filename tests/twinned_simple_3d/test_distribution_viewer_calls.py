"""Regression test for compare_ebsd_vs_sgs's forwarding of positional arguments to
assemble_ebsd_sgs_comparison_data.

compare_ebsd_vs_sgs kept its own ``base`` parameter for backward compatibility with
existing callers (the notebook passes it), but assemble_ebsd_sgs_comparison_data no
longer accepts it (removed by commit a065827, 2026-10-21). Forwarding ``base``
positionally shifted every later argument by one and duplicated n_slices_per_axis,
raising ``TypeError: ... got multiple values for argument 'n_slices_per_axis'``.
"""
import numpy as np
import pytest

from upxo.pxtal.twinned_simple_3d.steps import steps_distribution_viewer as sdv


def _dummy_ref():
    return {
        'miso_deg_full': np.array([1.0]),
        'twin_thick_um': np.array([1.0]),
        'host_eqdia_um': np.array([1.0]),
        'tvf_2d': 0.1,
    }, {
        'miso_deg_posttwin': np.array([1.0]),
        'twin_thick_3d_um': np.array([1.0]),
        'host_eqdia_um': np.array([1.0]),
        'tvf_2d_slices': np.array([0.1]),
    }


def test_base_is_accepted_but_not_forwarded(monkeypatch):
    calls = []

    def fake_assemble(*args, **kwargs):
        calls.append((args, kwargs))
        return _dummy_ref()

    monkeypatch.setattr(
        "upxo.pxtal.twinned_simple_3d.feature_props_3d.assemble_ebsd_sgs_comparison_data",
        fake_assemble)

    cleaner, tg, base, rg, parent_info, mdf, twin_thickness = object(), object(), object(), object(), object(), object(), object()
    result = sdv.compare_ebsd_vs_sgs(cleaner, tg, base, rg, parent_info, mdf, twin_thickness,
                                      n_per_axis=7, axes=('x', 'y'))

    assert len(calls) == 1
    args, kwargs = calls[0]
    # exactly the six positional arguments the callee's current signature takes -- base is
    # not among them, and n_slices_per_axis must appear only as a keyword, never twice.
    assert args == (cleaner, tg, rg, parent_info, mdf, twin_thickness)
    assert kwargs == {"n_slices_per_axis": 7, "axes": ["x", "y"]}
    assert set(result) == {"misorientation", "twin_thickness", "host_grain_size", "twin_volume_fraction"}


def test_against_the_real_callee_signature():
    """The real assemble_ebsd_sgs_comparison_data must accept exactly what
    compare_ebsd_vs_sgs now sends it -- a TypeError here means the forwarding broke again."""
    import inspect
    from upxo.pxtal.twinned_simple_3d.feature_props_3d import assemble_ebsd_sgs_comparison_data

    sig = inspect.signature(assemble_ebsd_sgs_comparison_data)
    params = list(sig.parameters)
    assert params[:6] == ["cleaner", "tg", "rg", "parent_info", "mdf", "twin_thickness"]
    assert "base" not in params
