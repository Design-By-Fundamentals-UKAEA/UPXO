"""Regression tests for upxo.repgen.repgen3dmcgs.repgen3d construction."""
import pytest

from upxo.repgen.repgen3dmcgs import repgen3d


def test_repgen3d_default_construction():
    """__slots__ must cover every attribute assigned in __init__."""
    rg = repgen3d()
    assert rg.tdim == 2
    assert rg.iroute == 'tgs.sgs'
    assert rg.sgstype == 'upxo.mc3d'
    assert rg.tgstype == 'upxo.mc3d'


def test_repgen3d_rm1_setup_attribute_is_assignable():
    """rm1tests is assigned outside __init__ and must be a declared slot."""
    rg = repgen3d()
    rg.rm1tests = dict(ed=True, ordern=1)
    assert rg.rm1tests['ed'] is True


@pytest.mark.parametrize('kw', [dict(iroute='bad'), dict(sgstype='bad'),
                                dict(tgstype='bad')])
def test_repgen3d_rejects_invalid_codes(kw):
    """Invalid route or gs-type codes raise ValueError."""
    with pytest.raises(ValueError):
        repgen3d(**kw)
