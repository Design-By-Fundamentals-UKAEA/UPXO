"""Tests for upxo.material.identity (MaterialIdentity)."""
import pytest
from upxo.material.identity import MaterialIdentity, KNOWN_MATERIALS


def test_material_identity_default():
    """MaterialIdentity with defaults."""
    mid = MaterialIdentity()
    assert mid.name == 'cu'
    assert mid.alloy == 'value'
    assert mid.comp == 'value'


def test_material_identity_custom_name():
    """MaterialIdentity with custom name."""
    mid = MaterialIdentity(name='AA7075')
    assert mid.name == 'AA7075'


def test_known_materials_exist():
    """KNOWN_MATERIALS set is populated."""
    assert isinstance(KNOWN_MATERIALS, set)
    assert len(KNOWN_MATERIALS) > 0


def test_known_materials_contains_examples():
    """KNOWN_MATERIALS includes expected examples."""
    expected = {'OFHC-Cu', 'CuCrZr', 'AA7075', 'AA2099'}
    assert expected.issubset(KNOWN_MATERIALS)


def test_material_identity_known_name():
    """MaterialIdentity with a name in KNOWN_MATERIALS."""
    mid = MaterialIdentity(name='OFHC-Cu')
    assert mid.name == 'OFHC-Cu'


def test_material_identity_with_alloy():
    """MaterialIdentity with explicit alloy grade."""
    mid = MaterialIdentity(name='AA7075', alloy='T73')
    assert mid.name == 'AA7075'
    assert mid.alloy == 'T73'
