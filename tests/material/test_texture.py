"""Tests for upxo.material.texture (TextureComponentProfile)."""
import pytest
from upxo.material.texture import TextureComponentProfile


def test_texture_component_profile_default():
    """TextureComponentProfile with minimal args."""
    tcp = TextureComponentProfile(crystal_family='FCC')
    assert tcp.crystal_family == 'FCC'
    assert tcp.component_fractions == {}
    assert tcp.component_widths == {}


def test_texture_component_profile_with_fractions():
    """TextureComponentProfile with component fractions dict."""
    fracs = {'Cube': 0.4, 'Goss': 0.3}
    tcp = TextureComponentProfile(crystal_family='FCC',
                                 component_fractions=fracs)
    assert tcp.component_fractions == fracs


def test_texture_component_profile_with_widths():
    """TextureComponentProfile with component widths dict."""
    widths = {'Cube': 15.0, 'Goss': 20.0}
    tcp = TextureComponentProfile(crystal_family='FCC',
                                 component_widths=widths)
    assert tcp.component_widths == widths


def test_texture_component_profile_with_random_fraction():
    """TextureComponentProfile can have a random_fraction field."""
    tcp = TextureComponentProfile(crystal_family='FCC',
                                 random_fraction=0.1)
    assert tcp.random_fraction == 0.1


def test_texture_component_profile_all_fields():
    """TextureComponentProfile with all fields populated."""
    tcp = TextureComponentProfile(
        crystal_family='BCC',
        component_fractions={'Alpha': 0.6, 'Gamma': 0.3},
        component_widths={'Alpha': 10.0, 'Gamma': 12.0},
        random_fraction=0.1
    )
    assert tcp.crystal_family == 'BCC'
    assert len(tcp.component_fractions) == 2
    assert len(tcp.component_widths) == 2
    assert tcp.random_fraction == 0.1


def test_texture_component_profile_crystal_families():
    """TextureComponentProfile accepts standard crystal families."""
    for family in ['FCC', 'BCC', 'HCP']:
        tcp = TextureComponentProfile(crystal_family=family)
        assert tcp.crystal_family == family


def test_texture_component_profile_empty_dicts():
    """TextureComponentProfile allows empty dicts."""
    tcp = TextureComponentProfile(crystal_family='FCC',
                                 component_fractions={},
                                 component_widths={})
    assert tcp.component_fractions == {}
    assert tcp.component_widths == {}
