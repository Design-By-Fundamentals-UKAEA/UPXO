"""Regression tests for the abstract geoEntities base classes."""
import inspect

from upxo.geoEntities import bases


def test_abstract_coords_bodies_do_not_raise_nameerror():
    """Abstract members must not reference names (e.g. np) the module lacks."""
    for _, cls in inspect.getmembers(bases, inspect.isclass):
        prop = cls.__dict__.get('coords')
        if isinstance(prop, property):
            assert prop.fget(object()) is None
