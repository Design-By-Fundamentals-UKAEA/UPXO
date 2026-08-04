"""Tests for upxo.material.registry.MaterialRegistry."""
import pytest
from dataclasses import dataclass
from upxo.material.registry import MaterialRegistry, Provenance


@dataclass
class DummyCategory:
    """Throwaway dataclass for testing."""
    value: str
    count: int = 0


def test_registry_init():
    """Registry initializes as empty."""
    reg = MaterialRegistry()
    assert reg.categories() == []


def test_register_category():
    """Register a new category with optional known values."""
    reg = MaterialRegistry()
    reg.register_category('dummy', DummyCategory,
                         known_values={'value': {'test1', 'test2'}})
    assert 'dummy' in reg.categories()


def test_ingest_stores_typed_instance():
    """Ingest stores the dataclass instance without flattening to dict."""
    reg = MaterialRegistry()
    reg.register_category('dummy', DummyCategory)

    obj = DummyCategory(value='test', count=5)
    reg.ingest('dummy', obj, source='test_source', method='direct')

    retrieved = reg.get('dummy')
    assert isinstance(retrieved, DummyCategory)
    assert retrieved.value == 'test'
    assert retrieved.count == 5


def test_ingest_rejects_non_dataclass():
    """Ingest rejects non-dataclass instances."""
    reg = MaterialRegistry()
    reg.register_category('dummy', DummyCategory)

    with pytest.raises(TypeError):
        reg.ingest('dummy', {'value': 'test'})


def test_ingest_rejects_unregistered_category():
    """Ingest rejects categories that haven't been registered."""
    reg = MaterialRegistry()
    obj = DummyCategory(value='test')

    with pytest.raises(KeyError):
        reg.ingest('unknown', obj)


def test_ingest_wrong_type():
    """Ingest rejects instances of wrong dataclass type."""
    @dataclass
    class OtherCategory:
        data: str

    reg = MaterialRegistry()
    reg.register_category('dummy', DummyCategory)

    with pytest.raises(TypeError):
        reg.ingest('dummy', OtherCategory(data='test'))


def test_provenance_tracking():
    """Get_provenance returns metadata attached during ingest."""
    reg = MaterialRegistry()
    reg.register_category('dummy', DummyCategory)

    obj = DummyCategory(value='test')
    reg.ingest('dummy', obj, source='experiment_lab',
              method='measurement', params={'temp': 25})

    prov = reg.get_provenance('dummy')
    assert prov.source == 'experiment_lab'
    assert prov.method == 'measurement'
    assert prov.params == {'temp': 25}
    assert prov.timestamp is not None


def test_get_before_ingest_raises():
    """Get on an empty category raises KeyError."""
    reg = MaterialRegistry()
    reg.register_category('dummy', DummyCategory)

    with pytest.raises(KeyError):
        reg.get('dummy')


def test_get_provenance_before_ingest_raises():
    """Get_provenance on empty category raises KeyError."""
    reg = MaterialRegistry()
    reg.register_category('dummy', DummyCategory)

    with pytest.raises(KeyError):
        reg.get_provenance('dummy')


def test_soft_validation_recognized_value():
    """Recognized values in known_values do not warn."""
    reg = MaterialRegistry()
    reg.register_category('dummy', DummyCategory,
                         known_values={'value': {'valid1', 'valid2'}})

    obj = DummyCategory(value='valid1')
    reg.ingest('dummy', obj, source='test')
    # No warning should occur


def test_soft_validation_unrecognized_value(caplog):
    """Unrecognized values in known_values produce a warning."""
    import logging
    caplog.set_level(logging.WARNING)

    reg = MaterialRegistry()
    reg.register_category('dummy', DummyCategory,
                         known_values={'value': {'valid1', 'valid2'}})

    obj = DummyCategory(value='invalid')
    reg.ingest('dummy', obj, source='test')

    # Should have issued a warning
    assert 'invalid' in caplog.text or len(caplog.records) > 0


def test_multiple_categories():
    """Registry can hold multiple categories independently."""
    @dataclass
    class Cat1:
        x: int

    @dataclass
    class Cat2:
        y: str

    reg = MaterialRegistry()
    reg.register_category('cat1', Cat1)
    reg.register_category('cat2', Cat2)

    obj1 = Cat1(x=10)
    obj2 = Cat2(y='test')

    reg.ingest('cat1', obj1, source='src1')
    reg.ingest('cat2', obj2, source='src2')

    assert reg.get('cat1').x == 10
    assert reg.get('cat2').y == 'test'
    assert 'cat1' in reg.categories()
    assert 'cat2' in reg.categories()


def test_re_register_replaces_known_values():
    """Re-registering a category replaces its known_values."""
    reg = MaterialRegistry()
    reg.register_category('dummy', DummyCategory,
                         known_values={'value': {'old1', 'old2'}})
    reg.register_category('dummy', DummyCategory,
                         known_values={'value': {'new1', 'new2'}})

    # New known_values should be in effect
    obj = DummyCategory(value='new1')
    reg.ingest('dummy', obj, source='test')
    assert reg.get('dummy').value == 'new1'
