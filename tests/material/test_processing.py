"""Tests for upxo.material.processing (ProcessingRoute, ProcessingStep)."""
import pytest
from upxo.material.processing import ProcessingStep, ProcessingRoute


def test_processing_step_creation():
    """Create a ProcessingStep with valid step_type."""
    step = ProcessingStep(step_type='heat_treatment',
                         params={'temp': 800, 'time_min': 30})
    assert step.step_type == 'heat_treatment'
    assert step.params == {'temp': 800, 'time_min': 30}


def test_processing_step_known_types():
    """ProcessingStep accepts all KNOWN_PROCESSING_STEP_TYPES."""
    known_types = [
        'rolling_symmetric', 'rolling_asymmetric', 'extrusion', 'forging',
        'heat_treatment', 'casting', 'machining', 'aging'
    ]
    for step_type in known_types:
        step = ProcessingStep(step_type=step_type)
        assert step.step_type == step_type


def test_processing_step_unknown_type(caplog):
    """ProcessingStep warns on unrecognized step_type but still constructs."""
    import logging
    caplog.set_level(logging.WARNING)

    step = ProcessingStep(step_type='unknown_process', params={})
    assert step.step_type == 'unknown_process'
    # Warning should be issued


def test_processing_route_empty():
    """ProcessingRoute initializes with empty steps list."""
    route = ProcessingRoute()
    assert route.steps == []


def test_processing_route_append_step():
    """Append steps to a route using append_step method."""
    route = ProcessingRoute()
    route.append_step('casting')
    route.append_step('heat_treatment', temp=900)

    assert len(route.steps) == 2
    assert route.steps[0].step_type == 'casting'
    assert route.steps[1].step_type == 'heat_treatment'
    assert route.steps[1].params == {'temp': 900}


def test_processing_route_last_deformation_step_empty():
    """last_deformation_step on empty route returns None."""
    route = ProcessingRoute()
    assert route.last_deformation_step is None


def test_processing_route_last_deformation_step_only_non_deformation():
    """last_deformation_step with only non-deformation steps returns None."""
    route = ProcessingRoute()
    route.append_step('heat_treatment')
    route.append_step('aging')

    assert route.last_deformation_step is None


def test_processing_route_last_deformation_step_single():
    """last_deformation_step returns the only deformation step."""
    route = ProcessingRoute()
    route.append_step('heat_treatment')
    route.append_step('rolling_symmetric', reduction=0.5)

    assert route.last_deformation_step is not None
    assert route.last_deformation_step.step_type == 'rolling_symmetric'


def test_processing_route_last_deformation_step_multiple():
    """last_deformation_step returns the most recent deformation step."""
    route = ProcessingRoute()
    route.append_step('rolling_symmetric')
    route.append_step('heat_treatment')
    route.append_step('forging')

    assert route.last_deformation_step is not None
    assert route.last_deformation_step.step_type == 'forging'


def test_processing_route_last_deformation_step_skips_later_non_deformation():
    """last_deformation_step ignores non-deformation steps that come after."""
    route = ProcessingRoute()
    route.append_step('extrusion')
    route.append_step('heat_treatment')
    route.append_step('aging')

    assert route.last_deformation_step is not None
    assert route.last_deformation_step.step_type == 'extrusion'


def test_processing_route_deformation_types():
    """Check which step types are considered deformation."""
    # Only these four are deformation per _DEFORMATION_STEP_TYPES
    deformation_types = [
        'rolling_symmetric', 'rolling_asymmetric', 'extrusion', 'forging'
    ]
    # Everything else is non-deformation
    non_deformation_types = ['heat_treatment', 'casting', 'aging', 'machining']

    for step_type in deformation_types:
        route = ProcessingRoute()
        route.append_step(step_type)
        assert route.last_deformation_step is not None, f"{step_type} should be deformation"
        assert route.last_deformation_step.step_type == step_type

    for step_type in non_deformation_types:
        route = ProcessingRoute()
        route.append_step(step_type)
        assert route.last_deformation_step is None, f"{step_type} should not be deformation"


def test_processing_step_with_notes():
    """ProcessingStep can include optional notes."""
    step = ProcessingStep(step_type='heat_treatment',
                         params={'temp': 850},
                         notes='Vacuum furnace, slow cool')
    assert step.notes == 'Vacuum furnace, slow cool'


def test_processing_route_order_preserved():
    """Steps remain in the order they were appended."""
    route = ProcessingRoute()
    types = ['casting', 'rolling_symmetric', 'heat_treatment', 'forging', 'aging']

    for step_type in types:
        route.append_step(step_type)

    for i, expected_type in enumerate(types):
        assert route.steps[i].step_type == expected_type
