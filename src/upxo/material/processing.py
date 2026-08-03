"""
processing.py
==============
A material's processing history, modeled as an ordered sequence of typed
steps -- replaces the old, flat ``ProcessingCondition`` (a single
``ht``/``pro``/``app``/``appLoc`` snapshot with no way to represent a
multi-step route). Texture is history- and sequence-dependent, so knowing
*which* step happened *and in what order* -- not just "some processing
happened" -- is what downstream consumers like ``tcVf_twinned_fcc`` need.
"""

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

# The four deformation-route types tcVf_twinned_fcc's processing-route ->
# texture-model mapping distinguishes (see
# admin/twinnedFccGui/texIntegration/tcVf_scoping.md §3.2), plus common
# non-deformation steps a route might also include.
KNOWN_PROCESSING_STEP_TYPES: Set[str] = {
    "rolling_symmetric", "rolling_asymmetric", "extrusion", "forging",
    "heat_treatment", "casting", "machining", "aging",
}

# The subset of KNOWN_PROCESSING_STEP_TYPES that ProcessingRoute's
# last_deformation_step looks for -- every other type is a non-deformation
# step it skips over when searching.
_DEFORMATION_STEP_TYPES: Set[str] = {
    "rolling_symmetric", "rolling_asymmetric", "extrusion", "forging",
}


@dataclass
class ProcessingStep:
    """One step in a material's processing route.

    Parameters
    ----------
    step_type : str
        What kind of step this is. Soft-validated against
        `KNOWN_PROCESSING_STEP_TYPES` at construction time -- an
        unrecognized value warns but is still accepted.

        This validates itself here, in `__post_init__`, rather than
        relying on `MaterialRegistry`'s `known_values` mechanism
        (`registry.py`): that mechanism only checks direct fields of a
        *registered* category's own top-level instance (e.g.
        `MaterialIdentity.name`) via `getattr`, and can't reach into a
        `ProcessingRoute`'s nested `steps` list to validate each step's
        `step_type` individually.
    params : dict
        Step-specific parameters (e.g. reduction ratio, temperature).
    notes : str
        Free-text notes.
    """

    step_type: str
    params: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    def __post_init__(self) -> None:
        if self.step_type not in KNOWN_PROCESSING_STEP_TYPES:
            warnings.warn(
                f'ProcessingStep: "{self.step_type}" is not a recognized '
                f'step type (known: {sorted(KNOWN_PROCESSING_STEP_TYPES)}). '
                f'Accepted anyway -- soft validation only.',
                stacklevel=2)


@dataclass
class ProcessingRoute:
    """A material's processing history as an ordered sequence of steps.

    Replaces the old, flat ``ProcessingCondition``. Order matters --
    texture depends on which step happened, and in what sequence, not
    just which step types occurred somewhere in the material's history.
    """

    steps: List[ProcessingStep] = field(default_factory=list)

    def append_step(self, step_type: str, **params: Any) -> None:
        """Append a new ``ProcessingStep(step_type, params)`` to the route."""
        self.steps.append(ProcessingStep(step_type=step_type, params=params))

    @property
    def last_deformation_step(self) -> Optional[ProcessingStep]:
        """The most recent step whose type is one of the four deformation
        routes (rolling_symmetric/rolling_asymmetric/extrusion/forging),
        skipping non-deformation steps (heat_treatment, machining, ...)
        that may follow it. Returns ``None`` if the route has no
        deformation step at all.

        This is what ``tcVf_twinned_fcc``'s processing-route ->
        texture-model mapping
        (admin/twinnedFccGui/texIntegration/tcVf_scoping.md §3.2) is
        meant to call.
        """
        for step in reversed(self.steps):
            if step.step_type in _DEFORMATION_STEP_TYPES:
                return step
        return None
