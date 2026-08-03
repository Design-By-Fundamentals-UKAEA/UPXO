"""
identity.py
============
``MaterialIdentity``, moved out of ``Material.py`` unchanged in shape --
the only addition is ``KNOWN_MATERIALS``, a soft-validation whitelist
consumed by ``MaterialRegistry`` when this category gets registered (see
``registry.py``'s ``known_values`` mechanism).
"""

from dataclasses import dataclass, field
from typing import Set

# Permitted material names, as scoped in
# admin/twinnedFccGui/texIntegration/tcVf_scoping.md §3.1. Not an
# exhaustive/enforced list -- MaterialRegistry's soft validation only
# warns on an unrecognized value, it never blocks.
KNOWN_MATERIALS: Set[str] = {"OFHC-Cu", "CuCrZr", "AA7075", "AA2099"}


@dataclass(frozen=False, repr=True)
class MaterialIdentity:
    """Identifies a specific material -- name, alloy grade, composition."""

    name: str = field(default='cu')       # Name of the material
    alloy: str = field(default='value')   # Alloy grade
    comp: str = field(default='value', compare=False)  # Composition
