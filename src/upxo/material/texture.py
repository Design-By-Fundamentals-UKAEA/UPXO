"""
texture.py
===========
Crystal-family-generic, dict-based texture-component profile -- replaces
the old, fixed-field, FCC-hardcoded ``TexCompVolFracFCC``/
``TexFibreVolFracFCC``/``TexCompWidth`` classes, whose ten hardcoded field
names didn't match ``FCCTexture``'s own component naming (see
``admin/material/scoping.md`` §2, item 2).
"""

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class TextureComponentProfile:
    """A material's texture, described as volume fractions and spreads
    over named crystallographic components.

    Parameters
    ----------
    crystal_family : str
        e.g. ``'FCC'`` -- not baked into the class name, unlike the
        classes this replaces, so a future BCC/steel profile needs only a
        different value here, not a structurally different class.
    component_fractions : dict
        Component name -> volume fraction, e.g.
        ``{'copper': 0.45, 'brass': 0.30, ...}``. Matches
        ``FCCTexture.tc_fractions``' shape directly
        (``src/upxo/texOps/fcc.py:236-237``), so a fitted profile can be
        passed straight into
        ``FCCTexture(tc_fractions=profile.component_fractions, ...)``
        without reshaping.
    component_widths : dict
        Component name -> angular spread in degrees. Matches
        ``FCCTexture``'s ``spread`` dict shape (``fcc.py:278-281``) the
        same way.
    random_fraction : float
        Volume fraction attributed to random/untextured background,
        separate from any named component.
    """

    crystal_family: str
    component_fractions: Dict[str, float] = field(default_factory=dict)
    component_widths: Dict[str, float] = field(default_factory=dict)
    random_fraction: float = 0.0
