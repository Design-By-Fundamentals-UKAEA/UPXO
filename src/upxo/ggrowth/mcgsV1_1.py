"""mcgsV1_1 -- retained for existing imports only.

The implementation moved to :mod:`upxo.ggrowth.mcgs_v2`, which adds 2D
(``dim=2``) and keeps 3D as the default. Import from there::

    from upxo.ggrowth.mcgs_v2 import mcgs_v2, MCGSConfig

``mcgsV1_1`` and ``MCGSConfig`` below are the same objects as ``mcgs_v2`` and
its ``MCGSConfig``. Constructor and ``simulate()`` signatures are unchanged;
``MCGSConfig`` has a new optional ``dim`` field (default 3) and z-axis fields
are keyword-friendly. Existing keyword-argument callers need no change.
"""
import warnings

from upxo.ggrowth.mcgs_v2 import (  # noqa: F401
    mcgs_v2 as mcgsV1_1,
    MCGSConfig,
    VALID_BOLTZMANN_MODES,
    _build_state_boltzmann_probabilities,
)

VALID_ALGORITHMS = ('300a', '300b')

warnings.warn(
    "upxo.ggrowth.mcgsV1_1 is superseded by upxo.ggrowth.mcgs_v2.",
    DeprecationWarning, stacklevel=2)
