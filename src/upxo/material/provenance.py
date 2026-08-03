"""
provenance.py
==============
Generic record of how a value stored in a :class:`MaterialRegistry` was
derived -- what produced it, with what parameters, and when. Attached
uniformly by :meth:`MaterialRegistry.ingest` to whatever is stored, rather
than each data category declaring its own source/method/timestamp fields.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class Provenance:
    """Where an ingested value came from.

    Parameters
    ----------
    source : str, optional
        Where the data originated -- a file path, a dataset identifier,
        "user input", etc.
    method : str, optional
        The function/module/algorithm that produced the value, e.g.
        ``"tcVf_twinned_fcc.fit_components"``.
    params : dict
        Whatever parameters that method was called with, e.g.
        ``{"misori_tol_deg": 5.0, "n_grains": 340}``.
    timestamp : datetime
        When the value was ingested. Defaults to the moment the
        ``Provenance`` instance is constructed.
    """

    source: Optional[str] = None
    method: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
