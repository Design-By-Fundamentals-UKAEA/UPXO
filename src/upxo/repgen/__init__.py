"""
Representative grain-structure generation / candidate ranking for UPXO.

Ranks Monte-Carlo time slices (and related candidates) against a reference
(often EBSD) for morphology and related metrics. Main modules:

* ``repgen2dmcgs`` — 2D MC slice ranking (``repgen2d``)
* ``repgen3dmcgs`` — 3D MC slice ranking (``repgen3d``)

Used by twin FCC workflows (host selection) and representativeness studies.
See also ``upxo.repqual``.
"""
