"""
Low-level Monte-Carlo iteration kernels for UPXO grain growth.

Selected by the Excel dashboard field ``mcalg`` from ``upxo.ggrowth.mcgs``:

**Dispatched by ``simulate()`` today**

* 2D: ``alg200``, ``alg201``, ``alg202`` (ids ``200`` / ``201`` / ``202``)
* 3D: ``alg300a``, ``alg300b``, ``alg301``, ``alg302`` (ids ``300a`` / ``300b`` /
  ``301.0`` / ``302.0``)

Other modules in this package (e.g. ``alg220``–``alg230``, ``alg310*``,
``alg300_old``) are experimental or legacy and are **not** selected by the
stock dispatcher unless you extend ``mcgs`` yourself.

Kernels are Numba-oriented; prefer the wiki *Monte Carlo Algorithms* page for
user-facing algorithm choice guidance.
"""
