"""
Grain-growth simulation package for UPXO (UKAEA Poly-XTAL Operations).

Provides Monte-Carlo Potts grain-structure generation:

* ``mcgs`` / ``grid`` — dashboard-driven 2D/3D simulations (``mcgs.py``)
* ``mcgs_v2`` — 2D/3D MC driven by a plain-Python config, no Excel dashboard
* ``make3d`` — helpers to stack 2D states into 3D volumes for visualisation

Typical import::

    from upxo.ggrowth.mcgs import mcgs
"""
