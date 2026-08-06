"""
Grain-growth simulation package for UPXO (UKAEA Poly-XTAL Operations).

Provides Monte-Carlo Potts grain-structure generation:

* ``mcgs`` / ``grid`` — dashboard-driven 2D/3D simulations (``mcgs.py``)
* ``mcgsV1_1`` — leaner lightweight MC entry (when used by interactive callers)
* ``make3d`` — helpers to stack 2D states into 3D volumes for visualisation

Typical import::

    from upxo.ggrowth.mcgs import mcgs
"""
