"""Thin, reusable pipeline-stage wrappers for the FM Steel 3D pipeline.

Each `steps_XXX` module wraps the core API for one GUI pipeline stage
(see `src/upxo/pxtal/fm_steel_3d/gui/` for the reference implementation
each wrapper mirrors). Backs the `demos/FMSteel3D/*.ipynb` notebooks, but
lives here -- inside the real `upxo.pxtal.fm_steel_3d` package, not under
`demos/` -- so it ships with the installed package and is directly usable
from automation scripts, not just from a dev checkout with the demo
notebooks' own `sys.path` manipulation.
"""
