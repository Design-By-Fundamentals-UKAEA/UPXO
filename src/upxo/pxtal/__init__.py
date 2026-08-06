"""
Polycrystal (pxtal) core package for UPXO grain-structure data models.

Houses:

* MCGS temporal slices — ``mcgs2_temporal_slice``, ``mcgs3_temporal_slice``
* Voronoi tessellation — ``vortess2d`` (``gtess2d``), ``vortess3d`` (``gtess3d``)
* Hierarchical FM steel — ``fm_steel_3d``
* Twinned FCC — ``twinned_simple_3d``
* Supporting geometry / grid / image utilities

Generation drivers live in ``upxo.ggrowth``; this package focuses on structure
objects, specialised pipelines, and related operations.
"""
