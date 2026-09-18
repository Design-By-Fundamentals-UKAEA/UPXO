"""Transformations -- Part G of the Twinned FCC walkthrough (OPTIONAL).

Rescales then stretches the selected temporal slice's base grain
structure to introduce non-equiaxiality before Host Allocation. All
factors default to 1.0 (identity -- no change), so this stage is
genuinely optional: if you never call `apply_transform`, just use
TwinnedSimple3DBase.from_mcgs(pxt, tslice_key) directly and Host
Allocation gets the unmodified structure.
"""


def apply_transform(pxt, tslice_key, scale_factor=1.0, sf_x=1.0, sf_y=1.0, sf_z=1.0):
    """Rescales (isotropic `scale_factor`) then stretches (per-axis
    sf_x/sf_y/sf_z) the base grain structure at `tslice_key`, always
    composed FROM THE ORIGINAL structure (never chained onto a previous
    transform) so re-calling this with new factors replaces, rather than
    compounds, the previous result.

    Returns
    -------
    (new_base, original_base) : both TwinnedSimple3DBase instances.
    """
    from upxo.pxtal.twinned_simple_3d.base_3d import TwinnedSimple3DBase
    import upxo.gsdataops.grid_ops as gridOps

    original = TwinnedSimple3DBase.from_mcgs(pxt, int(tslice_key))

    lgi = gridOps.rescale_grid_3d(original.lgi, scale_factor, method='nearest')
    voxel_size = original.voxel_size / scale_factor
    lgi, _px_x, _px_y, _px_z, _new_shape = gridOps.stretch_grid_3d(
        lgi, stretch_x=sf_x, stretch_y=sf_y, stretch_z=sf_z,
        px_size_x=voxel_size, px_size_y=voxel_size, px_size_z=voxel_size)

    new_base = TwinnedSimple3DBase(lgi, voxel_size, original.units)
    new_base._tslice_key = original._tslice_key
    return new_base, original
