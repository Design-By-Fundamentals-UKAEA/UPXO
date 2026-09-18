"""Transformations -- Part D of the FM Steel 3D walkthrough (OPTIONAL).

Two independent, order-dependent operations, both defaulting to the
identity (no change): an isotropic rescale/anisotropic stretch, and a
"Sweep Clean" resolution round-trip (majority-vote downscale then
nearest-neighbour upscale back to the original resolution) intended to
smooth a structure for later conformal meshing. Neither function has a
skip flag -- simply don't call the one(s) you want to omit; downstream
stages (PAG Clustering) just take whichever FMSteel3DBase you hand them.
"""


def apply_transform(fm_base, scale_factor=1.0, sf_x=1.0, sf_y=1.0, sf_z=1.0, cleanup_threshold=0):
    """Isotropic rescale (`scale_factor`, nearest-neighbour resampling, RVE
    physical size held constant) followed by an anisotropic stretch
    (`sf_x`/`sf_y`/`sf_z`, each axis independently). Both default to 1.0
    (no change). `cleanup_threshold` > 0 additionally dissolves any grain
    below that many voxels into its largest neighbour after resampling
    (see transform_shared.build_clean_structure).

    Returns
    -------
    FMSteel3DBase
    """
    from upxo.gsdataops.grid_ops import rescale_grid_3d, stretch_grid_3d
    from upxo.pxtal.fm_steel_3d.gui.transform_shared import build_clean_structure

    seed = fm_base._random_seed or 42

    lgi = rescale_grid_3d(fm_base.lgi, scale_factor, method='nearest')
    voxel_size = fm_base.voxel_size / scale_factor
    physical_dimensions = tuple(n * fm_base.voxel_size for n in fm_base.lgi.shape)

    if (sf_x, sf_y, sf_z) != (1.0, 1.0, 1.0):
        lgi_zyx = lgi.transpose(2, 1, 0)
        stretched_zyx, px_x, _px_y, _px_z, _new_shape = stretch_grid_3d(
            lgi_zyx, stretch_x=sf_x, stretch_y=sf_y, stretch_z=sf_z,
            px_size_x=voxel_size, px_size_y=voxel_size, px_size_z=voxel_size,
            method='nearest')
        lgi = stretched_zyx.transpose(2, 1, 0)
        voxel_size = px_x
        physical_dimensions = tuple(n * voxel_size for n in lgi.shape)

    return build_clean_structure(
        lgi, voxel_size, physical_dimensions, fm_base.units, fm_base.connectivity,
        seed, cleanup_threshold)


def sweep_clean(fm_base, factor=2.0, cleanup_threshold=0):
    """Resolution round-trip: downscale all three dimensions by `factor`
    with majority-vote resampling (minimizes small-feature loss), then
    upscale back to the ORIGINAL voxel resolution with nearest-neighbour
    resampling. Net effect: same resolution as `fm_base`, boundary detail
    coarsened by the round trip. `cleanup_threshold` > 0 additionally
    dissolves any grain below that many voxels into its largest neighbour.

    Returns
    -------
    FMSteel3DBase
    """
    from upxo.gsdataops.grid_ops import resample_grid_3d
    from upxo.pxtal.fm_steel_3d.gui.transform_shared import build_clean_structure

    seed = fm_base._random_seed or 42
    orig_shape = fm_base.lgi.shape
    orig_voxel_size = fm_base.voxel_size
    down_shape = tuple(max(1, int(round(n / factor))) for n in orig_shape)

    down_lgi = resample_grid_3d(fm_base.lgi, down_shape, method='majority')
    up_lgi = resample_grid_3d(down_lgi, orig_shape, method='nearest')
    physical_dimensions = tuple(n * orig_voxel_size for n in orig_shape)

    return build_clean_structure(
        up_lgi, orig_voxel_size, physical_dimensions, fm_base.units, fm_base.connectivity,
        seed, cleanup_threshold)
