"""Base Grain Structure -- Part B of the FM Steel 3D walkthrough.

Two independent generation methods, both terminating in the same
`build_base` step: Voronoi (a single nearest-seed tessellation, instant)
and Monte Carlo (a Potts-model grain-growth simulation -- one candidate
grain structure per saved temporal slice, pick one afterward).
"""
import numpy as np


def generate_voronoi_lfi(nx, ny, nz, n_seeds, seed):
    """Nearest-seed-argmin Voronoi tessellation -- aperiodic, random seed
    placement only (matches the FM Steel GUI's current Voronoi path exactly;
    periodic/regular seed layouts are not wired up anywhere in the pipeline
    today despite the GUI exposing dropdowns for them).

    Returns
    -------
    np.ndarray, shape (nx, ny, nz), dtype int32 : 1-indexed grain-ID field.
    """
    np.random.seed(seed)
    cx = np.random.randint(0, nx, n_seeds).astype(np.float32)
    cy = np.random.randint(0, ny, n_seeds).astype(np.float32)
    cz = np.random.randint(0, nz, n_seeds).astype(np.float32)
    xi = np.arange(nx, dtype=np.float32)[:, None, None, None]
    yi = np.arange(ny, dtype=np.float32)[None, :, None, None]
    zi = np.arange(nz, dtype=np.float32)[None, None, :, None]
    sq_dist = (xi - cx) ** 2 + (yi - cy) ** 2 + (zi - cz) ** 2
    return (np.argmin(sq_dist, axis=3) + 1).astype(np.int32)


def run_mc_simulation(nx=50, ny=50, nz=50, q_states=10, mcsteps=100, save_interval=5,
                       mcalg='300b', consider_boltzmann=True, boltzmann_temp_factor=0.01,
                       print_interval=10, rng_seed=0, verbose=True):
    """Runs a 3D Potts-model Monte Carlo grain-growth simulation.

    `nx`/`ny`/`nz` are voxel counts (not physical dimensions); xmax/ymax/zmax
    are derived as `n - 1` with unit increment, matching the FM Steel GUI's
    own MCGSConfig construction exactly (pages_basegrain.py).

    Returns
    -------
    pxt : the mcgsV1_1 simulation object. `pxt.m` lists saved temporal-slice
    indices; `pxt.gs[t].s` is the raw state array (shape (nz, ny, nx),
    z-first) for slice `t`.
    """
    from upxo.ggrowth.mcgsV1_1 import mcgsV1_1, MCGSConfig
    config = MCGSConfig(
        xmin=0.0, xmax=float(nx - 1), xinc=1.0,
        ymin=0.0, ymax=float(ny - 1), yinc=1.0,
        zmin=0.0, zmax=float(nz - 1), zinc=1.0,
        Q=q_states, mcalg=mcalg, mcsteps=mcsteps, save_interval=save_interval,
        print_interval=print_interval,
        consider_boltzmann=consider_boltzmann, boltzmann_mode='q_unrelated',
        boltzmann_temp_factor=boltzmann_temp_factor if consider_boltzmann else None,
        boltzmann_temp_factors=None, rng_seed=rng_seed,
    )
    pxt = mcgsV1_1(config, verbose=verbose)
    pxt.simulate(verbose=verbose)
    return pxt


def extract_mc_tslice_lfi(pxt, tslice_key, connectivity=6, randomize_ids=True):
    """Connected-component-labels one saved MC temporal slice into a
    grain-ID field, transposed into the (nx, ny, nz) axis order the rest of
    the pipeline expects (the raw MC state array is z-first).

    Returns
    -------
    np.ndarray, shape (nx, ny, nz), dtype int32
    """
    state_s = pxt.gs[tslice_key].s  # (nz, ny, nx)
    try:
        import cc3d
        lfi_zyx = cc3d.connected_components(state_s, connectivity=connectivity).astype(np.int32)
    except ImportError:
        from scipy.ndimage import label as _label
        struct = np.ones((3, 3, 3), dtype=int) if connectivity >= 26 else None
        lfi_zyx, _ = _label(state_s, structure=struct)
        lfi_zyx = lfi_zyx.astype(np.int32)
    lfi = np.transpose(lfi_zyx, (2, 1, 0))
    if randomize_ids:
        from upxo.gsdataops.gid_ops import shuffleLFIIDs
        lfi = shuffleLFIIDs(None, lfi)
    return lfi


def compute_morphology_stats(lfi):
    """Full-structure morphology metrics (volume, surface area, neighbour
    counts, junction lines/points per grain) -- the same call the FM Steel
    GUI uses to populate its "3D SGS Morphology Parameters" plots.

    Returns
    -------
    dict : geom_metrics_3d.all_metrics(lfi)'s own return value
    (keys include 'vol', 'surf', 'jl', 'jp', 'nb_ids').
    """
    from upxo.pxtal.fm_steel_3d import geom_metrics_3d
    return geom_metrics_3d.all_metrics(lfi)


def build_base(lfi, physical_dimensions, voxel_size, units='microns', connectivity=6,
               min_grain_nvoxels=-1, random_seed=None, verbosity=0):
    """Terminal step shared by BOTH generation methods: wraps a raw
    grain-ID field into the pipeline's foundation object.

    Returns
    -------
    upxo.pxtal.fm_steel_3d.base_3d.FMSteel3DBase
    """
    from upxo.pxtal.fm_steel_3d.base_3d import FMSteel3DBase
    return FMSteel3DBase.from_lfi(
        lfi, physical_dimensions=physical_dimensions, voxel_size=voxel_size,
        units=units, connectivity=connectivity, min_grain_nvoxels=min_grain_nvoxels,
        random_seed=random_seed, verbosity=verbosity)
