"""Synthetic .ctf files for testing and smoke-checking EBSDReader.

No real or restricted EBSD data is read, referenced, or shipped by this module. Every value
is generated from a small upxo.ggrowth.mcgs_v2 grain structure with one uniformly random
cubic orientation per grain (upxo.xtalphy.crystal_orientation.rand_uniform_SO3).
"""
import pathlib

import numpy as np

from upxo.ggrowth.mcgs_v2 import MCGSConfig, mcgs_v2
from upxo.xtalphy.crystal_orientation import rand_uniform_SO3, matrix_to_euler_bunge

_HEADER_TEMPLATE = """Channel Text File
Prj\tUPXO synthetic grain structure
Author\tUPXO
JobMode\tGrid
XCells\t{nx}
YCells\t{ny}
XStep\t{step}
YStep\t{step}
AcqE1\t0
AcqE2\t0
AcqE3\t0
Euler angles refer to Sample Coordinate system (CS0)!\tMag\t100\tCoverage\t100\tDevice\t0\tKV\t20\tTiltAngle\t70\tTiltAxis\t0
Phases\t1
3.614;3.614;3.614\t90.000;90.000;90.000\t{phase_name}\t11\t225
Phase\tX\tY\tBands\tError\tEuler1\tEuler2\tEuler3\tMAD\tBC\tBS
"""

_COLUMN_FORMATS = ('%d', '%.3f', '%.3f', '%d', '%d', '%.5f', '%.5f', '%.5f', '%.5f', '%d', '%d')


def make_synthetic_grain_structure(shape=(20, 30), Q=8, mcsteps=6, rng_seed=0):
    """A small, fast mcgs_v2 2D grain structure, grain-labelled.

    Returns ``(lgi, gid)`` of the last saved slice: an (ny, nx) int array and the list of
    grain ids present in it.
    """
    ny, nx = shape
    config = MCGSConfig(xmin=0, xmax=nx - 1, xinc=1, ymin=0, ymax=ny - 1, yinc=1,
                        Q=Q, mcalg='200', mcsteps=mcsteps,
                        save_interval=max(1, mcsteps // 2), consider_boltzmann=False,
                        dim=2, rng_seed=rng_seed)
    pxt = mcgs_v2(config, verbose=False)
    pxt.simulate(verbose=False)
    pxt.detect_grains(verbose=False)
    tslice = max(pxt.gs)
    gs = pxt.gs[tslice]
    return np.asarray(gs.lgi, dtype=np.int64), list(gs.gid)


def write_synthetic_ctf(path, shape=(20, 30), Q=8, mcsteps=6, rng_seed=0, step_size=1.0,
                        phase_name='Copper'):
    """Write a synthetic .ctf file to *path* and return *path*.

    A small mcgs_v2 grain structure (see make_synthetic_grain_structure) with one uniformly
    random cubic orientation per grain, every pixel indexed, single phase. Suitable for
    exercising EBSDReader.load/detect_grains end to end without any real EBSD data.
    """
    path = pathlib.Path(path)
    lgi, gids = make_synthetic_grain_structure(shape=shape, Q=Q, mcsteps=mcsteps, rng_seed=rng_seed)
    ny, nx = lgi.shape

    rng = np.random.default_rng(rng_seed)
    euler_by_gid = {gid: matrix_to_euler_bunge(rand_uniform_SO3(rng), degrees=True) for gid in gids}

    flat_lgi = lgi.ravel()  # Y slowest, X fastest -- matches the CTF row order below
    euler = np.array([euler_by_gid[g] for g in flat_lgi])
    iy, ix = np.divmod(np.arange(flat_lgi.size), nx)

    rows = np.column_stack([
        np.ones_like(flat_lgi),          # Phase: every pixel indexed, single phase
        ix.astype(float) * step_size,    # X
        iy.astype(float) * step_size,    # Y
        np.full(flat_lgi.size, 8),       # Bands
        np.zeros(flat_lgi.size),         # Error
        euler[:, 0], euler[:, 1], euler[:, 2],
        np.full(flat_lgi.size, 0.3),     # MAD
        np.full(flat_lgi.size, 150),     # BC
        np.full(flat_lgi.size, 50),      # BS
    ])

    header = _HEADER_TEMPLATE.format(nx=nx, ny=ny, step=step_size, phase_name=phase_name)
    with path.open('w', encoding='latin-1') as fh:
        fh.write(header)
        np.savetxt(fh, rows, fmt=_COLUMN_FORMATS, delimiter='\t')
    return path
