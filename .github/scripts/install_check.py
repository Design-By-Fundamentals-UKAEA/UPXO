"""Check one installation level of an installed upxo.

Usage:
    python install_check.py <level>                 level: base, viz, mesh, ebsd, all
    python install_check.py --base-requirements <wheel>   print the core requirements of a wheel

Run it from a directory that is not the source checkout, so the installed package is the one imported.
Exits with status 1 when a module fails to import for a reason other than a missing optional package,
or when a functional check fails.
"""
import importlib
import importlib.metadata as md
import pkgutil
import sys
import time
import traceback
import warnings
import zipfile

EXTRAS = {'base': [], 'viz': ['viz'], 'mesh': ['mesh'], 'ebsd': ['ebsd'], 'all': ['viz', 'mesh', 'ebsd']}
# top-level import name of each optional package, by extra
OPTIONAL_PACKAGES = {'viz': {'plotly'}, 'mesh': {'tetgen', 'gmsh'}, 'ebsd': {'defdap'}}


def base_requirements(wheel):
    z = zipfile.ZipFile(wheel)
    meta = z.read([n for n in z.namelist() if n.endswith('METADATA')][0]).decode()
    for line in meta.splitlines():
        if line.startswith('Requires-Dist:') and 'extra ==' not in line:
            print(line.split(':', 1)[1].strip())


def import_sweep(level):
    import upxo
    allowed_missing = set()
    for extra, packages in OPTIONAL_PACKAGES.items():
        if extra not in EXTRAS[level]:
            allowed_missing |= packages
    names = sorted(m.name for m in pkgutil.walk_packages(upxo.__path__, 'upxo.') if not m.name.endswith('__main__'))
    failed = {}
    for name in names:
        try:
            importlib.import_module(name)
        except ModuleNotFoundError as e:
            if (e.name or '').split('.')[0] not in allowed_missing:
                failed[name] = f'ModuleNotFoundError: {e}'
        except BaseException as e:      # noqa: BLE001 - any import-time failure is a finding
            failed[name] = f'{type(e).__name__}: {str(e)[:150]}'
    return names, failed


def check(label, fn):
    t0 = time.time()
    try:
        out = fn()
        print(f'  PASS {label} ({time.time() - t0:.1f} s){": " + str(out) if out is not None else ""}', flush=True)
        return True
    except BaseException:               # noqa: BLE001
        print(f'  FAIL {label}\n' + traceback.format_exc(limit=4), flush=True)
        return False


def base_checks():
    import numpy as np
    from upxo.meshing.gbconformant.d3v2p0.surface_intersections import find_surface_intersections as ref
    from upxo.meshing.gbconformant.d3v2p1 import backend
    from upxo.meshing.gbconformant.d3v2p1.surface_intersections import find_surface_intersections as fast

    def search():
        rng = np.random.default_rng(0)
        pts = (rng.uniform(0, 8, (1500, 1, 3)) + rng.normal(0, .4, (1500, 3, 3))).reshape(-1, 3)
        tri = np.arange(len(pts)).reshape(-1, 3)
        r = ref(pts, tri)
        for tier in ('numpy', 'numba', 'parallel', 'auto'):
            assert np.array_equal(fast(pts, tri, backend=tier, n_workers=2), r), tier
        return f'{len(r)} pairs, numba={backend.numba_available()}'

    def mcgs_slice():
        from upxo.ggrowth.mcgs import mcgs
        pxt = mcgs()
        return type(pxt).__name__
    return all([check('d3v2p1 intersection search, all tiers identical', search),
                check('mcgs object creation', mcgs_slice)])


def mesh_checks():
    import numpy as np
    import pyvista as pv

    def conformal():
        from upxo.meshing.gbconformant.d3v2p0.interfaces import smooth_interfaces
        from upxo.meshing.gbconformant.d3v2p0.rve_caps import close_rve_faces
        from upxo.meshing.gbconformant.d3v2p1.gmsh_tets import mesh_repaired_rve_gmsh
        from upxo.meshing.gbconformant.d3v2p1.tet_smoothing import smooth_grain_tetrahedra
        from upxo.meshing.gbconformant.d3v2p1.tet_swaps import swap_grain_tetrahedra
        i, j, k = np.indices((8, 8, 8)) // 4
        labels = 1 + i * 4 + j * 2 + k
        s = smooth_interfaces(labels, iterations=0)
        xyz = s.original_points[s.triangles]
        normal = np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0])
        s.triangles = s.triangles.copy()
        s.grain_pairs = s.grain_pairs.copy()
        flip = (normal.sum(axis=1) < 0) != (s.grain_pairs[:, 0] > s.grain_pairs[:, 1])
        s.triangles[flip] = s.triangles[flip][:, [0, 2, 1]]
        s.grain_pairs = np.sort(s.grain_pairs, axis=1)
        closed = close_rve_faces(s, labels, mesh_size=1.)
        tets = mesh_repaired_rve_gmsh(closed, mesh_size=1.2, minimum_quality=0., n_workers=2)
        tets = smooth_grain_tetrahedra(tets, len(closed.points), surface=closed, target=30., n_workers=2)
        tets = swap_grain_tetrahedra(tets, closed, target=30.)
        assert abs(tets.report['total_volume'] - 512.) < 1e-6
        return f'{len(tets.tetrahedra)} tets in 8 grains, volume 512'

    def tetgen_run():
        import tetgen
        t = tetgen.TetGen(pv.Cube().triangulate())
        elems = t.tetrahedralize()[1]
        return f'{len(elems)} tets'
    return all([check('3D conformal pipeline with Gmsh (d3v2p0 + d3v2p1)', conformal),
                check('TetGen tetrahedralisation', tetgen_run)])


def ebsd_checks():
    def reader():
        import defdap  # noqa: F401
        from upxo.interfaces.defdap.ebsd_reader import EBSDReader  # noqa: F401
        from upxo.pxtal.twinned_simple_3d import orientation_3d  # noqa: F401
        return f'defdap {md.version("defdap")}'
    return check('DefDAP and EBSDReader import', reader)


def viz_checks():
    def plot():
        import plotly.graph_objects as go
        import upxo.analysis.analysis2d  # noqa: F401
        go.Figure(go.Scatter(x=[0, 1], y=[1, 0])).to_json()
        return f'plotly {md.version("plotly")}'
    return check('plotly figure and analysis2d import', plot)


def main(level):
    warnings.filterwarnings('ignore')
    import matplotlib
    matplotlib.use('Agg')
    print(f'== level {level}: upxo {md.version("upxo")}, python {sys.version.split()[0]}', flush=True)
    t0 = time.time()
    names, failed = import_sweep(level)
    print(f'  import sweep: {len(names) - len(failed)}/{len(names)} modules import ({time.time() - t0:.0f} s)', flush=True)
    for name, why in sorted(failed.items()):
        print(f'    FAIL {name}: {why}')
    results = [not failed, base_checks()]
    if 'mesh' in EXTRAS[level]:
        results.append(mesh_checks())
    if 'ebsd' in EXTRAS[level]:
        results.append(ebsd_checks())
    if 'viz' in EXTRAS[level]:
        results.append(viz_checks())
    print('  LEVEL', level, 'PASSED' if all(results) else 'FAILED')
    return 0 if all(results) else 1


if __name__ == '__main__':
    if sys.argv[1] == '--base-requirements':
        base_requirements(sys.argv[2])
    else:
        sys.exit(main(sys.argv[1]))
