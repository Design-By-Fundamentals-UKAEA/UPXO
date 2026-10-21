"""Tet face and edge swaps (d3v2p0.tet_swaps) with numba kernels.

The swap loop, the candidates, their order and the acceptance rule are
d3v2p0's. With numba available ('auto' or 'numba'), each candidate is
oriented and tested in one compiled call and tets are measured by a compiled
kernel; the 'numpy' tier runs d3v2p0.tet_swaps.swap_tets unchanged. The
re-verification after swapping is d3v2p0's.
"""
from collections import defaultdict
from dataclasses import replace
import numpy as np
from ..d3v2p0 import tet_swaps as ref
from ..d3v2p0.tet_swaps import _Mesh, insert_nodes  # noqa: F401 (re-exported)
from .backend import plan, numba_threads
from .mesh_tables import unique_rows, node_element_index


class _Tets:
    """Read-only sequence of the tets of an _ArrayMesh: row k as a list of Python ints."""

    def __init__(self, mesh):
        self.mesh = mesh

    def __len__(self):
        return len(self.mesh.base) + len(self.mesh.added)

    def __getitem__(self, k):
        n = len(self.mesh.base)
        return self.mesh.base[k].tolist() if k < n else self.mesh.added[k - n]


class _ArrayMesh:
    """d3v2p0's _Mesh with the tets kept in an array.

    The same interface for the swap loop (t, g, alive, star, add), the same values and the same order;
    a tet becomes a Python list only when the loop reads it, instead of every tet at the start.
    """

    def __init__(self, tetrahedra, grain_ids, n_points):
        self.base = np.asarray(tetrahedra)
        self.added = []
        self.t = _Tets(self)
        self.g = list(map(int, grain_ids))
        self.alive = [True] * len(self.base)
        self.order, self.offsets = node_element_index(self.base, n_points)
        self.extra = defaultdict(list)

    def star(self, n):
        base = self.order[self.offsets[n]:self.offsets[n + 1]].tolist()
        return [k for k in base + self.extra.get(n, []) if self.alive[k]]

    def add(self, tet, grain):
        k = len(self.t)
        self.added.append(list(map(int, tet)))
        self.g.append(int(grain))
        self.alive.append(True)
        for n in tet:
            self.extra[int(n)].append(k)
        return k

    def array(self, dtype=np.int64):
        """All tets, original ones first, then the added ones in order, as an (n, 4) array."""
        base = self.base.astype(dtype, copy=False)
        if not self.added:
            return base
        return np.vstack((base, np.asarray(self.added, dtype=dtype)))


def swap_tets(points, tetrahedra, grain_ids, surface_triangles, *, enabled=True, target=30., max_angle=150.,
              max_passes=5, progress=None, backend='auto', n_workers=None):
    """d3v2p0.tet_swaps.swap_tets; backend / n_workers: see d3v2p1.backend
    (the swap loop is serial; n_workers sets the numba threads used to measure
    the whole mesh). Returns (tetrahedra, grain_ids, report)."""
    chosen = plan(backend, n_workers, numba_kernels=True)
    if not chosen.numba:
        out_t, out_g, report = ref.swap_tets(points, tetrahedra, grain_ids, surface_triangles, enabled=enabled,
                                             target=target, max_angle=max_angle, max_passes=max_passes,
                                             progress=progress)
        report['backend'] = chosen.report()
        return out_t, out_g, report
    with numba_threads(chosen.workers):
        out_t, out_g, report = _swap_tets_kernels(points, tetrahedra, grain_ids, surface_triangles, enabled,
                                                  target, max_angle, max_passes, progress)
    report['backend'] = chosen.report()
    return out_t, out_g, report


def _swap_tets_kernels(points, tetrahedra, grain_ids, surface_triangles, enabled, target, max_angle, max_passes,
                       progress):
    from .numba_swaps import evaluate_swap, measure_tets
    p = np.ascontiguousarray(points, dtype=float)
    t0 = np.asarray(tetrahedra)
    if t0.ndim != 2 or t0.shape[1] != 4 or len(grain_ids) != len(t0):
        raise ValueError('Expected (m, 4) tetrahedra with one grain ID each')
    if not np.isfinite(target) or not 0 < target < 70.5:
        raise ValueError('target must lie in (0, 70.5) degrees')
    if not np.isfinite(max_angle) or not 70.5 < max_angle < 180:
        raise ValueError('max_angle must lie in (70.5, 180) degrees')
    if isinstance(max_passes, bool) or not isinstance(max_passes, (int, np.integer)) or max_passes < 0:
        raise ValueError('max_passes must be a nonnegative integer')
    if not isinstance(enabled, (bool, np.bool_)):
        raise ValueError('enabled must be boolean')
    surf = np.sort(np.asarray(surface_triangles), axis=1)
    surface_faces = set(map(tuple, surf.tolist()))
    surface_edges = set(map(tuple, np.sort(surf[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1).tolist()))
    span = 180. - max_angle

    def measure(tets):
        return measure_tets(p, tets, target, span)

    def summary(mn, mx):
        return dict(minimum_dihedral=float(mn.min()), below_target=int(np.sum(mn < target)),
                    above_max_angle=int(np.sum(mx > max_angle)), below_10=int(np.sum(mn < 10)),
                    below_15=int(np.sum(mn < 15)), below_20=int(np.sum(mn < 20)))

    mesh = _ArrayMesh(t0, grain_ids, len(p))
    mn, mx, q = measure(t0)
    before = summary(mn, mx)
    quality = q.tolist()
    history = []
    counts = dict(swaps_2_3=0, swaps_3_2=0)
    for pass_number in range(max_passes if enabled else 0):
        bad = [k for k in range(len(mesh.t)) if mesh.alive[k] and quality[k] < 1.]
        bad.sort(key=lambda k: quality[k])
        done = {'2-3': 0, '3-2': 0}
        for k in bad:
            if not mesh.alive[k] or quality[k] >= 1.:
                continue
            best = None
            tet, grain = mesh.t[k], mesh.g[k]
            # 2-3 across each face
            for i in range(4):
                face = [tet[j] for j in range(4) if j != i]
                if tuple(sorted(face)) in surface_faces:
                    continue
                d = tet[i]
                shared = set(mesh.star(face[0])) & set(mesh.star(face[1])) & set(mesh.star(face[2]))
                others = [o for o in shared if o != k]
                if len(others) != 1 or mesh.g[others[0]] != grain:
                    continue
                o = others[0]
                e = [n for n in mesh.t[o] if n not in face][0]
                ok, val, new = evaluate_swap(p, [tet, mesh.t[o]],
                                             [[face[0], face[1], d, e], [face[1], face[2], d, e],
                                              [face[2], face[0], d, e]], target, span)
                if ok and (best is None or val > best[0]):
                    best = (val, '2-3', [k, o], new.tolist())
            # 3-2 around each edge
            for i in range(4):
                for j in range(i + 1, 4):
                    d, e = tet[i], tet[j]
                    if tuple(sorted((d, e))) in surface_edges:
                        continue
                    ring = sorted(set(mesh.star(d)) & set(mesh.star(e)))
                    if len(ring) != 3 or any(mesh.g[r] != grain for r in ring):
                        continue
                    apex = defaultdict(int)
                    for r in ring:
                        for n in mesh.t[r]:
                            if n not in (d, e):
                                apex[n] += 1
                    if len(apex) != 3 or any(c != 2 for c in apex.values()):
                        continue                               # ring not closed: edge on a boundary
                    a, b, c = list(apex)
                    if tuple(sorted((a, b, c))) in surface_faces:
                        continue
                    ok, val, new = evaluate_swap(p, [mesh.t[r] for r in ring], [[a, b, c, d], [a, b, c, e]],
                                                 target, span)
                    if ok and (best is None or val > best[0]):
                        best = (val, '3-2', ring, new.tolist())
            if best is None:
                continue
            _, kind, old_ids, new = best
            for r in old_ids:
                mesh.alive[r] = False
            _, _, nq = measure(new)
            for x, qq in zip(new, nq):
                mesh.add(x, grain)
                quality.append(float(qq))
            done[kind] += 1
        counts['swaps_2_3'] += done['2-3']
        counts['swaps_3_2'] += done['3-2']
        alive = np.flatnonzero(mesh.alive)
        amn, amx, _ = measure(mesh.array()[alive])
        entry = dict(pass_number=pass_number + 1, swaps_2_3=done['2-3'], swaps_3_2=done['3-2'],
                     below_target=int(np.sum(amn < target)), below_15=int(np.sum(amn < 15)),
                     below_10=int(np.sum(amn < 10)), minimum_dihedral=float(amn.min()))
        history.append(entry)
        if progress is not None:
            progress(entry)
        if not done['2-3'] and not done['3-2']:
            break
    alive = np.flatnonzero(mesh.alive)
    tets = mesh.array(t0.dtype)[alive]
    grains = np.asarray(mesh.g, dtype=np.asarray(grain_ids).dtype)[alive]
    mn, mx, _ = measure(tets)
    report = dict(enabled=bool(enabled), target_min_dihedral=float(target), max_dihedral=float(max_angle),
                  before=before, after=summary(mn, mx), history=history, **counts)
    return tets, grains, report


def swap_grain_tetrahedra(tets, surface, *, enabled=True, target=30., max_angle=150., max_passes=5,
                          progress=None, backend='auto', n_workers=None):
    """d3v2p0.tet_swaps.swap_grain_tetrahedra on the d3v2p1 swap_tets
    (backend / n_workers: see swap_tets); re-verification is _reverify below."""
    new_t, new_g, swaps = swap_tets(tets.points, tets.tetrahedra, tets.grain_ids, surface.triangles,
                                    enabled=enabled, target=target, max_angle=max_angle,
                                    max_passes=max_passes, progress=progress, backend=backend, n_workers=n_workers)
    report = dict(tets.report)
    report['tet_swaps'] = swaps
    if not enabled:
        return replace(tets, report=report)
    return _reverify(tets, tets.points, new_t, new_g, surface, report, 'swaps')


def _reverify(tets, points, new_t, new_g, surface, report, what):
    """d3v2p0.tet_swaps._reverify: the same checks, report and errors.

    The face tables come from integer keys and the per-grain sums run over grain-sorted tet numbers,
    which are the same tets in the same order as d3v2p0's masks, so every sum is the same number.
    """
    from ..d3v2p0.tet_smoothing import minimum_sicn_gmsh, _volumes
    from ..d3v2p0.tet_angles import dihedral_summary
    volumes = _volumes(points[new_t])
    if np.any(~np.isfinite(volumes)) or np.any(volumes <= 0):
        raise RuntimeError(f'{what} produced a nonpositive tetrahedron volume')
    # conformity: each grain's single-use tet faces must be exactly its surface triangles
    pairs = np.asarray(surface.grain_pairs)
    exterior = np.asarray(surface.exterior, bool)
    pattern = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]
    faces = np.sort(new_t[:, pattern].reshape(-1, 3), axis=1)
    key = np.column_stack((np.repeat(new_g, 4), faces))
    uniq, counts = unique_rows(key, False, True)
    if np.any(counts > 2):
        raise RuntimeError(f'Nonmanifold tet faces after {what}')
    boundary = uniq[counts == 1]
    st = np.sort(np.asarray(surface.triangles), axis=1)
    expected = unique_rows(np.vstack((np.column_stack((pairs[:, 0], st)),
                                      np.column_stack((pairs[~exterior, 1], st[~exterior])))))
    if not np.array_equal(boundary, expected):
        raise RuntimeError(f'Tet boundary differs from the grain surfaces after {what}')
    quality = minimum_sicn_gmsh(points, new_t)
    if np.any(~np.isfinite(quality)) or np.any(quality <= 0):
        raise RuntimeError(f'Invalid tetrahedron quality after {what}')
    per_grain = {}
    by_grain = np.argsort(new_g, kind='stable')                 # tet numbers of a grain, ascending, as a mask gives
    sorted_grain = new_g[by_grain]
    for gid, entry in tets.report['per_grain'].items():
        first = np.searchsorted(sorted_grain, int(gid), side='left')
        last = np.searchsorted(sorted_grain, int(gid), side='right')
        selected = by_grain[first:last]
        volume = float(volumes[selected].sum())
        if not np.isclose(volume, entry['volume'], rtol=1e-8):
            raise RuntimeError(f'Volume of grain {gid} changed during {what}')
        per_grain[gid] = dict(entry, tetrahedra=int(len(selected)), volume=volume,
                              minimum_quality=float(quality[selected].min()))
    dihedral = dihedral_summary(points, new_t)
    passed = bool(quality.min() >= report['quality_threshold'])
    if report.get('minimum_dihedral_threshold') is not None:
        passed = passed and dihedral['minimum'] >= report['minimum_dihedral_threshold']
    report.update(status='TET_MESH_VERIFIED' if passed else 'TET_QUALITY_BELOW_TARGET', ready=passed,
                  per_grain=per_grain, tetrahedra=int(len(new_t)), nodes=int(len(points)),
                  total_volume=float(volumes.sum()), minimum_quality=float(quality.min()),
                  quality_percentiles_1_5_50=np.percentile(quality, [1, 5, 50]).tolist(),
                  below_quality_threshold=int(np.sum(quality < report['quality_threshold'])),
                  dihedral_angles=dihedral)
    return replace(tets, points=points, tetrahedra=new_t, grain_ids=new_g, quality=quality, report=report)


def insert_grain_tetrahedra(tets, surface, *, enabled=True, target=30., max_angle=150., cavity_rings=(1, 2, 3),
                            max_passes=3, progress=None):
    """d3v2p0.tet_swaps.insert_grain_tetrahedra: d3v2p0's insertion, re-verified by _reverify above."""
    points, new_t, new_g, insertion = insert_nodes(
        tets.points, tets.tetrahedra, tets.grain_ids, surface.triangles, enabled=enabled, target=target,
        max_angle=max_angle, cavity_rings=cavity_rings, max_passes=max_passes, progress=progress)
    report = dict(tets.report)
    report['node_insertion'] = insertion
    if not enabled or not insertion['inserted_nodes']:
        return replace(tets, report=report)
    return _reverify(tets, points, new_t, new_g, surface, report, 'node insertion')
