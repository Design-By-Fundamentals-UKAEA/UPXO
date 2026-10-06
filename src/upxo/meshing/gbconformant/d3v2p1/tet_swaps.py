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
from ..d3v2p0.tet_swaps import _Mesh, _reverify, insert_nodes, insert_grain_tetrahedra  # noqa: F401 (re-exported)
from .backend import plan, numba_threads


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

    mesh = _Mesh(t0, grain_ids, len(p))
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
        amn, amx, _ = measure(np.asarray(mesh.t)[alive])
        entry = dict(pass_number=pass_number + 1, swaps_2_3=done['2-3'], swaps_3_2=done['3-2'],
                     below_target=int(np.sum(amn < target)), below_15=int(np.sum(amn < 15)),
                     below_10=int(np.sum(amn < 10)), minimum_dihedral=float(amn.min()))
        history.append(entry)
        if progress is not None:
            progress(entry)
        if not done['2-3'] and not done['3-2']:
            break
    alive = np.flatnonzero(mesh.alive)
    tets = np.asarray(mesh.t, dtype=t0.dtype)[alive]
    grains = np.asarray(mesh.g, dtype=np.asarray(grain_ids).dtype)[alive]
    mn, mx, _ = measure(tets)
    report = dict(enabled=bool(enabled), target_min_dihedral=float(target), max_dihedral=float(max_angle),
                  before=before, after=summary(mn, mx), history=history, **counts)
    return tets, grains, report


def swap_grain_tetrahedra(tets, surface, *, enabled=True, target=30., max_angle=150., max_passes=5,
                          progress=None, backend='auto', n_workers=None):
    """d3v2p0.tet_swaps.swap_grain_tetrahedra on the d3v2p1 swap_tets
    (backend / n_workers: see swap_tets); re-verification is d3v2p0's."""
    new_t, new_g, swaps = swap_tets(tets.points, tets.tetrahedra, tets.grain_ids, surface.triangles,
                                    enabled=enabled, target=target, max_angle=max_angle,
                                    max_passes=max_passes, progress=progress, backend=backend, n_workers=n_workers)
    report = dict(tets.report)
    report['tet_swaps'] = swaps
    if not enabled:
        return replace(tets, report=report)
    return _reverify(tets, tets.points, new_t, new_g, surface, report, 'swaps')
