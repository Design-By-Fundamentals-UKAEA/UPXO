"""Block-parallel tet dihedral smoothing (d3v2p0.tet_smoothing rules, unchanged).

The RVE is cut into cubic blocks of fixed size, coloured by the parity of
their grid indices (8 colours), so blocks of one colour are a block apart.
Each pass runs 8 phases, one per colour; the blocks of a phase run in
parallel. A block moves the nodes it contains (membership from the positions
at the start of the pass) and reads the nodes around them, which may lie in
neighbouring blocks of other colours; those stay fixed during the phase.
Every node is thus worked on once per pass, including nodes at block faces.
A node is skipped only if another block of the same phase would read it
(a star wider than a block). Each block runs d3v2p0's node search and
acceptance rule; tet and surface-triangle floors are anchored to the start of
the whole call and the surface deviation limit is computed once for the whole
surface, so the d3v2p0 guarantees hold across blocks. Block layout does not
depend on the worker count, so results are identical for any n_workers.
"""
import math
from dataclasses import replace
import numpy as np
from .backend import plan, WorkerPool
from .mesh_tables import node_element_index
from ..d3v2p0.tet_smoothing import surface_after_smoothing  # unchanged; re-exported for notebooks


def _check_arguments(p, t, movable, *, enabled, target, max_angle, neighbour_floor, max_passes, node_iterations,
                     max_halvings, initial_step, softness, max_deviation, minimum_facet_angle, triangle_target,
                     triangle_floor, search_directions):
    """The argument checks of d3v2p0.tet_smoothing.smooth_tet_dihedrals, copied from it in its order and with
    its messages, without its setup (the caller does that once)."""
    from ..d3v2p0.tet_smoothing import _check_int
    if p.ndim != 2 or p.shape[1] != 3 or t.ndim != 2 or t.shape[1] != 4:
        raise ValueError('Expected (n, 3) points and (m, 4) tetrahedra')
    if movable.shape != (len(p),):
        raise ValueError('movable must be a boolean mask over points')
    if not np.isfinite(target) or not 0 < target < 70.5:
        raise ValueError('target must lie in (0, 70.5) degrees')
    if not np.isfinite(max_angle) or not 70.5 < max_angle < 180:
        raise ValueError('max_angle must lie in (70.5, 180) degrees')
    neighbour_floor = target if neighbour_floor is None else neighbour_floor
    if not np.isfinite(neighbour_floor) or not 0 <= neighbour_floor <= target:
        raise ValueError('neighbour_floor must lie in [0, target]')
    if not np.isfinite(triangle_target) or not 0 < triangle_target < 60:
        raise ValueError('triangle_target must lie in (0, 60) degrees')
    triangle_floor = triangle_target if triangle_floor is None else triangle_floor
    if not np.isfinite(triangle_floor) or not 0 <= triangle_floor <= triangle_target:
        raise ValueError('triangle_floor must lie in [0, triangle_target]')
    _check_int('max_passes', max_passes, 0)
    _check_int('node_iterations', node_iterations, 1)
    _check_int('max_halvings', max_halvings, 0)
    if not np.isfinite(initial_step) or initial_step <= 0 or not np.isfinite(softness) or softness <= 0:
        raise ValueError('initial_step and softness must be positive')
    if max_deviation is not None and (not np.isfinite(max_deviation) or max_deviation < 0):
        raise ValueError('max_deviation must be finite and nonnegative')
    if not np.isfinite(minimum_facet_angle) or not 0 <= minimum_facet_angle < 180:
        raise ValueError('minimum_facet_angle must lie in [0, 180)')
    if not all(isinstance(v, (bool, np.bool_)) for v in (enabled, search_directions)):
        raise ValueError('enabled and search_directions must be boolean')


def _improve_node_kernels(n, r, p, t, mn0, mx0, target, floor, max_angle, quality, max_halvings, initial_step,
                          softness, project=None, check=None, final=None, search_directions=True):
    """d3v2p0.tet_smoothing._improve_node with trial positions evaluated by the
    numba star kernel (same trials, rule and ranking)."""
    from ..d3v2p0.tet_smoothing import _DIRECTIONS
    from .numba_smoothing import star_trials
    x0 = p[t[r]]
    at = t[r] == n                                   # (k, 4) positions of the node in its star
    slot = np.argmax(at, axis=1)
    origin = p[n].copy()

    def evaluate(y):
        return star_trials(x0, slot, y)

    def softmin(a_min, a_max):
        g = quality(a_min, a_max)
        lo = g.min(axis=1, keepdims=True)
        return lo[:, 0] - np.log(np.exp(-softness * (g - lo)).sum(axis=1)) / softness

    others = x0[~at]
    h = float(np.linalg.norm(others - origin, axis=1).mean())
    if not np.isfinite(h) or h <= 0:
        return None
    eps = 1e-4 * h
    probes = origin + np.vstack((np.eye(3), -np.eye(3))) * eps
    f = softmin(*evaluate(probes)[:2])
    grad = (f[:3] - f[3:]) / (2 * eps)
    norm = np.linalg.norm(grad)
    steps = initial_step * h * .5 ** np.arange(max_halvings + 1)
    if np.isfinite(norm) and norm > 0:
        trial = origin + steps[:, None] * (grad / norm)
    elif search_directions:
        trial = np.empty((0, 3))
    else:
        return None
    if search_directions:
        extra = (origin + (steps[:4, None, None] * _DIRECTIONS[None]).reshape(-1, 3))
        trial = np.vstack((trial, extra))
    if project is not None:
        trial = project(trial)
        moved = np.linalg.norm(trial - origin, axis=1) > 1e-12 * h
        if not moved.any():
            return None
        trial = trial[moved]
    a_min, a_max, vol = evaluate(trial)
    lower = np.where(mn0 >= target, floor, mn0)
    q0 = quality(mn0, mx0).min()
    ok = (np.all(vol > 0, axis=1)
          & np.all(a_min >= lower - 1e-9, axis=1)
          & np.all(a_max <= np.maximum(mx0, max_angle) + 1e-9, axis=1)
          & (quality(a_min, a_max).min(axis=1) > q0 + 1e-9))
    if not ok.any():
        return None
    candidates = np.flatnonzero(ok)
    candidates = candidates[np.argsort(-quality(a_min[candidates], a_max[candidates]).min(axis=1), kind='stable')]
    for best in candidates:
        if check is not None and not check(trial[best][None])[0]:
            continue
        if final is None or final(trial[best]):
            return trial[best], a_min[best], a_max[best]
    return None


def _keeps_openings_scalar(moves, n, y, p):
    """d3v2p0 _SurfaceMoves.keeps_openings with scalar opening arithmetic."""
    if moves.wedge_limit is None:
        return True
    from .surface_angles import _opening
    f = moves.f
    ring = moves.topo.ring(n)
    edges = sorted({(min(a, b), max(a, b)) for tri in f[ring].tolist()
                    for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0]))})
    cache = {}

    def before(m):
        c = cache.get(m)
        if c is None:
            c = cache[m] = tuple(p[m].tolist())
        return c
    moved = tuple(float(v) for v in y)

    def after(m):
        return moved if m == n else before(m)
    for e in edges:
        tris = [tuple(f[r].tolist()) for r in moves.topo.edge_tri.get(e, [])]
        if _opening(e, tris, after) < min(_opening(e, tris, before), moves.wedge_limit) - 1e-6:
            return False
    return True


def _projector_kernels(moves, n, mode, p):
    """d3v2p0 _SurfaceMoves.projector with the closest-point kernels."""
    from .numba_smoothing import closest_on_triangles, closest_on_segments
    ring = moves.topo.ring(n)
    planes = mode[3] if mode[0] == 'curve' else mode[1]
    if mode[0] == 'corner':
        def project(y):
            out = y.copy()
            for axis, side in planes:
                out[:, axis] = side * moves.extent[axis]
            return out
        return project
    if mode[0] == 'surface':
        x = np.ascontiguousarray(p[moves.f[ring]])

        def candidates(y):
            return closest_on_triangles(np.ascontiguousarray(y, dtype=float), x)
    else:
        a = np.array([p[mode[1]], p[mode[2]]])
        b = np.array([p[n], p[n]])

        def candidates(y):
            c = closest_on_segments(np.ascontiguousarray(y, dtype=float), a, b)
            return c, np.linalg.norm(c - y[:, None, :], axis=2)

    def project(y):
        c, d = candidates(y)
        out = c[np.arange(len(y)), np.argmin(d, axis=1)]
        for axis, side in planes:
            out[:, axis] = side * moves.extent[axis]
        return out
    return project


def _constraint_kernels(moves, n, mode, p):
    """d3v2p0 _SurfaceMoves.constraint with the triangle-angle and
    closest-point kernels."""
    from .numba_smoothing import triangle_min_angles, closest_on_triangles, closest_on_segments
    ring = moves.topo.ring(n)
    x0 = np.ascontiguousarray(p[moves.f[ring]])
    at = moves.f[ring] == n
    old = triangle_min_angles(x0)
    lower = np.where(old >= moves.triangle_target, moves.triangle_base[ring], old)
    origin = p[n].copy()

    def check(y):
        x = np.broadcast_to(x0, (len(y),) + x0.shape).copy()
        x[:, at] = y[:, None, :]
        ok = np.all(triangle_min_angles(x.reshape(-1, 3, 3)).reshape(len(y), -1) >= lower - 1e-9, axis=1)
        limit = moves.corner_max_deviation if mode[0] == 'corner' else moves.max_deviation
        for i in np.flatnonzero(ok):
            if mode[0] in ('surface', 'corner'):
                dev = closest_on_triangles(origin[None], np.ascontiguousarray(x[i]))[1].min()
            else:
                a = np.array([p[mode[1]], y[i]]); b = np.array([y[i], p[mode[2]]])
                c = closest_on_segments(origin[None], a, b)[0]
                dev = np.linalg.norm(c - origin, axis=1).min()
            ok[i] = dev <= limit
        return ok
    return check


def _facets_ok_scalar(moves, n, p):
    """d3v2p0 _SurfaceMoves.facets_ok, checking only the edges of the node's
    triangles (the only edges where a fold can involve them), with
    small_facet_angles' construction and degenerate-case rules."""
    if moves.minimum_facet_angle <= 0:
        return True
    f = moves.f
    ring = moves.topo.ring(n)
    edges = {(min(a, b), max(a, b)) for tri in f[ring].tolist()
             for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0]))}
    ring_set = set(ring.tolist())
    for e in edges:
        rows = moves.topo.edge_tri.get(e, [])
        if len(rows) < 2 or not any(r in ring_set for r in rows):
            continue
        a0, a1, a2 = p[e[0]].tolist()
        b0, b1, b2 = p[e[1]].tolist()
        x, y, z = b0 - a0, b1 - a1, b2 - a2
        length = math.sqrt(x * x + y * y + z * z)
        if length <= 0:
            continue                                        # small_facet_angles ignores zero-length edges
        x, y, z = x / length, y / length, z / length
        radial = []
        for r in rows:
            o = [v for v in f[r].tolist() if v != e[0] and v != e[1]][0]
            c0, c1, c2 = p[o].tolist()
            v0, v1, v2 = c0 - a0, c1 - a1, c2 - a2
            d = v0 * x + v1 * y + v2 * z
            v0, v1, v2 = v0 - d * x, v1 - d * y, v2 - d * z
            nrm = math.sqrt(v0 * v0 + v1 * v1 + v2 * v2)
            radial.append((r, nrm, (v0 / nrm, v1 / nrm, v2 / nrm) if nrm > 0 else (0., 0., 0.)))
        for i in range(len(radial)):
            for j in range(i + 1, len(radial)):
                ri, ni, ui = radial[i]
                rj, nj, uj = radial[j]
                if ni <= 0 or nj <= 0 or (ri not in ring_set and rj not in ring_set):
                    continue
                cx = ui[1] * uj[2] - ui[2] * uj[1]
                cy = ui[2] * uj[0] - ui[0] * uj[2]
                cz = ui[0] * uj[1] - ui[1] * uj[0]
                angle = math.degrees(math.atan2(math.sqrt(cx * cx + cy * cy + cz * cz),
                                                ui[0] * uj[0] + ui[1] * uj[1] + ui[2] * uj[2]))
                if angle < moves.minimum_facet_angle:
                    return False
    return True


def _block_task(task):
    """Worker: d3v2p0's smoothing loop restricted to the block's own nodes."""
    from ..d3v2p0.tet_angles import dihedral_angles_xyz
    from ..d3v2p0 import tet_smoothing as ref
    (p, t, own, surface, base, tri_base, params) = task
    target, max_angle = params['target'], params['max_angle']
    kernels = params.get('kernels') == 'numba'
    improve = _improve_node_kernels if kernels else ref._improve_node
    if kernels:
        from .backend import single_thread_numba
        single_thread_numba()
    span = 180. - max_angle

    def quality(a_min, a_max):
        return np.minimum(a_min / target, (180. - a_max) / span)

    a = dihedral_angles_xyz(p[t])
    mn, mx = a.min(axis=1), a.max(axis=1)
    moves = None
    if surface is not None:
        from ..d3v2p0.rve_caps import ClosedRVE
        rve = ClosedRVE(p[:surface['count']], surface['triangles'], surface['pairs'], surface['exterior'],
                        surface['rve_face'], dict(rve_dimensions=surface['extent']))
        moves = ref._SurfaceMoves(rve, p, surface['frozen'], params['max_deviation'], params['minimum_facet_angle'],
                                  params['triangle_target'], params['triangle_floor'],
                                  params['corner_max_deviation'], params['wedge_limit'])
        moves.triangle_base = tri_base
    order = np.argsort(t.ravel(), kind='stable')
    offsets = np.r_[0, np.cumsum(np.bincount(t.ravel(), minlength=len(p)))]
    star_of = lambda n: order[offsets[n]:offsets[n + 1]] // 4
    outside = (mn < target) | (mx > max_angle)
    nodes = np.flatnonzero(own)
    score = np.full(len(p), np.inf)
    np.minimum.at(score, t[outside].ravel(), np.repeat(quality(mn[outside], mx[outside]), 4))
    nodes = nodes[np.argsort(score[nodes], kind='stable')]
    moved = {}
    for n in nodes:
        r = star_of(n)
        for _ in range(params['node_iterations']):
            if not ((mn[r] < target) | (mx[r] > max_angle)).any():
                break
            if moves is not None and n < moves.count:
                mode = moves.mode(n)
                if mode is None:
                    break
                if kernels:
                    project = _projector_kernels(moves, n, mode, p)
                    check = _constraint_kernels(moves, n, mode, p)
                    final = lambda y, n=n: (_keeps_openings_scalar(moves, n, y, p)
                                            and ref._with_point(p, n, y, lambda m, q: _facets_ok_scalar(moves, m, q)))
                else:
                    project = moves.projector(n, mode, p)
                    check = moves.constraint(n, mode, p)
                    final = lambda y, n=n: (moves.keeps_openings(n, y, p) and ref._with_point(p, n, y, moves.facets_ok))
            else:
                project = check = final = None
            result = improve(n, r, p, t, mn[r], mx[r], target, base[r], max_angle, quality,
                                       params['max_halvings'], params['initial_step'], params['softness'],
                                       project, check, final, params['search_directions'])
            if result is None:
                break
            p[n], mn[r], mx[r] = result
            moved[int(n)] = p[n].copy()
    return moved


def _block_ids(points, size):
    """(block id, colour) per point; the colour is the parity of the block's
    three grid indices (0-7), so blocks of one colour are a block apart."""
    cells = np.floor(points / size).astype(np.int64)
    cells -= cells.min(axis=0)
    dims = cells.max(axis=0) + 1
    block = cells[:, 0] + dims[0] * (cells[:, 1] + dims[1] * cells[:, 2])
    colour = (cells[:, 0] % 2) + 2 * (cells[:, 1] % 2) + 4 * (cells[:, 2] % 2)
    return block, colour


def smooth_tet_dihedrals(points, tetrahedra, movable, *, enabled=True, target=30., max_angle=150.,
                         neighbour_floor=None, max_passes=10, node_iterations=3, max_halvings=8,
                         initial_step=.3, softness=20., surface=None, frozen_grain_ids=(),
                         max_deviation=None, minimum_facet_angle=0., triangle_target=30.,
                         triangle_floor=None, search_directions=True, progress=None,
                         corner_max_deviation=None, wedge_limit=None, n_workers=None, block_size=12.,
                         minimum_gain=0., backend='auto'):
    """Block-parallel version of d3v2p0.tet_smoothing.smooth_tet_dihedrals.

    Same arguments and rules, plus backend / n_workers (see d3v2p1.backend;
    None = automatic; the result does not depend on the worker count). Blocks
    run in worker processes; with numba available ('auto' or 'numba') each
    block evaluates trial positions with compiled kernels, which can change
    the last bit of some volumes and so, rarely, a borderline choice
    ('numpy' keeps d3v2p0's arithmetic). block_size
    (physical units; fixed so results do not depend on n_workers) and
    minimum_gain (stop when a pass reduces the number of tets outside the
    limits by less than this fraction; 0 runs until no move is accepted).
    Returns (points, report).
    """
    from ..d3v2p0 import tet_smoothing as ref
    from ..d3v2p0.tet_angles import dihedral_angles_xyz
    p = np.array(points, dtype=float)
    t = np.asarray(tetrahedra)
    movable = np.asarray(movable, dtype=bool)
    neighbour_floor = target if neighbour_floor is None else neighbour_floor
    triangle_floor = triangle_target if triangle_floor is None else triangle_floor
    if movable.shape != (len(p),):
        raise ValueError('movable must be a boolean mask over points')
    if not np.isfinite(block_size) or block_size <= 0:
        raise ValueError('block_size must be positive')
    if not 0 <= minimum_gain < 1:
        raise ValueError('minimum_gain must lie in [0, 1)')
    # process tier for the blocks; numba kernels inside each block when available
    kernels = 'numba' if plan(backend, n_workers, numba_kernels=True).numba else 'numpy'
    chosen = plan('parallel' if backend == 'numba' else backend, n_workers)
    # Validate every shared argument as d3v2p0 does and in its order, then set up once.
    # enabled=False, as the validation call this replaces passed it: this function has never validated `enabled`
    _check_arguments(p, t, movable, enabled=False, target=target, max_angle=max_angle,
                     neighbour_floor=neighbour_floor, max_passes=max_passes, node_iterations=node_iterations,
                     max_halvings=max_halvings, initial_step=initial_step, softness=softness,
                     max_deviation=max_deviation, minimum_facet_angle=minimum_facet_angle,
                     triangle_target=triangle_target, triangle_floor=triangle_floor,
                     search_directions=search_directions)
    moves = None
    if surface is not None:
        if corner_max_deviation is not None and (not np.isfinite(corner_max_deviation) or corner_max_deviation < 0):
            raise ValueError('corner_max_deviation must be finite and nonnegative')
        moves = ref._SurfaceMoves(surface, p, frozen_grain_ids, max_deviation, minimum_facet_angle,
                                  triangle_target, triangle_floor, corner_max_deviation, wedge_limit)
        if movable[:moves.count].any():
            raise ValueError('Surface nodes must not be in the movable mask; they move under surface rules')
    a = dihedral_angles_xyz(p[t])
    mn, mx = a.min(axis=1), a.max(axis=1)
    if np.any(ref._volumes(p[t]) <= 0):
        raise ValueError('All tetrahedra must have positive volume')
    before = ref._summary(mn, mx, target, max_angle)
    base = np.where(mn >= target, neighbour_floor, np.maximum(mn, neighbour_floor))
    candidate = movable.copy()
    tri_base_global = None
    surf = None
    if surface is not None:
        for n in range(moves.count):
            if moves.mode(n) is not None:
                candidate[n] = True
        max_deviation = moves.max_deviation
        tri_base_global = moves.triangle_base
        surf = dict(count=moves.count, f=np.asarray(surface.triangles), pairs=np.asarray(surface.grain_pairs),
                    exterior=np.asarray(surface.exterior, bool), rve_face=np.asarray(surface.rve_face),
                    extent=surface.report.get('rve_dimensions'), frozen=tuple(frozen_grain_ids))
        node_tris, tri_offsets = node_element_index(surf['f'], len(p))
    params = dict(target=target, max_angle=max_angle, node_iterations=node_iterations, max_halvings=max_halvings,
                  initial_step=initial_step, softness=softness, search_directions=search_directions,
                  max_deviation=max_deviation, minimum_facet_angle=minimum_facet_angle,
                  triangle_target=triangle_target, triangle_floor=triangle_floor,
                  corner_max_deviation=corner_max_deviation, wedge_limit=wedge_limit, kernels=kernels)
    order, offsets = node_element_index(t, len(p))
    history = []
    moved_nodes, moved_surface = set(), set()
    pool = WorkerPool(chosen)
    try:
        for pass_number in range(max_passes if enabled else 0):
            outside_before = int(np.sum((mn < target) | (mx > max_angle)))
            if not outside_before:
                break
            accepted = 0
            skipped = 0
            # block membership from the positions at the start of the pass: every
            # node belongs to exactly one block, and is worked on in that block's phase
            block, colour = _block_ids(p, block_size)
            for phase in range(8):
                outside = (mn < target) | (mx > max_angle)
                nodes = np.unique(t[outside])
                nodes = nodes[candidate[nodes] & (colour[nodes] == phase)]
                if not len(nodes):
                    continue
                groups = []
                for b in np.unique(block[nodes]):
                    own_nodes = nodes[block[nodes] == b]
                    star = np.unique(np.concatenate([order[offsets[n]:offsets[n + 1]] for n in own_nodes]))
                    local_nodes = np.unique(t[star])
                    tri_rows = None
                    if surf is not None:
                        touching = local_nodes[local_nodes < surf['count']]
                        if len(touching):
                            tri_rows = np.unique(np.concatenate(
                                [node_tris[tri_offsets[n]:tri_offsets[n + 1]] for n in touching]))
                            local_nodes = np.union1d(local_nodes, surf['f'][tri_rows].ravel())
                    groups.append([b, own_nodes, local_nodes])
                # Blocks of one colour run together. A block's own node may not be
                # read by another block of this phase (only if a star spans a block).
                readers = np.zeros(len(p), np.int64)
                for _, _, local_nodes in groups:
                    readers[local_nodes] += 1
                for g in groups:
                    keep = readers[g[1]] == 1
                    skipped += int(np.sum(~keep))
                    g[1] = g[1][keep]
                tasks, maps = [], []
                lookup = np.full(len(p), -1, dtype=np.int64)
                for b, own_nodes, _ in groups:
                    if not len(own_nodes):
                        continue
                    star = np.unique(np.concatenate([order[offsets[n]:offsets[n + 1]] for n in own_nodes]))
                    local_nodes = np.unique(t[star])
                    tri_rows = None
                    if surf is not None:
                        touching = local_nodes[local_nodes < surf['count']]
                        if len(touching):
                            tri_rows = np.unique(np.concatenate(
                                [node_tris[tri_offsets[n]:tri_offsets[n + 1]] for n in touching]))
                            local_nodes = np.union1d(local_nodes, surf['f'][tri_rows].ravel())
                    # surface nodes first, as d3v2p0's surface moves require
                    surface_nodes = local_nodes[local_nodes < (surf['count'] if surf else 0)]
                    other_nodes = local_nodes[local_nodes >= (surf['count'] if surf else 0)]
                    glob = np.concatenate((surface_nodes, other_nodes))
                    lookup[glob] = np.arange(len(glob))
                    lt = lookup[t[star]]
                    own = np.zeros(len(glob), bool)
                    own[lookup[own_nodes]] = True
                    sub = None
                    tri_base = None
                    if surf is not None and tri_rows is not None and len(surface_nodes):
                        lf = lookup[surf['f'][tri_rows]]
                        sub = dict(count=len(surface_nodes), triangles=lf, pairs=surf['pairs'][tri_rows],
                                   exterior=surf['exterior'][tri_rows], rve_face=surf['rve_face'][tri_rows],
                                   extent=surf['extent'], frozen=surf['frozen'])
                        tri_base = tri_base_global[tri_rows]
                    elif surf is not None:
                        own[:len(surface_nodes)] = False
                    if sub is None:
                        own[:len(surface_nodes)] = False     # without its triangles a surface node cannot move
                    tasks.append((p[glob].copy(), lt, own, sub, base[star], tri_base, params))
                    maps.append(glob)
                results = pool.map(_block_task, tasks)
                changed = []
                for glob, moved in zip(maps, results):
                    for i, y in moved.items():
                        g = int(glob[i])
                        p[g] = y
                        changed.append(g)
                        moved_nodes.add(g)
                        if surf is not None and g < surf['count']:
                            moved_surface.add(g)
                accepted += len(changed)
                if changed:
                    rows = np.unique(np.concatenate([order[offsets[g]:offsets[g + 1]] for g in changed]))
                    a = dihedral_angles_xyz(p[t[rows]])
                    mn[rows], mx[rows] = a.min(axis=1), a.max(axis=1)
            outside_after = int(np.sum((mn < target) | (mx > max_angle)))
            entry = dict(pass_number=pass_number + 1, outside_limits=outside_before, accepted_moves=accepted,
                         below_target=int(np.sum(mn < target)), minimum_dihedral=float(mn.min()),
                         skipped_shared_nodes=skipped)
            history.append(entry)
            if progress is not None:
                progress(entry)
            if not accepted or (outside_before - outside_after) < minimum_gain * outside_before:
                break
    finally:
        pool.close()
    report = dict(enabled=bool(enabled), target_min_dihedral=float(target), max_dihedral=float(max_angle),
                  neighbour_floor=float(neighbour_floor), movable_nodes=int(candidate.sum()),
                  moved_nodes=len(moved_nodes), moved_surface_nodes=len(moved_surface),
                  surface_nodes_movable=surface is not None,
                  maximum_node_move=float(np.linalg.norm(p - np.asarray(points, float), axis=1).max()) if len(p) else 0.,
                  before=before, after=ref._summary(mn, mx, target, max_angle), history=history,
                  n_workers=int(chosen.workers), block_size=float(block_size), parallel_by='coloured blocks',
                  backend=dict(chosen.report(), kernels=kernels))
    if surface is not None:
        report.update(triangle_target=float(triangle_target), triangle_floor=float(triangle_floor),
                      max_deviation=float(max_deviation), minimum_facet_angle=float(minimum_facet_angle),
                      corner_max_deviation=corner_max_deviation, wedge_limit=wedge_limit)
    return p, report


def _verify_smoothed(moved):
    """d3v2p0.tet_smoothing.smooth_grain_tetrahedra's verification of ``moved`` (the smoothed points with the
    report of the unsmoothed mesh, whose per-grain volumes are the volumes the moved surface encloses).

    The checks, their order, their messages and the report (its keys in the same order) are d3v2p0's. What is
    left out is the zero-pass smoothing call d3v2p0 makes first, which repeats the setup of the smoothing that
    has just run; its one visible effect, the positive-volume check, is made here first, as it is there.
    The per-grain sums run over grain-sorted tet numbers, the same tets in the same order as d3v2p0's masks.
    """
    from ..d3v2p0 import tet_smoothing as ref
    from ..d3v2p0.tet_angles import dihedral_summary
    points = moved.points
    volumes = ref._volumes(points[moved.tetrahedra])
    if np.any(volumes <= 0):
        raise ValueError('All tetrahedra must have positive volume')
    if np.any(~np.isfinite(volumes)) or np.any(volumes <= 0):
        raise RuntimeError('Smoothing produced a nonpositive tetrahedron volume')
    if not np.isclose(volumes.sum(), moved.report['total_volume'], rtol=1e-8):
        raise RuntimeError('Tetrahedron volumes no longer fill the RVE')
    quality = ref.minimum_sicn_gmsh(points, moved.tetrahedra)
    if np.any(~np.isfinite(quality)) or np.any(quality <= 0):
        raise RuntimeError('Invalid tetrahedron quality after smoothing')
    report = dict(moved.report)
    report['dihedral_smoothing'] = None             # holds the key's place; the caller sets the report
    per_grain = {}
    by_grain = np.argsort(moved.grain_ids, kind='stable')
    sorted_grain = moved.grain_ids[by_grain]
    for gid, entry in moved.report['per_grain'].items():
        first = np.searchsorted(sorted_grain, int(gid), side='left')
        last = np.searchsorted(sorted_grain, int(gid), side='right')
        selected = by_grain[first:last]
        volume = float(volumes[selected].sum())
        if not np.isclose(volume, entry['volume'], rtol=1e-8):
            raise RuntimeError(f'Volume of grain {gid} does not match its surface after smoothing')
        per_grain[gid] = dict(entry, volume=volume, minimum_quality=float(quality[selected].min()))
    dihedral = dihedral_summary(points, moved.tetrahedra)
    passed = bool(quality.min() >= report['quality_threshold'])
    if report.get('minimum_dihedral_threshold') is not None:
        passed = passed and dihedral['minimum'] >= report['minimum_dihedral_threshold']
    report.update(status='TET_MESH_VERIFIED' if passed else 'TET_QUALITY_BELOW_TARGET', ready=passed,
                  per_grain=per_grain, total_volume=float(volumes.sum()),
                  minimum_quality=float(quality.min()),
                  quality_percentiles_1_5_50=np.percentile(quality, [1, 5, 50]).tolist(),
                  below_quality_threshold=int(np.sum(quality < report['quality_threshold'])),
                  dihedral_angles=dihedral)
    return replace(moved, quality=quality, report=report)


def smooth_grain_tetrahedra(tets, surface_node_count, *, n_workers=None, block_size=12., minimum_gain=0.,
                            backend='auto', **kwargs):
    """Block-parallel version of d3v2p0.tet_smoothing.smooth_grain_tetrahedra.

    Smoothing runs in d3v2p1 blocks; verification (positive volumes, RVE and
    grain volumes against the moved surface, moved-surface intersections, Gmsh
    minSICN, dihedral statistics, readiness) is d3v2p0's (see _verify_smoothed), run on the result.
    """
    from ..d3v2p0 import tet_smoothing as ref
    surface = kwargs.get('surface')
    if isinstance(surface_node_count, bool) or not isinstance(surface_node_count, (int, np.integer)) \
            or not 0 <= surface_node_count <= len(tets.points):
        raise ValueError('surface_node_count must be an integer within the node count')
    if surface is not None and len(surface.points) != surface_node_count:
        raise ValueError('surface_node_count must equal the number of surface points')
    movable = np.arange(len(tets.points)) >= surface_node_count
    points, smoothing = smooth_tet_dihedrals(tets.points, tets.tetrahedra, movable, n_workers=n_workers,
                                             block_size=block_size, minimum_gain=minimum_gain, backend=backend,
                                             **kwargs)
    if not kwargs.get('enabled', True):
        report = dict(tets.report)
        report['dihedral_smoothing'] = smoothing
        return replace(tets, report=report)
    # d3v2p0's verification on the new points: run its wrapper with zero passes on a
    # GrainTetrahedra carrying the new points but the old report (volumes reference).
    reference = dict(tets.report)
    if surface is not None:
        # grain volumes must equal what the moved surface encloses
        enclosed = ref._enclosed_volumes(points, surface)
        reference['per_grain'] = {gid: dict(entry, volume=enclosed[int(gid)])
                                  for gid, entry in tets.report['per_grain'].items()}
    moved = replace(tets, points=points, report=reference)
    out = _verify_smoothed(moved)
    report = dict(out.report)
    if surface is not None:
        from .surface_intersections import find_surface_intersections
        changed = np.flatnonzero(np.any(np.any(points[:surface_node_count] != tets.points[:surface_node_count],
                                               axis=1)[surface.triangles], axis=1))
        if len(changed) and len(find_surface_intersections(points[:surface_node_count], surface.triangles,
                                                           triangle_ids=changed, n_workers=n_workers,
                                                           backend=backend)):
            raise RuntimeError('Moved surface triangles intersect')
        smoothing['checked_surface_triangles'] = int(len(changed))
    report['dihedral_smoothing'] = smoothing
    return replace(out, report=report)
