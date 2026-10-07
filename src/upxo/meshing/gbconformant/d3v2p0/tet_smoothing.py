"""Raise tetrahedral dihedral angles by moving mesh nodes.

Phase 1 moves nodes inside grains. Phase 2 (optional, ``surface`` given) also
moves grain-surface nodes along their own surface: patch-interior nodes on
their surrounding triangles, junction-line and RVE-trace nodes along their
line; RVE-plane coordinates stay exact; curve corners and nodes of frozen
grains stay fixed. A node's move changes only the tets that contain it (its
star) and, for a surface node, the surface triangles around it.

Acceptance rule for a move, on every tet of the star:
- the worst normalised angle quality of the star rises (smallest dihedral
  relative to ``target``, largest dihedral relative to ``max_angle``);
- a tet whose smallest dihedral is at or above ``target`` may lose angle but
  stays at or above its floor; a tet below ``target`` loses none. The floor is
  ``neighbour_floor`` (default: ``target``) for a tet that started at or above
  ``target``, and the larger of its starting angle and ``neighbour_floor`` for
  one that started below, so no tet ends below min(start, target) or below
  ``neighbour_floor`` if it started above it;
- a tet whose largest dihedral is at or below ``max_angle`` stays at or below
  it; a tet above ``max_angle`` gains none;
- every tet keeps a positive volume.
With corner_max_deviation, curve corners (junction points where three or
more junction lines meet) also move, in any direction but within that
deviation of the previous surface; RVE corners stay fixed.
With wedge_limit, a surface node's move must also keep the no-sharpening rule
at every surface edge around it (facet_angles.sharpens).
For a surface node, additionally: each surrounding surface triangle obeys the
same rule on its smallest angle (``triangle_target``, ``triangle_floor``); the
old node position stays within ``max_deviation`` of the new surface; facet
openings around the node stay at or above ``minimum_facet_angle``.
Trial moves follow the gradient of a soft minimum of the star quality at
halving step lengths and, with search_directions, 26 fixed directions at the
first four step lengths; for surface nodes they are projected onto the surface
or line. The passing move with the best star quality is kept.
"""
from dataclasses import replace
import uuid
import numpy as np
from .tet_angles import dihedral_angles_xyz, dihedral_summary


def _volumes(x):
    return np.einsum('...i,...i->...', x[..., 1, :] - x[..., 0, :],
                     np.cross(x[..., 2, :] - x[..., 0, :], x[..., 3, :] - x[..., 0, :])) / 6


def _triangle_min_angles(x):
    """Smallest angle (degrees) of triangles x (..., 3, 3)."""
    out = None
    for k in range(3):
        u, v = x[..., (k + 1) % 3, :] - x[..., k, :], x[..., (k + 2) % 3, :] - x[..., k, :]
        nu, nv = np.linalg.norm(u, axis=-1), np.linalg.norm(v, axis=-1)
        c = np.divide(np.sum(u * v, axis=-1), nu * nv, out=np.ones(u.shape[:-1]), where=(nu * nv) > 0)
        a = np.degrees(np.arccos(np.clip(c, -1, 1)))
        out = a if out is None else np.minimum(out, a)
    return out


def _closest_on_segments_batch(y, a, b):
    """Closest points from y (B, 3) to segments a-b (m, 3); shape (B, m, 3)."""
    d = b - a
    l2 = np.einsum('ij,ij->i', d, d)
    t = np.divide(np.einsum('bmi,mi->bm', y[:, None, :] - a[None], d), l2[None],
                  out=np.zeros((len(y), len(a))), where=l2[None] > 0)
    return a[None] + np.clip(t, 0, 1)[..., None] * d[None]


def _closest_on_triangles_batch(y, x):
    """Closest points from y (B, 3) to triangles x (m, 3, 3); shape (B, m, 3)."""
    n = np.cross(x[:, 1] - x[:, 0], x[:, 2] - x[:, 0])
    nn = np.einsum('ij,ij->i', n, n)
    dist = np.divide(np.einsum('bmi,mi->bm', y[:, None, :] - x[None, :, 0], n), nn[None],
                     out=np.zeros((len(y), len(x))), where=nn[None] > 0)
    proj = y[:, None, :] - dist[..., None] * n[None]
    inside = np.broadcast_to(nn[None] > 0, dist.shape).copy()
    for k in range(3):
        e = x[:, (k + 1) % 3] - x[:, k]
        inside &= np.einsum('bmi,mi->bm', np.cross(e[None], proj - x[None, :, k]), n) >= 0
    best = proj
    best_d = np.where(inside, np.linalg.norm(proj - y[:, None, :], axis=2), np.inf)
    for k in range(3):
        c = _closest_on_segments_batch(y, x[:, k], x[:, (k + 1) % 3])
        dc = np.linalg.norm(c - y[:, None, :], axis=2)
        better = ~inside & (dc < best_d)
        best = np.where(better[..., None], c, best)
        best_d = np.where(better, dc, best_d)
    return best, best_d


def _check_int(name, value, low):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < low:
        raise ValueError(f'{name} must be an integer >= {low}')


class _SurfaceMoves:
    """Classification and constraints of grain-surface nodes (Phase 2)."""

    def __init__(self, surface, points, frozen_grain_ids, max_deviation, minimum_facet_angle,
                 triangle_target, triangle_floor, corner_max_deviation=None, wedge_limit=None):
        from .surface_angles import _Topology
        n = len(surface.points)
        if not np.array_equal(np.asarray(points)[:n], surface.points):
            raise ValueError('The first surface-node-count points must be the surface points')
        self.f = np.asarray(surface.triangles)
        pairs = np.asarray(surface.grain_pairs)
        exterior = np.asarray(surface.exterior, dtype=bool)
        rve_face = np.asarray(surface.rve_face)
        extent = surface.report.get('rve_dimensions')
        self.extent = None if extent is None else np.asarray(extent, dtype=float)
        frozen = np.isin(pairs, np.asarray(list(frozen_grain_ids), dtype=pairs.dtype)).any(axis=1) \
            if len(frozen_grain_ids) else np.zeros(len(self.f), bool)
        _, patch = np.unique(np.column_stack((pairs, rve_face)), axis=0, return_inverse=True)
        self.topo = _Topology(surface.points, self.f, np.ones(len(self.f), bool), patch.ravel(),
                              frozen, exterior, rve_face)
        self.count = n
        edges = np.linalg.norm(surface.points[self.f[:, 1]] - surface.points[self.f[:, 0]], axis=1)
        self.max_deviation = .1 * float(np.median(edges)) if max_deviation is None else float(max_deviation)
        # None keeps curve corners (junction points) fixed
        self.corner_max_deviation = None if corner_max_deviation is None else float(corner_max_deviation)
        self.wedge_limit = None if wedge_limit is None else float(wedge_limit)
        self.minimum_facet_angle = float(minimum_facet_angle)
        self.triangle_target, self.triangle_floor = float(triangle_target), float(triangle_floor)
        start = _triangle_min_angles(surface.points[self.f])
        self.triangle_base = np.where(start >= triangle_target, triangle_floor, np.maximum(start, triangle_floor))

    def mode(self, n):
        """None (fixed), ('surface', planes), ('curve', n1, n2, planes) or ('corner', planes)."""
        if n >= self.count:
            return None
        kind = self.topo.kind(n)
        planes = self.topo.planes(n)
        if planes and self.extent is None:
            return None
        if kind is None:
            # A curve corner (junction point) may move freely within its own
            # deviation limit, keeping its RVE planes; RVE corners stay fixed.
            if self.corner_max_deviation is None or self.topo.fixed[n] or len(planes) >= 3:
                return None
            return ('corner', planes)
        if kind == 'interior':
            return ('surface', planes)
        return ('curve', kind[1], kind[2], planes)

    def projector(self, n, mode, p):
        """Return project(y_batch) -> y_batch onto the node's current surface or line."""
        ring = self.topo.ring(n)
        planes = mode[3] if mode[0] == 'curve' else mode[1]
        if mode[0] == 'corner':

            def project(y):
                out = y.copy()
                for axis, side in planes:
                    out[:, axis] = side * self.extent[axis]
                return out
            return project
        if mode[0] == 'surface':
            x = p[self.f[ring]]

            def candidates(y):
                c, d = _closest_on_triangles_batch(y, x)
                return c, d
        else:
            a = np.array([p[mode[1]], p[mode[2]]])
            b = np.array([p[n], p[n]])

            def candidates(y):
                c = _closest_on_segments_batch(y, a, b)
                return c, np.linalg.norm(c - y[:, None, :], axis=2)

        def project(y):
            c, d = candidates(y)
            out = c[np.arange(len(y)), np.argmin(d, axis=1)]
            for axis, side in planes:
                out[:, axis] = side * self.extent[axis]
            return out
        return project

    def constraint(self, n, mode, p):
        """Return check(y_batch) -> bool mask for triangle angles and deviation."""
        from .surface_angles import _closest_on_triangles, _closest_on_segments
        ring = self.topo.ring(n)
        x0 = p[self.f[ring]]
        at = self.f[ring] == n
        old = _triangle_min_angles(x0)
        lower = np.where(old >= self.triangle_target, self.triangle_base[ring], old)
        origin = p[n].copy()

        def check(y):
            x = np.broadcast_to(x0, (len(y),) + x0.shape).copy()
            x[:, at] = y[:, None, :]
            ok = np.all(_triangle_min_angles(x) >= lower - 1e-9, axis=1)
            limit = self.corner_max_deviation if mode[0] == 'corner' else self.max_deviation
            for i in np.flatnonzero(ok):
                if mode[0] in ('surface', 'corner'):
                    dev = np.linalg.norm(_closest_on_triangles(origin, x[i]) - origin, axis=1).min()
                else:
                    a = np.array([p[mode[1]], y[i]]); b = np.array([y[i], p[mode[2]]])
                    dev = np.linalg.norm(_closest_on_segments(origin, a, b) - origin, axis=1).min()
                ok[i] = dev <= limit
            return ok
        return check

    def keeps_openings(self, n, y, p):
        """No-sharpening rule at the edges around the node when it moves to y."""
        if self.wedge_limit is None:
            return True
        from .facet_angles import edge_openings, sharpens
        ring = self.topo.ring(n)
        edges = sorted({(min(a, b), max(a, b)) for t in self.f[ring]
                        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0]))})
        before = edge_openings(p, self.f, edges, self.topo.edge_tri)
        old = p[n].copy()
        p[n] = y
        try:
            after = edge_openings(p, self.f, edges, self.topo.edge_tri)
        finally:
            p[n] = old
        return not np.any(sharpens(before, after, self.wedge_limit))

    def facets_ok(self, n, p):
        """Facet openings at edges around the node stay at or above the minimum."""
        if self.minimum_facet_angle <= 0:
            return True
        from .facet_angles import small_facet_angles
        ring = self.topo.ring(n)
        nodes = np.unique(self.f[ring])
        local = np.unique(np.concatenate([self.topo.ring(m) for m in nodes]))
        hits, _ = small_facet_angles(p, self.f[local], self.minimum_facet_angle)
        if not len(hits):
            return True
        return not np.isin(local[np.unique(hits)], ring).any()


def smooth_tet_dihedrals(points, tetrahedra, movable, *, enabled=True, target=30., max_angle=150.,
                         neighbour_floor=None, max_passes=10, node_iterations=3, max_halvings=8,
                         initial_step=.3, softness=20., surface=None, frozen_grain_ids=(),
                         max_deviation=None, minimum_facet_angle=0., triangle_target=30.,
                         triangle_floor=None, search_directions=True, progress=None,
                         corner_max_deviation=None, wedge_limit=None):
    """Return (points, report) with nodes relocated under the rule above.

    movable: boolean mask of free nodes inside grains (Phase 1). With surface
    (a ClosedRVE whose points are the first len(surface.points) mesh points),
    grain-surface nodes also move (Phase 2). All tets must have positive
    volume. initial_step is the first trial step as a fraction of the mean
    distance from the node to its star neighbours. neighbour_floor and
    triangle_floor default to target and triangle_target (no relaxation).
    """
    p = np.array(points, dtype=float)
    t = np.asarray(tetrahedra)
    movable = np.asarray(movable, dtype=bool)
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
    moves = None
    if surface is not None:
        if corner_max_deviation is not None and (not np.isfinite(corner_max_deviation) or corner_max_deviation < 0):
            raise ValueError('corner_max_deviation must be finite and nonnegative')
        moves = _SurfaceMoves(surface, p, frozen_grain_ids, max_deviation, minimum_facet_angle,
                              triangle_target, triangle_floor, corner_max_deviation, wedge_limit)
        if movable[:moves.count].any():
            raise ValueError('Surface nodes must not be in the movable mask; they move under surface rules')

    angles = dihedral_angles_xyz(p[t])
    mn, mx = angles.min(axis=1), angles.max(axis=1)
    if np.any(_volumes(p[t]) <= 0):
        raise ValueError('All tetrahedra must have positive volume')
    before = _summary(mn, mx, target, max_angle)
    # Floors are anchored to the starting angles, so a tet that starts below
    # the target can never end below its start, even if a move first lifts it.
    base = np.where(mn >= target, neighbour_floor, np.maximum(mn, neighbour_floor))
    history = []
    order = np.argsort(t.ravel(), kind='stable')
    offsets = np.r_[0, np.cumsum(np.bincount(t.ravel(), minlength=len(p)))]
    star_of = lambda n: order[offsets[n]:offsets[n + 1]] // 4
    span = 180. - max_angle

    def quality(a_min, a_max):
        return np.minimum(a_min / target, (180. - a_max) / span)

    candidate_mask = movable.copy()
    if moves is not None:
        surface_mode = {}
        for n in range(moves.count):
            m = moves.mode(n)
            if m is not None:
                surface_mode[n] = m
                candidate_mask[n] = True
    moved_nodes, moved_surface = set(), set()
    for pass_number in range(max_passes if enabled else 0):
        outside = (mn < target) | (mx > max_angle)
        if not outside.any():
            break
        nodes = np.unique(t[outside])
        nodes = nodes[candidate_mask[nodes]]
        node_score = np.full(len(p), np.inf)
        np.minimum.at(node_score, t[outside].ravel(), np.repeat(quality(mn[outside], mx[outside]), 4))
        nodes = nodes[np.argsort(node_score[nodes], kind='stable')]
        accepted = attempted = 0
        for n in nodes:
            r = star_of(n)
            for _ in range(node_iterations):
                if not ((mn[r] < target) | (mx[r] > max_angle)).any():
                    break
                attempted += 1
                if moves is not None and n < moves.count:
                    mode = surface_mode[n]
                    project = moves.projector(n, mode, p)
                    check = moves.constraint(n, mode, p)
                    final = lambda y, n=n: (moves.keeps_openings(n, y, p) and _with_point(p, n, y, moves.facets_ok))
                else:
                    project = check = final = None
                result = _improve_node(n, r, p, t, mn[r], mx[r], target, base[r], max_angle,
                                       quality, max_halvings, initial_step, softness, project, check, final,
                                       search_directions)
                if result is None:
                    break
                p[n], mn[r], mx[r] = result
                accepted += 1
                moved_nodes.add(int(n))
                if moves is not None and n < moves.count:
                    moved_surface.add(int(n))
        history.append(dict(pass_number=pass_number + 1, outside_limits=int(outside.sum()),
                            candidate_nodes=int(len(nodes)), attempts=attempted, accepted_moves=accepted))
        if progress is not None:
            progress(dict(history[-1], below_target=int(np.sum(mn < target)), below_10=int(np.sum(mn < 10)),
                          below_20=int(np.sum(mn < 20)), minimum_dihedral=float(mn.min())))
        if not accepted:
            break
    report = dict(enabled=bool(enabled), target_min_dihedral=float(target), max_dihedral=float(max_angle),
                  search_directions=bool(search_directions),
                  neighbour_floor=float(neighbour_floor), movable_nodes=int(candidate_mask.sum()),
                  moved_nodes=len(moved_nodes), moved_surface_nodes=len(moved_surface),
                  surface_nodes_movable=moves is not None,
                  maximum_node_move=float(np.linalg.norm(p - np.asarray(points, float), axis=1).max()) if len(p) else 0.,
                  before=before, after=_summary(mn, mx, target, max_angle), history=history)
    if moves is not None:
        report.update(triangle_target=float(triangle_target), triangle_floor=float(triangle_floor),
                      max_deviation=moves.max_deviation, minimum_facet_angle=float(minimum_facet_angle),
                      corner_max_deviation=moves.corner_max_deviation, wedge_limit=moves.wedge_limit)
    return p, report


def _with_point(p, n, y, test):
    old = p[n].copy()
    p[n] = y
    try:
        return test(n, p)
    finally:
        p[n] = old


def _summary(mn, mx, target, max_angle):
    return dict(minimum_dihedral=float(mn.min()), maximum_dihedral=float(mx.max()),
                below_target=int(np.sum(mn < target)), above_max_angle=int(np.sum(mx > max_angle)),
                below_10=int(np.sum(mn < 10)), below_20=int(np.sum(mn < 20)))


_DIRECTIONS = np.array([d for d in np.array(np.meshgrid(*[[-1, 0, 1]] * 3)).reshape(3, -1).T if np.any(d)], float)
_DIRECTIONS /= np.linalg.norm(_DIRECTIONS, axis=1)[:, None]          # 26 lattice directions


def _improve_node(n, r, p, t, mn0, mx0, target, floor, max_angle, quality, max_halvings, initial_step,
                  softness, project=None, check=None, final=None, search_directions=True):
    x0 = p[t[r]]
    at = t[r] == n                                   # (k, 4) positions of the node in its star
    origin = p[n].copy()

    def evaluate(y):
        x = np.broadcast_to(x0, (len(y),) + x0.shape).copy()
        x[:, at] = y[:, None, :]
        a = dihedral_angles_xyz(x)
        return a.min(axis=2), a.max(axis=2), _volumes(x)

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
        # Fixed extra directions at the first few step lengths: the gradient
        # of the soft minimum can point into a move the rule rejects.
        extra = (origin + (steps[:4, None, None] * _DIRECTIONS[None]).reshape(-1, 3))
        trial = np.vstack((trial, extra))
    if project is not None:
        trial = project(trial)
        moved = np.linalg.norm(trial - origin, axis=1) > 1e-12 * h
        if not moved.any():
            return None
        trial = trial[moved]
    a_min, a_max, vol = evaluate(trial)
    lower = np.where(mn0 >= target, floor, mn0)          # floor: per-tet anchored floor
    q0 = quality(mn0, mx0).min()
    ok = (np.all(vol > 0, axis=1)
          & np.all(a_min >= lower - 1e-9, axis=1)
          & np.all(a_max <= np.maximum(mx0, max_angle) + 1e-9, axis=1)
          & (quality(a_min, a_max).min(axis=1) > q0 + 1e-9))
    if not ok.any():
        return None
    # Best tet quality first; surface constraints are checked lazily.
    candidates = np.flatnonzero(ok)
    candidates = candidates[np.argsort(-quality(a_min[candidates], a_max[candidates]).min(axis=1), kind='stable')]
    for best in candidates:
        if check is not None and not check(trial[best][None])[0]:
            continue
        if final is None or final(trial[best]):
            return trial[best], a_min[best], a_max[best]
    return None


def minimum_sicn_gmsh(points, tetrahedra):
    """Gmsh minSICN of linear tetrahedra (same metric as the tet mesher)."""
    from upxo._sup.optional_imports import import_gmsh
    gmsh = import_gmsh()
    points = np.asarray(points, dtype=float)
    tetrahedra = np.asarray(tetrahedra)
    owned = not gmsh.isInitialized()
    if owned:
        gmsh.initialize()
    previous_model = gmsh.model.getCurrent()
    terminal = gmsh.option.getNumber('General.Terminal')
    name = 'tet_quality_' + uuid.uuid4().hex
    gmsh.model.add(name)
    try:
        gmsh.option.setNumber('General.Terminal', 0)
        tag = gmsh.model.addDiscreteEntity(3)
        gmsh.model.mesh.addNodes(3, tag, np.arange(1, len(points) + 1), points.ravel())
        ids = np.arange(1, len(tetrahedra) + 1)
        gmsh.model.mesh.addElementsByType(tag, 4, ids, (tetrahedra + 1).ravel())
        return np.asarray(gmsh.model.mesh.getElementQualities(ids, 'minSICN'))
    finally:
        gmsh.model.setCurrent(name)
        gmsh.model.remove()
        gmsh.option.setNumber('General.Terminal', terminal)
        if owned:
            gmsh.finalize()
        elif previous_model in gmsh.model.list():
            gmsh.model.setCurrent(previous_model)


def _enclosed_volumes(points, surface):
    """Volume enclosed by each grain's surface triangles (signed as in gmsh_closed)."""
    f, pairs, exterior = surface.triangles, np.asarray(surface.grain_pairs), np.asarray(surface.exterior, bool)
    x = points[f]
    v = np.einsum('ij,ij->i', x[:, 0], np.cross(x[:, 1], x[:, 2])) / 6
    out = {}
    for gid in np.unique(pairs[:, 0].tolist() + pairs[~exterior, 1].tolist()):
        sel0 = pairs[:, 0] == gid
        sel1 = ~exterior & (pairs[:, 1] == gid)
        out[int(gid)] = float(v[sel0].sum() - v[sel1].sum())
    return out


def smooth_grain_tetrahedra(tets, surface_node_count, *, enabled=True, target=30., max_angle=150.,
                            neighbour_floor=None, max_passes=10, node_iterations=3, max_halvings=8,
                            initial_step=.3, softness=20., surface=None, frozen_grain_ids=(),
                            max_deviation=None, minimum_facet_angle=0., triangle_target=30.,
                            triangle_floor=None, search_directions=True, progress=None,
                            corner_max_deviation=None, wedge_limit=None):
    """Smooth a verified GrainTetrahedra result and re-verify it.

    Nodes 0..surface_node_count-1 are the grain-surface nodes of the mesher
    output. Without surface they stay fixed (Phase 1). With surface (the
    ClosedRVE the tets were generated from) they move under the surface rules
    (Phase 2); use surface_after_smoothing to obtain the moved surface.
    Quality (Gmsh minSICN), per-grain minima and volumes, dihedral statistics
    and readiness are recomputed. Without surface moves grain volumes must be
    unchanged; with them each grain's tet volume must equal the volume its
    moved surface encloses, the total must equal the RVE volume, and moved
    surface triangles must not intersect. Readiness keeps the input thresholds.
    """
    if isinstance(surface_node_count, bool) or not isinstance(surface_node_count, (int, np.integer)) \
            or not 0 <= surface_node_count <= len(tets.points):
        raise ValueError('surface_node_count must be an integer within the node count')
    if surface is not None and len(surface.points) != surface_node_count:
        raise ValueError('surface_node_count must equal the number of surface points')
    movable = np.arange(len(tets.points)) >= surface_node_count
    points, smoothing = smooth_tet_dihedrals(
        tets.points, tets.tetrahedra, movable, enabled=enabled, target=target, max_angle=max_angle,
        neighbour_floor=neighbour_floor, max_passes=max_passes, node_iterations=node_iterations,
        max_halvings=max_halvings, initial_step=initial_step, softness=softness, surface=surface,
        frozen_grain_ids=frozen_grain_ids, max_deviation=max_deviation,
        minimum_facet_angle=minimum_facet_angle, triangle_target=triangle_target,
        triangle_floor=triangle_floor, search_directions=search_directions, progress=progress,
        corner_max_deviation=corner_max_deviation, wedge_limit=wedge_limit)
    report = dict(tets.report)
    report['dihedral_smoothing'] = smoothing
    if not enabled:
        return replace(tets, report=report)
    surface_moved = not np.array_equal(points[~movable], tets.points[~movable])
    if surface_moved and surface is None:
        raise RuntimeError('Grain-surface nodes moved')
    x = points[tets.tetrahedra]
    volumes = _volumes(x)
    if np.any(~np.isfinite(volumes)) or np.any(volumes <= 0):
        raise RuntimeError('Smoothing produced a nonpositive tetrahedron volume')
    if not np.isclose(volumes.sum(), tets.report['total_volume'], rtol=1e-8):
        raise RuntimeError('Tetrahedron volumes no longer fill the RVE')
    enclosed = _enclosed_volumes(points, surface) if surface_moved else None
    if surface_moved:
        from .surface_intersections import find_surface_intersections
        changed = np.flatnonzero(np.any(np.any(points[:surface_node_count] != tets.points[:surface_node_count],
                                               axis=1)[surface.triangles], axis=1))
        if len(find_surface_intersections(points[:surface_node_count], surface.triangles, triangle_ids=changed)):
            raise RuntimeError('Moved surface triangles intersect')
        smoothing['checked_surface_triangles'] = int(len(changed))
    quality = minimum_sicn_gmsh(points, tets.tetrahedra)
    if np.any(~np.isfinite(quality)) or np.any(quality <= 0):
        raise RuntimeError('Invalid tetrahedron quality after smoothing')
    per_grain = {}
    for gid, entry in tets.report['per_grain'].items():
        selected = tets.grain_ids == int(gid)
        volume = float(volumes[selected].sum())
        expected = enclosed[int(gid)] if surface_moved else entry['volume']
        if not np.isclose(volume, expected, rtol=1e-8):
            raise RuntimeError(f'Volume of grain {gid} does not match its surface after smoothing')
        per_grain[gid] = dict(entry, volume=volume, minimum_quality=float(quality[selected].min()))
    dihedral = dihedral_summary(points, tets.tetrahedra)
    passed = bool(quality.min() >= report['quality_threshold'])
    if report.get('minimum_dihedral_threshold') is not None:
        passed = passed and dihedral['minimum'] >= report['minimum_dihedral_threshold']
    report.update(status='TET_MESH_VERIFIED' if passed else 'TET_QUALITY_BELOW_TARGET', ready=passed,
                  per_grain=per_grain, total_volume=float(volumes.sum()),
                  minimum_quality=float(quality.min()),
                  quality_percentiles_1_5_50=np.percentile(quality, [1, 5, 50]).tolist(),
                  below_quality_threshold=int(np.sum(quality < report['quality_threshold'])),
                  dihedral_angles=dihedral)
    return replace(tets, points=points, quality=quality, report=report)


def surface_after_smoothing(surface, tets):
    """The ClosedRVE with its points taken from smoothed tets (same numbering)."""
    points = np.asarray(tets.points)[:len(surface.points)].copy()
    report = dict(surface.report)
    if 'enclosed_grain_volumes' in report:
        report['enclosed_grain_volumes'] = {str(k): v for k, v in _enclosed_volumes(points, surface).items()}
    return replace(surface, points=points, report=report)
