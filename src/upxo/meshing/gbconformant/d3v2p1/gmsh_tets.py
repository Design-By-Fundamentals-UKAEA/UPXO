"""Grain-parallel conforming tetrahedral meshing of repaired grain shells.

Same result contract as d3v2p0.gmsh_tets.mesh_repaired_rve_gmsh: surface
triangles stay fixed, so each grain's volume is meshed independently of every
other grain (their shared interface triangles cannot change). Grains are
distributed over worker processes, largest first; each worker meshes its
grains one Gmsh model at a time with the d3v2p0 options. Results are merged on
the fixed surface-node numbering and verified exactly as in d3v2p0
(conformity per grain, positive volumes, grain and RVE volumes, quality,
dihedral statistics, readiness).
"""
import uuid
import numpy as np
from ..d3v2p0.gmsh_tets import GrainTetrahedra, save_grain_tetrahedra  # noqa: F401  (re-exported)
from .backend import plan, check_workers
from .parallel import balanced_chunks, run_chunks


def _grain_shells(points, triangles, pairs, gid):
    """Outward-oriented triangles of one grain, split into connected shells."""
    first = pairs[:, 0] == gid
    second = pairs[:, 1] == gid
    tri = np.vstack((triangles[first], triangles[second][:, [0, 2, 1]]))
    # connected components over shared nodes
    nodes, local = np.unique(tri, return_inverse=True)
    local = local.reshape(-1, 3)
    parent = np.arange(len(nodes))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for a, b, c in local:
        ra, rb, rc = find(a), find(b), find(c)
        parent[rb] = ra
        parent[find(rc)] = ra
    roots = np.array([find(x) for x in local[:, 0]])
    shells = []
    for r in np.unique(roots):
        t = tri[roots == r]
        x = points[t]
        volume = float(np.einsum('ij,ij->i', x[:, 0], np.cross(x[:, 1], x[:, 2])).sum() / 6)
        shells.append((t, volume))
    return shells


def _contains(points, shell_tri, point):
    """Solid-angle winding test: point inside the closed shell."""
    a, b, c = (points[shell_tri] - point).transpose(1, 0, 2)
    la, lb, lc = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1), np.linalg.norm(c, axis=1)
    num = np.einsum('ij,ij->i', a, np.cross(b, c))
    den = la * lb * lc + np.einsum('ij,ij->i', a, b) * lc + np.einsum('ij,ij->i', b, c) * la \
        + np.einsum('ij,ij->i', c, a) * lb
    return abs(2 * np.arctan2(num, den).sum()) > 2 * np.pi


def _mesh_grain_jobs(task):
    """Worker: mesh a list of grain jobs; returns per-volume results."""
    from upxo._sup.optional_imports import import_gmsh
    gmsh = import_gmsh()
    jobs, options, optimize_netgen, netgen_all_volumes, minimum_quality = task
    owned = not gmsh.isInitialized()
    if owned:
        gmsh.initialize()
    out = []
    try:
        gmsh.option.setNumber('General.Terminal', 0)
        for key, value in options.items():
            gmsh.option.setNumber(key, value)
        for gid, global_ids, local_points, shells in jobs:
            name = 'd3v2p1_tets_' + uuid.uuid4().hex
            gmsh.model.add(name)
            try:
                n = len(local_points)
                tags = []
                for k, (tri, _) in enumerate(shells):
                    tag = gmsh.model.addDiscreteEntity(2)
                    if k == 0:
                        gmsh.model.mesh.addNodes(2, tag, np.arange(1, n + 1), local_points.ravel())
                    first = 1 + sum(len(s[0]) for s in shells[:k])
                    gmsh.model.mesh.addElementsByType(tag, 2, np.arange(first, first + len(tri)), (tri + 1).ravel())
                    tags.append(tag)
                gmsh.model.mesh.reclassifyNodes()
                gmsh.model.mesh.createTopology(True, True)
                loops = [gmsh.model.geo.addSurfaceLoop([t]) for t in tags]
                outer = [k for k, (_, v) in enumerate(shells) if v > 0]
                inner = [k for k, (_, v) in enumerate(shells) if v < 0]
                if not outer:
                    raise RuntimeError(f'Grain {gid} has no outer shell')
                holes = {k: [] for k in outer}
                for k in inner:
                    point = local_points[shells[k][0][0, 0]]
                    owners = [j for j in outer if _contains(local_points, shells[j][0], point)]
                    if len(owners) != 1:
                        raise RuntimeError(f'Ambiguous cavity containment in grain {gid}')
                    holes[owners[0]].append(loops[k])
                volumes = [gmsh.model.geo.addVolume([loops[k]] + holes[k]) for k in outer]
                gmsh.model.geo.synchronize()
                gmsh.model.mesh.generate(3)

                def snapshot():
                    node_tags, coords, _ = gmsh.model.mesh.getNodes()
                    node_tags = np.asarray(node_tags, dtype=np.int64)
                    coords = np.asarray(coords).reshape(-1, 3)
                    order = np.argsort(node_tags)
                    node_tags, coords = node_tags[order], coords[order]
                    result = {}
                    for v in volumes:
                        elements, conn = gmsh.model.mesh.getElementsByType(4, v)
                        if not len(elements):
                            raise RuntimeError(f'No tetrahedra in grain {gid}')
                        conn = np.asarray(conn, dtype=np.int64).reshape(-1, 4)
                        used, inv = np.unique(conn, return_inverse=True)
                        quality = np.asarray(gmsh.model.mesh.getElementQualities(elements, 'minSICN'))
                        result[v] = (used, coords[np.searchsorted(node_tags, used)], inv.reshape(-1, 4), quality)
                    return result
                baseline = snapshot()
                chosen = baseline
                retained = 0
                if optimize_netgen:
                    poor = [(3, v) for v, d in baseline.items() if netgen_all_volumes or d[3].min() < minimum_quality]
                    if poor:
                        gmsh.model.mesh.optimize('Netgen', dimTags=poor)
                        chosen = snapshot()
                        for v in chosen:
                            if chosen[v][3].min() < baseline[v][3].min():
                                chosen[v] = baseline[v]
                                retained += 1
                for v, (used, coords, conn, quality) in chosen.items():
                    out.append((gid, global_ids, n, used, coords, conn, quality, retained))
            finally:
                gmsh.model.setCurrent(name)
                gmsh.model.remove()
    finally:
        if owned:
            gmsh.finalize()
    return out


def mesh_repaired_rve_gmsh(surface, mesh_size=1.5, verbose=False, optimize_netgen=True, minimum_quality=.05,
                           netgen_all_volumes=False, optimize_threshold=None, minimum_dihedral=None,
                           n_workers=None, precheck=True, backend='auto'):
    """Grain-parallel version of d3v2p0.gmsh_tets.mesh_repaired_rve_gmsh.

    backend / n_workers: see d3v2p1.backend (None = automatic); grains are
    meshed in separate Gmsh models, so the mesh does not depend on them.
    precheck: run the d3v2p0 surface validation first (as d3v2p0 does).
    Other arguments as in d3v2p0. Returns a GrainTetrahedra whose first
    len(surface.points) nodes are the surface points.
    """
    from .tet_validation import validate_tet_surfaces
    from ..d3v2p0.tet_angles import dihedral_summary
    if not np.isfinite(mesh_size) or mesh_size <= 0:
        raise ValueError('mesh_size must be finite and positive')
    if not 0 <= minimum_quality <= 1:
        raise ValueError('minimum_quality must lie between 0 and 1')
    if not isinstance(netgen_all_volumes, (bool, np.bool_)):
        raise ValueError('netgen_all_volumes must be boolean')
    if optimize_threshold is not None and (not np.isfinite(optimize_threshold) or not 0 < optimize_threshold <= 1):
        raise ValueError('optimize_threshold must lie in (0, 1]')
    if minimum_dihedral is not None and (not np.isfinite(minimum_dihedral) or not 0 <= minimum_dihedral < 70.5):
        raise ValueError('minimum_dihedral must lie in [0, 70.5) degrees')
    check_workers(n_workers)
    p = np.asarray(surface.points, float)
    f = np.asarray(surface.triangles)
    pairs = np.asarray(surface.grain_pairs)
    exterior = np.asarray(surface.exterior, bool)
    expected_rve_volume = float(np.prod(surface.report['rve_dimensions']))
    if precheck:
        validation = validate_tet_surfaces(surface, check_intersections=True, n_workers=n_workers,
                                           backend=backend)
        if validation['blockers']:
            raise ValueError('Repair surface blockers first: ' + '; '.join(validation['blockers']))
        expected_rve_volume = validation['expected_rve_volume']
    # Internal triangles count for both grains, RVE faces for their own grain only.
    internal_pairs = np.where(exterior[:, None], np.column_stack((pairs[:, 0], np.full(len(pairs), -1))), pairs)
    grains = np.unique(np.concatenate((pairs[:, 0], pairs[~exterior, 1])))
    jobs, weights = [], []
    for gid in grains:
        shells = _grain_shells(p, f, internal_pairs, gid)
        nodes = np.unique(np.concatenate([t.ravel() for t, _ in shells]))
        remap = np.full(len(p), -1, dtype=np.int64)
        remap[nodes] = np.arange(len(nodes))
        local_shells = [(remap[t], v) for t, v in shells]
        jobs.append((int(gid), nodes, p[nodes], local_shells))
        weights.append(sum(abs(v) for _, v in shells))
    options = {'Mesh.MeshOnlyEmpty': 1, 'Mesh.MeshSizeMax': mesh_size, 'Mesh.MeshSizeMin': 0,
               'Mesh.Algorithm3D': 1, 'Mesh.ElementOrder': 1, 'Mesh.Optimize': 1, 'Mesh.OptimizeNetgen': 0,
               'Mesh.Renumber': 0, 'General.NumThreads': 1}
    if optimize_threshold is not None:
        options['Mesh.OptimizeThreshold'] = float(optimize_threshold)
    chosen = plan(backend, n_workers, items=len(jobs))
    chunks = balanced_chunks(weights, chosen.workers)
    tasks = [([jobs[i] for i in c], options, bool(optimize_netgen), bool(netgen_all_volumes), float(minimum_quality))
             for c in chunks]
    results = [r for chunk in run_chunks(_mesh_grain_jobs, tasks, chosen) for r in chunk]
    results.sort(key=lambda r: r[0])                         # deterministic order: by grain id
    point_blocks, count = [p.copy()], len(p)
    tets, labels, quality, retained = [], [], [], 0
    for gid, global_ids, n, used, coords, conn, q, kept in results:
        mapping = np.empty(len(used), dtype=np.int64)
        surface_node = used <= n
        mapping[surface_node] = global_ids[used[surface_node] - 1]
        interior = ~surface_node
        mapping[interior] = np.arange(count, count + int(interior.sum()))
        point_blocks.append(coords[interior])
        count += int(interior.sum())
        tets.append(mapping[conn])
        labels.append(np.full(len(conn), gid, dtype=pairs.dtype))
        quality.append(q)
        retained += kept
    points = np.vstack(point_blocks)
    tets, labels, quality = np.vstack(tets), np.concatenate(labels), np.concatenate(quality)
    x = points[tets]
    volumes = np.einsum('ij,ij->i', x[:, 1] - x[:, 0], np.cross(x[:, 2] - x[:, 0], x[:, 3] - x[:, 0])) / 6
    if np.any(~np.isfinite(volumes)) or np.any(volumes <= 0):
        raise RuntimeError('Nonpositive/nonfinite tetrahedron volumes')
    if np.any(~np.isfinite(quality)) or np.any(quality <= 0):
        raise RuntimeError('Invalid tetrahedron quality')
    pattern = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]
    faces = np.sort(tets[:, pattern].reshape(-1, 3), axis=1)
    key = np.column_stack((np.repeat(labels, 4), faces))
    uniq, counts = np.unique(key, axis=0, return_counts=True)
    if np.any(counts > 2):
        raise RuntimeError('Nonmanifold tet faces')
    st = np.sort(f, axis=1)
    expected = np.unique(np.vstack((np.column_stack((pairs[:, 0], st)),
                                    np.column_stack((pairs[~exterior, 1], st[~exterior])))), axis=0)
    if not np.array_equal(uniq[counts == 1], expected):
        raise RuntimeError('Tet boundary differs from repaired surface')
    signed = np.einsum('ij,ij->i', p[f][:, 0], np.cross(p[f][:, 1], p[f][:, 2])) / 6
    per_grain = {}
    for gid in grains:
        sel = labels == gid
        actual = float(volumes[sel].sum())
        expected_volume = float(signed[pairs[:, 0] == gid].sum() - signed[~exterior & (pairs[:, 1] == gid)].sum())
        if not np.isclose(actual, expected_volume, rtol=1e-8):
            raise RuntimeError(f'Volume mismatch in grain {gid}')
        per_grain[str(int(gid))] = dict(tetrahedra=int(sel.sum()), volume=actual,
                                        minimum_quality=float(quality[sel].min()))
    if not np.isclose(volumes.sum(), expected_rve_volume, rtol=1e-8):
        raise RuntimeError('Tetrahedron volumes do not fill the RVE')
    dihedral = dihedral_summary(points, tets)
    passed = bool(quality.min() >= minimum_quality)
    if minimum_dihedral is not None:
        passed = passed and dihedral['minimum'] >= minimum_dihedral
    report = dict(status='TET_MESH_VERIFIED' if passed else 'TET_QUALITY_BELOW_TARGET', ready=passed,
                  grains=len(per_grain), volume_entities=len(results), tetrahedra=len(tets), nodes=len(points),
                  per_grain=per_grain, positive_volumes=True, source_surface_conformity_verified=True,
                  surface_intersection_check_passed=bool(precheck), total_volume=float(volumes.sum()),
                  minimum_quality=float(quality.min()),
                  quality_percentiles_1_5_50=np.percentile(quality, [1, 5, 50]).tolist(),
                  quality_metric='Gmsh minSICN', quality_threshold=float(minimum_quality),
                  baseline_retained_volumes=int(retained), below_quality_threshold=int(np.sum(quality < minimum_quality)),
                  netgen_all_volumes=bool(netgen_all_volumes),
                  optimize_threshold=None if optimize_threshold is None else float(optimize_threshold),
                  dihedral_angles=dihedral,
                  minimum_dihedral_threshold=None if minimum_dihedral is None else float(minimum_dihedral),
                  n_workers=int(chosen.workers), parallel_by='grain', backend=chosen.report())
    return GrainTetrahedra(points, tets, labels, quality, report)
