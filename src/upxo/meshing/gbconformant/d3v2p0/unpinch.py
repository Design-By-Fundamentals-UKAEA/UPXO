"""Separate point-contact surface fans and locally relax their tips before Gmsh."""
from dataclasses import replace
from numbers import Integral
import numpy as np


def _fans(triangles, npoints):
    incident = [[] for _ in range(npoints)]
    for face_id, tri in enumerate(triangles):
        for node in tri:
            incident[node].append(face_id)
    contacts = {}
    for node, faces in enumerate(incident):
        if len(faces) < 2:
            continue
        # Two incident triangles belong to the same fan when they share an
        # entire edge through this vertex, not just this vertex itself.
        parent = list(range(len(faces)))
        def root(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i
        neighbor_owner = {}
        for i, f in enumerate(faces):
            for other in triangles[f]:
                if other == node:
                    continue
                if other in neighbor_owner:
                    parent[root(i)] = root(neighbor_owner[other])
                else:
                    neighbor_owner[other] = i
        groups = {}
        for i, f in enumerate(faces):
            groups.setdefault(root(i), []).append(f)
        if len(groups) > 1:
            contacts[node] = list(groups.values())
    return contacts


def separate_pinch_fans(interfaces, enabled=True, radius=1.5, max_displacement=.5,
                        iterations=30, relaxation=.25, min_separation=.02):
    """Return (new Interfaces, report, node audit) without changing the input.

    Split vertices whose full triangle stars have disconnected fans. This
    intentionally removes point contacts, including legitimate ones. A shared
    edge or connected multi-grain junction is NOT cut: that would require new
    interface patches to avoid cracks. Only split tips and their radius-limited
    neighborhoods move. Other frozen junction points and RVE plane coordinates
    remain fixed. Length controls are physical units.

    Collapse/reversal checks and fixed edge incidence guard local validity;
    this is not a global surface-intersection or closed-volume certification.
    A batch that fails to separate every duplicated tip by min_separation is
    rolled back rather than returning coincident disconnected nodes.
    """
    if not isinstance(enabled, (bool, np.bool_)):
        raise ValueError('enabled must be boolean')
    if not isinstance(iterations, Integral) or isinstance(iterations, (bool, np.bool_)) or iterations < 0:
        raise ValueError('iterations must be a nonnegative integer')
    if any(not np.isfinite(x) or x <= 0 for x in (radius, max_displacement, min_separation)) or not 0 < relaxation <= .5:
        raise ValueError('Lengths must be positive and 0 < relaxation <= 0.5')
    source = interfaces
    result = replace(source, points=source.points.copy(), triangles=source.triangles.copy(),
                     grain_pairs=source.grain_pairs.copy(), original_points=source.original_points.copy(),
                     node_kind=source.node_kind.copy(), fixed_axes=source.fixed_axes.copy(),
                     junction_edges=source.junction_edges.copy())
    if not enabled:
        return result, {'enabled': False, 'split_contacts': 0, 'stop_reason': 'disabled'}, []
    contacts = _fans(source.triangles, len(source.points))
    if not contacts:
        return result, {'enabled': True, 'candidate_contacts': 0, 'split_contacts': 0,
                        'stop_reason': 'no disconnected surface fans'}, []
    # A fan may have open edges only where the source already has them: fan
    # splitting cannot split an existing edge, so its triangle incidence stays
    # unchanged. All grain-pair assignments are copied without alteration.
    points, originals = source.points.tolist(), source.original_points.tolist()
    kinds, fixed = source.node_kind.tolist(), source.fixed_axes.tolist()
    audit = []
    for node, fans in contacts.items():
        copies = [node]
        for faces in fans[1:]:
            new_node = len(points)
            points.append(source.points[node].tolist())
            originals.append(source.original_points[node].tolist())
            kinds.append(int(source.node_kind[node]))
            fixed.append(source.fixed_axes[node].tolist())
            f = result.triangles[faces].copy()
            f[f == node] = new_node
            result.triangles[faces] = f
            copies.append(new_node)
        audit.append({'source_node': int(node), 'fan_nodes': copies, 'fan_count': len(fans)})
    result.points = np.asarray(points, dtype=float)
    result.original_points = np.asarray(originals, dtype=float)
    result.node_kind = np.asarray(kinds, dtype=np.uint8)
    result.fixed_axes = np.asarray(fixed, dtype=bool)
    anchor = result.points.copy()
    tips = np.array([node for item in audit for node in item['fan_nodes']], dtype=int)
    edges, edge_inverse = np.unique(np.sort(result.triangles[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1), axis=0, return_inverse=True)
    edge_grains = [set() for _ in edges]
    for e, pair in zip(edge_inverse, np.repeat(result.grain_pairs, 3, axis=0)):
        edge_grains[e].update(pair.tolist())
    result.junction_edges = edges[np.array([len(s) >= 3 for s in edge_grains])]
    # Keep line directions for ordinary junction nodes; separated tips are
    # explicitly released from their former frozen/common-node constraint.
    neighbors = np.vstack([edges, edges[:, ::-1]])
    line = np.vstack([result.junction_edges, result.junction_edges[:, ::-1]])
    free_tip = np.zeros(len(anchor), dtype=bool)
    free_tip[tips] = True
    neighbors = neighbors[(result.node_kind[neighbors[:, 0]] != 2) | free_tip[neighbors[:, 0]]]
    line = line[(result.node_kind[line[:, 0]] == 2) & ~free_tip[line[:, 0]]]
    neighbors = np.vstack([neighbors, line])
    degree = np.bincount(neighbors[:, 0], minlength=len(anchor))
    # Geodesic distances keep relaxation on the affected surface sheets,
    # rather than moving unrelated surfaces that happen to pass nearby.
    import heapq
    adjacency = [[] for _ in anchor]
    for a, b in edges:
        length = float(np.linalg.norm(anchor[a]-anchor[b]))
        adjacency[a].append((b, length))
        adjacency[b].append((a, length))
    distance = np.full(len(anchor), np.inf)
    distance[tips] = 0.
    queue = [(0., int(t)) for t in tips]
    heapq.heapify(queue)
    while queue:
        d, node = heapq.heappop(queue)
        if d != distance[node]:
            continue
        for other, length in adjacency[node]:
            nd = d+length
            if nd < radius and nd < distance[other]:
                distance[other] = nd
                heapq.heappush(queue, (nd, int(other)))
    weight = np.maximum(0., 1-distance/radius)**2
    weight[(result.node_kind == 3) & ~free_tip] = 0.
    weight[degree == 0] = 0.
    def normals(p):
        tri = p[result.triangles]
        return np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])
    area_floor = .1*np.linalg.norm(normals(anchor), axis=1)
    accepted = 0
    for _ in range(iterations):
        mean = np.column_stack([np.bincount(neighbors[:, 0], weights=result.points[neighbors[:, 1], axis], minlength=len(anchor))
                                for axis in range(3)])/np.maximum(degree[:, None], 1)
        delta = relaxation*weight[:, None]*(mean-result.points)
        delta[result.fixed_axes] = 0
        offset = result.points+delta-anchor
        offset *= np.minimum(1., max_displacement/np.maximum(np.linalg.norm(offset, axis=1, keepdims=True), 1e-30))
        delta = anchor+offset-result.points
        current_normals = normals(result.points)
        for _ in range(20):
            n = normals(result.points+delta)
            bad = (np.linalg.norm(n, axis=1) < area_floor) | (np.einsum('ij,ij->i', current_normals, n) <= 0)
            if not np.any(bad):
                break
            delta[np.unique(result.triangles[bad])] = 0
        for _ in range(20):
            n = normals(result.points+delta)
            if np.all(np.linalg.norm(n, axis=1) >= area_floor) and np.all(np.einsum('ij,ij->i', current_normals, n) > 0):
                break
            delta *= .5
        else:
            break
        if np.linalg.norm(delta, axis=1).max() < 1e-12:
            break
        result.points += delta
        accepted += 1
    unresolved = []
    for item in audit:
        p = result.points[item['fan_nodes']]
        separation = min(float(np.linalg.norm(p[i]-p[j])) for i in range(len(p)) for j in range(i))
        item['minimum_separation'] = separation
        item['displacements'] = (p-source.points[item['source_node']]).tolist()
        if separation < min_separation:
            unresolved.append(item['source_node'])
    report = {'enabled': True, 'candidate_contacts': len(audit), 'split_contacts': len(audit),
              'added_nodes': len(result.points)-len(source.points), 'relaxation_steps': accepted,
              'maximum_displacement': float(np.linalg.norm(result.points-anchor, axis=1).max()),
              'minimum_tip_separation': min(item['minimum_separation'] for item in audit),
              'unresolved_contacts': unresolved, 'stop_reason': 'completed'}
    if unresolved:
        report.update(split_contacts=0, added_nodes=0, maximum_displacement=0.,
                      stop_reason='batch rolled back: some tips could not be safely separated')
        return replace(source, points=source.points.copy(), triangles=source.triangles.copy()), report, audit
    # Verify exact incidence preservation: no surface edges were cut open.
    def incidence(triangles):
        _, counts = np.unique(np.sort(triangles[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1), axis=0, return_counts=True)
        return np.sort(counts)
    if not np.array_equal(incidence(source.triangles), incidence(result.triangles)):
        raise RuntimeError('Fan splitting changed edge incidence')
    if not np.array_equal(result.points[result.fixed_axes], anchor[result.fixed_axes]):
        raise RuntimeError('RVE-plane coordinates changed')
    report['edge_incidence_preserved'] = True
    report['rve_planes_preserved'] = True
    return result, report, audit
