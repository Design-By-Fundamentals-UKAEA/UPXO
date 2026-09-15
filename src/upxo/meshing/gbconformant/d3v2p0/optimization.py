"""Constrained mean-ratio optimization without changing mesh connectivity."""
import numpy as np

from .mesh import quality


def _energy_gradient(points, tets, weights):
    """Weighted inverse-square mean ratio and its analytic node gradient."""
    p = points[tets]
    a, b, c = p[:, 1]-p[:, 0], p[:, 2]-p[:, 0], p[:, 3]-p[:, 0]
    det = np.einsum('ij,ij->i', a, np.cross(b, c))
    ddet = np.stack([np.cross(b, c), np.cross(c, a), np.cross(a, b)], axis=1)
    ddet = np.concatenate([-ddet.sum(axis=1, keepdims=True), ddet], axis=1)
    edge2 = sum(np.sum((p[:, i]-p[:, j])**2, axis=1)
                for i in range(4) for j in range(i))
    q = 12*np.cbrt((det/2)**2)/edge2
    energy = weights/q**2
    # Sum of the six squared edge lengths has derivative 8*(p-centroid).
    dlogq = (2/3)*ddet/det[:, None, None] - 8*(p-p.mean(axis=1, keepdims=True))/edge2[:, None, None]
    local = -2*energy[:, None, None]*dlogq
    gradient = np.column_stack([np.bincount(tets.ravel(), weights=local[:, :, axis].ravel(),
                                          minlength=len(points)) for axis in range(3)])
    return float(energy.sum()), gradient


def _projectors(mesh):
    """Tangent subspaces at the post-Laplacian geometry, including RVE constraints."""
    n = len(mesh.points)
    constraints = np.zeros((n, 3, 3))
    triangles = mesh.boundary_triangles[~mesh.exterior]
    p = mesh.points[triangles]
    normals = np.cross(p[:, 1]-p[:, 0], p[:, 2]-p[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-30)
    # Outer products avoid dependence on triangle orientation.
    covariance = np.zeros_like(constraints)
    for corner in range(3):
        np.add.at(covariance, triangles[:, corner], normals[:, :, None]*normals[:, None, :])
    _, vectors = np.linalg.eigh(covariance)
    normal = vectors[:, :, -1]
    surface = mesh.node_kind == 1
    constraints[surface] = (normal[:, :, None]*normal[:, None, :])[surface]

    # A junction edge is incident to at least three distinct grains.
    signatures = [set() for _ in range(n)]
    for face, pair in zip(triangles, mesh.boundary_grains[~mesh.exterior]):
        for node in face:
            signatures[node].update(pair.tolist())
    edges = np.unique(np.sort(triangles[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1), axis=0)
    edges = np.array([(a, b) for a, b in edges if len(signatures[a] & signatures[b]) >= 3], dtype=int).reshape(-1, 2)
    covariance[:] = 0
    if len(edges):
        tangents = mesh.points[edges[:, 1]]-mesh.points[edges[:, 0]]
        tangents /= np.maximum(np.linalg.norm(tangents, axis=1, keepdims=True), 1e-30)
        for end in range(2):
            np.add.at(covariance, edges[:, end], tangents[:, :, None]*tangents[:, None, :])
    _, vectors = np.linalg.eigh(covariance)
    tangent = vectors[:, :, -1]
    line = mesh.node_kind == 2
    constraints[line] = (np.eye(3)-tangent[:, :, None]*tangent[:, None, :])[line]
    for axis in range(3):
        constraints[:, axis, axis] += mesh.fixed_axes[:, axis]
    constraints[mesh.node_kind == 3] = np.eye(3)
    values, vectors = np.linalg.eigh(constraints)
    return np.einsum('nik,nk,njk->nij', vectors, values < 1e-8, vectors)


def optimize_quality(mesh, iterations=40, max_displacement=0.35):
    """Improve tetrahedral quality in place with bounded tangential node moves.

    ``max_displacement`` is in physical coordinate units, measured from the
    post-Laplacian mesh. Surface and junction nodes stay in their initial local
    tangent subspaces; this approximates shape preservation on curved surfaces.
    Junction points and RVE planes stay exact. Backtracking accepts only positive
    elements, nondecreasing global minimum quality and decreasing inverse-square
    quality energy (boundary-node tetrahedra carry three times the weight).
    Returns accepted-step diagnostics. No optional dependencies are required.
    """
    if not isinstance(iterations, int) or iterations < 0:
        raise ValueError('iterations must be a nonnegative integer')
    if not np.isfinite(max_displacement) or max_displacement <= 0:
        raise ValueError('max_displacement must be finite and positive')
    if iterations == 0:
        return []
    anchor = mesh.points.copy()
    projection = _projectors(mesh)
    tets = mesh.tetrahedra
    weights = np.where(np.any(mesh.node_kind[tets] > 0, axis=1), 3., 1.)
    incidence = np.bincount(tets.ravel(), weights=np.repeat(weights, 4), minlength=len(anchor))
    history = []
    for _ in range(iterations):
        energy, gradient = _energy_gradient(mesh.points, tets, weights)
        direction = -np.einsum('nij,nj->ni', projection, gradient)/np.maximum(incidence[:, None], 1)
        largest = np.linalg.norm(direction, axis=1).max()
        if largest < 1e-12:
            break
        direction *= min(1., max_displacement*0.2/largest)
        floor = quality(mesh.points, tets)[1].min()
        # Keep isolated constrained elements from stopping optimization elsewhere.
        for _ in range(20):
            volumes, q = quality(mesh.points + direction, tets)
            bad = (volumes <= 0) | (q < floor)
            if not np.any(bad):
                break
            direction[np.unique(tets[bad])] = 0
        step = 1.
        for _ in range(20):
            offset = mesh.points + step*direction-anchor
            offset *= np.minimum(1., max_displacement/np.maximum(np.linalg.norm(offset, axis=1, keepdims=True), 1e-30))
            candidate = anchor+offset
            candidate[mesh.fixed_axes] = anchor[mesh.fixed_axes]
            candidate[mesh.node_kind == 3] = anchor[mesh.node_kind == 3]
            volumes, q = quality(candidate, tets)
            new_energy = float(np.sum(weights/q**2))
            if np.all(volumes > 0) and q.min() >= floor and new_energy < energy:
                break
            step *= 0.5
        else:
            break
        mesh.points = candidate
        history.append({'energy': new_energy, 'minimum_quality': float(q.min()), 'step': step})
    return history
