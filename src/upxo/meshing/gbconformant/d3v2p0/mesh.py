"""Shared voxel tetrahedra with constrained Laplacian relaxation.

Array axes are physical x, y, z; voxels occupy [i, i+1] * spacing.
All integer labels, including zero and negative labels, are grains.
"""
from dataclasses import dataclass, field
from itertools import permutations
import json
from pathlib import Path

import numpy as np


def quality(points, tets):
    """Signed volume and mean-ratio quality (regular tetrahedron = 1)."""
    p = points[tets]
    det = np.einsum('ij,ij->i', np.cross(p[:, 1]-p[:, 0], p[:, 2]-p[:, 0]),
                    p[:, 3]-p[:, 0])
    edge2 = sum(np.sum((p[:, i]-p[:, j])**2, axis=1)
                for i in range(4) for j in range(i))
    return det / 6, 12 * np.cbrt((det / 2)**2) / edge2


def faces_of(tets):
    faces = np.sort(tets[:, [[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]]], axis=2)
    unique, inverse, counts = np.unique(faces.reshape(-1, 3), axis=0,
                                        return_inverse=True, return_counts=True)
    owners = np.repeat(np.arange(len(tets)), 4)
    order = np.argsort(inverse, kind='stable')
    starts = np.r_[0, np.cumsum(counts)[:-1]]
    return unique, counts, owners[order[starts]], owners[order[starts+counts-1]]


@dataclass
class Mesh:
    points: np.ndarray
    tetrahedra: np.ndarray
    grain_ids: np.ndarray
    boundary_triangles: np.ndarray
    boundary_grains: np.ndarray
    exterior: np.ndarray
    node_kind: np.ndarray
    fixed_axes: np.ndarray
    original_points: np.ndarray
    history: list
    quality_history: list = field(default_factory=list)
    quality_before_optimization: np.ndarray | None = None

    def boundary_quality(self):
        """Quality of tets touching internal interfaces, and interface triangles."""
        nodes = np.unique(self.boundary_triangles[~self.exterior])
        boundary_nodes = np.zeros(len(self.points), dtype=bool)
        boundary_nodes[nodes] = True
        mask = np.any(boundary_nodes[self.tetrahedra], axis=1)
        q = quality(self.points, self.tetrahedra)[1][mask]
        p = self.points[self.boundary_triangles[~self.exterior]]
        area2 = np.linalg.norm(np.cross(p[:, 1]-p[:, 0], p[:, 2]-p[:, 0]), axis=1)
        edge2 = sum(np.sum((p[:, i]-p[:, j])**2, axis=1) for i, j in [(0, 1), (1, 2), (2, 0)])
        tq = 2*np.sqrt(3)*area2/edge2
        def summary(values):
            return {'count': len(values), 'minimum': float(values.min()),
                    'mean': float(values.mean()), 'p05': float(np.percentile(values, 5)),
                    'below_0_4': int(np.sum(values < .4))} if len(values) else {'count': 0}
        return {'boundary_node_tetrahedra': summary(q), 'interface_triangles': summary(tq)}

    def validate(self):
        """Check incidence, per-grain closure, volume, and frozen constraints."""
        volumes, q = quality(self.points, self.tetrahedra)
        _, counts, _, _ = faces_of(self.tetrahedra)
        if np.any(counts > 2) or np.any(volumes <= 0):
            raise ValueError('Non-manifold or inverted volume mesh')
        nonmanifold_grains = []
        for gid in np.unique(self.grain_ids):
            mask = (self.boundary_grains[:, 0] == gid) | (
                ~self.exterior & (self.boundary_grains[:, 1] == gid))
            f = self.boundary_triangles[mask]
            edges = np.sort(f[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1)
            _, n = np.unique(edges, axis=0, return_counts=True)
            if np.any(n % 2):
                raise ValueError(f'Open grain boundary: {gid}')
            if np.any(n != 2):
                nonmanifold_grains.append(int(gid))
        if not np.array_equal(self.points[self.node_kind == 3],
                              self.original_points[self.node_kind == 3]):
            raise ValueError('Junction points moved')
        if not np.array_equal(self.points[self.fixed_axes], self.original_points[self.fixed_axes]):
            raise ValueError('RVE plane constraints violated')
        expected = np.prod(np.ptp(self.original_points, axis=0))
        if not np.isclose(volumes.sum(), expected, rtol=1e-10):
            raise ValueError('RVE volume not preserved')
        return {'nodes': len(self.points), 'tetrahedra': len(self.tetrahedra),
                'grains': len(np.unique(self.grain_ids)), 'volume': float(volumes.sum()),
                'minimum_quality': float(q.min()), 'mean_quality': float(q.mean()),
                'quality_percentiles': np.percentile(q, [1, 5, 50, 95]).tolist(),
                'frozen_junction_points': int(np.sum(self.node_kind == 3)),
                'junction_line_nodes': int(np.sum(self.node_kind == 2)),
                'grains_with_nonmanifold_surface_edges': nonmanifold_grains,
                'boundary_quality': self.boundary_quality(),
                'quality_optimization_steps': len(self.quality_history),
                'accepted_iterations': len(self.history)}

    def save(self, prefix):
        """Write lossless NPZ, ParaView VTU, and validation JSON."""
        prefix = Path(prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        report = self.validate()
        np.savez_compressed(prefix.with_suffix('.npz'), points=self.points,
                            tetrahedra=self.tetrahedra, grain_ids=self.grain_ids,
                            boundary_triangles=self.boundary_triangles,
                            boundary_grains=self.boundary_grains, exterior=self.exterior,
                            node_kind=self.node_kind, fixed_axes=self.fixed_axes,
                            original_points=self.original_points)
        def array(name, values, dtype, components=1):
            return (f'<DataArray type="{dtype}" Name="{name}" NumberOfComponents="{components}" format="ascii">'
                    + ' '.join(map(str, np.asarray(values).ravel())) + '</DataArray>')
        xml = ('<?xml version="1.0"?><VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">'
               f'<UnstructuredGrid><Piece NumberOfPoints="{len(self.points)}" NumberOfCells="{len(self.tetrahedra)}">'
               '<Points>' + array('Points', self.points, 'Float64', 3) + '</Points><Cells>'
               + array('connectivity', self.tetrahedra, 'Int64')
               + array('offsets', np.arange(1, len(self.tetrahedra)+1)*4, 'Int64')
               + array('types', np.full(len(self.tetrahedra), 10), 'UInt8')
               + '</Cells><CellData>' + array('grain_id', self.grain_ids, 'Int64')
               + array('mean_ratio', quality(self.points, self.tetrahedra)[1], 'Float64')
               + '</CellData><PointData>' + array('node_kind', self.node_kind, 'UInt8')
               + '</PointData></Piece></UnstructuredGrid></VTKFile>')
        prefix.with_suffix('.vtu').write_text(xml)
        prefix.with_suffix('.json').write_text(json.dumps(report, indent=2))
        return report


def mesh_voxels(labels, spacing=1., iterations=20, relaxation=0.35, min_quality=0.5,
                quality_iterations=60, quality_displacement=0.35):
    """Mesh every voxel with six conforming tetrahedra, then relax globally.

    Surface nodes average surface neighbors; junction line nodes average only
    neighbors on their junction line. Branch/end points and >=4-grain nodes
    are frozen. Interior nodes average volume neighbors. Backtracking enforces
    positive volumes and the requested mean-ratio quality floor globally.
    The initial mesh must meet that floor; otherwise a ValueError is raised.
    A subsequent tangential quality optimization runs for quality_iterations
    steps (0 disables it), with motion bounded by quality_displacement times
    the smallest voxel spacing from the post-Laplacian geometry.
    """
    labels = np.asarray(labels)
    spacing = np.broadcast_to(np.asarray(spacing, dtype=float), (3,))
    if labels.ndim != 3 or not labels.size or labels.dtype.kind not in 'iu':
        raise ValueError('labels must be a nonempty 3D integer array')
    if np.any(~np.isfinite(spacing)) or np.any(spacing <= 0):
        raise ValueError('spacing must be finite and positive')
    if not isinstance(iterations, int) or iterations < 0 or not 0 < relaxation <= 1 or not 0 < min_quality <= 1:
        raise ValueError('Invalid smoothing parameters')
    if not isinstance(quality_iterations, int) or quality_iterations < 0 or not np.isfinite(quality_displacement) or quality_displacement <= 0:
        raise ValueError('Invalid quality optimization parameters')
    shape = np.array(labels.shape)+1
    lattice = np.indices(shape).reshape(3, -1).T
    points = lattice.astype(float)*spacing
    corners = np.indices(labels.shape).reshape(3, -1).T
    blocks = []
    for perm in permutations(range(3)):
        offsets = np.vstack([np.zeros(3, dtype=int), np.cumsum(np.eye(3, dtype=int)[list(perm)], axis=0)])
        blocks.append(np.ravel_multi_index((corners[:, None, :]+offsets).transpose(2, 0, 1), shape))
    tets = np.stack(blocks, axis=1).reshape(-1, 4)
    negative = quality(points, tets)[0] < 0
    tets[negative] = tets[negative][:, [0, 2, 1, 3]]
    grain_ids = np.repeat(labels.ravel(), 6)
    if quality(points, tets)[1].min() < min_quality:
        raise ValueError('Initial mesh is below min_quality; reduce spacing anisotropy or quality floor')
    faces, counts, first, last = faces_of(tets)
    selected = (counts == 1) | (grain_ids[first] != grain_ids[last])
    boundary = faces[selected]
    exterior = counts[selected] == 1
    pairs = np.column_stack([grain_ids[first[selected]], grain_ids[last[selected]]])
    signatures = [set() for _ in points]
    for tet, gid in zip(tets, grain_ids):
        for node in tet:
            signatures[node].add(int(gid))
    kind = np.array([min(len(s)-1, 3) for s in signatures], dtype=np.uint8)
    fixed = (lattice == 0) | (lattice == shape-1)
    all_edges = np.unique(np.sort(tets[:, [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]].reshape(-1, 2), axis=1), axis=0)
    surface_edges = np.unique(np.sort(boundary[~exterior][:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1), axis=0)
    junction_edges = np.array([(a, b) for a, b in surface_edges if len(signatures[a] & signatures[b]) >= 3], dtype=int).reshape(-1, 2)
    degree = np.bincount(junction_edges.ravel(), minlength=len(points))
    kind[(kind == 2) & (degree != 2)] = 3
    # Different three-grain signatures meeting at a node mark a junction point.
    for a, b in junction_edges:
        if signatures[a] != signatures[b]:
            if kind[a] == 2 and len(signatures[b]) == 3:
                kind[a] = 3
            if kind[b] == 2 and len(signatures[a]) == 3:
                kind[b] = 3
    directed = []
    for category, edges in enumerate([all_edges, surface_edges, junction_edges]):
        e = np.vstack([edges, edges[:, ::-1]])
        directed.append(e[kind[e[:, 0]] == category])
    edges = np.vstack(directed)
    degree = np.bincount(edges[:, 0], minlength=len(points))
    original = points.copy()
    history = []
    for _ in range(iterations):
        average = np.column_stack([np.bincount(edges[:, 0], weights=points[edges[:, 1], axis], minlength=len(points)) for axis in range(3)])
        delta = average / np.maximum(degree[:, None], 1)-points
        delta[(degree == 0) | (kind == 3)] = 0
        delta[fixed] = 0
        # Suppress only moves participating in unacceptable elements. Repeat
        # because suppressing one corner changes neighboring element shapes.
        for local_pass in range(30):
            volume, q = quality(points + relaxation*delta, tets)
            bad = (volume <= 0) | (q < min_quality)
            if not np.any(bad):
                break
            delta[np.unique(tets[bad])] = 0
        step = relaxation
        for attempt in range(18):
            candidate = points + step*delta
            volume, q = quality(candidate, tets)
            if np.all(volume > 0) and q.min() >= min_quality:
                break
            step *= 0.5
        else:
            break
        movement = float(np.max(np.linalg.norm(candidate-points, axis=1)))
        if movement < 1e-9*spacing.min():
            break
        points = candidate
        history.append({'step': step, 'maximum_displacement': movement, 'minimum_quality': float(q.min())})
    result = Mesh(points, tets, grain_ids, boundary, pairs, exterior, kind, fixed, original, history)
    result.quality_before_optimization = quality(points, tets)[1]
    if quality_iterations:
        from .optimization import optimize_quality
        result.quality_history = optimize_quality(result, iterations=quality_iterations,
                                                   max_displacement=quality_displacement*spacing.min())
    result.validate()
    return result
