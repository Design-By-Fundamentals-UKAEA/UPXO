"""Close all six RVE faces with labelled, conforming planar Gmsh caps."""
from dataclasses import dataclass
import uuid
import numpy as np


@dataclass
class ClosedRVE:
    points: np.ndarray
    triangles: np.ndarray
    grain_pairs: np.ndarray
    exterior: np.ndarray
    rve_face: np.ndarray
    report: dict


def close_rve_faces(surface, labels, spacing=1., mesh_size=.75, verbose=False,
                    edge_size=None):
    """Cap a remeshed internal interface complex without moving its nodes.

    Planar regions are polygonized from existing interface traces and the box
    perimeter. Grain IDs follow the oriented interface normals. Gmsh generates
    each planar region with existing interface traces preserved. Box perimeter
    segments are subdivided to edge_size (default mesh_size), using shared
    curves on adjacent faces. Existing internal triangle edges are retained.
    This adds no tetrahedra and does not repair inherited nonmanifoldness.
    Source triangles must point from grain_pairs[:,0] toward grain_pairs[:,1].
    """
    import gmsh
    edge_size = mesh_size if edge_size is None else edge_size
    if not np.isfinite(edge_size) or edge_size <= 0:
        raise ValueError('edge_size must be finite and positive')
    from shapely.geometry import LineString
    from shapely.geometry.polygon import orient
    from shapely.ops import polygonize
    labels = np.asarray(labels)
    spacing = np.broadcast_to(np.asarray(spacing, dtype=float), (3,))
    if labels.ndim != 3 or not labels.size or labels.dtype.kind not in 'iu':
        raise ValueError('labels must be a nonempty 3D integer array')
    if np.any(~np.isfinite(spacing)) or np.any(spacing <= 0) or not np.isfinite(mesh_size) or mesh_size <= 0:
        raise ValueError('spacing and mesh_size must be finite and positive')
    extent = np.array(labels.shape)*spacing
    tol = 1e-9*max(1., float(extent.max()))
    source_points = np.asarray(surface.points, dtype=float).reshape(-1, 3).copy()
    points = source_points.tolist()
    triangles = np.asarray(surface.triangles, dtype=int).reshape(-1, 3)
    grain_pairs = np.asarray(surface.grain_pairs).reshape(-1, 2)
    if len(triangles) != len(grain_pairs):
        raise ValueError('Each interface triangle needs a grain pair')
    p = source_points[triangles]
    normals = np.cross(p[:, 1]-p[:, 0], p[:, 2]-p[:, 0])
    # Keep only edges lying on an RVE face, retaining the incident triangle
    # normal and pair to label the planar region on either side of the edge.
    all_edges = np.sort(triangles[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1)
    owners = np.repeat(np.arange(len(triangles)), 3)
    patches = []
    if np.any(source_points < -tol) or np.any(source_points > extent+tol):
        raise ValueError('Internal interface nodes lie outside the RVE')
    # Share perimeter vertices between adjacent box faces as well.
    global_lookup = {}
    for i, xyz in enumerate(points):
        key = tuple(np.rint(np.array(xyz)/tol).astype(np.int64))
        if key in global_lookup and global_lookup[key] != i:
            if np.linalg.norm(np.array(points[global_lookup[key]])-xyz) <= tol:
                raise ValueError('Coincident independent interface nodes must be resolved before capping')
        global_lookup[key] = i
    def node_at(xyz):
        key = tuple(np.rint(np.asarray(xyz)/tol).astype(np.int64))
        if key not in global_lookup:
            global_lookup[key] = len(points)
            points.append(list(map(float, xyz)))
        return global_lookup[key]
    for axis in range(3):
        uv = [j for j in range(3) if j != axis]
        for side in (0, 1):
            plane = extent[axis]*side
            face_id = 2*axis+side
            mask = np.all(np.abs(source_points[all_edges, axis]-plane) <= tol, axis=1)
            traces = {}
            for edge, owner in zip(all_edges[mask], owners[mask]):
                traces.setdefault(tuple(map(int, edge)), []).append(int(owner))
            face_nodes = {node for edge in traces for node in edge}
            on_face = np.abs(source_points[:, axis]-plane) <= tol
            on_perimeter = np.any((np.abs(source_points[:, uv]) <= tol)
                                  | (np.abs(source_points[:, uv]-extent[uv]) <= tol), axis=1)
            face_nodes.update(map(int, np.flatnonzero(on_face & on_perimeter)))
            for u, v in [(0., 0.), (extent[uv[0]], 0.), (extent[uv[0]], extent[uv[1]]), (0., extent[uv[1]])]:
                xyz = np.zeros(3)
                xyz[axis], xyz[uv[0]], xyz[uv[1]] = plane, u, v
                face_nodes.add(node_at(xyz))
            segments = set(traces)
            # Every interface endpoint on a box edge subdivides the perimeter.
            for fixed_axis, varying_axis in [(uv[0], uv[1]), (uv[1], uv[0])]:
                for value in (0., extent[fixed_axis]):
                    ordered = sorted((node for node in face_nodes if abs(points[node][fixed_axis]-value) <= tol),
                                     key=lambda node: points[node][varying_axis])
                    segments.update(tuple(sorted((a, b))) for a, b in zip(ordered, ordered[1:]))
            lookup = {tuple(np.rint(np.array(points[node])[uv]/tol).astype(np.int64)): node for node in face_nodes}
            lines = [LineString([np.array(points[a])[uv], np.array(points[b])[uv]]) for a, b in segments]
            regions = [orient(poly, sign=1.) for poly in polygonize(lines)]
            face_area = float(extent[uv[0]]*extent[uv[1]])
            if not np.isclose(sum(poly.area for poly in regions), face_area, rtol=1e-8, atol=tol**2):
                raise ValueError(f'RVE face {face_id}: traces do not partition the complete rectangle')
            # Edge usage verifies no trace was ignored by polygonization.
            used_edges = {}
            for polygon in regions:
                rings = []
                inferred = set()
                for ring in [polygon.exterior, *polygon.interiors]:
                    nodes = [lookup[tuple(np.rint(np.asarray(xy)/tol).astype(np.int64))] for xy in list(ring.coords)[:-1]]
                    rings.append(nodes)
                    for a, b in zip(nodes, nodes[1:]+nodes[:1]):
                        edge = tuple(sorted((a, b)))
                        used_edges[edge] = used_edges.get(edge, 0)+1
                        tangent = np.array(points[b])[uv]-np.array(points[a])[uv]
                        left = np.zeros(3)
                        left[uv] = [-tangent[1], tangent[0]]
                        for owner in traces.get(edge, []):
                            dot = np.dot(left, normals[owner])
                            if abs(dot) <= tol*np.linalg.norm(normals[owner])*np.linalg.norm(left):
                                raise ValueError(f'RVE face {face_id}: an internal triangle is coplanar with the cap')
                            inferred.add(int(grain_pairs[owner, 1 if dot > 0 else 0]))
                if len(inferred) > 1:
                    raise ValueError(f'RVE face {face_id}: inconsistent grain labels around a cap region: {sorted(inferred)}')
                if inferred:
                    gid = inferred.pop()
                else:
                    # A face without traces is occupied by its corner grain.
                    index = [0, 0, 0]
                    index[axis] = labels.shape[axis]-1 if side else 0
                    gid = int(labels[tuple(index)])
                patches.append((face_id, gid, rings, float(polygon.area)))
            if not set(traces).issubset(used_edges):
                raise ValueError(f'RVE face {face_id}: dangling or crossing interface traces')
    owned = not gmsh.isInitialized()
    if owned:
        gmsh.initialize()
    previous_model = gmsh.model.getCurrent()
    options = {'General.Terminal': int(verbose), 'Mesh.Algorithm': 6,
               'Mesh.MeshSizeMin': mesh_size, 'Mesh.MeshSizeMax': mesh_size,
               'Mesh.ElementOrder': 1, 'Mesh.RecombineAll': 0, 'Mesh.Renumber': 0}
    previous_options = {key: gmsh.option.getNumber(key) for key in options}
    model_name = 'CM02_RVE_caps_'+uuid.uuid4().hex
    gmsh.model.add(model_name)
    try:
        for key, value in options.items():
            gmsh.option.setNumber(key, value)
        point_tags, line_tags, surface_info = {}, {}, {}
        for face_id, gid, rings, area in patches:
            loops = []
            for ring in rings:
                curves = []
                for a, b in zip(ring, ring[1:]+ring[:1]):
                    for node in (a, b):
                        if node not in point_tags:
                            point_tags[node] = gmsh.model.geo.addPoint(*points[node], mesh_size)
                    edge = tuple(sorted((a, b)))
                    if edge not in line_tags:
                        line_tags[edge] = gmsh.model.geo.addLine(point_tags[edge[0]], point_tags[edge[1]])
                    curves.append(line_tags[edge]*(1 if a < b else -1))
                loops.append(gmsh.model.geo.addCurveLoop(curves))
            tag = gmsh.model.geo.addPlaneSurface(loops)
            surface_info[tag] = (face_id, gid, area)
        gmsh.model.geo.synchronize()
        source_edges = set(map(tuple, all_edges))
        subdivided_segments = 0
        added_edge_nodes = 0
        boundary_points = np.asarray(points)
        for edge, tag in line_tags.items():
            xyz = boundary_points[list(edge)]
            # Both endpoints must share two box planes. Keep source interface
            # edges intact to avoid a hanging node on an internal triangle.
            box_axes = np.all(np.abs(xyz) <= tol, axis=0) | np.all(np.abs(xyz-extent) <= tol, axis=0)
            intervals = 1
            if np.sum(box_axes) >= 2 and edge not in source_edges:
                intervals = max(1, int(np.ceil(np.linalg.norm(xyz[1]-xyz[0])/edge_size)))
            subdivided_segments += int(intervals > 1)
            added_edge_nodes += intervals-1
            gmsh.model.mesh.setTransfiniteCurve(tag, intervals+1)
        gmsh.model.mesh.generate(2)
        tags, coords, _ = gmsh.model.mesh.getNodes()
        tag_to_node = {}
        for node, tag in point_tags.items():
            node_tags, xyz, _ = gmsh.model.mesh.getNodes(0, tag)
            if len(node_tags) != 1 or not np.allclose(xyz, points[node], atol=tol, rtol=0):
                raise RuntimeError('Gmsh moved a cap boundary point')
            tag_to_node[int(node_tags[0])] = node
        for tag, xyz in zip(tags, np.asarray(coords).reshape(-1, 3)):
            if int(tag) not in tag_to_node:
                tag_to_node[int(tag)] = len(points)
                points.append(xyz.tolist())
        cap_faces, cap_labels, cap_ids = [], [], []
        for tag, (face_id, gid, expected_area) in surface_info.items():
            _, conn = gmsh.model.mesh.getElementsByType(2, tag)
            f = np.array([tag_to_node[int(node)] for node in conn], dtype=int).reshape(-1, 3)
            if not len(f):
                raise RuntimeError(f'Empty cap region for grain {gid}')
            p = np.asarray(points)[f]
            axis, side = divmod(face_id, 2)
            normal = np.cross(p[:, 1]-p[:, 0], p[:, 2]-p[:, 0])
            flip = normal[:, axis]*(1 if side else -1) < 0
            f[flip] = f[flip][:, [0, 2, 1]]
            if not np.isclose(np.linalg.norm(normal, axis=1).sum()/2, expected_area, rtol=1e-8, atol=tol**2):
                raise RuntimeError('Cap mesh area differs from its planar region')
            if not np.all(np.abs(p[:, :, axis]-extent[axis]*side) <= tol):
                raise RuntimeError('Cap nodes left the RVE plane')
            cap_faces.append(f)
            cap_labels.append(np.full((len(f), 2), gid, dtype=grain_pairs.dtype))
            cap_ids.append(np.full(len(f), face_id, dtype=int))
        points = np.asarray(points)
        faces = np.vstack([triangles, *cap_faces])
        pairs = np.vstack([grain_pairs, *cap_labels])
        face_ids = np.concatenate([np.full(len(triangles), -1), *cap_ids])
        exterior = face_ids >= 0
        nonmanifold = []
        grain_volumes = {}
        for gid in np.unique(labels):
            selected = (pairs[:, 0] == gid) | (~exterior & (pairs[:, 1] == gid))
            f = faces[selected]
            if not len(f):
                raise RuntimeError(f'Grain {gid} has no boundary surface')
            _, counts = np.unique(np.sort(f[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1), axis=0, return_counts=True)
            if np.any(counts % 2):
                raise RuntimeError(f'Grain {gid} has open/unbalanced surface edges after capping')
            if np.any(counts != 2):
                nonmanifold.append(int(gid))
            p = points[f]
            sign = np.where(~exterior[selected] & (pairs[selected, 1] == gid), -1., 1.)
            volume = float(np.sum(sign*np.einsum('ij,ij->i', p[:, 0], np.cross(p[:, 1], p[:, 2])))/6)
            if volume <= 0:
                raise RuntimeError(f'Grain {gid} has nonpositive enclosed signed volume')
            grain_volumes[str(int(gid))] = volume
        cap = faces[exterior]
        _, cap_counts = np.unique(np.sort(cap[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1), axis=0, return_counts=True)
        if np.any(cap_counts != 2):
            raise RuntimeError('The exterior RVE shell is not closed/manifold')
        if not np.isclose(sum(grain_volumes.values()), np.prod(extent), rtol=1e-8):
            raise RuntimeError('Total enclosed grain volume differs from RVE volume')
        report = {'grains': len(np.unique(labels)), 'cap_regions': len(patches),
                  'rve_edge_size': float(edge_size),
                  'subdivided_rve_edge_segments': subdivided_segments,
                  'added_rve_edge_nodes': added_edge_nodes,
                  'cap_triangles': int(exterior.sum()), 'total_triangles': len(faces),
                  'all_six_faces_closed': True, 'grain_surface_edge_closure_verified': True,
                  'grains_with_nonmanifold_edges': nonmanifold,
                  'rve_dimensions': extent.tolist(), 'rve_volume': float(np.prod(extent)),
                  'enclosed_grain_volumes': grain_volumes,
                  'source_interface_nodes_unchanged': bool(np.array_equal(points[:len(source_points)], source_points))}
        return ClosedRVE(points, faces, pairs, exterior, face_ids, report)
    finally:
        gmsh.model.setCurrent(model_name)
        gmsh.model.remove()
        for key, value in previous_options.items():
            gmsh.option.setNumber(key, value)
        if owned:
            gmsh.finalize()
        elif previous_model in gmsh.model.list():
            gmsh.model.setCurrent(previous_model)
