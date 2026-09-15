"""Gmsh remeshing of shared, labelled discrete grain interfaces."""
from dataclasses import dataclass
from numbers import Integral
import uuid
import numpy as np


@dataclass
class RemeshedInterfaces:
    points: np.ndarray
    triangles: np.ndarray
    grain_pairs: np.ndarray
    curve_edges: np.ndarray
    report: dict


def _directed_boundary(triangles):
    edges = triangles[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2)
    unique, inverse, counts = np.unique(np.sort(edges, axis=1), axis=0,
                                       return_inverse=True, return_counts=True)
    direction = np.bincount(inverse, weights=np.where(edges[:, 0] < edges[:, 1], 1, -1))
    if np.any(counts > 2) or np.any(direction[counts == 2] != 0):
        raise RuntimeError('Inconsistent triangle winding within a surface chart')
    return {tuple(edge): int(sign) for edge, sign in zip(unique[counts == 1], direction[counts == 1])}


def _align_chart_orientation(source_faces, source_points, new_faces, new_points,
                             source_tags=None, new_tags=None):
    """Restore winding using retained directed edges, or closed-shell volume.

    Optional node-tag arrays identify shared boundary nodes across the meshes.
    Returns a copy and whether the entire chart had to be reversed.
    """
    expected = _directed_boundary(source_faces if source_tags is None else source_tags[source_faces])
    actual = _directed_boundary(new_faces if new_tags is None else new_tags[new_faces])
    if expected.keys() != actual.keys():
        raise RuntimeError('Remeshed chart boundary differs from its source')
    if expected:
        agreement = {expected[e]*actual[e] for e in expected}
        if len(agreement) != 1:
            raise RuntimeError('Remeshed chart has mixed boundary orientations')
        flip = agreement.pop() < 0
    else:
        def volume(points, faces):
            xyz = points[faces]-points[faces].mean(axis=(0, 1))
            return np.einsum('ij,ij->i', xyz[:, 0], np.cross(xyz[:, 1], xyz[:, 2])).sum()/6
        before, after = volume(source_points, source_faces), volume(new_points, new_faces)
        if before == 0 or after == 0:
            raise RuntimeError('Closed chart has zero signed volume')
        flip = np.sign(before) != np.sign(after)
    return new_faces[:, [0, 2, 1]].copy() if flip else new_faces.copy(), bool(flip)


def remesh_interfaces_gmsh(interfaces, mesh_size=1., verbose=False, algorithm=6,
                           *, check_intersections=False, intersection_retries=2,
                           rve_dimensions=None, max_chart_triangles=None, minimum_facet_angle=.1,
                           _patch_keys=None, _oriented=False, _retry_depth=0,
                           frozen_grain_ids=()):
    """Reparametrize and regenerate every interface, retaining shared curve mesh.

    Existing 1D mesh nodes and edges are kept, so junctions and RVE-face traces
    remain exactly at their smoothed positions. Only the 2D mesh is regenerated.
    Gmsh may split an interface into multiple parametrizable charts; grain-pair
    labels are recovered from original element tags before remeshing.
    Charts that cannot be parametrized are split along existing triangle edges
    into disk-shaped charts and retried, without altering grain geometry.
    max_chart_triangles optionally partitions the input into bounded disks
    before Gmsh topology construction. Smaller values retain more source seams.
    check_intersections enables geometric checks and local source-chart fallback;
    rve_dimensions also rejects internal triangles flattened onto an RVE plane.
    algorithm selects a triangular surface mesher: 1 MeshAdapt, 2 Automatic,
    5 Delaunay, or 6 Frontal-Delaunay. Gmsh may fall back internally on failure.
    """
    import gmsh
    if len(frozen_grain_ids):
        # Give each protected voxel plane its own patch so parametrization
        # cannot round a stair-step corner. Shared curve construction is unchanged.
        base = np.sort(interfaces.grain_pairs, axis=1) if _patch_keys is None else np.asarray(_patch_keys)
        selected = np.any(np.isin(interfaces.grain_pairs[:, :2], frozen_grain_ids), axis=1)
        tags = np.zeros((len(base), 2))
        for i in np.flatnonzero(selected):
            p = interfaces.points[interfaces.triangles[i]]
            axes = np.flatnonzero(np.ptp(p, axis=0) < 1e-9)
            if len(axes) != 1: raise ValueError('Frozen surface must have planar voxel facets')
            axis = int(axes[0]); tags[i] = (axis+1, round(float(p[0,axis]), 12))
        _, plane_ids = np.unique(tags, axis=0, return_inverse=True)
        result = remesh_interfaces_gmsh(interfaces, mesh_size, verbose, algorithm,
            check_intersections=check_intersections, intersection_retries=intersection_retries,
            rve_dimensions=rve_dimensions, max_chart_triangles=max_chart_triangles,
            minimum_facet_angle=minimum_facet_angle, _patch_keys=np.column_stack((base,plane_ids)),
            _oriented=_oriented, _retry_depth=_retry_depth)
        result.grain_pairs = result.grain_pairs[:, :base.shape[1]]
        result.report['frozen_grain_ids'] = list(map(int, frozen_grain_ids))
        return result
    if not np.isfinite(minimum_facet_angle) or not 0 <= minimum_facet_angle < 180:
        raise ValueError('minimum_facet_angle must lie in [0, 180)')
    if max_chart_triangles is not None and (isinstance(max_chart_triangles,bool) or not isinstance(max_chart_triangles,Integral) or max_chart_triangles<1):
        raise ValueError('max_chart_triangles must be a positive integer or None')
    if not isinstance(intersection_retries,Integral) or intersection_retries<0:
        raise ValueError('intersection_retries must be a nonnegative integer')
    if not np.isfinite(mesh_size) or mesh_size <= 0:
        raise ValueError('mesh_size must be finite and positive')
    if isinstance(algorithm, (bool, np.bool_)) or not isinstance(algorithm, Integral) or algorithm not in (1, 2, 5, 6):
        raise ValueError('algorithm must be 1 (MeshAdapt), 2 (Automatic), 5 (Delaunay), or 6 (Frontal-Delaunay)')
    if not len(interfaces.triangles):
        raise ValueError('No internal grain interfaces to remesh')
    points = np.asarray(interfaces.points)
    triangles = interfaces.triangles.copy()
    pairs, pair_id = np.unique(np.sort(interfaces.grain_pairs, axis=1) if _patch_keys is None
                               else _patch_keys, axis=0, return_inverse=True)
    # Extraction winds x/z faces positively and y faces negatively. Orient
    # consistently from the lower grain ID toward the higher grain ID.
    if not _oriented:
        p = interfaces.original_points[triangles]
        normal = np.cross(p[:, 1]-p[:, 0], p[:, 2]-p[:, 0])
        flip = (normal.sum(axis=1) < 0) != (interfaces.grain_pairs[:, 0] > interfaces.grain_pairs[:, 1])
        triangles[flip] = triangles[flip][:, [0, 2, 1]]
    if max_chart_triangles is not None:
        from types import SimpleNamespace
        from .surface_charts import disk_charts
        partitions=np.empty(len(triangles),dtype=int);chart=0
        grouped=np.split(np.argsort(pair_id,kind='stable'),np.cumsum(np.bincount(pair_id))[:-1])
        for ids in grouped:
            for group in disk_charts(triangles[ids],max_triangles=max_chart_triangles):
                partitions[ids[group]]=chart;chart+=1
        base_keys=np.sort(interfaces.grain_pairs,axis=1) if _patch_keys is None else np.asarray(_patch_keys)
        source=SimpleNamespace(**vars(interfaces));source.triangles=triangles
        result=remesh_interfaces_gmsh(source,mesh_size,verbose,algorithm,
            check_intersections=check_intersections,intersection_retries=intersection_retries,rve_dimensions=rve_dimensions,minimum_facet_angle=minimum_facet_angle,
            _patch_keys=np.column_stack((base_keys,partitions)),_oriented=True,_retry_depth=max(1,_retry_depth))
        result.grain_pairs=result.grain_pairs[:,:base_keys.shape[1]]
        result.report['grain_pairs']=len(pairs)
        result.report['explicit_input_charts']=chart
        result.report['max_input_chart_triangles']=int(max_chart_triangles)
        return result
    owned = not gmsh.isInitialized()
    if owned:
        gmsh.initialize()
        gmsh.logger.start()
    previous_model = gmsh.model.getCurrent()
    options = {'General.Terminal': int(verbose), 'Mesh.MeshOnlyEmpty': 1,
               'Mesh.MeshOnlyVisible': 0,
               'Mesh.MeshSizeMin': mesh_size, 'Mesh.MeshSizeMax': mesh_size,
               'Mesh.Algorithm': int(algorithm), 'Mesh.ElementOrder': 1, 'Mesh.RecombineAll': 0,
               'Mesh.Renumber': 0}
    previous_options = {name: gmsh.option.getNumber(name) for name in options}
    name = 'CM02_interfaces_'+uuid.uuid4().hex
    gmsh.model.add(name)
    try:
        for key, value in options.items():
            gmsh.option.setNumber(key, value)
        grouped_ids = np.split(np.argsort(pair_id,kind='stable'),
                               np.cumsum(np.bincount(pair_id))[:-1])
        if _retry_depth >= 1:
            from .discrete_topology import add_chart_topology
            add_chart_topology(gmsh,points,triangles,grouped_ids)
        else:
            for i, ids in enumerate(grouped_ids):
                tag = gmsh.model.addDiscreteEntity(2)
                if i == 0:
                    gmsh.model.mesh.addNodes(2, tag, np.arange(1, len(points)+1), points.ravel())
                gmsh.model.mesh.addElementsByType(tag, 2, ids+1, (triangles[ids]+1).ravel())
            gmsh.model.mesh.reclassifyNodes()
        # Preserve labelled entities when creating their shared topology. Global
        # angle-based classifySurfaces is inappropriate for these multi-grain
        # nonmanifold junctions and crashes in Gmsh 4.14.1 on the sample.
            gmsh.model.mesh.createTopology(True, True)
        surface_pairs = {}
        source_charts = {}
        source_chart_ids = {}
        for _, tag in gmsh.model.getEntities(2):
            element_tags, _ = gmsh.model.mesh.getElementsByType(2, tag)
            source_ids = np.unique(pair_id[np.asarray(element_tags, dtype=int)-1])
            if len(source_ids) != 1:
                raise RuntimeError('Gmsh classification merged distinct grain interfaces')
            surface_pairs[tag] = int(source_ids[0])
            source_charts[tag] = triangles[np.asarray(element_tags, dtype=int)-1].copy()
            source_chart_ids[tag] = np.asarray(element_tags,dtype=int)-1
        if set(surface_pairs.values()) != set(range(len(pairs))):
            raise RuntimeError('Gmsh classification lost an interface')
        # Some voxel grains touch only at a point. Such frozen junctions can
        # lie inside a Gmsh chart instead of on a curve: embed one shared 0D
        # entity into every incident chart so it cannot be removed or relocated.
        constrained = np.flatnonzero((interfaces.node_kind == 3) | np.any(interfaces.fixed_axes, axis=1))+1
        on_boundary = set()
        for dim in (0, 1):
            for _, tag in gmsh.model.getEntities(dim):
                on_boundary.update(map(int, gmsh.model.mesh.getNodes(dim, tag, True, False)[0]))
        missing = set(map(int, constrained))-on_boundary
        if missing:
            # Explicit topology imports point nodes directly. Give these points
            # elements before reclassification, which otherwise only sees edges.
            for _, point_tag in gmsh.model.getEntities(0):
                if not len(gmsh.model.mesh.getElementsByType(15,point_tag)[0]):
                    point_nodes=gmsh.model.mesh.getNodes(0,point_tag)[0]
                    if len(point_nodes):gmsh.model.mesh.addElementsByType(point_tag,15,[],point_nodes)
            incident = {node: [] for node in missing}
            for tag in surface_pairs:
                _, conn = gmsh.model.mesh.getElementsByType(2, tag)
                for node in missing.intersection(map(int, conn)):
                    incident[node].append(tag)
            for node, surfaces in incident.items():
                point_tag = gmsh.model.addDiscreteEntity(0)
                gmsh.model.mesh.addElementsByType(point_tag, 15, [], [node])
                for tag in surfaces:
                    gmsh.model.mesh.embed(0, [point_tag], 2, tag)
            gmsh.model.mesh.reclassifyNodes()
        curves = gmsh.model.getEntities(1)
        if curves:gmsh.model.mesh.createGeometry(curves)
        failed = []
        from .surface_charts import is_disk_chart
        for tag in surface_pairs:
            if not is_disk_chart(source_charts[tag]):
                failed.append((tag,'Source chart is not a manifold disk'))
                continue
            try:
                gmsh.model.mesh.createGeometry([(2,tag)])
            except Exception as error:
                failed.append((tag,str(error)))
        if failed:
            if _retry_depth >= 4:
                raise RuntimeError(f'Parametrization failed after disk subdivision: {failed[:5]}')
            from types import SimpleNamespace
            from .surface_charts import disk_charts
            base_keys = np.sort(interfaces.grain_pairs,axis=1) if _patch_keys is None else np.asarray(_patch_keys)
            partitions = np.zeros(len(triangles),dtype=int)
            for chart_index,ids in enumerate(source_chart_ids.values(),1):
                partitions[ids]=chart_index
            next_partition = len(source_chart_ids)+1
            first_replacement = next_partition
            repaired_triangles = 0
            for tag,_ in failed:
                ids = source_chart_ids[tag]
                repaired_triangles += len(ids)
                for group in disk_charts(triangles[ids],max_triangles=64 if _retry_depth==0 else 1):
                    partitions[ids[group]] = next_partition;next_partition += 1
            retry_source = SimpleNamespace(**vars(interfaces))
            retry_source.triangles = triangles
            result = remesh_interfaces_gmsh(
                retry_source,mesh_size,verbose,algorithm,
                check_intersections=check_intersections,intersection_retries=intersection_retries,rve_dimensions=rve_dimensions,minimum_facet_angle=minimum_facet_angle,
                _patch_keys=np.column_stack((base_keys,partitions)),_oriented=True,
                _retry_depth=_retry_depth+1)
            result.grain_pairs = result.grain_pairs[:,:base_keys.shape[1]]
            result.report.setdefault('parametrization_repairs',[]).insert(0,{
                'failed_charts':len(failed),'replacement_disk_charts':next_partition-first_replacement,
                'source_triangles':int(repaired_triangles),
                'errors':[message for _,message in failed[:5]]})
            result.report['grain_pairs'] = len(np.unique(base_keys,axis=0))
            return result
        # Snapshot every curve and point node, including RVE traces. Their
        # identities and coordinates must survive clearing/remeshing 2D only.
        frozen = {}
        curve_edges = []
        for dim in (0, 1):
            for _, tag in gmsh.model.getEntities(dim):
                tags, coords, _ = gmsh.model.mesh.getNodes(dim, tag, True, False)
                frozen.update(zip(map(int, tags), np.asarray(coords).reshape(-1, 3)))
                if dim == 1:
                    _, conn = gmsh.model.mesh.getElementsByType(1, tag)
                    curve_edges.extend(np.asarray(conn, dtype=int).reshape(-1, 2).tolist())
        if not set(map(int, constrained)).issubset(frozen):
            missing = np.array(sorted(set(map(int, constrained))-set(frozen)))
            raise RuntimeError(f'Gmsh did not classify {len(missing)} frozen/RVE nodes on preserved boundaries; '
                               f'RVE={np.sum(np.any(interfaces.fixed_axes[missing-1], axis=1))}, examples={missing[:10].tolist()}')
        gmsh.model.mesh.clear(gmsh.model.getEntities(2))
        log_start = len(gmsh.logger.get())
        invalid_charts = []
        meshing_warnings = []
        try:
            gmsh.model.mesh.generate(2)
            import re
            messages = gmsh.logger.get()[log_start:]
            meshing_warnings = [message for message in messages if message.lower().startswith('warning')]
            invalid_charts = sorted({int(match.group(1)) for message in meshing_warnings
                if (match := re.search(r'elements remain invalid in surface (\d+)',message))})
            if invalid_charts:
                raise RuntimeError(f'Gmsh reported invalid elements in surfaces {invalid_charts}')
        except Exception as error:
            if _retry_depth >= 4:
                raise RuntimeError(f'Surface meshing failed after chart subdivision: {error}') from error
            from types import SimpleNamespace
            from .surface_charts import disk_charts
            base_keys = np.sort(interfaces.grain_pairs,axis=1) if _patch_keys is None else np.asarray(_patch_keys)
            partitions = np.zeros(len(triangles),dtype=int)
            for chart_index,ids in enumerate(source_chart_ids.values(),1):
                partitions[ids]=chart_index
            if invalid_charts:
                retry_tags = invalid_charts
                limit = 32 if _retry_depth <= 1 else 1
            else:
                # Continue meshing with each failed chart temporarily hidden.
                # Collect all failures instead of subdividing every good chart.
                import re
                retry_tags=[]
                while True:
                    messages=gmsh.logger.get()[log_start:]
                    tags=[int(m.group(1)) for message in messages
                          if (m:=re.search(r'Meshing surface (\d+)',message))]
                    tag=next((t for t in reversed(tags) if t in source_chart_ids and t not in retry_tags),None)
                    if tag is None:
                        empty=[t for t in source_chart_ids if t not in retry_tags and
                               not len(gmsh.model.mesh.getElementsByType(2,t)[0])]
                        if not empty:raise
                        tag=min(empty)
                    retry_tags.append(tag)
                    gmsh.model.setVisibility([(2,tag)],0)
                    gmsh.option.setNumber('Mesh.MeshOnlyVisible',1)
                    log_start=len(gmsh.logger.get())
                    try:
                        gmsh.model.mesh.generate(2)
                        # Some later charts may complete with invalid-element warnings.
                        for message in gmsh.logger.get()[log_start:]:
                            m=re.search(r'elements remain invalid in surface (\d+)',message)
                            if m and int(m.group(1)) not in retry_tags:retry_tags.append(int(m.group(1)))
                        break
                    except Exception:
                        if len(retry_tags)>=len(source_chart_ids):break
                limit=32 if _retry_depth<=1 else 1
            partition = len(source_chart_ids)+1
            first_replacement = partition
            repaired_triangles = 0
            for tag in retry_tags:
                ids = source_chart_ids[tag]
                repaired_triangles += len(ids)
                for group in disk_charts(triangles[ids],max_triangles=limit):
                    partitions[ids[group]] = partition;partition += 1
            retry_source = SimpleNamespace(**vars(interfaces));retry_source.triangles = triangles
            result = remesh_interfaces_gmsh(retry_source,mesh_size,verbose,algorithm,
                check_intersections=check_intersections,intersection_retries=intersection_retries,rve_dimensions=rve_dimensions,minimum_facet_angle=minimum_facet_angle,
                _patch_keys=np.column_stack((base_keys,partitions)),_oriented=True,_retry_depth=_retry_depth+1)
            result.grain_pairs = result.grain_pairs[:,:base_keys.shape[1]]
            result.report.setdefault('parametrization_repairs',[]).insert(0,{
                'stage':'surface_generation','replacement_disk_charts':partition-first_replacement,
                'source_triangles':int(repaired_triangles),'error':str(error)})
            result.report['grain_pairs'] = len(np.unique(base_keys,axis=0))
            return result
        node_tags, coords, _ = gmsh.model.mesh.getNodes()
        node_tags = np.asarray(node_tags, dtype=int)
        sort = np.argsort(node_tags)
        node_tags, new_points = node_tags[sort], np.asarray(coords).reshape(-1, 3)[sort]
        faces, labels, face_charts = [], [], []
        reoriented_charts = []
        source_node_tags = np.arange(1, len(points)+1)
        for tag, i in surface_pairs.items():
            _, conn = gmsh.model.mesh.getElementsByType(2, tag)
            conn = np.asarray(conn, dtype=int).reshape(-1, 3)
            if not len(conn):
                raise RuntimeError(f'Gmsh produced no triangles on interface {pairs[i].tolist()}')
            new_faces = np.searchsorted(node_tags, conn)
            source_faces = source_charts[tag]
            # Source interior nodes need not survive remeshing. Use original
            # node tags for boundary comparisons and separate coordinate arrays
            # for the closed-shell fallback.
            new_faces, flip = _align_chart_orientation(
                source_faces, points, new_faces, new_points, source_node_tags, node_tags)
            if flip:
                reoriented_charts.append(int(tag))
            faces.append(new_faces)
            labels.append(np.tile(pairs[i], (len(conn), 1)))
            face_charts.append(np.full(len(new_faces),tag,dtype=int))
        faces, labels = np.vstack(faces), np.vstack(labels)
        frozen_tags = np.array(sorted(frozen))
        positions = np.searchsorted(node_tags, frozen_tags)
        if np.any(positions >= len(node_tags)) or not np.array_equal(node_tags[positions], frozen_tags):
            raise RuntimeError('Gmsh discarded shared boundary nodes')
        maximum_curve_error = float(np.max(np.abs(new_points[positions]-np.array([frozen[t] for t in frozen_tags])))) if len(positions) else 0.
        if maximum_curve_error > 1e-12:
            raise RuntimeError('Gmsh moved a shared curve or junction point')
        if not np.allclose(new_points[np.searchsorted(node_tags, constrained)], points[constrained-1], rtol=0, atol=1e-12):
            raise RuntimeError('A source junction point or RVE trace moved')
        curve_edges = np.asarray(curve_edges, dtype=int).reshape(-1, 2)
        new_curves = np.searchsorted(node_tags, curve_edges)
        # Verify each chart boundary uses exactly the retained shared curve mesh.
        for tag in surface_pairs:
            _, conn = gmsh.model.mesh.getElementsByType(2, tag)
            f = np.asarray(conn, dtype=int).reshape(-1, 3)
            e, counts = np.unique(np.sort(f[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1), axis=0, return_counts=True)
            actual = {tuple(edge) for edge in e[counts == 1]}
            expected = set()
            for dim, curve_tag in gmsh.model.getBoundary([(2, tag)], oriented=False):
                if dim == 1:
                    _, conn = gmsh.model.mesh.getElementsByType(1, curve_tag)
                    expected.update(map(tuple, np.sort(np.asarray(conn, dtype=int).reshape(-1, 2), axis=1)))
            if actual != expected or np.any(counts > 2):
                raise RuntimeError(f'Nonconforming boundary on Gmsh surface {tag}')
            for dim, point_tag in gmsh.model.mesh.getEmbedded(2, tag):
                if dim == 0:
                    point_nodes = gmsh.model.mesh.getNodes(0, point_tag)[0]
                    if not set(map(int, point_nodes)).issubset(map(int, f.ravel())):
                        raise RuntimeError(f'Frozen junction is disconnected from Gmsh surface {tag}')
        p = new_points[faces]
        area2 = np.linalg.norm(np.cross(p[:, 1]-p[:, 0], p[:, 2]-p[:, 0]), axis=1)
        if np.any(~np.isfinite(new_points)) or np.any(~np.isfinite(area2)) or np.any(area2 <= 0):
            raise RuntimeError('Gmsh produced degenerate triangles')
        if check_intersections:
            from .surface_intersections import find_surface_intersections
            hits=find_surface_intersections(new_points,faces)
            bad_geometry=np.zeros(len(faces),dtype=bool)
            bad_geometry[np.unique(hits)]=True
            from .facet_angles import small_facet_angles
            folds,angles=small_facet_angles(new_points,faces,minimum_facet_angle)
            if len(folds):bad_geometry[np.unique(folds)]=True
            tangencies=0
            if rve_dimensions is not None:
                extent=np.asarray(rve_dimensions,dtype=float)
                if extent.shape!=(3,) or np.any(extent<=0) or np.any(~np.isfinite(extent)):raise ValueError('Invalid RVE dimensions')
                tol=1e-9*max(1.,float(extent.max()))
                coplanar=np.zeros(len(faces),dtype=bool)
                normals=np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0])
                for axis in range(3):
                    angle=np.degrees(np.arctan2(np.linalg.norm(np.delete(normals,axis,axis=1),axis=1),np.abs(normals[:,axis])))
                    for value in (0.,extent[axis]):
                        on=np.abs(new_points[faces,axis]-value)<=tol
                        coplanar |= np.all(on,axis=1) | ((on.sum(axis=1)>=2)&(angle<minimum_facet_angle))
                coplanar &= labels[:,0]!=labels[:,1]
                tangencies=int(coplanar.sum());bad_geometry |= coplanar
                bad_geometry |= np.any((p < -tol)|(p > extent+tol),axis=(1,2))
            if np.any(bad_geometry):
                if not intersection_retries:
                    raise RuntimeError(f'{len(hits)} intersections, {len(folds)} small facet angles and {tangencies} cap tangencies remain after Gmsh chart repair')
                # Restore the exact piecewise-planar source on intersecting
                # charts. Every triangle becomes a chart with shared edges;
                # other charts retain their previous computational boundaries.
                from types import SimpleNamespace
                affected=np.unique(np.concatenate(face_charts)[bad_geometry])
                base_keys=np.sort(interfaces.grain_pairs,axis=1) if _patch_keys is None else np.asarray(_patch_keys)
                partitions=np.zeros(len(triangles),dtype=int);partition=1
                for tag,ids in source_chart_ids.items():
                    if tag in affected:
                        partitions[ids]=np.arange(partition,partition+len(ids));partition+=len(ids)
                    else:partitions[ids]=partition;partition+=1
                retry_source=SimpleNamespace(**vars(interfaces));retry_source.triangles=triangles
                # Geometric restoration creates new charts. Give them a fresh
                # bounded parametrization budget; intersection_retries remains
                # the separate limit on geometric repair passes.
                result=remesh_interfaces_gmsh(retry_source,mesh_size,verbose,algorithm,
                    check_intersections=True,intersection_retries=intersection_retries-1,
                    rve_dimensions=rve_dimensions,minimum_facet_angle=minimum_facet_angle,
                    _patch_keys=np.column_stack((base_keys,partitions)),_oriented=True,_retry_depth=1)
                result.grain_pairs=result.grain_pairs[:,:base_keys.shape[1]]
                result.report.setdefault('geometric_repairs',[]).insert(0,dict(
                    detected_intersections=len(hits),small_facet_angles=len(folds),cap_tangencies=tangencies,restored_source_charts=len(affected)))
                result.report['grain_pairs']=len(np.unique(base_keys,axis=0))
                return result
        report = {'input_triangles': len(triangles), 'output_triangles': len(faces),
                  'requested_algorithm': int(algorithm),
                  'intersections_checked': bool(check_intersections),
                  'remaining_intersections': 0 if check_intersections else None,
                  'remaining_small_facet_angles': 0 if check_intersections else None,
                  'minimum_facet_angle': float(minimum_facet_angle),
                  'source_chart_orientation_verified': True,
                  'reoriented_surface_charts': reoriented_charts,
                  'grain_pairs': len(pairs), 'surface_charts': len(surface_pairs),
                  'preserved_curve_nodes': len(frozen), 'shared_boundaries_verified': True,
                  'curve_coordinates_preserved': True, 'mesh_size': float(mesh_size),
                  'rve_trace_nodes_preserved': int(np.sum(np.any(interfaces.fixed_axes, axis=1))),
                  'frozen_junction_points_preserved': int(np.sum(interfaces.node_kind == 3)),
                  'surface_meshing_warnings': meshing_warnings,
                  'surface_warning_capture_available': bool(messages),
                  'maximum_curve_coordinate_error': maximum_curve_error}
        return RemeshedInterfaces(new_points, faces, labels, new_curves, report)
    finally:
        gmsh.model.setCurrent(name)
        gmsh.model.remove()
        for key, value in previous_options.items():
            gmsh.option.setNumber(key, value)
        if owned:
            gmsh.logger.stop()
            gmsh.finalize()
        elif previous_model in gmsh.model.list():
            gmsh.model.setCurrent(previous_model)
