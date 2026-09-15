"""Conforming tetrahedral meshing and verification of repaired grain shells."""
from dataclasses import dataclass
import uuid
import numpy as np
from .tet_validation import validate_tet_surfaces


@dataclass
class GrainTetrahedra:
    points: np.ndarray
    tetrahedra: np.ndarray
    grain_ids: np.ndarray
    quality: np.ndarray
    report: dict


def mesh_repaired_rve_gmsh(surface, mesh_size=1.5, verbose=False,
                           optimize_netgen=True, minimum_quality=.05):
    """Mesh all grains together; verify exact surface conformity and volumes.

    Disconnected grain pieces receive separate Gmsh volumes with the same
    output grain ID. Negative shells are attached as cavities to their enclosing
    positive shell. Existing triangles stay fixed. Quality is Gmsh minSICN.
    Optional Netgen optimization targets only volumes below minimum_quality.
    Readiness requires conformity, positive volumes, complete grain coverage,
    volume agreement and the requested quality threshold. This is a meshing
    verification, not a general surface-intersection certificate.
    """
    import gmsh
    if not np.isfinite(mesh_size) or mesh_size <= 0:
        raise ValueError('mesh_size must be finite and positive')
    if not 0 <= minimum_quality <= 1:
        raise ValueError('minimum_quality must lie between 0 and 1')
    validation = validate_tet_surfaces(surface, check_intersections=True)
    if validation['blockers']:
        raise ValueError('Repair surface blockers first: '+'; '.join(validation['blockers']))
    p, f = surface.points, surface.triangles
    pairs = surface.grain_pairs
    keys, inverse = np.unique(np.column_stack((pairs, surface.rve_face)), axis=0, return_inverse=True)
    owned = not gmsh.isInitialized()
    if owned: gmsh.initialize()
    previous_model = gmsh.model.getCurrent()
    options = {'General.Terminal': int(verbose), 'Mesh.MeshOnlyEmpty': 1,
               'Mesh.MeshSizeMax': mesh_size, 'Mesh.MeshSizeMin': 0,
               'Mesh.Algorithm3D': 1, 'Mesh.ElementOrder': 1,
               'Mesh.Optimize': 1, 'Mesh.OptimizeNetgen': 0,
               'Mesh.Renumber': 0}
    previous_options = {k: gmsh.option.getNumber(k) for k in options}
    name = 'CM02_tets_'+uuid.uuid4().hex
    gmsh.model.add(name)
    try:
        for key,value in options.items():gmsh.option.setNumber(key,value)
        for i in range(len(keys)):
            tag = gmsh.model.addDiscreteEntity(2)
            if i == 0:gmsh.model.mesh.addNodes(2,tag,np.arange(1,len(p)+1),p.ravel())
            ids = np.flatnonzero(inverse == i)
            gmsh.model.mesh.addElementsByType(tag,2,ids+1,(f[ids]+1).ravel())
        gmsh.model.mesh.reclassifyNodes()
        gmsh.model.mesh.createTopology(True,True)
        surfaces, nodes, triangles = {}, {}, {}
        for _, tag in gmsh.model.getEntities(2):
            ids, conn = gmsh.model.mesh.getElementsByType(2,tag)
            patch_ids = np.unique(inverse[np.asarray(ids,dtype=int)-1])
            if len(patch_ids) != 1:raise RuntimeError('Gmsh mixed labelled patches')
            surfaces[tag] = keys[patch_ids[0]]
            triangles[tag] = np.asarray(conn,dtype=int).reshape(-1,3)-1
            nodes[tag] = set(triangles[tag].ravel())
        # Surface meshes are already complete: volume recovery uses their
        # triangles directly, avoiding unnecessary surface parametrization.
        volume_grain = {}
        for gid in np.unique(pairs):
            signed_tags = [tag*(1 if key[0] == gid else -1)
                           for tag,key in surfaces.items() if gid in key[:2]]
            remaining = set(map(abs,signed_tags))
            shells = []
            while remaining:
                component = {remaining.pop()}
                shell_nodes = set().union(*(nodes[t] for t in component))
                while True:
                    added = {t for t in remaining if shell_nodes & nodes[t]}
                    if not added:break
                    remaining -= added;component |= added
                    shell_nodes.update(set().union(*(nodes[t] for t in added)))
                tags = [t for t in signed_tags if abs(t) in component]
                tri = np.vstack([triangles[abs(t)] if t > 0 else triangles[-t][:, [0,2,1]] for t in tags])
                xyz = p[tri]
                volume = float(np.einsum('ij,ij->i',xyz[:,0],np.cross(xyz[:,1],xyz[:,2])).sum()/6)
                shells.append(dict(volume=volume,loop=gmsh.model.geo.addSurfaceLoop(tags),tri=tri))
            outer = [s for s in shells if s['volume'] > 0]
            inner = [s for s in shells if s['volume'] < 0]
            if not outer:raise RuntimeError(f'Grain {gid} has no outer shell')
            holes = [[] for _ in outer]
            for cavity in inner:
                point = p[cavity['tri'][0,0]]
                containers = []
                for j,shell in enumerate(outer):
                    a,b,c = (p[shell['tri']]-point).transpose(1,0,2)
                    la,lb,lc = np.linalg.norm(a,axis=1),np.linalg.norm(b,axis=1),np.linalg.norm(c,axis=1)
                    num = np.einsum('ij,ij->i',a,np.cross(b,c))
                    den = la*lb*lc+np.einsum('ij,ij->i',a,b)*lc+np.einsum('ij,ij->i',b,c)*la+np.einsum('ij,ij->i',c,a)*lb
                    if abs(2*np.arctan2(num,den).sum()) > 2*np.pi:containers.append(j)
                if len(containers) != 1:raise RuntimeError(f'Ambiguous cavity containment in grain {gid}')
                holes[containers[0]].append(cavity['loop'])
            for shell,cavities in zip(outer,holes):
                tag = gmsh.model.geo.addVolume([shell['loop']]+cavities)
                volume_grain[tag] = int(gid)
        gmsh.model.geo.synchronize()
        for gid in np.unique(pairs):
            volumes = [v for v,g in volume_grain.items() if g == gid]
            physical = gmsh.model.addPhysicalGroup(3,volumes)
            gmsh.model.setPhysicalName(3,physical,f'grain_{gid}')
        gmsh.model.mesh.generate(3)
        def snapshot():
            node_tags,coords,_ = gmsh.model.mesh.getNodes()
            node_tags = np.asarray(node_tags,dtype=int)
            order = np.argsort(node_tags)
            node_tags,coords = node_tags[order],np.asarray(coords).reshape(-1,3)[order]
            meshes = {}
            for volume,gid in volume_grain.items():
                elements,conn = gmsh.model.mesh.getElementsByType(4,volume)
                if not len(elements):raise RuntimeError(f'No tetrahedra in grain {gid}, volume {volume}')
                used,local = np.unique(np.asarray(conn,dtype=int),return_inverse=True)
                meshes[volume] = (coords[np.searchsorted(node_tags,used)],local.reshape(-1,4),
                                  np.asarray(gmsh.model.mesh.getElementQualities(elements,'minSICN')))
            return meshes
        baseline = snapshot()
        chosen = baseline
        retained_volumes = []
        if optimize_netgen:
            poor_volumes = [(3,v) for v,data in baseline.items() if data[2].min() < minimum_quality]
            if poor_volumes:
                gmsh.model.mesh.optimize('Netgen',dimTags=poor_volumes)
                chosen = snapshot()
                # Some Gmsh versions optimize more than the requested entities.
                # Retain the baseline for any volume whose minimum deteriorates.
                for v in chosen:
                    if chosen[v][2].min() < baseline[v][2].min():
                        chosen[v] = baseline[v];retained_volumes.append(v)
        # Assemble volumes using the original surface node IDs. Interior nodes
        # belong to one volume, so baseline and optimized interiors can safely
        # be combined while all shared interface nodes stay identical.
        from scipy.spatial import cKDTree
        source_tree = cKDTree(p)
        point_blocks = [p.copy()];point_count = len(p)
        tets, labels, quality = [], [], []
        for volume,gid in volume_grain.items():
            coords,conn,q = chosen[volume]
            distance,mapping = source_tree.query(coords)
            interior = distance > 1e-10
            mapping[interior] = np.arange(point_count,point_count+int(interior.sum()))
            point_blocks.append(coords[interior]);point_count += int(interior.sum())
            tets.append(mapping[conn]);labels.append(np.full(len(conn),gid,dtype=pairs.dtype));quality.append(q)
        points = np.vstack(point_blocks)
        tets,labels,quality = np.vstack(tets),np.concatenate(labels),np.concatenate(quality)
        xyz = points[tets]
        volumes = np.einsum('ij,ij->i',xyz[:,1]-xyz[:,0],np.cross(xyz[:,2]-xyz[:,0],xyz[:,3]-xyz[:,0]))/6
        if np.any(~np.isfinite(volumes)) or np.any(volumes <= 0):raise RuntimeError('Nonpositive/nonfinite tetrahedron volumes')
        if np.any(~np.isfinite(quality)) or np.any(quality <= 0):raise RuntimeError('Invalid tetrahedron quality')
        # Match every per-grain tet boundary to the original surface by node ID.
        # This detects missing tiny components, cracks, and changed interfaces.
        source_positions = np.arange(len(p))
        per_grain = {}
        face_pattern = [[0,1,2],[0,1,3],[0,2,3],[1,2,3]]
        for gid in np.unique(pairs):
            selected = labels == gid
            tet_faces,counts = np.unique(np.sort(tets[selected][:,face_pattern].reshape(-1,3),axis=1),axis=0,return_counts=True)
            if np.any(counts > 2):raise RuntimeError(f'Nonmanifold tet faces in grain {gid}')
            expected = np.sort(source_positions[f[np.any(pairs == gid,axis=1)]],axis=1)
            expected = np.unique(expected,axis=0)
            if not np.array_equal(tet_faces[counts == 1],expected):
                raise RuntimeError(f'Tet boundary differs from repaired surface for grain {gid}')
            actual = float(volumes[selected].sum())
            expected_volume = validation['per_grain'][str(int(gid))]['signed_volume']
            if not np.isclose(actual,expected_volume,rtol=1e-8):raise RuntimeError(f'Volume mismatch in grain {gid}')
            per_grain[str(int(gid))] = dict(tetrahedra=int(selected.sum()),volume=actual,
                                            minimum_quality=float(quality[selected].min()))
        if not np.isclose(volumes.sum(),validation['expected_rve_volume'],rtol=1e-8):
            raise RuntimeError('Tetrahedron volumes do not fill the RVE')
        passed = bool(quality.min() >= minimum_quality)
        report = dict(status='TET_MESH_VERIFIED' if passed else 'TET_QUALITY_BELOW_TARGET',
                      ready=passed, grains=len(per_grain), volume_entities=len(volume_grain),
                      tetrahedra=len(tets), nodes=len(points), per_grain=per_grain,
                      positive_volumes=True, source_surface_conformity_verified=True,
                      surface_intersection_check_passed=True,
                      surface_intersection_tolerance=validation['coordinate_tolerance'],
                      total_volume=float(volumes.sum()), minimum_quality=float(quality.min()),
                      quality_percentiles_1_5_50=np.percentile(quality,[1,5,50]).tolist(),
                      quality_metric='Gmsh minSICN', quality_threshold=float(minimum_quality),
                      baseline_retained_for_volumes=retained_volumes,
                      below_quality_threshold=int(np.sum(quality < minimum_quality)))
        return GrainTetrahedra(points,tets,labels,quality,report)
    finally:
        gmsh.model.setCurrent(name);gmsh.model.remove()
        for key,value in previous_options.items():gmsh.option.setNumber(key,value)
        if owned:gmsh.finalize()
        elif previous_model in gmsh.model.list():gmsh.model.setCurrent(previous_model)
