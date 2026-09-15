"""Partitioned, mesh-only Abaqus export of existing linear tetrahedra."""
from pathlib import Path
import json
import re
import numpy as np


def export_tets_abaqus(mesh, directory, *, prefix='grn_', units='unspecified',
                       expected_grains=None):
    """Write C3D4 nodes, connectivity and disjoint per-grain ELSETs.

    Coordinates/connectivity are preserved, numbering becomes one-based.
    A new output directory is required so earlier exports cannot be overwritten.
    This validates export inputs; it does not rerun mesh conformity verification.
    """
    p = np.asarray(mesh.points)
    t = np.asarray(mesh.tetrahedra)
    g = np.asarray(mesh.grain_ids)
    if p.ndim != 2 or p.shape[1] != 3 or not len(p) or not np.isfinite(p).all():
        raise ValueError('Expected finite Nx3 points')
    if t.ndim != 2 or t.shape[1] != 4 or not len(t) or t.dtype.kind not in 'iu':
        raise ValueError('Expected nonempty integer Mx4 tetrahedra')
    if t.min() < 0 or t.max() >= len(p):
        raise ValueError('Connectivity references missing nodes')
    if g.shape != (len(t),) or g.dtype.kind not in 'iu' or np.any(g < 0):
        raise ValueError('Expected one nonnegative grain ID per tetrahedron')
    ids, counts = np.unique(g, return_counts=True)
    if expected_grains is not None and not np.array_equal(ids, np.unique(expected_grains)):
        raise ValueError('Mesh grain IDs differ from expected retained grains')
    if not isinstance(prefix, str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*', prefix):
        raise ValueError('Prefix must begin with a letter and contain letters, digits or underscores')
    if len(prefix + str(int(ids.max()))) > 80:
        raise ValueError('ELSET name exceeds 80 characters')
    volume_sum = 0.; volume_min = float('inf')
    for start in range(0, len(t), 100000):
        v = p[t[start:start + 100000]]
        volumes = np.einsum('ij,ij->i', v[:, 1]-v[:, 0],
                           np.cross(v[:, 2]-v[:, 0], v[:, 3]-v[:, 0])) / 6
        if not np.isfinite(volumes).all() or np.any(volumes <= 0):
            raise ValueError('Nonpositive or nonfinite tetrahedron volume')
        volume_sum += float(volumes.sum()); volume_min = min(volume_min, float(volumes.min()))
    quality = np.asarray(mesh.quality)
    if quality.shape != (len(t),) or not np.isfinite(quality).all():
        raise ValueError('Expected one finite quality value per tetrahedron')
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=False)
    with (out / '01_nodes.inp').open('w', encoding='ascii') as f:
        f.write('*NODE\n')
        for i, xyz in enumerate(p, 1):
            f.write(f'{i}, {xyz[0]:.17g}, {xyz[1]:.17g}, {xyz[2]:.17g}\n')
    with (out / '02_elements.inp').open('w', encoding='ascii') as f:
        f.write('*ELEMENT, TYPE=C3D4\n')
        for i, row in enumerate(t, 1):
            f.write(f'{i}, '+', '.join(str(int(n)+1) for n in row)+'\n')
    order = np.argsort(g, kind='stable')
    with (out / '03a_elsets_grains.inp').open('w', encoding='ascii') as f:
        offset = 0
        for gid, count in zip(ids, counts):
            f.write(f'*ELSET, ELSET={prefix}{int(gid)}\n')
            elements = order[offset:offset + count] + 1
            for start in range(0, len(elements), 16):
                f.write(', '.join(map(str, elements[start:start+16]))+'\n')
            offset += count
    (out / 'model_master.inp').write_text(
        '*HEADING\n** UPXO CM04 tetrahedral mesh only\n'
        '*INCLUDE, INPUT=01_nodes.inp\n*INCLUDE, INPUT=02_elements.inp\n'
        '*INCLUDE, INPUT=03a_elsets_grains.inp\n', encoding='ascii')
    info = dict(nodes=len(p), tetrahedra=len(t), grains=len(ids), element_type='C3D4',
                coordinate_units=str(units), bounds=[p.min(axis=0).tolist(), p.max(axis=0).tolist()],
                total_tet_volume=volume_sum, minimum_tet_volume=volume_min,
                minSICN_min=float(quality.min()), minSICN_median=float(np.median(quality)),
                minSICN_max=float(quality.max()),
                element_sets={prefix+str(int(gid)): int(n) for gid, n in zip(ids, counts)},
                source_verification_report=mesh.report)
    def encode(value):
        if isinstance(value, np.ndarray): return value.tolist()
        if isinstance(value, np.generic): return value.item()
        return str(value)
    (out / 'mesh_info.json').write_text(json.dumps(info, indent=2, default=encode), encoding='utf8')
    rows = '\n'.join(f'| {prefix}{int(gid)} | {int(n)} |' for gid, n in zip(ids, counts))
    (out / 'README.md').write_text(f'''# CM04 Abaqus tetrahedral mesh

Open `model_master.inp`; keep all include files together. This follows the
partitioned UPXO twinned/FM-steel file layout, restricted to mesh and grain sets.

- Nodes: {len(p):,}; tetrahedra: {len(t):,}; grains: {len(ids):,}.
- Element type: C3D4 (four-node linear tetrahedron).
- Coordinate units: {units}. No scaling or axis transformation was applied.
- Bounds (minimum/maximum XYZ): {info['bounds']}.
- Sum of tet volumes: {volume_sum:.17g}; minimum: {volume_min:.17g} (coordinate units cubed).
- Gmsh minSICN minimum/median/maximum: {quality.min():.6g} / {np.median(quality):.6g} / {quality.max():.6g}.

`01_nodes.inp` contains nodes; `02_elements.inp` contains tetrahedra only;
`03a_elsets_grains.inp` contains one disjoint set per retained grain.
Each tetrahedron belongs to exactly one grain set. Disconnected parts of a
grain share its set. Grain IDs are the current shuffled/retained IDs, not
original feature names. Node/element labels are source array row indices + 1.
Shared interface nodes are retained; no surface triangles are exported.

No materials, sections, loads, boundary conditions or analysis steps are included.
This is a mesh import deck, not a complete runnable analysis.
`mesh_info.json` includes set sizes and the source verification report; export
checks positive volumes and expected IDs but does not certify mesh conformity
or attainment of the requested quality threshold.

| Element set | Tetrahedra |
|---|---:|
{rows}
''', encoding='utf8')
    return out
