"""Mesh files that carry the orientation of every tetrahedron: Gmsh ``.msh`` and VTK ``.vtu``.

Both hold the nodes and tetrahedra of the deck in the deck's numbering and units, the feature each
tetrahedron belongs to, the feature's Bunge-Euler angles in degrees and its IPF colour for the three
sample axes (the colour of :func:`upxo.xtalphy.crystal_orientation.ipf_color`).
"""
from pathlib import Path
import numpy as np

from upxo._sup.text_rows import write_rows

_CHUNK = 250_000
IPF_AXES = {'x': (1., 0., 0.), 'y': (0., 1., 0.), 'z': (0., 0., 1.)}


def tetrahedron_fields(tet_feature, features, names, ids=None):
    """Per-tetrahedron arrays from the feature of each tetrahedron.

    ``features`` maps a feature label to an object with ``euler`` and ``global_id``; ``names`` maps the
    label to the name of its element set; ``ids`` is ``{column: {label: int}}`` of further ids (the hierarchy
    levels, see :func:`.features.id_columns`). Returns a dictionary with ``feature_label`` (M,), ``global_id`` (M,),
    ``euler_deg`` (M, 3), ``ipf_x``/``ipf_y``/``ipf_z`` (M, 3, floats in 0..1), ``ids`` ({column: (M,)}) and
    ``names`` ({label: name}).
    """
    from upxo.xtalphy.crystal_orientation import ipf_color

    labels, inverse = np.unique(np.asarray(tet_feature, dtype=np.int64), return_inverse=True)
    euler = np.array([features[int(label)].euler for label in labels], dtype=float).reshape(-1, 3)
    global_id = np.array([features[int(label)].global_id for label in labels], dtype=np.int64)
    fields = {'feature_label': labels[inverse], 'global_id': global_id[inverse], 'euler_deg': euler[inverse],
              'names': {int(label): names[int(label)] for label in labels}}
    fields['ids'] = {name: np.array([int(table.get(int(label), -1)) for label in labels], dtype=np.int64)[inverse]
                     for name, table in (ids or {}).items()}
    for letter, direction in IPF_AXES.items():
        colours = np.array([ipf_color(row, direction) for row in euler], dtype=float).reshape(-1, 3)
        fields[f'ipf_{letter}'] = colours[inverse]
    return fields


def write_vtu(path, points, tetrahedra, fields):
    """VTK unstructured grid with the cell data of :func:`tetrahedron_fields`; IPF colours as 8-bit RGB."""
    import pyvista as pv

    t = np.asarray(tetrahedra, dtype=np.int64)
    cells = np.column_stack((np.full(len(t), 4, dtype=np.int64), t)).ravel()
    grid = pv.UnstructuredGrid(cells, np.full(len(t), 10, dtype=np.uint8), np.asarray(points, dtype=float))
    grid.cell_data['feature_label'] = fields['feature_label'].astype(np.int64)
    grid.cell_data['global_id'] = fields['global_id'].astype(np.int64)
    grid.cell_data['euler_deg'] = fields['euler_deg']
    for name, values in fields.get('ids', {}).items():
        grid.cell_data[name] = values
    for letter in IPF_AXES:
        grid.cell_data[f'ipf_{letter}'] = np.rint(fields[f'ipf_{letter}'] * 255).astype(np.uint8)
    grid.save(str(path))
    return Path(path)


def _quoted(text):
    return '"' + str(text).replace('"', "'") + '"'


def write_msh(path, points, tetrahedra, fields):
    """Gmsh MSH 2.2 ASCII: physical groups named after the element sets, element data for the orientations."""
    p = np.asarray(points, dtype=float)
    t = np.asarray(tetrahedra, dtype=np.int64) + 1
    tags = fields['feature_label']
    with Path(path).open('w', encoding='ascii') as f:
        f.write('$MeshFormat\n2.2 0 8\n$EndMeshFormat\n')
        names = fields['names']
        f.write(f'$PhysicalNames\n{len(names)}\n')
        for label in sorted(names):
            f.write(f'3 {label} {_quoted(names[label])}\n')
        f.write('$EndPhysicalNames\n')
        f.write(f'$Nodes\n{len(p)}\n')
        for start in range(0, len(p), _CHUNK):
            block = p[start:start + _CHUNK]
            ids = np.arange(start + 1, start + 1 + len(block))
            write_rows(f, np.column_stack((ids, block)), ['%d', '%.9e', '%.9e', '%.9e'], ' ')
        f.write(f'$EndNodes\n$Elements\n{len(t)}\n')
        for start in range(0, len(t), _CHUNK):
            block = t[start:start + _CHUNK]
            ids = np.arange(start + 1, start + 1 + len(block))
            tag = tags[start:start + _CHUNK]
            rows = np.column_stack((ids, np.full(len(block), 4), np.full(len(block), 2), tag, tag, block))
            write_rows(f, rows, '%d', ' ')
        f.write('$EndElements\n')
        arrays = dict(fields.get('ids', {}))
        listed = [('euler_deg', 'euler_deg', 3), ('feature_label', 'feature_label', 1),
                  ('global_id', 'global_id', 1), ('ipf_x', 'ipf_x', 3), ('ipf_y', 'ipf_y', 3), ('ipf_z', 'ipf_z', 3)]
        listed += [(name, name, 1) for name in arrays]
        for name, key, width in listed:
            values = np.asarray(arrays[key] if key in arrays else fields[key])
            values = values.reshape(len(t), width)
            f.write(f'$ElementData\n1\n{_quoted(name)}\n1\n0.0\n3\n0\n{width}\n{len(t)}\n')
            integer = values.dtype.kind in 'iu'
            for start in range(0, len(t), _CHUNK):
                block = values[start:start + _CHUNK]
                ids = np.arange(start + 1, start + 1 + len(block))
                if integer:
                    write_rows(f, np.column_stack((ids, block)), '%d', ' ')
                else:
                    write_rows(f, np.column_stack((ids, block)), ['%d'] + ['%.6f'] * width, ' ')
            f.write('$EndElementData\n')
    return Path(path)
