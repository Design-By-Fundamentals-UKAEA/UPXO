"""The table that ties each feature's element set, material and orientation to its place in the hierarchy.

``features.csv`` has one row per feature that holds elements: the names of its element set and material,
the UMAT grain number, the Bunge-Euler angles, its PAG, packet, block and sub-block (as ids and as the
names of their element sets), and what the mesh made of it: tetrahedra, volume and share of the volume.
The same ids ride along as cell data in the mesh files, so a result can be summed by hierarchy level.

Datasets that name their own attributes (twinned FCC: role and host family) give those columns instead.
"""
import csv
from pathlib import Path
import numpy as np

from .twinned import FAMILY_PREFIX

_CHUNK = 250_000
LEVEL_IDS = ('pag_id', 'packet_id', 'block_id', 'subblock_id')


def tet_volumes(points, tetrahedra):
    """Volume of each tetrahedron in the units of ``points``."""
    out = np.empty(len(tetrahedra))
    for start in range(0, len(tetrahedra), _CHUNK):
        v = points[tetrahedra[start:start + _CHUNK]]
        out[start:start + _CHUNK] = np.einsum('ij,ij->i', v[:, 1] - v[:, 0],
                                              np.cross(v[:, 2] - v[:, 0], v[:, 3] - v[:, 0])) / 6
    return out


def _numbered(keys):
    """{key: 1-based number} in ascending key order."""
    return {key: i for i, key in enumerate(sorted(set(keys)), start=1)}


def level_keys(feature):
    """The keys of a feature's PAG, packet, block and sub-block (``None`` where the level does not apply)."""
    return (feature.pag_id, feature.pck_key, feature.blk_key, feature.key if feature.sb_local is not None else None)


def id_columns(hierarchy, labels=None):
    """``{column: {label: int}}``: the ids a mesh file carries for each feature, ``-1`` where none applies.

    FM steel: ``pag_id`` is the PAG number of the element set name; ``packet_id``, ``block_id`` and ``subblock_id``
    number the packets, blocks and sub-blocks present, from 1 in ascending key order. A dataset that names its
    own attributes gives those (twinned FCC: ``family_id``, the host label, and ``role_code``).
    """
    labels = sorted(hierarchy.features if labels is None else labels)
    if hierarchy.extra:
        return {'family_id': {l: int(hierarchy.extra[l]['family']) for l in labels},
                'role_code': {l: int(hierarchy.extra[l]['role_code']) for l in labels}}
    keys = {l: level_keys(hierarchy.features[l]) for l in labels}
    numbers = [_numbered(k[i] for k in keys.values() if k[i] is not None) for i in range(4)]
    out = {}
    for i, name in enumerate(LEVEL_IDS):
        out[name] = {l: (-1 if keys[l][i] is None else (keys[l][i] if i == 0 else numbers[i][keys[l][i]]))
                     for l in labels}
    return out


def feature_table(hierarchy, names, material_names, tets_of, volume_of, prefixes=None):
    """``(columns, rows)`` of the feature table.

    ``names`` and ``material_names`` map a label to its element set and material; ``tets_of`` and ``volume_of``
    map a label to the tetrahedra and the volume the mesh made of it (output units cubed).
    """
    labels = [f.label for f in hierarchy.ordered_features() if f.label in tets_of]
    ids = id_columns(hierarchy, labels)
    total = float(sum(volume_of[l] for l in labels)) or 1.0
    head = ['label', 'elset', 'material', 'umat_grain_number', 'phi1_deg', 'Phi_deg', 'phi2_deg']
    tail = ['voxels_raw', 'tetrahedra', 'volume', 'volume_fraction']
    if hierarchy.extra:
        middle = ['role', 'family_id', 'family_elset']
        extra = lambda l, f: [hierarchy.extra[l]['role'], ids['family_id'][l],
                              f"{FAMILY_PREFIX}{ids['family_id'][l]}" if ids['family_id'][l] >= 0 else '']
    else:
        middle = list(LEVEL_IDS) + ['pag_elset', 'packet_elset', 'block_elset']
        extra = lambda l, f: ['' if ids[c][l] < 0 else ids[c][l] for c in LEVEL_IDS] + _level_elsets(hierarchy, f, prefixes)
    rows = []
    for label in labels:
        feature = hierarchy.features[label]
        rows.append([label, names[label], material_names[label], feature.global_id,
                     *(f'{a:.6f}' for a in feature.euler), *extra(label, feature),
                     hierarchy.voxel_counts.get(label, ''), tets_of[label], f'{volume_of[label]:.9g}',
                     f'{volume_of[label] / total:.9g}'])
    return head + middle + tail, rows


def _level_elsets(hierarchy, feature, prefixes=None):
    from .export import elset_name
    from . import constants as C

    prefixes = prefixes or C.ELSET_PREFIXES
    pck, blk = feature.pck_key, feature.blk_key
    return [elset_name(prefixes, 'pag', (feature.pag_id,)),
            '' if pck is None else elset_name(prefixes, 'pck', pck),
            '' if blk is None else elset_name(prefixes, 'blk', blk)]


def write_feature_table(path, columns, rows):
    with Path(path).open('w', encoding='ascii', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)
    return Path(path)
