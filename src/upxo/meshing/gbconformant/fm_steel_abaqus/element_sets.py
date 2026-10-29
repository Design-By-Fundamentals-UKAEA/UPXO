"""Every element set and node set of an FM-steel deck, computed in one place.

:func:`deck_sets` returns the sets file by file in the order the deck writes them. The element sets are the
feature sets of the hierarchy (PAG, packet, block, sub-block), the sets that join features (twinned FCC: by
role and by host family) and the boundary-layer pair; the node sets are the six box faces and the grain-boundary
and junction nodes. A deck can also take further sets from a provider (see :mod:`.extra_sets`).
Element and node indices are 0-based; the writer adds one.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import constants as C
from .extra_sets import SetFile, SetRequest, check_files, load_provider

FACES6 = ('XMIN', 'XMAX', 'YMIN', 'YMAX', 'ZMIN', 'ZMAX')
BASIC_FAMILIES = ('levels', 'groups', 'layers', 'bc', 'boundaries')


@dataclass
class DeckSets:
    files: list
    skipped: list
    summary: dict

    def file(self, name):
        return next((f for f in self.files if f.file == name), None)

    def element_sets(self):
        """``{name: 0-based element indices}`` of every element set."""
        return {name: ids for f in self.files if f.keyword == 'Elset' for name, ids in f.entries}

    def node_sets(self):
        return {name: ids for f in self.files if f.keyword == 'Nset' for name, ids in f.entries}


def _groups(keys):
    """``{key: ascending 0-based indices}`` per distinct key."""
    keys = np.asarray(keys)
    if not len(keys):
        return {}
    order = np.argsort(keys, kind='stable')
    sorted_keys = keys[order]
    cuts = np.flatnonzero(np.diff(sorted_keys)) + 1
    starts, stops = np.concatenate(([0], cuts)), np.concatenate((cuts, [len(keys)]))
    return {int(sorted_keys[a]): np.sort(order[a:b]) for a, b in zip(starts, stops)}


def box_face_nodes(points, face_tolerance=1e-8):
    """``({face: 0-based node indices}, tolerance)`` of the nodes on each of the six box faces."""
    p = np.asarray(points, dtype=float)
    low, high = p.min(axis=0), p.max(axis=0)
    tol = face_tolerance * float((high - low).max())
    out = {}
    for axis, (lo_key, hi_key) in enumerate((('XMIN', 'XMAX'), ('YMIN', 'YMAX'), ('ZMIN', 'ZMAX'))):
        out[lo_key] = np.flatnonzero(np.abs(p[:, axis] - low[axis]) <= tol)
        out[hi_key] = np.flatnonzero(np.abs(p[:, axis] - high[axis]) <= tol)
    return out, tol


def deck_sets(points, elements, assigned, hierarchy, settings, *, mesh_voxel_length=1.0, families=None,
              face_nodes=None) -> DeckSets:
    """The sets of the deck. ``points``/``elements`` are the deck's (compacted) mesh, ``assigned`` the feature
    of every element; ``families`` limits the work to some families (default all)."""
    from .export import elset_name, feature_elset_name
    from .hierarchy import parent_tables

    wanted = set(BASIC_FAMILIES if families is None else families)
    p = np.asarray(points, dtype=float)
    e = np.asarray(elements, dtype=np.int64)
    a = np.asarray(assigned, dtype=np.int64)
    n_nodes = len(p)
    files, skipped, summary = [], [], {}
    level = hierarchy.level
    pag_lut, pck_lut, pck_keys, blk_lut, blk_keys = parent_tables(hierarchy)
    leaf = _groups(a)
    prefixes = settings.elset_prefixes
    has_pck, has_blk = level in ('pck', 'blk', 'sblk'), level in ('blk', 'sblk')

    if 'levels' in wanted:
        sets = _groups(pag_lut[a])
        entries = []
        for pag in hierarchy.pag_order:
            if pag in sets:
                named = hierarchy.names.get(pag) if level == 'pag' else None
                entries.append((named or elset_name(prefixes, 'pag', (pag,)), sets[pag]))
        files.append(SetFile('03a_elsets_pag.inp', 'Elset', 'PAG elsets', 'levels', entries))
        for present, lv, lut, keys, order_of, file, title in (
                (has_pck, 'pck', pck_lut, pck_keys, hierarchy.pck_order, '03b_elsets_pck.inp', 'Packet elsets'),
                (has_blk, 'blk', blk_lut, blk_keys, hierarchy.blk_order, '03c_elsets_blk.inp', 'Block elsets')):
            if not present:
                continue
            sets = _groups(lut[a])
            rank = {key: i for i, key in enumerate(order_of)}
            entries = [(elset_name(prefixes, lv, key), sets[index])
                       for index, key in sorted(enumerate(keys), key=lambda kv: rank[kv[1]]) if index in sets]
            files.append(SetFile(file, 'Elset', title, 'levels', entries))
        if level == 'sblk':
            entries = [(feature_elset_name(hierarchy, prefixes, f), leaf[f.label])
                       for f in hierarchy.ordered_features() if f.label in leaf]
            files.append(SetFile('03d_elsets_subblk.inp', 'Elset', 'Sub-block elsets', 'levels', entries))
    if 'groups' in wanted and hierarchy.groups:
        entries = []
        for name, members in hierarchy.groups.items():
            ids = [leaf[label] for label in members if label in leaf]
            if ids:
                entries.append((name, np.sort(np.concatenate(ids))))
        files.append(SetFile('03e_elsets_groups.inp', 'Elset',
                             'Element sets that join features (by role and by host family)', 'groups', entries))

    boundary = None
    if ('layers' in wanted and settings.layer_sets) or ('boundaries' in wanted and settings.boundary_sets):
        from .boundary_sets import boundary_nodes
        boundary = boundary_nodes(e, a, n_nodes)
    if 'layers' in wanted and settings.layer_sets:
        touches = np.zeros(n_nodes, dtype=bool)
        touches[boundary.grain_boundary] = True
        in_layer = touches[e].any(axis=1)
        layer_ids, interior_ids = np.flatnonzero(in_layer), np.flatnonzero(~in_layer)
        entries = [(name, ids) for name, ids in ((settings.prefix_gb_layer, layer_ids),
                                                 (settings.prefix_interior, interior_ids)) if len(ids)]
        files.append(SetFile('03f_elsets_layers.inp', 'Elset',
                             'Elements with a node on a grain boundary, and the elements with none', 'layers', entries))
        summary['layer_sets'] = {'gb_layer_elements': int(len(layer_ids)), 'interior_elements': int(len(interior_ids)),
                                 'gb_layer_fraction': float(len(layer_ids) / len(e))}

    if 'bc' in wanted:
        if face_nodes is None:
            face_nodes, _tol = box_face_nodes(p, settings.face_tolerance)
        files.append(SetFile('04_nsets_bc.inp', 'Nset', '', 'bc',
                             [(settings.nset_names[key], face_nodes[key]) for key in FACES6]))
    if 'boundaries' in wanted and settings.boundary_sets:
        entries = [(name, nodes) for name, nodes in ((C.GRAIN_BOUNDARY_NSET, boundary.grain_boundary),
                                                     (C.JUNCTION_NSET, boundary.junction)) if len(nodes)]
        for feature in hierarchy.ordered_features():
            nodes = boundary.by_feature.get(feature.label)
            if nodes is not None and len(nodes):
                entries.append((C.FEATURE_BOUNDARY_NSET_PREFIX + feature_elset_name(hierarchy, prefixes, feature), nodes))
        files.append(SetFile('04b_nsets_boundaries.inp', 'Nset',
                             'Grain-boundary and junction node sets (nodes shared by 2 or more / 3 or more features)',
                             'boundaries', entries))
        summary['boundary_sets'] = boundary.counts()

    if getattr(settings, 'set_provider', ''):
        if face_nodes is None:
            face_nodes, _tol = box_face_nodes(p, settings.face_tolerance)
        request = SetRequest(p, e, a, hierarchy, settings, float(mesh_voxel_length), face_nodes, families)
        extra = check_files(settings.set_provider, load_provider(settings.set_provider).extra_sets(request))
        taken = {f.file for f in files} | {'04c_nsets_boundary_classes.inp'}
        clash = sorted(taken & {f.file for f in extra.files})
        if clash:
            raise ValueError(f'The set provider {settings.set_provider!r} writes a file the deck already has: '
                             f'{", ".join(clash)}.')
        files.extend(extra.files)
        skipped.extend(extra.skipped)
        summary['extra_sets'] = dict(extra.summary)
    return DeckSets(files, skipped, summary)
