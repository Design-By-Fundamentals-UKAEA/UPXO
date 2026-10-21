"""Hierarchy levels of a raw feature export (PAG, packet, block, sub-block).

A raw export folder holds one label array and one orientation file per level
(``<prefix>_lgi.npy``, ``<prefix>_orientations.pkl``) and a ``raw_export.pkl``
bundle with the maps between level identifiers, parentage, units and voxel
size. This module reads it, reports which levels are present and builds the
feature table (orientation, parentage, global ID) for one chosen level.
"""
from dataclasses import dataclass, field
from pathlib import Path
import pickle
import numpy as np

LEVELS = ('pag', 'pck', 'blk', 'sblk')
LEVEL_LABELS = {'pag': 'PAG', 'pck': 'Packet', 'blk': 'Block', 'sblk': 'Sub-block'}
DEFAULT_PREFIXES = {level: level for level in LEVELS}
BUNDLE_NAME = 'raw_export.pkl'


@dataclass(frozen=True)
class Feature:
    """One feature of the chosen level, identified by its label-array value."""
    label: int
    level: str
    source_id: object
    pag_id: int
    pkt_idx: int | None
    blk_local: int | None
    sb_local: int | None
    euler: tuple
    global_id: int

    @property
    def pck_key(self):
        return None if self.pkt_idx is None else (self.pag_id, self.pkt_idx)

    @property
    def blk_key(self):
        return None if self.blk_local is None else (self.pag_id, self.pkt_idx, self.blk_local)

    @property
    def key(self):
        parts = (self.pag_id, self.pkt_idx, self.blk_local, self.sb_local)
        return tuple(p for p in parts if p is not None)


@dataclass
class Hierarchy:
    level: str
    lgi: np.ndarray
    features: dict
    voxel_counts: dict
    units: str
    voxel_size: float
    pag_order: list = field(default_factory=list)
    pck_order: list = field(default_factory=list)
    blk_order: list = field(default_factory=list)
    sblk_order: list = field(default_factory=list)
    global_ids: dict = field(default_factory=dict)
    origin: tuple | None = None          # first voxel of a crop in the parent array
    parent_shape: tuple | None = None    # shape of the parent array of a crop
    names: dict = field(default_factory=dict)    # {label: element-set name} where a dataset names its own features
    material_names: dict = field(default_factory=dict)   # {label: material name} likewise
    extra: dict = field(default_factory=dict)    # {label: {column: value}} the dataset adds to each feature (twinned: role, family)
    groups: dict = field(default_factory=dict)   # {element-set name: labels} for sets that join features
    system: str = 'FM steel'             # what the structure is, for the deck's headings

    def ordered_features(self):
        """Features in global-ID order, which is the deck's material order."""
        return sorted(self.features.values(), key=lambda f: f.global_id)


def _prefixes(prefixes):
    merged = dict(DEFAULT_PREFIXES)
    merged.update(prefixes or {})
    return merged


def read_bundle(folder):
    path = Path(folder) / BUNDLE_NAME
    if not path.is_file():
        raise FileNotFoundError(f'{BUNDLE_NAME} not found in {folder}')
    with path.open('rb') as handle:
        return pickle.load(handle)


def detect_levels(folder, prefixes=None):
    """Levels (coarse to fine) with both a label array and an orientation file."""
    folder = Path(folder)
    prefix = _prefixes(prefixes)
    return [level for level in LEVELS
            if (folder / f'{prefix[level]}_lgi.npy').is_file()
            and (folder / f'{prefix[level]}_orientations.pkl').is_file()]


def _structure_maps(bundle):
    g2pag = bundle['pag_grain_to_pag_id']
    g2pkt = bundle['pag_grain_to_local_pkt_idx']
    blk_map = {}
    for grain, blocks in bundle.get('pck_grain_to_blocks_map', {}).items():
        pag = g2pag.get(grain)
        if pag is None:
            continue
        for local, block_id in enumerate(blocks, start=1):
            blk_map[block_id] = (pag, g2pkt[grain], local)
    sb_map = {}
    block_to_sub = bundle.get('sblk_block_to_subblocks_map', {})
    for block_id, (pag, pkt, local) in blk_map.items():
        for sb_local, sb_id in enumerate(block_to_sub.get(block_id, []), start=1):
            sb_map[sb_id] = (pag, pkt, local, sb_local)
    return g2pag, g2pkt, blk_map, sb_map


def _global_ids(bundle, g2pag, g2pkt, blk_map, sb_map):
    """Sequential IDs in write order: PAGs, packets, blocks, sub-blocks."""
    gmap, counter = {}, 0
    for pag in bundle['pag_clusters_dict']:
        gmap[('pag', pag)] = counter
        counter += 1
    pck_order = []
    for grain in bundle['base_grain_locs']:
        if grain not in g2pag:
            continue
        key = (g2pag[grain], g2pkt[grain])
        if ('pck',) + key not in gmap:
            gmap[('pck',) + key] = counter
            pck_order.append(key)
            counter += 1
    for block_id in blk_map:
        gmap[('blk', block_id)] = counter
        counter += 1
    for sb_id in sb_map:
        gmap[('sb', sb_id)] = counter
        counter += 1
    return gmap, pck_order


def load_hierarchy(folder, level, prefixes=None):
    """Read ``level`` of the raw export in ``folder`` into a :class:`Hierarchy`."""
    if level not in LEVELS:
        raise ValueError(f'level must be one of {LEVELS}')
    folder = Path(folder)
    prefix = _prefixes(prefixes)
    available = detect_levels(folder, prefixes)
    if level not in available:
        raise FileNotFoundError(
            f'Level {level!r} is not in {folder}; available levels: {available}')
    bundle = read_bundle(folder)
    if bundle.get('pag_retained_austenite_pag_ids'):
        raise NotImplementedError('Retained austenite is not supported yet.')
    if bundle.get('pag_isolated_grains'):
        raise NotImplementedError('Isolated grains are not supported yet.')
    lgi = np.load(folder / f'{prefix[level]}_lgi.npy', allow_pickle=False)
    if lgi.ndim != 3 or lgi.dtype.kind not in 'iu':
        raise ValueError('The label array must be a 3D integer array.')
    with (folder / f'{prefix[level]}_orientations.pkl').open('rb') as handle:
        orientations = pickle.load(handle)
    labels, counts = np.unique(lgi, return_counts=True)
    if labels[0] == 0:
        raise NotImplementedError(
            'Label 0 (void or retained austenite) is present in the label array.')
    g2pag, g2pkt, blk_map, sb_map = _structure_maps(bundle)
    gmap, pck_order = _global_ids(bundle, g2pag, g2pkt, blk_map, sb_map)

    int_to_id = None
    if level == 'blk':
        int_to_id = bundle['blk_lgi_int_to_id']
    elif level == 'sblk':
        int_to_id = bundle['sblk_lgi_int_to_id']
    features, missing = {}, []
    for label in labels.tolist():
        source = label if int_to_id is None else int_to_id.get(label)
        if source not in orientations:
            missing.append(label)
            continue
        euler = tuple(float(a) for a in orientations[source])
        if level == 'pag':
            feature = Feature(label, level, source, label, None, None, None, euler,
                              gmap[('pag', label)])
        elif level == 'pck':
            if label not in g2pag:
                missing.append(label)
                continue
            pag, pkt = g2pag[label], g2pkt[label]
            feature = Feature(label, level, source, pag, pkt, None, None, euler,
                              gmap[('pck', pag, pkt)])
        elif level == 'blk':
            pag, pkt, local = blk_map[source]
            feature = Feature(label, level, source, pag, pkt, local, None, euler,
                              gmap[('blk', source)])
        else:
            pag, pkt, local, sb_local = sb_map[source]
            feature = Feature(label, level, source, pag, pkt, local, sb_local, euler,
                              gmap[('sb', source)])
        features[label] = feature
    if missing:
        raise ValueError(
            f'{len(missing)} label(s) have no orientation or parentage entry '
            f'(first few: {missing[:5]}).')
    keys = {f.key for f in features.values()}
    if len(keys) != len(features):
        raise ValueError('Two features share one element-set name.')
    return Hierarchy(
        level=level, lgi=lgi, features=features,
        voxel_counts=dict(zip(labels.tolist(), counts.tolist())),
        units=str(bundle['base_units']), voxel_size=float(bundle['base_voxel_size']),
        pag_order=list(bundle['pag_clusters_dict']), pck_order=pck_order,
        blk_order=list(blk_map.values()), sblk_order=list(sb_map.values()),
        global_ids=gmap)


def parent_tables(hierarchy):
    """Lookup tables from a feature label to its parents.

    Returns ``(pag_lut, pck_lut, pck_keys, blk_lut, blk_keys)``; ``pag_lut``
    holds PAG IDs, the packet and block tables hold indices into the key
    lists (``-1`` where the level does not apply).
    """
    size = max(hierarchy.features) + 1
    pag = np.full(size, -1, dtype=np.int64)
    pck = np.full(size, -1, dtype=np.int64)
    blk = np.full(size, -1, dtype=np.int64)
    pck_keys, blk_keys = [], []
    pck_index, blk_index = {}, {}
    for label, f in hierarchy.features.items():
        pag[label] = f.pag_id
        if f.pck_key is not None:
            if f.pck_key not in pck_index:
                pck_index[f.pck_key] = len(pck_keys)
                pck_keys.append(f.pck_key)
            pck[label] = pck_index[f.pck_key]
        if f.blk_key is not None:
            if f.blk_key not in blk_index:
                blk_index[f.blk_key] = len(blk_keys)
                blk_keys.append(f.blk_key)
            blk[label] = blk_index[f.blk_key]
    return pag, pck, pck_keys, blk, blk_keys
