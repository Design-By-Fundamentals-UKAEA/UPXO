"""Cuboid subset of an input dataset: box entry, the crop, and what the crop cuts.

The crop is taken from the input label array (and the feature table that goes
with it) before anything is cleaned or meshed. Raw labels are kept, so grain
IDs, element-set names, global IDs and orientations of the features that
remain are those of the full structure.
"""
from dataclasses import dataclass, field, replace
import json
from pathlib import Path
import re
import numpy as np

from . import constants as C

MODES = ('index', 'percent', 'physical')
AXES = 'xyz'


@dataclass(frozen=True)
class BoxSpec:
    """A box entered in one of three ways.

    ``index``: voxel indices, ``hi`` exclusive. ``percent``: percent of the
    dataset along each axis. ``physical``: lengths from the dataset origin in
    ``units``. A ``hi`` of ``None`` means the full extent.
    """
    mode: str = 'index'
    lo: tuple = (0, 0, 0)
    hi: tuple | None = None
    units: str | None = None

    def validate(self):
        if self.mode not in MODES:
            raise ValueError(f'mode must be one of {MODES}')
        if len(self.lo) != 3 or (self.hi is not None and len(self.hi) != 3):
            raise ValueError('lo and hi need three values (x, y, z)')
        if self.units is not None and self.units not in C.UNITS:
            raise ValueError(f'units must be one of {C.UNITS}')


def resolve_box(spec, shape, voxel_size=1.0, source_units=None):
    """Voxel indices ``(lo, hi)`` of ``spec`` in a dataset of ``shape``.

    ``voxel_size`` is one voxel edge in ``source_units``; physical entry is
    converted from ``spec.units`` (default: ``source_units``). Indices are
    rounded to the nearest voxel boundary.
    """
    spec.validate()
    shape = tuple(int(n) for n in shape)
    if len(shape) != 3:
        raise ValueError('shape must have three values')

    def to_index(values):
        values = np.asarray(values, dtype=float)
        if spec.mode == 'percent':
            return values / 100.0 * shape
        if spec.mode == 'physical':
            factor = 1.0 if source_units is None else C.unit_scale(
                spec.units or source_units, source_units)
            return values * factor / voxel_size
        return values

    lo_i = np.floor(to_index(spec.lo) + 0.5).astype(int)
    hi_i = (np.asarray(shape) if spec.hi is None
            else np.floor(to_index(spec.hi) + 0.5).astype(int))
    for axis in range(3):
        name = AXES[axis]
        if lo_i[axis] < 0 or hi_i[axis] > shape[axis]:
            raise ValueError(
                f'{name}: the box [{lo_i[axis]}, {hi_i[axis]}) lies outside the dataset '
                f'[0, {shape[axis]}) voxels')
        if hi_i[axis] <= lo_i[axis]:
            raise ValueError(f'{name}: the box has no voxel (lo {lo_i[axis]}, hi {hi_i[axis]})')
    return tuple(int(v) for v in lo_i), tuple(int(v) for v in hi_i)


def describe_box(lo, hi, shape, voxel_size=1.0, source_units=None, units=None):
    """The same box in all three entry forms, for display and for mode switches."""
    lo, hi, shape = np.asarray(lo), np.asarray(hi), np.asarray(shape)
    factor = 1.0 if source_units is None else C.unit_scale(source_units, units or source_units)
    size = (hi - lo) * voxel_size * factor
    return {
        'index': {'lo': lo.tolist(), 'hi': hi.tolist()},
        'percent': {'lo': (100.0 * lo / shape).tolist(), 'hi': (100.0 * hi / shape).tolist()},
        'physical': {'lo': (lo * voxel_size * factor).tolist(),
                     'hi': (hi * voxel_size * factor).tolist(), 'units': units or source_units},
        'shape': (hi - lo).tolist(),
        'voxels': int(np.prod(hi - lo)),
        'size': size.tolist(),
    }


def extract_labels(labels, lo, hi):
    """Crop of ``labels`` as a new contiguous array."""
    a = np.asarray(labels)
    window = tuple(slice(int(l), int(h)) for l, h in zip(lo, hi))
    return np.ascontiguousarray(a[window])


def extract_subset(hierarchy, lo, hi):
    """Hierarchy of the crop: same features, IDs and orientations, sizes recomputed."""
    lgi = extract_labels(hierarchy.lgi, lo, hi)
    labels, counts = np.unique(lgi, return_counts=True)
    present = set(labels.tolist())
    features = {k: f for k, f in hierarchy.features.items() if k in present}
    pcks = {f.pck_key for f in features.values() if f.pck_key is not None}
    blks = {f.blk_key for f in features.values() if f.blk_key is not None}
    sblks = {f.key for f in features.values() if f.sb_local is not None}
    pags = {f.pag_id for f in features.values()}
    return replace(
        hierarchy, lgi=lgi, features=features,
        voxel_counts=dict(zip(labels.tolist(), counts.tolist())),
        pag_order=[p for p in hierarchy.pag_order if p in pags],
        pck_order=[k for k in hierarchy.pck_order if k in pcks],
        blk_order=[k for k in hierarchy.blk_order if k in blks],
        sblk_order=[k for k in hierarchy.sblk_order if k in sblks],
        origin=tuple(int(v) for v in lo),
        parent_shape=tuple(int(n) for n in hierarchy.lgi.shape))


def cut_statistics(parent_counts, crop_counts, small_threshold=4):
    """What the box cuts: grains partly inside, and fragments below ``small_threshold``."""
    parent_counts = {int(k): int(v) for k, v in parent_counts.items()}
    crop_counts = {int(k): int(v) for k, v in crop_counts.items()}
    cut = sorted(k for k, v in crop_counts.items() if v < parent_counts.get(k, v))
    fragments = sorted(k for k, v in crop_counts.items() if v < small_threshold)
    return {
        'features_in_parent': len(parent_counts),
        'features_in_crop': len(crop_counts),
        'features_dropped': len(parent_counts) - len(crop_counts),
        'cut_features': len(cut),
        'whole_features': len(crop_counts) - len(cut),
        'small_threshold': int(small_threshold),
        'fragments_below_threshold': len(fragments),
        'smallest_in_crop': min(crop_counts.values()) if crop_counts else 0,
        'fragment_labels': fragments[:50],
    }


COUNT_NAMES = {'pag': 'PAGs', 'packet': 'packets', 'block': 'blocks', 'subblock': 'sub-blocks',
               'grains': 'grains', 'families': 'host families'}
LEVEL_COUNTS = {'pag': ('pag',), 'pck': ('pag', 'packet'), 'blk': ('pag', 'packet', 'block'),
                'sblk': ('pag', 'packet', 'block', 'subblock')}
MINIMUM_LEVELS = ('pag', 'packet', 'block')


def hierarchy_counts(hierarchy, labels=None):
    """How many PAGs, packets, blocks and sub-blocks the features hold (those of the level meshed), as ``{name: count}``.

    ``labels`` limits the count to those features (the ones in a crop); a structure that names its own groups
    (twinned FCC) is counted as grains and host families.
    """
    chosen = sorted(hierarchy.features if labels is None else (int(l) for l in labels if int(l) in hierarchy.features))
    features = [hierarchy.features[label] for label in chosen]
    if hierarchy.extra:
        families = {hierarchy.extra[f.label]['family'] for f in features} - {-1}
        return {'grains': len(features), 'families': len(families)}
    levels = LEVEL_COUNTS.get(hierarchy.level, LEVEL_COUNTS['sblk'])
    keys = {'pag': lambda f: f.pag_id, 'packet': lambda f: f.pck_key, 'block': lambda f: f.blk_key,
            'subblock': lambda f: f.key if f.sb_local is not None else None}
    return {name: len({keys[name](f) for f in features} - {None}) for name in levels}


def crop_adequacy(counts, minimums):
    """Warnings for the levels whose count is below the minimum the person asked for (``None`` or 0: no minimum)."""
    warnings = []
    for level in MINIMUM_LEVELS:
        wanted = (minimums or {}).get(level)
        if level in counts and wanted and counts[level] < int(wanted):
            warnings.append(f"Only {counts[level]} {COUNT_NAMES[level]} in the crop; at least {int(wanted)} were asked for.")
    return warnings


def subset_record(spec, lo, hi, shape, voxel_size=1.0, source_units=None, units=None,
                  statistics=None):
    """JSON-ready description of a crop (what was cut, in every entry form)."""
    record = {
        'entered': {'mode': spec.mode, 'lo': list(spec.lo),
                    'hi': None if spec.hi is None else list(spec.hi), 'units': spec.units},
        'box': describe_box(lo, hi, shape, voxel_size, source_units, units),
        'origin_voxels': [int(v) for v in lo],
        'parent_shape': [int(n) for n in shape],
        'voxel_size': voxel_size, 'source_units': source_units,
    }
    if statistics is not None:
        record['statistics'] = statistics
    return record


def write_subset_record(folder, record, name='subset.json'):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    path.write_text(json.dumps(record, indent=2), encoding='utf8')
    return path


_INDEXED = re.compile(r'^(\d+)(?:_.*)?$')
_SUFFIXED = re.compile(r'^(.*)_(\d+)$')


def next_index(parent, width=3):
    """One more than the largest ``NNN`` or ``NNN_suffix`` entry in ``parent``."""
    parent = Path(parent)
    taken = [int(m.group(1)) for p in parent.glob('*')
             if p.is_dir() and (m := _INDEXED.match(p.name))] if parent.is_dir() else []
    return max(taken, default=0) + 1


def indexed_name(index, suffix='', width=3):
    return f'{int(index):0{width}d}{suffix}'


def new_indexed_directory(parent, stem, width=3):
    """Create ``parent/<stem>_NNN`` with the next free number and return its path."""
    parent = Path(parent)
    parent.mkdir(parents=True, exist_ok=True)
    taken = [int(m.group(2)) for p in parent.glob(f'{stem}_*')
             if p.is_dir() and (m := _SUFFIXED.match(p.name)) and m.group(1) == stem]
    number = max(taken, default=0) + 1
    while True:
        path = parent / f'{stem}_{number:0{width}d}'
        try:
            path.mkdir()
            return path
        except FileExistsError:
            number += 1
