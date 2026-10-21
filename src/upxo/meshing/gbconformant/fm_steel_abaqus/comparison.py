"""Before/after report: raw voxel features against the conformal tetrahedral mesh.

Nothing here judges the result; it reports what the raw structure had and what
the mesh has. Both sides use the same features: the label array of the chosen
hierarchy level (before) and the resolved identity of every meshed grain
(after). Lengths are in output units.
"""
from dataclasses import dataclass
import numpy as np

from . import constants as C
from .junctions import (
    junction_lines, tet_junction_edges, totals_per_grain, totals_per_set,
    voxel_junction_edges)

_CHUNK = 500_000
BEFORE_TITLE, AFTER_TITLE = 'Voxel structure', 'Tetrahedral mesh'


@dataclass
class ComparisonSettings:
    include_rve_faces: bool = False   # count grain-boundary traces on the box faces
    output_units: str | None = None   # None keeps the raw folder's units
    length_scale: float = 1.0
    mesh_voxel_length: float = 1.0    # one raw voxel edge in mesh coordinates
    symmetry: str = 'cubic'           # crystal symmetry of the misorientations


def describe(values):
    """Count, total and distribution summary of a 1D array (NaN when empty)."""
    v = np.asarray(values, dtype=float)
    if not v.size:
        nan = float('nan')
        return {'count': 0, 'total': 0.0, 'mean': nan, 'std': nan, 'min': nan,
                'p05': nan, 'p25': nan, 'median': nan, 'p75': nan, 'p95': nan, 'max': nan}
    q = np.percentile(v, [5, 25, 50, 75, 95])
    return {'count': int(v.size), 'total': float(v.sum()), 'mean': float(v.mean()),
            'std': float(v.std()), 'min': float(v.min()), 'p05': float(q[0]),
            'p25': float(q[1]), 'median': float(q[2]), 'p75': float(q[3]),
            'p95': float(q[4]), 'max': float(v.max())}


def tet_volumes(points, tetrahedra, chunk=_CHUNK):
    p = np.asarray(points, dtype=float)
    t = np.asarray(tetrahedra)
    out = np.empty(len(t))
    for start in range(0, len(t), chunk):
        v = p[t[start:start + chunk]]
        out[start:start + chunk] = np.abs(np.einsum(
            'ij,ij->i', v[:, 1] - v[:, 0], np.cross(v[:, 2] - v[:, 0], v[:, 3] - v[:, 0]))) / 6
    return out


def sphere_equivalent_diameter(volume):
    """Diameter of the sphere with the given volume."""
    return (6.0 * np.asarray(volume, dtype=float) / np.pi) ** (1.0 / 3.0)


def _units(hierarchy, settings):
    dst = settings.output_units or hierarchy.units
    per_voxel = hierarchy.voxel_size * C.unit_scale(hierarchy.units, dst) * settings.length_scale
    return dst, per_voxel


def _line_summary(lines, scale):
    length = lines.length * scale
    per_grain = {k: v * scale for k, v in totals_per_grain(lines).items()}
    return {
        'lines': describe(length),
        'per_grain_total': describe(list(per_grain.values())),
        'length': length,
        'per_grain': per_grain,
        'per_set': {k: v * scale for k, v in totals_per_set(lines).items()},
    }


def _matched(before, after):
    keys_before, keys_after = set(before), set(after)
    both = sorted(keys_before & keys_after)
    rows = [{'set': k, 'before': before[k], 'after': after[k],
             'ratio': after[k] / before[k] if before[k] else float('nan')} for k in both]
    return {'matched': rows,
            'only_before': {k: before[k] for k in sorted(keys_before - keys_after)},
            'only_after': {k: after[k] for k in sorted(keys_after - keys_before)}}


def compare_junction_lines(hierarchy, points, tetrahedra, resolved_grains, settings=None):
    """Junction line lengths before (voxels) and after (tetrahedra)."""
    settings = settings or ComparisonSettings()
    units, per_voxel = _units(hierarchy, settings)
    before_edges = voxel_junction_edges(hierarchy.lgi, settings.include_rve_faces)
    after_edges = tet_junction_edges(points, tetrahedra, resolved_grains, settings.include_rve_faces)
    before = _line_summary(junction_lines(before_edges), per_voxel)
    after = _line_summary(junction_lines(after_edges), per_voxel / settings.mesh_voxel_length)
    return {
        'units': units,
        'include_rve_faces': settings.include_rve_faces,
        'before': before, 'after': after,
        'truncated_edges': {'before': before_edges.truncated, 'after': after_edges.truncated},
        'by_region_set': _matched(before['per_set'], after['per_set']),
        'by_grain': _matched(before['per_grain'], after['per_grain']),
    }


def feature_diameters(hierarchy, points, tetrahedra, resolved_grains, settings=None):
    """Sphere-equivalent diameter of every feature: raw voxels against mesh volume."""
    settings = settings or ComparisonSettings()
    units, per_voxel = _units(hierarchy, settings)
    before = {int(label): float(sphere_equivalent_diameter(count * per_voxel ** 3))
              for label, count in hierarchy.voxel_counts.items()}
    volume = tet_volumes(points, tetrahedra) * (per_voxel / settings.mesh_voxel_length) ** 3
    labels = np.asarray(resolved_grains, dtype=np.int64)
    sums = np.bincount(labels, weights=volume)
    after = {int(label): float(sphere_equivalent_diameter(sums[label]))
             for label in np.flatnonzero(sums)}
    rows = [{'label': k, 'before': before[k], 'after': after[k],
             'ratio': after[k] / before[k]} for k in sorted(set(before) & set(after))]
    return {'units': units,
            'before': before, 'after': after,
            'before_stats': describe(list(before.values())),
            'after_stats': describe(list(after.values())),
            'matched': rows,
            'ratio_stats': describe([r['ratio'] for r in rows]),
            'lost': sorted(set(before) - set(after))}


def compare_before_after(hierarchy, points, tetrahedra, tet_grain, identity, settings=None):
    """Junction line lengths and sphere-equivalent diameters, before and after."""
    settings = settings or ComparisonSettings()
    lut = identity.lookup()
    grains = np.asarray(tet_grain)
    if grains.max() >= len(lut) or np.any(lut[grains] < 0):
        raise ValueError('A tetrahedron carries a grain with no resolved identity')
    resolved = lut[grains]
    return {
        'level': hierarchy.level,
        'junction_lines': compare_junction_lines(hierarchy, points, tetrahedra, resolved, settings),
        'diameters': feature_diameters(hierarchy, points, tetrahedra, resolved, settings),
        'names': {'before': BEFORE_TITLE, 'after': AFTER_TITLE},
        'features_before': len(hierarchy.features),
        'features_after': int(len(np.unique(resolved))),
        'labels_before': sorted(int(k) for k in hierarchy.features),
        'labels_after': np.unique(resolved).tolist(),
    }


def report_summary(result):
    """JSON-friendly digest of :func:`compare_before_after` (no per-line arrays)."""
    lines = result['junction_lines']
    diam = result['diameters']
    def side(name):
        return {'lines': lines[name]['lines'], 'per_grain_total': lines[name]['per_grain_total']}
    return {
        'level': result['level'],
        'units': lines['units'],
        'include_rve_faces': lines['include_rve_faces'],
        'features_before': result['features_before'],
        'features_after': result['features_after'],
        'junction_lines_before': side('before'),
        'junction_lines_after': side('after'),
        'junction_sets_matched': len(lines['by_region_set']['matched']),
        'junction_sets_only_before': len(lines['by_region_set']['only_before']),
        'junction_sets_only_after': len(lines['by_region_set']['only_after']),
        'diameter_before': diam['before_stats'],
        'diameter_after': diam['after_stats'],
        'diameter_ratio_matched': diam['ratio_stats'],
        'features_lost': diam['lost'],
    }


_ROWS = (('count', '{:.0f}'), ('total', '{:.6g}'), ('mean', '{:.6g}'), ('median', '{:.6g}'),
         ('std', '{:.6g}'), ('min', '{:.6g}'), ('p05', '{:.6g}'), ('p25', '{:.6g}'),
         ('p75', '{:.6g}'), ('p95', '{:.6g}'), ('max', '{:.6g}'))


def _table(title, before, after, names):
    width = max(16, len(names['before']) + 2, len(names['after']) + 2)
    lines = [title, f"{'':8s}{names['before']:>{width}s}{names['after']:>{width}s}"]
    for key, fmt in _ROWS:
        lines.append(f"{key:8s}{fmt.format(before[key]):>{width}s}{fmt.format(after[key]):>{width}s}")
    return lines


def format_summary(result):
    """Plain-text tables of a :func:`compare_before_after` result, the columns named for what they hold."""
    lines = result['junction_lines']
    diam = result['diameters']
    units = lines['units']
    names = result.get('names') or {'before': BEFORE_TITLE, 'after': AFTER_TITLE}
    out = [f"Level: {result['level']}   grains: {result['features_before']} in the {names['before'].lower()}, "
           f"{result['features_after']} in the {names['after'].lower()}   lengths in {units}",
           'RVE-face traces: ' + ('included' if lines['include_rve_faces'] else 'excluded'), '']
    out += _table(f'Junction line length per line ({units})',
                  lines['before']['lines'], lines['after']['lines'], names)
    out += [''] + _table(f'Junction line length per grain ({units})',
                         lines['before']['per_grain_total'], lines['after']['per_grain_total'], names)
    out += [''] + _table(f'Sphere-equivalent diameter ({units})',
                         diam['before_stats'], diam['after_stats'], names)
    sets = lines['by_region_set']
    out += ['', f"Junction region sets on both sides: {len(sets['matched'])}; "
                f"only in the {names['before'].lower()}: {len(sets['only_before'])}; "
                f"only in the {names['after'].lower()}: {len(sets['only_after'])}"]
    ratio = diam['ratio_stats']
    out += [f"Diameter ({names['after'].lower()} / {names['before'].lower()}) on {ratio['count']} shared grains: "
            f"median {ratio['median']:.4g}, min {ratio['min']:.4g}, max {ratio['max']:.4g}",
            f"Grains with no mesh: {diam['lost']}"]
    return '\n'.join(out)
