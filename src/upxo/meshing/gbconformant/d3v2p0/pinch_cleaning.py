"""Targeted repair of local edge/corner contacts, without majority filtering."""
from itertools import product
from numbers import Integral
import numpy as np


_CORNERS = list(product(range(2), repeat=3))
_NEIGHBORS = [[j for j, b in enumerate(_CORNERS) if sum(abs(x-y) for x, y in zip(a, b)) == 1]
              for a in _CORNERS]
_OFFSETS = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]


def _block_contacts(block):
    values = block.ravel().tolist()
    defects = set()
    for gid in dict.fromkeys(values):
        remaining = {i for i, value in enumerate(values) if value == gid}
        components = 0
        while remaining:
            components += 1
            todo = [remaining.pop()]
            while todo:
                for j in _NEIGHBORS[todo.pop()]:
                    if j in remaining:
                        remaining.remove(j)
                        todo.append(j)
        if components > 1:
            defects.add(gid)
    return defects


def _validate(labels):
    a = np.asarray(labels)
    if a.ndim != 3 or not a.size or a.dtype.kind not in 'iu':
        raise ValueError('labels must be a nonempty 3D integer array')
    return a


def detect_pinch_candidates(labels):
    """Map 2x2x2 block origins to labels with disconnected local face components.

    These are candidate edge/corner contacts, not proof of a physical defect.
    This detector does not cover every thin neck or smoothing-induced pinch.
    """
    a = _validate(labels)
    result = {}
    for origin in np.ndindex(tuple(max(0, n-1) for n in a.shape)):
        sl = tuple(slice(i, i+2) for i in origin)
        contacts = _block_contacts(a[sl])
        if contacts:
            result[origin] = contacts
    return result


def clean_pinch_contacts(labels, enabled=True, max_passes=5, max_changes=200,
                         max_volume_fraction=.03, protect_grains_up_to=8):
    """Greedily repair detected contacts, returning (copy, report, change_log).

    Each accepted single-voxel reassignment must strictly reduce candidate
    (block, grain) pairs, introduce none nearby, retain every grain's GLOBAL
    six-connected component count and respect per-grain net voxel-count limits.
    Grains initially <= protect_grains_up_to cannot donate voxels. Each grain's
    volume allowance is max(1, floor(initial_count * max_volume_fraction)); zero
    fraction gives zero allowance. Positions/labels are recorded for every edit.
    Input is never mutated. Disabled mode returns an unchanged copy immediately.
    """
    original = _validate(labels)
    a = original.copy()
    if not isinstance(enabled, (bool, np.bool_)):
        raise ValueError('enabled must be boolean')
    for name, value in [('max_passes', max_passes), ('max_changes', max_changes),
                        ('protect_grains_up_to', protect_grains_up_to)]:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 0:
            raise ValueError(f'{name} must be a nonnegative integer')
    if not np.isfinite(max_volume_fraction) or not 0 <= max_volume_fraction <= 1:
        raise ValueError('max_volume_fraction must be between 0 and 1')
    if not enabled:
        return a, {'enabled': False, 'accepted_changes': 0, 'stop_reason': 'disabled'}, []
    from scipy.ndimage import label, generate_binary_structure
    structure = generate_binary_structure(3, 1)
    ids, counts = np.unique(a, return_counts=True)
    initial = dict(zip(ids.tolist(), counts.tolist()))
    current = initial.copy()
    allowance = {g: max(1, int(n*max_volume_fraction)) if max_volume_fraction else 0 for g, n in initial.items()}
    component_counts = {g: label(a == g, structure)[1] for g in initial}
    candidates = detect_pinch_candidates(a)
    initial_count = sum(map(len, candidates.values()))
    changes, pass_history = [], []
    reason = 'maximum passes reached'
    for pass_number in range(1, max_passes+1):
        # Candidate locations, not grain IDs, determine traversal order.
        positions = sorted({tuple(np.array(origin)+corner) for origin in candidates for corner in _CORNERS})
        accepted = 0
        for pos in positions:
            if len(changes) >= max_changes:
                reason = 'maximum changes reached'
                break
            donor = int(a[pos])
            if initial[donor] <= protect_grains_up_to or current[donor] <= 1:
                continue
            if abs(current[donor]-1-initial[donor]) > allowance[donor]:
                continue
            origins = [tuple(np.array(pos)-corner) for corner in _CORNERS
                       if all(0 <= pos[j]-corner[j] < a.shape[j]-1 for j in range(3))]
            before = {(o, g) for o in origins for g in candidates.get(o, ())}
            if not before:
                continue
            neighbors = []
            for offset in _OFFSETS:
                n = tuple(x+d for x, d in zip(pos, offset))
                if all(0 <= n[j] < a.shape[j] for j in range(3)):
                    g = int(a[n])
                    if g != donor and g not in neighbors:
                        neighbors.append(g)
            best = None
            for recipient in neighbors:
                if abs(current[recipient]+1-initial[recipient]) > allowance[recipient]:
                    continue
                a[pos] = recipient
                after = {o: _block_contacts(a[tuple(slice(i, i+2) for i in o)]) for o in origins}
                defects = {(o, g) for o, gs in after.items() for g in gs}
                improvement = len(before)-len(defects)
                if improvement > 0 and defects.issubset(before):
                    if (label(a == donor, structure)[1] == component_counts[donor]
                            and label(a == recipient, structure)[1] == component_counts[recipient]):
                        if best is None or improvement > best[0]:
                            best = (improvement, recipient, after)
                a[pos] = donor
            if best is None:
                continue
            improvement, recipient, after = best
            a[pos] = recipient
            current[donor] -= 1
            current[recipient] += 1
            for origin, defects in after.items():
                if defects:
                    candidates[origin] = defects
                else:
                    candidates.pop(origin, None)
            changes.append({'pass': pass_number, 'voxel': list(map(int, pos)),
                            'from_grain': donor, 'to_grain': recipient,
                            'candidates_removed': improvement})
            accepted += 1
        pass_history.append({'pass': pass_number, 'accepted': accepted,
                             'remaining_candidates': sum(map(len, candidates.values()))})
        if not candidates:
            reason = 'no candidates remain'
            break
        if len(changes) >= max_changes:
            reason = 'maximum changes reached'
            break
        if accepted == 0:
            reason = 'no admissible local repair'
            break
    report = {'enabled': True, 'initial_candidates': initial_count,
              'remaining_candidates': sum(map(len, candidates.values())),
              'accepted_changes': len(changes), 'net_changed_voxels': int(np.count_nonzero(a != original)),
              'grains_retained': len(initial), 'face_component_counts_preserved': True,
              'stop_reason': reason, 'passes': pass_history,
              'grain_voxel_changes': {str(g): current[g]-initial[g] for g in initial if current[g] != initial[g]}}
    return a, report, changes
