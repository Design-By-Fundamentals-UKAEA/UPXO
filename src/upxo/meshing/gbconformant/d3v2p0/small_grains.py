"""Conservative whole-grain absorption before voxel refinement.

Each sequential merge uses current face adjacency, never batch merge chains.
New cubical boundary-link defects are forbidden. Existing defects are reported
and must still pass the downstream topology cleaner before surface extraction.
"""
from numbers import Integral
import numpy as np
from scipy import ndimage
from .voxel_topology import detect_voxel_topology


def remove_small_grains(labels, *, enabled=True, min_voxels=4,
                        protected_ids=(), protect_rve=False, max_passes=10,
                        selection='shared_area', spacing=(1., 1., 1.),
                        preserve_recipient_components=True):
    """Absorb entire IDs with count < min_voxels; return copy and JSON report.

    All nonnegative labels (including zero) are grains. Threshold is measured
    on the input grid, before upscaling. Protected IDs cannot be removed but
    may receive voxels. Disconnected pieces of an ID are removed together.
    A protected island should be listed explicitly. No IDs are renumbered.
    Exact checks favour correctness; this first version is not optimized for
    thousands of candidates. Failed proposals leave the working array intact.
    """
    a = np.asarray(labels)
    for name, value in [('min_voxels', min_voxels), ('max_passes', max_passes)]:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 1:
            raise ValueError(name + ' must be a positive integer')
    for value in (enabled, protect_rve, preserve_recipient_components):
        if not isinstance(value, (bool, np.bool_)):
            raise ValueError('switches must be boolean')
    spacing = np.asarray(spacing, dtype=float)
    if spacing.shape != (3,) or not np.all(np.isfinite(spacing)) or np.any(spacing <= 0):
        raise ValueError('spacing must contain three positive finite values')
    if selection not in ('shared_area', 'largest'):
        raise ValueError('selection must be shared_area or largest')
    defects = detect_voxel_topology(a)
    a = a.copy()
    ids, sizes = np.unique(a, return_counts=True)
    initial = dict(zip(map(int, ids), map(int, sizes)))
    protected = set(protected_ids)
    if not protected <= initial.keys():
        raise ValueError('protected_ids contains unknown IDs')
    if protect_rve:
        for axis in range(3):
            protected.update(map(int, np.unique(np.take(a, [0, -1], axis=axis))))
    mapping = {g: g for g in initial}
    events, rejected = [], []
    initial_defects = sum(map(len, defects.values()))
    structure = ndimage.generate_binary_structure(3, 1)
    stop = 'disabled'
    if enabled:
        stop = 'max_passes'
        for iteration in range(max_passes):
            ids, sizes = np.unique(a, return_counts=True)
            candidates = sorted((int(n), int(g)) for g, n in zip(ids, sizes)
                                if n < min_voxels and g not in protected)
            changed = False
            for _, source in candidates:
                mask = a == source
                count = int(mask.sum())
                if not count or count >= min_voxels:
                    continue
                contacts = {}
                for axis in range(3):
                    left = [slice(None)] * 3; right = left.copy()
                    left[axis] = slice(None, -1); right[axis] = slice(1, None)
                    u, v = a[tuple(left)], a[tuple(right)]
                    neighbours = np.concatenate((v[(u == source) & (v != source)],
                                                 u[(v == source) & (u != source)]))
                    gs, ns = np.unique(neighbours, return_counts=True)
                    area = float(np.prod(np.delete(spacing, axis)))
                    for g, n in zip(gs, ns):
                        contacts[int(g)] = contacts.get(int(g), 0.) + int(n) * area
                current = dict(zip(*np.unique(a, return_counts=True)))
                ranked = sorted(contacts, key=lambda g: (
                    -contacts[g] if selection == 'shared_area' else -current[g],
                    -current[g] if selection == 'shared_area' else -contacts[g], g))
                for target in ranked:
                    proposal = a.copy(); proposal[mask] = target
                    after = detect_voxel_topology(proposal)
                    new_defects = any(gs - defects.get(pos, set()) for pos, gs in after.items())
                    reason = 'new_boundary_link_defect' if new_defects else None
                    if reason is None and preserve_recipient_components:
                        if ndimage.label(a == target, structure)[1] != ndimage.label(proposal == target, structure)[1]:
                            reason = 'recipient_component_count_changed'
                    if reason:
                        rejected.append(dict(source=source, target=target, reason=reason))
                        continue
                    a, defects = proposal, after
                    mapping = {g: target if sink == source else sink for g, sink in mapping.items()}
                    events.append(dict(source=source, target=target, voxels=count,
                                       shared_area=contacts[target], pass_number=iteration + 1))
                    changed = True
                    break
            if not changed:
                stop = 'no_acceptable_merges'
                break
    final = dict(zip(*[list(map(int, x)) for x in np.unique(a, return_counts=True)]))
    report = dict(enabled=bool(enabled), min_voxels=int(min_voxels), stop_reason=stop,
                  removed_ids=sorted(set(initial) - set(final)),
                  retained_ids=sorted(final), original_to_survivor=mapping,
                  protected_ids=sorted(map(int, protected)), merges=events,
                  rejected_proposals=rejected, initial_defects=initial_defects,
                  remaining_defects=sum(map(len, defects.values())),
                  unresolved_small_ids=sorted(g for g, n in final.items() if n < min_voxels),
                  voxel_changes={g: final.get(g, 0) - n for g, n in initial.items()},
                  total_voxels_preserved=sum(initial.values()) == sum(final.values()))
    return a, report
