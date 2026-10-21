"""Iterative removal of spike voxels from a label array before meshing.

A spike is a voxel with at most one face neighbour of its own label; the
cleaner reassigns it to the neighbouring label. The passes repeat until a pass
changes nothing, a label configuration repeats, or the pass limit is reached.
Every grain ID must survive.
"""
import hashlib
from numbers import Integral

import numpy as np


def clean_spikes(labels, max_passes=100, progress=None):
    """Return ``(cleaned_labels, report)``; the input array is not modified.

    ``report`` has ``stop_reason``, ``history`` (voxels changed per pass),
    ``remaining_candidates`` (spike candidates left, including protected last
    voxels) and ``net_changed_voxels`` (voxels that differ from the input).
    ``progress(message, current, total)`` is called before every pass.
    """
    if isinstance(max_passes, bool) or not isinstance(max_passes, Integral) or max_passes < 1:
        raise ValueError('max_passes must be a positive integer')
    original = np.asarray(labels)
    if original.ndim != 3 or original.dtype.kind not in 'iu':
        raise ValueError('labels must be a 3D integer array')
    if np.any(original < 0):
        raise ValueError('The spike cleaner requires nonnegative labels.')
    # The routine lives in the twinned-structure cleaner; it is used here only.
    from upxo.pxtal.twinned_simple_3d.cleaning_3d import StructureCleaner3D

    cleaner = StructureCleaner3D(upscale_fallback=False, do_lobe_split=False)
    cleaned = original.copy()
    ids_before = np.unique(cleaned)
    seen = {hashlib.sha256(cleaned.tobytes()).digest()}
    history, stop_reason = [], 'maximum passes reached'
    for number in range(1, int(max_passes) + 1):
        if progress is not None:
            progress(f'Spike removal pass {number}/{max_passes}', number, int(max_passes))
        cleaned, changed = cleaner._fast_clean_spikes(cleaned)
        history.append({'pass': number, 'changed_voxels': int(changed)})
        if changed == 0:
            stop_reason = 'no further changes'
            break
        digest = hashlib.sha256(cleaned.tobytes()).digest()
        if digest in seen:
            stop_reason = 'repeated label configuration'
            break
        seen.add(digest)
    if not np.array_equal(np.unique(cleaned), ids_before):
        raise RuntimeError('Spike cleaning lost a grain ID')
    remaining = int(np.count_nonzero(
        (cleaned > 0) & (cleaner._count_face_same_grain(cleaned)
                         <= cleaner._SPIKE_SAME_COUNT_THRESHOLD)))
    return cleaned, {
        'stop_reason': stop_reason,
        'history': history,
        'remaining_candidates': remaining,
        'net_changed_voxels': int(np.count_nonzero(cleaned != original)),
    }
