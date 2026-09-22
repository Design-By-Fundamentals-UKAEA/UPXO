"""Pixel-level detection of island regions in a labelled 2D grain image.

A grain whose pixels enclose other pixels (the enclosed set is a union of
whole grains) is a *host*; the enclosed set is an *island region*. The
geometrification pipeline only handles hole-free structures, so a structure
with islands is split into:

* the *filled* structure: every top-level host absorbs its island regions;
* one *island cluster* per region: the region cropped to its bounding box,
  with the pixels outside the region relabelled as filler grains so the crop
  is a complete rectangular partition.

Nested islands (an island grain that itself hosts an island) stay inside the
cluster and are resolved when the cluster is processed in turn.
"""
import numpy as np
from scipy.ndimage import binary_fill_holes, find_objects
from scipy.ndimage import label as cc_label


def _relabel_consecutive(lgi):
    """Return (relabelled image with labels 1..k, sorted original labels)."""
    uniq, inv = np.unique(lgi, return_inverse=True)
    return (inv.reshape(lgi.shape) + 1).astype(lgi.dtype), uniq


def find_island_regions(lgi):
    """
    Split ``lgi`` into a hole-free filled image plus island clusters.

    Returns
    -------
    filled_lgi : ndarray
        Labels 1..m, no holes. Top-level hosts have absorbed their regions;
        grains inside a region are absent.
    filled_labels : ndarray
        ``filled_labels[k-1]`` is the original label of filled label ``k``.
    clusters : list of dict
        Keys ``host`` (original label), ``offset`` (row0, col0),
        ``lgi`` (crop, labels 1..k+f), ``labels`` (original label of each
        crop label; ``0`` marks a filler grain).
        Empty when ``lgi`` has no holes.
    """
    lgi = np.asarray(lgi)
    labels = np.unique(lgi)
    slices = find_objects(lgi) if lgi.min() >= 0 else None
    if slices is None:
        raise ValueError("Island detection needs non-negative integer labels.")

    regions = []                      # (host, region mask over full image)
    for lab in labels:
        if lab <= 0 or lab > len(slices) or slices[lab - 1] is None:
            continue
        sl = slices[lab - 1]
        m = lgi[sl] == lab
        holes = binary_fill_holes(m) & ~m
        if not holes.any():
            continue
        comp, n = cc_label(holes)
        for k in range(1, n + 1):
            full = np.zeros(lgi.shape, dtype=bool)
            full[sl][comp == k] = True
            regions.append((int(lab), full))
    if not regions:
        return None, labels, []

    nested = set()
    for host, region in regions:
        nested.update(int(v) for v in np.unique(lgi[region]))
    top = [(h, r) for h, r in regions if h not in nested]

    filled = lgi.copy()
    for host, region in top:
        filled[region] = host
    filled_lgi, filled_labels = _relabel_consecutive(filled)

    clusters = []
    for host, region in top:
        rows, cols = np.nonzero(region)
        r0, r1, c0, c1 = rows.min(), rows.max() + 1, cols.min(), cols.max() + 1
        rm = region[r0:r1, c0:c1]
        crop = lgi[r0:r1, c0:c1].astype(np.int64)
        outside, nf = cc_label(~rm)
        crop = np.where(rm, crop, outside + (int(lgi.max()) + 1))
        crop_lgi, uniq = _relabel_consecutive(crop)
        orig = np.where(uniq > lgi.max(), 0, uniq)
        clusters.append({'host': host, 'offset': (int(r0), int(c0)),
                         'lgi': crop_lgi, 'labels': orig})
    return filled_lgi, filled_labels, clusters
