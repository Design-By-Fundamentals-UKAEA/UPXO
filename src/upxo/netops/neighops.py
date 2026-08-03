"""Neighbour graph operations for labelled 2D/3D arrays."""

from collections import defaultdict
import numpy as np

try:  # cc3d is preferred when available.
    import cc3d
except ImportError:  # pragma: no cover - depends on optional runtime package.
    cc3d = None


def default_connectivity(ndim):
    """Return UPXO default connectivity for labelled arrays."""
    if ndim == 3:
        return 6
    if ndim == 2:
        return 4
    raise ValueError('Only 2D and 3D arrays are supported.')


def _axis_offsets(ndim):
    """Return face-neighbour offsets for 2D/3D arrays."""
    offsets = []
    for axis in range(ndim):
        for step in (-1, 1):
            offset = [0]*ndim
            offset[axis] = step
            offsets.append(tuple(offset))
    return offsets


def _adjacency_by_shifts(label_image, include_self=False,
                         ignore_labels=(0,)):
    """Fallback adjacency extraction using face-neighbour shifts."""
    label_image = np.asarray(label_image)
    ignore = set(ignore_labels or ())
    neigh = defaultdict(set)
    for label in np.unique(label_image):
        if label not in ignore:
            neigh[int(label)] = set()

    for axis in range(label_image.ndim):
        left = [slice(None)]*label_image.ndim
        right = [slice(None)]*label_image.ndim
        left[axis] = slice(0, -1)
        right[axis] = slice(1, None)
        a = label_image[tuple(left)]
        b = label_image[tuple(right)]
        mask = a != b
        for la, lb in np.unique(np.column_stack((a[mask], b[mask])), axis=0):
            if la not in ignore and lb not in ignore:
                neigh[int(la)].add(int(lb))
                neigh[int(lb)].add(int(la))

    if include_self:
        for label in list(neigh):
            neigh[label].add(label)
    return {label: sorted(vals) for label, vals in neigh.items()}


def adjacency_from_labels(label_image, connectivity=None, include_self=False,
                          ignore_labels=(0,)):
    """Return first-order label adjacency for a labelled 2D/3D array."""
    label_image = np.asarray(label_image)
    connectivity = default_connectivity(label_image.ndim) if connectivity is None else connectivity

    if cc3d is not None and label_image.ndim == 3:
        contacts = cc3d.contacts(label_image, connectivity=connectivity,
                                 surface_area=False)
        ignore = set(ignore_labels or ())
        neigh = {int(label): set() for label in np.unique(label_image)
                 if label not in ignore}
        for label_pair in contacts.keys():
            left, right = map(int, label_pair)
            if left in ignore or right in ignore or left == right:
                continue
            neigh.setdefault(left, set()).add(right)
            neigh.setdefault(right, set()).add(left)
        if include_self:
            for label in list(neigh):
                neigh[label].add(label)
        return {label: sorted(vals) for label, vals in neigh.items()}

    return _adjacency_by_shifts(label_image, include_self=include_self,
                                ignore_labels=ignore_labels)


def has_neighbor(neigh_map, parent_label, other_label):
    """Return whether ``other_label`` is an O(1) neighbour of ``parent_label``."""
    return other_label in neigh_map.get(parent_label, ())


def upto_nth_order_neighbors(neigh_map, label, order, include_parent=False):
    """Return neighbours up to the requested graph distance."""
    visited = {label} if include_parent else set()
    frontier = {label}
    for _ in range(int(order)):
        next_frontier = set()
        for item in frontier:
            next_frontier.update(neigh_map.get(item, ()))
        next_frontier -= visited
        visited.update(next_frontier)
        frontier = next_frontier
    if not include_parent:
        visited.discard(label)
    return sorted(visited)


def nth_order_neighbors(neigh_map, label, order, include_parent=False):
    """Return neighbours exactly at the requested graph distance."""
    previous = {label}
    frontier = {label}
    for _ in range(int(order)):
        next_frontier = set()
        for item in frontier:
            next_frontier.update(neigh_map.get(item, ()))
        next_frontier -= previous
        previous.update(next_frontier)
        frontier = next_frontier
    if include_parent and order == 0:
        frontier.add(label)
    return sorted(frontier)
