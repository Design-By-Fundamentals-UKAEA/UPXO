"""Label and connected-component operations for UPXO image data."""

import numpy as np
from scipy.ndimage import generate_binary_structure
from scipy.ndimage import label as scipy_label


def reindex_labels(label_image, background=0, start=1):
    """Reindex non-background labels to consecutive integers."""
    label_image = np.asarray(label_image)
    unique_labels = np.unique(label_image)
    unique_labels = unique_labels[unique_labels != background]
    label_map = {old: new for new, old in enumerate(unique_labels,
                                                    start=start)}
    vectorized_map = np.vectorize(lambda x: label_map.get(x, background))
    return vectorized_map(label_image)


def reindex_arrays_by_image_pair(old_image, new_image, old_arrays):
    """Reindex ID arrays using the old-image to new-image mapping."""
    old_image = np.asarray(old_image)
    new_image = np.asarray(new_image)
    unique_pairs = np.unique(
        np.vstack((old_image.ravel(), new_image.ravel())).T, axis=0
    )
    old_to_new = {old_id: new_id for old_id, new_id in unique_pairs}
    return [np.array([old_to_new.get(old_id) for old_id in old_array],
                     dtype=np.int32)
            for old_array in old_arrays]


def binary_structure(ndim=3, connectivity=1):
    """Return a SciPy binary structure for connected-component labelling."""
    if connectivity not in tuple(range(1, ndim+1)):
        raise ValueError(f"connectivity must be in [1, {ndim}].")
    return generate_binary_structure(ndim, connectivity)


def label_components(binary_image, connectivity=1, labeler=None):
    """Label one binary image using SciPy-compatible labelling."""
    structure = binary_structure(np.ndim(binary_image), connectivity)
    labeler = scipy_label if labeler is None else labeler
    return labeler(np.asarray(binary_image).astype(np.uint8),
                   structure=structure)


def label_multistate_components(state_image, states=None, connectivity=1,
                                labeler=None):
    """Label each state independently and merge labels into one image."""
    state_image = np.asarray(state_image)
    states = np.unique(state_image) if states is None else np.asarray(states)
    lgi = None
    s_gid = {}
    s_n = []
    gid_s = []

    for i, state in enumerate(states):
        labels, _ = label_components(state_image == state,
                                     connectivity=connectivity,
                                     labeler=labeler)
        labels = labels.astype(np.int32, copy=False)
        if i == 0:
            lgi = labels
        else:
            labels[labels > 0] += int(lgi.max())
            lgi = lgi + labels

        gids = tuple(np.delete(np.unique(labels), 0))
        s_gid[int(state)] = gids
        s_n.append(len(gids))
        gid_s.extend([int(state) for _ in gids])

    return lgi.astype(np.int32, copy=False), s_gid, s_n, gid_s


def coerce_lgi_uint8(lgi):
    """Return a uint8 2D labelled image or None when invalid."""
    if isinstance(lgi, np.ndarray) and np.size(lgi) > 0 and np.ndim(lgi) == 2:
        return lgi if lgi.dtype.name == 'uint8' else lgi.astype(np.uint8)
    return None
