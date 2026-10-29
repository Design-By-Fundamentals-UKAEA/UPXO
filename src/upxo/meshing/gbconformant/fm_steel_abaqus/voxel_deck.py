"""Voxel (non-conformal) FM-steel deck: one C3D8 hexahedron per voxel of a label array.

The deck is written by the same writer as the conformal tetrahedral deck (:func:`.export.write_deck`),
so it has the same files, Part and Instance, element and node sets, set names, materials, sections,
step and deck check. Boundaries follow the voxel faces (a staircase); nothing is smoothed or fitted.

The label array is one of:

* a cleaned label array (pipeline labels), optionally resampled by a scale factor
  (:func:`resample_labels`: nearest voxel when refining, majority vote when coarsening);
* the tetrahedral mesh turned back into voxels (:func:`voxelize_tetrahedra`), which shows how far the
  conformal mesh moved from the voxels it was built from.

Coordinates are in mesh units where one raw voxel edge is ``mesh_voxel_length`` long, as for the
conformal deck.
"""
from __future__ import annotations

import numpy as np

from .elements import grid_hexahedra


def label_hexahedra(labels, voxel_size=1.0):
    """Nodes, C3D8 connectivity and label of every voxel of ``labels``.

    ``voxel_size`` is one edge length, or three (x, y, z). Element ``e`` is voxel
    ``np.unravel_index(e, labels.shape)``.
    """
    a = np.asarray(labels)
    if a.ndim != 3 or a.dtype.kind not in 'iu':
        raise ValueError('labels must be a 3D integer array')
    size = np.broadcast_to(np.asarray(voxel_size, dtype=float), (3,))
    if not np.all(size > 0):
        raise ValueError('voxel_size must be > 0')
    points, hexahedra = grid_hexahedra(a.shape, tuple(size))
    return points, hexahedra, a.ravel().astype(np.int64)


def _axis_targets(n_old, n_new):
    """Target index of every source voxel along one axis (coarsening: several sources per target)."""
    return np.minimum((np.arange(n_old) * n_new) // n_old, n_new - 1)


def resample_labels(labels, factor):
    """``(resampled labels, report)``: ``labels`` with ``round(n * factor)`` voxels along each axis.

    A factor above 1 refines (each new voxel takes the label of the old voxel under its centre); below 1
    it coarsens, and each new voxel takes the label most of the old voxels inside it carry (ties go to the
    smaller label). The report lists the labels that no voxel carries any more.
    """
    a = np.asarray(labels)
    if a.ndim != 3 or a.dtype.kind not in 'iu':
        raise ValueError('labels must be a 3D integer array')
    factor = float(factor)
    if not factor > 0:
        raise ValueError('The scale factor must be > 0.')
    old = np.array(a.shape)
    new = np.maximum(1, np.rint(old * factor).astype(np.int64))
    out = a
    # refine first, axis by axis, by the nearest old voxel
    for axis in range(3):
        if new[axis] > old[axis]:
            source = np.minimum(((np.arange(new[axis]) + 0.5) * old[axis] / new[axis]).astype(np.int64), old[axis] - 1)
            out = np.take(out, source, axis=axis)
    # then coarsen the remaining axes together by majority vote
    shape = np.array(out.shape)
    if np.any(new < shape):
        targets = [_axis_targets(shape[ax], new[ax]) if new[ax] < shape[ax] else np.arange(shape[ax])
                   for ax in range(3)]
        ti, tj, tk = np.meshgrid(*targets, indexing='ij')
        target = ((ti * new[1] + tj) * new[2] + tk).ravel()
        values, index = np.unique(out.ravel(), return_inverse=True)
        n_values = len(values)
        keys, counts = np.unique(target * n_values + index.ravel(), return_counts=True)
        cell, value = keys // n_values, keys % n_values
        order = np.lexsort((value, -counts, cell))
        first = order[np.r_[True, cell[order][1:] != cell[order][:-1]]]
        result = np.empty(int(np.prod(new)), dtype=a.dtype)
        result[cell[first]] = values[value[first]]
        out = result.reshape(tuple(new))
    else:
        out = np.ascontiguousarray(out)
    before, after = np.unique(a), np.unique(out)
    report = {'factor': factor, 'shape_before': [int(n) for n in old], 'shape_after': [int(n) for n in out.shape],
              'voxel_scale': [float(o / n) for o, n in zip(old, out.shape)],
              'labels_before': int(len(before)), 'labels_after': int(len(after)),
              'labels_lost': [int(v) for v in np.setdiff1d(before, after)]}
    return out, report


def voxelize_tetrahedra(points, tetrahedra, grains, voxel_size, *, low=None, high=None, chunk=100_000):
    """``(labels, report)``: the grain of the tetrahedron that holds each voxel centre.

    The grid covers the box ``low``..``high`` (the mesh's bounding box by default) with voxels as close to
    ``voxel_size`` as fit a whole number of times along each axis. A centre that no tetrahedron holds
    (numerical gaps on shared faces) takes the grain of the nearest tetrahedron centroid.
    """
    p = np.asarray(points, dtype=float)
    t = np.asarray(tetrahedra, dtype=np.int64)
    g = np.asarray(grains, dtype=np.int64)
    if t.ndim != 2 or t.shape[1] != 4 or g.shape != (len(t),):
        raise ValueError('Expected Mx4 tetrahedra and one grain per tetrahedron')
    if not float(voxel_size) > 0:
        raise ValueError('voxel_size must be > 0')
    low = p.min(axis=0) if low is None else np.asarray(low, dtype=float)
    high = p.max(axis=0) if high is None else np.asarray(high, dtype=float)
    shape = np.maximum(1, np.rint((high - low) / float(voxel_size)).astype(np.int64))
    h = (high - low) / shape
    labels = np.zeros(int(np.prod(shape)), dtype=np.int64)
    found = np.zeros(len(labels), dtype=bool)
    eps = 1e-9
    for start in range(0, len(t), chunk):
        v = p[t[start:start + chunk]]
        lo = np.clip(np.ceil((v.min(axis=1) - low) / h - 0.5 - eps), 0, shape - 1).astype(np.int64)
        hi = np.clip(np.floor((v.max(axis=1) - low) / h - 0.5 + eps), -1, shape - 1).astype(np.int64)
        extent = np.maximum(hi - lo + 1, 0)
        n_box = extent.prod(axis=1)
        if not n_box.sum():
            continue
        owner = np.repeat(np.arange(len(v)), n_box)
        local = np.arange(int(n_box.sum())) - np.repeat(np.cumsum(n_box) - n_box, n_box)
        ny, nz = extent[owner, 1], extent[owner, 2]
        ijk = lo[owner] + np.column_stack((local // (ny * nz), (local // nz) % ny, local % nz))
        centre = low + (ijk + 0.5) * h
        base = v[:, 0]
        frame = np.stack((v[:, 1] - base, v[:, 2] - base, v[:, 3] - base), axis=2)
        inverse = np.linalg.inv(frame)
        lam = np.einsum('nij,nj->ni', inverse[owner], centre - base[owner])
        inside = (lam >= -eps).all(axis=1) & (lam.sum(axis=1) <= 1 + eps)
        flat = (ijk[:, 0] * shape[1] + ijk[:, 1]) * shape[2] + ijk[:, 2]
        flat, tet = flat[inside], owner[inside] + start
        fresh = ~found[flat]
        flat, tet = flat[fresh], tet[fresh]
        flat, first = np.unique(flat, return_index=True)
        labels[flat] = g[tet[first]]
        found[flat] = True
    missing = np.flatnonzero(~found)
    if len(missing):
        from scipy.spatial import cKDTree

        centroid = p[t].mean(axis=1)
        mi, mj, mk = np.unravel_index(missing, tuple(shape))
        _, nearest = cKDTree(centroid).query(low + (np.column_stack((mi, mj, mk)) + 0.5) * h)
        labels[missing] = g[nearest]
    labels = labels.reshape(tuple(shape))
    present = np.unique(labels)
    report = {'shape': [int(n) for n in shape], 'voxel_size': [float(x) for x in h],
              'low': [float(x) for x in low], 'high': [float(x) for x in high],
              'voxels_filled_by_nearest': int(len(missing)),
              'grains_in_mesh': int(len(np.unique(g))), 'grains_in_voxels': int(len(present)),
              'grains_lost': [int(v) for v in np.setdiff1d(np.unique(g), present)]}
    return labels, report


def write_voxel_deck(directory, labels, hierarchy, identity, settings=None, *, voxel_size=1.0,
                     mesh_voxel_length=1.0, meshed_labels=None, audit=True, source=None):
    """Write the voxel deck of ``labels`` into the new ``directory``; return the summary of :func:`.export.write_deck`.

    ``voxel_size`` is the edge of one voxel of ``labels`` in mesh units and ``mesh_voxel_length`` that of one
    raw voxel (they are equal for the cleaned labels at their own resolution). ``source`` (a dictionary)
    is stored in ``deck_info.json`` as ``voxel_source``.
    """
    import json
    from pathlib import Path

    from .export import _encode, write_deck

    points, hexahedra, grains = label_hexahedra(labels, voxel_size)
    result = write_deck(directory, points, hexahedra, grains, hierarchy, identity, settings,
                        mesh_voxel_length=mesh_voxel_length, meshed_labels=meshed_labels, audit=audit)
    info = result['info']
    info['voxel_source'] = dict(source or {})
    info['voxel_grid'] = {'shape': [int(n) for n in np.shape(labels)],
                          'voxel_size_mesh_units': [float(x) for x in np.broadcast_to(np.asarray(voxel_size, float), (3,))]}
    (Path(result['directory']) / 'deck_info.json').write_text(json.dumps(info, indent=2, default=_encode), encoding='utf8')
    return result
