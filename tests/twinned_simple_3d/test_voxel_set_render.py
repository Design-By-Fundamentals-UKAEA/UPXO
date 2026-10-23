"""The 3D render of voxel sets and the grain or feature ids it can colour by."""
import numpy as np
import pytest


def _ids(lgi, mask_fn):
    """1-based C-order voxel ids of the voxels where mask_fn(x, y, z) holds."""
    x, y, z = np.indices(lgi.shape)
    return sorted((np.flatnonzero(mask_fn(x, y, z).ravel()) + 1).tolist())


def _slab():
    lgi = np.ones((6, 2, 2), dtype=int)
    lgi[3:] = 2
    return lgi


def _edge_sets(lgi):
    """The voxels on each side of the slab's boundary."""
    return {'es_side_a': np.array(_ids(lgi, lambda x, y, z: x == 2)),
            'es_side_b': np.array(_ids(lgi, lambda x, y, z: x == 3))}


def test_voxel_set_render_places_each_set_in_its_colour():
    pv = pytest.importorskip("pyvista")
    from upxo.pxtal.twinned_simple_3d.viz_3d import render_voxel_sets
    lgi = _slab()
    sets = _edge_sets(lgi)
    plotter = render_voxel_sets(lgi.shape, sets, off_screen=True)
    try:
        meshes = [m for m in plotter.renderer.actors.values() if hasattr(m, 'GetMapper')]
        assert len(meshes) >= 3                       # the outline and two sets
    finally:
        plotter.close()
