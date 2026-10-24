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


def test_feature_ids_by_grain_use_the_global_grain_ids():
    from upxo.pxtal.twinned_simple_3d.viz_3d import build_feature_ids
    lgi = _slab()
    sets = _edge_sets(lgi)
    ids, n = build_feature_ids(lgi, sets, mode='grain')
    assert n == 2
    assert set(np.unique(ids)) == {0, 1, 2}
    assert (ids[2] == 1).all() and (ids[3] == 2).all() and (ids[:2] == 0).all()


def test_feature_ids_by_feature_split_clusters_and_grains():
    from upxo.pxtal.twinned_simple_3d.viz_3d import build_feature_ids
    lgi = np.ones((9, 3, 3), dtype=int)
    lgi[4:] = 2
    sets = {'a': np.flatnonzero(np.isin(np.indices(lgi.shape)[0].ravel(), [0, 1, 7, 8])) + 1}   # two clusters in two grains
    ids, n = build_feature_ids(lgi, sets, mode='feature')
    assert n == 2 and set(np.unique(ids)) == {0, 1, 2}
    both = {'a': np.flatnonzero(np.isin(np.indices(lgi.shape)[0].ravel(), [3, 4])) + 1}          # one cluster across two grains
    ids, n = build_feature_ids(lgi, both, mode='feature')
    assert n == 2                                                                                  # features stay inside one grain
    with pytest.raises(ValueError):
        build_feature_ids(lgi, sets, mode='colour')


@pytest.mark.parametrize("mode", ['grain', 'feature'])
def test_voxel_set_render_by_id_runs(mode):
    pytest.importorskip("pyvista")
    from upxo.pxtal.twinned_simple_3d.viz_3d import render_voxel_sets
    lgi = _slab()
    sets = _edge_sets(lgi)
    plotter = render_voxel_sets(lgi.shape, sets, off_screen=True, lgi=lgi, color_by=mode)
    try:
        assert plotter is not None
    finally:
        plotter.close()
    with pytest.raises(ValueError):
        render_voxel_sets(lgi.shape, sets, off_screen=True, color_by=mode)
