import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from upxo.meshing.gbconformant.d3v2p0.surface_export import (
    save_surface, surface_polydata, tet_grid)


def a_surface(**extra):
    return SimpleNamespace(
        points=np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1.]]),
        triangles=np.array([[0, 1, 2], [0, 1, 3]]),
        grain_pairs=np.array([[1, 2], [2, 3]]), **extra)


class SurfaceExportTests(unittest.TestCase):
    def test_surface_arrays(self):
        mesh = surface_polydata(a_surface(exterior=np.array([True, False]),
                                          rve_face=np.array([1, 0])))
        self.assertEqual(mesh.n_cells, 2)
        self.assertEqual(mesh.cell_data['grain_a'].tolist(), [1, 2])
        self.assertEqual(mesh.cell_data['grain_b'].tolist(), [2, 3])
        self.assertEqual(mesh.cell_data['is_rve_cap'].tolist(), [1, 0])
        self.assertEqual(mesh.cell_data['rve_face'].tolist(), [1, 0])

    def test_optional_arrays_and_overrides(self):
        mesh = surface_polydata(a_surface())
        self.assertNotIn('is_rve_cap', mesh.cell_data)
        mesh = surface_polydata(a_surface(), exterior=np.array([0, 1]), rve_face=np.array([3, 4]))
        self.assertEqual(mesh.cell_data['is_rve_cap'].tolist(), [0, 1])
        self.assertEqual(mesh.cell_data['rve_face'].tolist(), [3, 4])
        self.assertIs(surface_polydata(mesh), mesh)

    def test_save(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as folder:
            path = save_surface(Path(folder) / 'sub' / 'surface.vtp', a_surface())
            self.assertTrue(path.is_file())

    def test_tet_grid(self):
        tets = SimpleNamespace(
            points=np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1.]]),
            tetrahedra=np.array([[0, 1, 2, 3]]), grain_ids=np.array([7]),
            quality=np.array([0.5]))
        grid = tet_grid(tets)
        self.assertEqual(grid.n_cells, 1)
        self.assertEqual(grid.cell_data['grain_id'].tolist(), [7])
        self.assertEqual(grid.cell_data['minSICN'].tolist(), [0.5])
        self.assertIn('quality', tet_grid(tets, quality_name='quality').cell_data)
        tets.quality = None
        self.assertNotIn('minSICN', tet_grid(tets).cell_data)


if __name__ == '__main__':
    unittest.main()
