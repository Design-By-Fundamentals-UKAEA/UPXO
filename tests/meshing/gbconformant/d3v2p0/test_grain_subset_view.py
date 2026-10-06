import json
import unittest
from unittest.mock import patch
import numpy as np
import pyvista as pv
from upxo.meshing.gbconformant.d3v2p0 import grain_subset_view as gsv

MODULE = 'upxo.meshing.gbconformant.d3v2p0.grain_subset_view'


def surface():
    """Four triangles: grains 1|2 shared, 2|3 shared, a cap of grain 1, a cap of grain 3."""
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1]], float)
    faces = np.array([[0, 1, 2], [1, 3, 2], [0, 1, 4], [1, 5, 4]])
    s = pv.PolyData(points, np.column_stack((np.full(4, 3), faces)).ravel())
    s.cell_data['grain_a'] = np.array([1, 2, 1, 3])
    s.cell_data['grain_b'] = np.array([2, 3, -1, -1])
    s.cell_data['is_rve_cap'] = np.array([0, 0, 1, 1], np.uint8)
    return s


class GrainSubsetTests(unittest.TestCase):
    def test_subset_mesh_lists_each_selected_grain_shell(self):
        mesh = gsv.grain_subset_mesh(surface(), [1, 2])
        self.assertEqual(sorted(mesh.cell_data['grain'].tolist()), [1, 1, 2, 2])   # 1|2 twice, 1-cap, 2|3
        mesh = gsv.grain_subset_mesh(surface(), [3])
        self.assertEqual(sorted(mesh.cell_data['grain'].tolist()), [3, 3])
        self.assertEqual(gsv.grain_subset_mesh(surface(), [9]).n_cells, 0)

    def test_cap_owner_is_not_taken_from_grain_b(self):
        s = surface()
        s.cell_data['grain_b'] = np.array([2, 3, 1, 3])                 # caps repeat the owner
        self.assertEqual(sorted(gsv.grain_subset_mesh(s, [1]).cell_data['grain'].tolist()), [1, 1])

    def test_percentile_selection(self):
        vols = {str(i): float(i) for i in range(1, 11)}
        self.assertEqual(gsv.grains_above_percentile(vols, 80), [9, 10])
        self.assertEqual(gsv.grains_above_percentile({k: {'volume': v} for k, v in vols.items()}, 80), [9, 10])
        self.assertEqual(gsv.grains_above_percentile({}, 90), [])
        for bad in (-1, 100, float('nan')):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                gsv.grains_above_percentile(vols, bad)

    def test_launch_runs_a_separate_process(self):
        import tempfile, os
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 's.vtp')
            surface().save(path)
            with patch(MODULE + '.subprocess.Popen') as popen:
                gsv.launch_grain_subset_view(path, [3, 1], title='T', note='N')
            command = popen.call_args[0][0]
            self.assertEqual(command[1:3], ['-m', MODULE])
            self.assertEqual(command[3], path)
            self.assertEqual(json.load(open(command[4])), {'ids': [1, 3], 'note': 'N'})
            os.remove(command[4])
        with self.assertRaises(FileNotFoundError):
            gsv.launch_grain_subset_view('missing.vtp', [1])

    def test_command_line_reads_ids(self):
        import tempfile, os
        with tempfile.TemporaryDirectory() as tmp:
            ids = os.path.join(tmp, 'ids.json')
            json.dump([2, 3], open(ids, 'w'))
            with patch(MODULE + '.show_grain_subset') as show:
                gsv.main(['s.vtp', ids, '--title', 'X'])
            show.assert_called_once_with('s.vtp', [2, 3], title='X', note=None, screenshot=None)


if __name__ == '__main__':
    unittest.main()
