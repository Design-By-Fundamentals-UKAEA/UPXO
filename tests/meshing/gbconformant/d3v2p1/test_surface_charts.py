"""d3v2p1.surface_charts.is_disk_chart and gmsh_interfaces._directed_boundary answer as d3v2p0's do."""
import unittest
import numpy as np
from upxo.meshing.gbconformant.d3v2p0.surface_charts import is_disk_chart as reference, disk_charts
from upxo.meshing.gbconformant.d3v2p0.gmsh_interfaces import _directed_boundary as reference_boundary
from upxo.meshing.gbconformant.d3v2p1 import surface_charts as fast
from upxo.meshing.gbconformant.d3v2p1.gmsh_interfaces import _directed_boundary as fast_boundary


def grid(n):
    """An n x n square of triangles (a disk), consistently wound."""
    f = []
    for i in range(n):
        for j in range(n):
            a, b, c, d = i * (n + 1) + j, i * (n + 1) + j + 1, (i + 1) * (n + 1) + j, (i + 1) * (n + 1) + j + 1
            f += [[a, b, d], [a, d, c]]
    return np.array(f)


def shapes():
    out = {'disk': grid(3), 'single': np.array([[0, 1, 2]]), 'two_disjoint': np.vstack((grid(1), grid(1) + 100)),
           'bowtie': np.array([[0, 1, 2], [0, 3, 4]]),                                 # two triangles meeting at a vertex
           'tetra_shell': np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]),       # closed: no boundary
           'branching_edge': np.array([[0, 1, 2], [0, 1, 3], [0, 1, 4]]),               # an edge on three triangles
           'annulus': np.array([[0, 1, 4], [1, 5, 4], [1, 2, 5], [2, 6, 5], [2, 3, 6], [3, 7, 6], [3, 0, 7], [0, 4, 7]])}
    rng = np.random.default_rng(0)
    big = grid(6)
    for k in range(12):
        keep = rng.random(len(big)) < .7
        out[f'random{k}'] = big[keep] if keep.any() else big[:1]
    for k, patch in enumerate(disk_charts(big, max_triangles=9)):
        out[f'chart{k}'] = big[patch]
    return out


class DiskChartTests(unittest.TestCase):
    def test_every_shape_and_dtype(self):
        seen = set()
        for dtype in (np.int64, np.int32):
            for name, tri in shapes().items():
                tri = tri.astype(dtype)
                with self.subTest(shape=name, dtype=dtype.__name__):
                    answer = reference(tri)
                    self.assertEqual(fast.is_disk_chart(tri), answer)
                    seen.add(answer)
        self.assertEqual(seen, {True, False})                       # both answers occur

    def test_disk_charts_is_d3v2p0s(self):
        self.assertIs(fast.disk_charts, disk_charts)


class DirectedBoundaryTests(unittest.TestCase):
    def test_boundary_signs_match(self):
        for dtype in (np.int64, np.int32):
            for name, tri in shapes().items():
                tri = tri.astype(dtype)
                with self.subTest(shape=name, dtype=dtype.__name__):
                    try:
                        want = reference_boundary(tri)
                    except RuntimeError as error:
                        with self.assertRaises(RuntimeError) as got:
                            fast_boundary(tri)
                        self.assertEqual(str(got.exception), str(error))
                    else:
                        self.assertEqual(fast_boundary(tri), want)

    def test_inconsistent_winding_raises_in_both(self):
        tri = grid(2).copy()
        tri[1] = tri[1][[0, 2, 1]]
        for function in (reference_boundary, fast_boundary):
            with self.assertRaises(RuntimeError):
                function(tri)


if __name__ == '__main__':
    unittest.main()
