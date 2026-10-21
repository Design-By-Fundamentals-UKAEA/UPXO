"""unique_rows and EdgeIncidence give np.unique(axis=0)'s results, element for element."""
import unittest
import numpy as np
from upxo.meshing.gbconformant.d3v2p1 import mesh_tables
from upxo.meshing.gbconformant.d3v2p1.mesh_tables import unique_rows, EdgeIncidence, triangle_edges


def reference(rows, inverse, counts):
    """np.unique(axis=0) with the inverse flattened (its shape differs between numpy versions)."""
    out = np.unique(rows, axis=0, return_inverse=inverse, return_counts=counts)
    if inverse:
        out = (out[0], np.asarray(out[1]).reshape(-1)) + tuple(out[2:])
    return out


class UniqueRowsTests(unittest.TestCase):
    def assertSame(self, rows, inverse=True, counts=True):
        got = unique_rows(rows, return_inverse=inverse, return_counts=counts)
        want = reference(rows, inverse, counts)
        if not (inverse or counts):
            got, want = (got,), (want,)
        self.assertEqual(len(got), len(want))
        for g, w in zip(got, want):
            self.assertEqual(g.shape, w.shape)
            self.assertEqual(g.dtype, w.dtype)
            np.testing.assert_array_equal(g, w)

    def test_random_integer_rows_of_every_width_and_dtype(self):
        rng = np.random.default_rng(0)
        for dtype in (np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32):
            info = np.iinfo(dtype)
            for width in (1, 2, 3, 4):
                for high in (3, 50, 1000):
                    low = 0 if info.min == 0 else -min(high, -info.min)
                    top = min(high, info.max)
                    rows = rng.integers(low, top + 1, size=(500, width)).astype(dtype)
                    with self.subTest(dtype=dtype.__name__, width=width, high=high):
                        for flags in ((True, True), (True, False), (False, True), (False, False)):
                            self.assertSame(rows, *flags)

    def test_negative_values_and_duplicates(self):
        rows = np.array([[-1, 5], [-1, 5], [3, -7], [0, 0], [3, -7], [-1, 4]])
        self.assertSame(rows)
        self.assertSame(np.array([[-5], [2], [-5]]))

    def test_edge_cases(self):
        self.assertSame(np.empty((0, 2), int))
        self.assertSame(np.empty((0, 3), np.int32))
        self.assertSame(np.array([[4, 4]]))
        self.assertSame(np.zeros((6, 3), int))

    def test_rows_too_wide_to_pack_use_numpy(self):
        rng = np.random.default_rng(1)
        rows = rng.integers(0, 2 ** 40, size=(300, 3))                       # span product exceeds the key limit
        rows[::7] = rows[0]
        self.assertIsNone(mesh_tables._packing(rows))
        self.assertSame(rows)
        big = np.array([[2 ** 63 - 1, 0], [-2 ** 63, 5], [0, 0]], dtype=np.int64)   # span itself overflows int64
        self.assertSame(big)
        u64 = np.array([[2 ** 63, 1], [5, 2], [5, 2]], dtype=np.uint64)
        self.assertSame(u64)

    def test_non_integer_input_uses_numpy(self):
        rng = np.random.default_rng(2)
        self.assertSame(rng.integers(0, 4, size=(100, 2)).astype(float))
        self.assertSame(rng.integers(0, 2, size=(100, 2)).astype(bool))

    def test_three_column_faces_and_label_prefixed_faces(self):
        rng = np.random.default_rng(3)
        faces = np.sort(rng.integers(0, 30000, size=(5000, 3)), axis=1)
        faces = np.vstack((faces, faces[:1000]))
        self.assertSame(faces)
        labelled = np.column_stack((rng.integers(1, 200, size=len(faces)), faces))
        self.assertSame(labelled)

    def test_unchanged_input(self):
        rows = np.array([[3, 1], [1, 2], [3, 1]])
        before = rows.copy()
        unique_rows(rows, True, True)
        np.testing.assert_array_equal(rows, before)


def soup(seed, n_nodes=60, n_triangles=400, int_dtype=np.int64):
    """Triangles over few nodes: many shared edges, some edges on three or more triangles."""
    rng = np.random.default_rng(seed)
    f = np.array([rng.choice(n_nodes, 3, replace=False) for _ in range(n_triangles)]).astype(int_dtype)
    return f


class RankCompressionTests(UniqueRowsTests):
    """Rows too wide to pack at once: leading columns are replaced by their rank, order preserved."""

    def test_wide_rows_use_the_rank_path_and_match_numpy(self):
        rng = np.random.default_rng(10)
        wide = rng.integers(0, 2 ** 20, size=(2000, 4))                      # 2**80 does not pack; ranks do
        wide[::5] = wide[0]
        wide[1::7, :3] = wide[3, :3]                                          # shared leading columns
        self.assertIsNone(mesh_tables._packing(wide))
        self.assertIsNotNone(mesh_tables._ranked_key(wide))
        self.assertSame(wide)
        three = rng.integers(0, 2 ** 40, size=(900, 3))
        three[::4] = three[1]
        self.assertIsNotNone(mesh_tables._ranked_key(three))
        self.assertSame(three)

    def test_label_prefixed_faces_of_a_large_mesh(self):
        rng = np.random.default_rng(11)
        n_nodes = 1_500_000                                                   # n**3 * labels exceeds the key limit
        faces = np.sort(rng.integers(0, n_nodes, size=(4000, 3)), axis=1)
        faces = np.vstack((faces, faces[:1500]))
        rows = np.column_stack((rng.integers(1, 300, size=len(faces)), faces))
        rows[len(faces) // 2:] = rows[:len(faces) - len(faces) // 2][::-1][:len(faces) - len(faces) // 2]   # duplicates
        self.assertIsNone(mesh_tables._packing(rows))
        self.assertSame(rows)

    def test_forced_rank_path_matches_numpy_on_ordinary_data(self):
        """A tiny key limit sends every packable input down the rank path (and the numpy fallback)."""
        rng = np.random.default_rng(12)
        saved = mesh_tables._KEY_LIMIT
        try:
            for limit in (2 ** 6, 2 ** 10, 2 ** 16):
                mesh_tables._KEY_LIMIT = limit
                for trial in range(60):
                    width = int(rng.integers(1, 6))
                    high = int(rng.integers(2, 40))
                    rows = rng.integers(-3, high, size=(int(rng.integers(1, 300)), width))
                    for flags in ((True, True), (False, False), (True, False), (False, True)):
                        self.assertSame(rows, *flags)
        finally:
            mesh_tables._KEY_LIMIT = saved

    def test_extreme_column_ranges_do_not_wrap_around(self):
        rows = np.array([[-2 ** 63, 1], [2 ** 63 - 1, 1], [0, 2], [0, 2]], dtype=np.int64)
        self.assertIsNone(mesh_tables._ranked_key(rows))
        self.assertSame(rows)
        rows = np.array([[0, -2 ** 63], [0, 2 ** 63 - 1], [1, 0], [1, 0]], dtype=np.int64)
        self.assertSame(rows)


class NodeElementIndexTests(unittest.TestCase):
    def reference(self, t, n):
        flat = t.ravel()
        return np.argsort(flat, kind='stable') // t.shape[1], np.r_[0, np.cumsum(np.bincount(flat, minlength=n))]

    def assertSame(self, t, n):
        want_order, want_offsets = self.reference(t, n)
        order, offsets = mesh_tables.node_element_index(t, n)
        np.testing.assert_array_equal(order, want_order)
        np.testing.assert_array_equal(offsets, want_offsets)
        self.assertEqual(order.dtype, want_order.dtype)

    def test_random_meshes_of_every_dtype(self):
        for seed in range(4):
            for dtype in (np.int64, np.int32):
                t = soup4(seed, dtype=dtype)
                self.assertSame(t, 50 + seed)                       # extra, unused node numbers at the end
                self.assertSame(t[:1], 50)

    def test_a_node_repeated_inside_a_tet_uses_the_argsort(self):
        t = np.array([[0, 1, 2, 3], [1, 1, 4, 5], [2, 3, 4, 5], [0, 0, 0, 1]])
        self.assertSame(t, 8)

    def test_without_scipy_uses_the_argsort(self):
        import sys
        from unittest import mock
        t = soup4(0)
        with mock.patch.dict(sys.modules, {'scipy.sparse': None}):
            self.assertSame(t, 50)

    def test_triangles(self):
        rng = np.random.default_rng(5)
        f = np.array([rng.choice(30, 3, replace=False) for _ in range(200)])
        self.assertSame(f, 35)

    def test_no_tets(self):
        order, offsets = mesh_tables.node_element_index(np.empty((0, 4), int), 5)
        self.assertEqual(len(order), 0)
        np.testing.assert_array_equal(offsets, np.zeros(6, int))


def soup4(seed, n_nodes=40, n_tets=300, dtype=np.int64):
    rng = np.random.default_rng(seed)
    return np.array([rng.choice(n_nodes, 4, replace=False) for _ in range(n_tets)]).astype(dtype)


class EdgeIncidenceTests(unittest.TestCase):
    def test_matches_the_d3v2p0_edge_table(self):
        for seed in range(4):
            for dtype in (np.int64, np.int32):
                f = soup(seed, int_dtype=dtype)
                edges = np.sort(f[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1)
                unique, inverse, counts = np.unique(edges, axis=0, return_inverse=True, return_counts=True)
                inverse = np.asarray(inverse).reshape(-1)
                inc = EdgeIncidence(f)
                np.testing.assert_array_equal(triangle_edges(f), edges)
                np.testing.assert_array_equal(inc.edges, unique)
                self.assertEqual(inc.edges.dtype, unique.dtype)
                np.testing.assert_array_equal(inc.inverse, inverse)
                np.testing.assert_array_equal(inc.counts, counts)
                np.testing.assert_array_equal(inc.order, np.argsort(inverse, kind='stable'))
                np.testing.assert_array_equal(inc.offsets, np.r_[0, np.cumsum(counts)])
                self.assertGreater(counts.max(), 2)                              # junction edges are present

    def test_empty_triangle_list(self):
        inc = EdgeIncidence(np.empty((0, 3), int))
        self.assertEqual(len(inc.edges), 0)
        self.assertEqual(len(inc.offsets), 1)


if __name__ == '__main__':
    unittest.main()
