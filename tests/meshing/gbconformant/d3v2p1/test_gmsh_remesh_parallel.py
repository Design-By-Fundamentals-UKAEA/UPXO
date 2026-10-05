"""d3v2p0's Gmsh interface and joint remesh tests run against d3v2p1, plus
byte-identical chart import and a valid joint remesh."""
import unittest
import numpy as np
from tests.meshing.gbconformant.d3v2p0 import test_gmsh_interfaces as base_interfaces
from tests.meshing.gbconformant.d3v2p0 import test_gmsh_closed as base_closed
from upxo.meshing.gbconformant.d3v2p0 import discrete_topology as reference_topology
from upxo.meshing.gbconformant.d3v2p1 import discrete_topology
from upxo.meshing.gbconformant.d3v2p1.gmsh_interfaces import remesh_interfaces_gmsh
from upxo.meshing.gbconformant.d3v2p1.gmsh_closed import remesh_closed_rve_gmsh
from tests.meshing.gbconformant.d3v2p1.test_validation_parallel import closed_block_rve


class _Swap:
    targets = ()

    def setUp(self):
        self._saved = [(m, n, getattr(m, n)) for m, n, _ in self.targets]
        for m, n, v in self.targets:
            setattr(m, n, v)

    def tearDown(self):
        for m, n, v in self._saved:
            setattr(m, n, v)


class D3v2p1InterfaceTests(_Swap, base_interfaces.GmshInterfaceTests):
    targets = ((base_interfaces, 'remesh_interfaces_gmsh', remesh_interfaces_gmsh),)


class D3v2p1JointTests(_Swap, base_closed.JointRemeshTests):
    targets = ((base_closed, 'remesh_interfaces_gmsh', remesh_interfaces_gmsh),
               (base_closed, 'remesh_closed_rve_gmsh', remesh_closed_rve_gmsh))


class _FakeGmsh:
    def __init__(self):
        self.calls, self.text = [], None
        outer = self

        class Model:
            def addDiscreteEntity(self, *args):
                outer.calls.append(args)
        self.model = Model()

    def merge(self, path):
        with open(path, 'rb') as stream:
            self.text = stream.read()


class IdenticalOutputTests(unittest.TestCase):
    def test_chart_import_file_is_byte_identical(self):
        rve = closed_block_rve()
        keys = np.column_stack((rve.grain_pairs, rve.rve_face))
        _, patch = np.unique(keys, axis=0, return_inverse=True)
        patch = patch.ravel()
        for split in (False, True):                  # whole patches; one triangle per chart
            with self.subTest(split=split):
                groups = [np.flatnonzero(patch == k) for k in np.unique(patch)]
                if split:
                    groups = [np.array([t]) for g in groups for t in g]
                a, b = _FakeGmsh(), _FakeGmsh()
                reference_topology.add_chart_topology(a, rve.points, rve.triangles, groups)
                discrete_topology.add_chart_topology(b, rve.points, rve.triangles, groups)
                self.assertEqual(a.text, b.text)
                self.assertEqual(a.calls, b.calls)

    def test_joint_remesh_is_valid(self):
        # d3v2p0's joint remesh is not repeatable call to call on this fixture
        # (Gmsh state), so the d3v2p1 output is checked for validity instead.
        from upxo.meshing.gbconformant.d3v2p1.tet_validation import validate_tet_surfaces
        rve = closed_block_rve()
        kw = dict(mesh_size=.8, check_intersections=True, intersection_retries=2, minimum_facet_angle=15.,
                  minimum_remesh_opening=30.)
        for workers in (1, 3):
            with self.subTest(workers=workers):
                b = remesh_closed_rve_gmsh(rve, n_workers=workers, **kw)
                self.assertEqual(b.report['grains'], 64)
                self.assertAlmostEqual(sum(b.report['enclosed_grain_volumes'].values()), 512.)
                v = validate_tet_surfaces(b, check_intersections=True, minimum_facet_angle=15., n_workers=workers)
                self.assertEqual(v['blockers'], [])


if __name__ == '__main__':
    unittest.main()
