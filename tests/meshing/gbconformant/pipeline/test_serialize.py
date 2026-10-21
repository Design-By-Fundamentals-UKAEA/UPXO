import json
import unittest
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from upxo.meshing.gbconformant.d3v2p0.gmsh_interfaces import RemeshedInterfaces
from upxo.meshing.gbconformant.d3v2p0.gmsh_tets import GrainTetrahedra
from upxo.meshing.gbconformant.d3v2p0.interfaces import Interfaces
from upxo.meshing.gbconformant.d3v2p0.rve_caps import ClosedRVE
from upxo.meshing.gbconformant.d3v2p0.thin_freeze import FrozenGeometry
from upxo.meshing.gbconformant.pipeline import SerializeError, read_value, write_value
from upxo.meshing.gbconformant.pipeline.serialize import safe_name

from ..fm_steel_abaqus.helpers import work_dir, write_raw_folder


@dataclass
class Foreign:                     # not a result class of this package
    x: int = 1


def roundtrip(value):
    with work_dir() as root:
        write_value(root / 'v', value)
        return read_value(root / 'v')


class CodecTests(unittest.TestCase):
    def test_plain_structures(self):
        value = {'a': 1, 'b': [1.5, 'x', None, True], 't': (1, (2, 3)), 's': {3, 1, 2},
                 5: 'int key', (1, 'p'): [1, 2], 'path': Path('some/where'),
                 'nan': float('nan'), 'big': 2 ** 70}
        out = roundtrip(value)
        self.assertEqual({k: v for k, v in out.items() if k != 'nan'},
                         {k: v for k, v in value.items() if k != 'nan'})
        self.assertTrue(np.isnan(out['nan']))
        self.assertIsInstance(out['t'], tuple)
        self.assertIsInstance(out['s'], set)
        self.assertIsInstance(out['path'], Path)

    def test_arrays_keep_dtype_and_shape(self):
        arrays = {'i': np.arange(6, dtype=np.int32).reshape(2, 3), 'f': np.linspace(0, 1, 5),
                  'b': np.array([True, False]), 'e': np.empty((0, 3)), 'u': np.array([7], np.uint8)}
        out = roundtrip({'nested': [arrays]})['nested'][0]
        for key, array in arrays.items():
            self.assertEqual(out[key].dtype, array.dtype, key)
            np.testing.assert_array_equal(out[key], array)

    def test_numpy_scalars_become_python_numbers(self):
        out = roundtrip([np.int64(4), np.float32(0.5), np.bool_(True)])
        self.assertEqual(out, [4, 0.5, True])
        self.assertEqual([type(v) for v in out], [int, float, bool])

    def test_result_dataclasses(self):
        report = {'per_grain': {1: {'volume': 2.5}, 7: {'volume': 1.0}}, 'angles': np.arange(4.),
                  'status': 'ok', 'pairs': [(1, 2)]}
        pts = np.random.default_rng(0).random((5, 3))
        tri = np.array([[0, 1, 2], [2, 3, 4]])
        pairs = np.array([[1, 2], [2, 3]])
        objects = [
            Interfaces(pts, tri, pairs, pts.copy(), np.zeros(5, int), np.zeros((5, 3), bool), 4,
                       np.array([[0, 1]]), {'moved': 3}),
            RemeshedInterfaces(pts, tri, pairs, np.array([[0, 1]]), report),
            ClosedRVE(pts, tri, pairs, np.array([False, True]), np.array([-1, 2]), report),
            GrainTetrahedra(pts, np.array([[0, 1, 2, 3]]), np.array([5]), np.array([.4]), report),
        ]
        for original in objects:
            back = roundtrip(original)
            self.assertIs(type(back), type(original))
            for name in original.__dataclass_fields__:
                a, b = getattr(original, name), getattr(back, name)
                if isinstance(a, np.ndarray):
                    np.testing.assert_array_equal(a, b)
                    self.assertEqual(a.dtype, b.dtype)
                elif name == 'report':
                    self.assertEqual(sorted(b['per_grain']), [1, 7])
                    self.assertEqual(b['pairs'], [(1, 2)])
                    np.testing.assert_array_equal(b['angles'], np.arange(4.))
                else:
                    self.assertEqual(a, b)

    def test_frozen_geometry_is_rebuilt(self):
        labels = np.ones((3, 3, 3), dtype=np.int32)
        labels[2:] = 2
        original = FrozenGeometry(labels, [2], (1., 2., 3.))
        back = roundtrip(original)
        self.assertEqual(back.ids, original.ids)
        np.testing.assert_array_equal(back.labels, original.labels)
        np.testing.assert_array_equal(back.spacing, original.spacing)
        self.assertEqual(back.faces, original.faces)

    def test_hierarchy_of_a_raw_folder(self):
        from upxo.meshing.gbconformant.fm_steel_abaqus import (
            extract_subset, load_hierarchy)
        with work_dir() as root:
            write_raw_folder(root)
            hierarchy = extract_subset(load_hierarchy(root, 'blk'), (1, 0, 0), (5, 4, 4))
            back = roundtrip(hierarchy)
        self.assertEqual(back.origin, (1, 0, 0))
        self.assertEqual(back.features, hierarchy.features)
        self.assertEqual(back.global_ids, hierarchy.global_ids)
        self.assertEqual(back.pck_order, hierarchy.pck_order)
        np.testing.assert_array_equal(back.lgi, hierarchy.lgi)

    def test_files_and_names(self):
        with work_dir() as root:
            self.assertEqual(write_value(root / 'plain', {'a': 1}), {'json': 'plain.json', 'npz': None})
            self.assertFalse((root / 'plain.npz').exists())
            self.assertEqual(write_value(root / 'arr', np.zeros(2))['npz'], 'arr.npz')
        self.assertEqual(safe_name('load_lgi.hierarchy'), 'load_lgi.hierarchy')
        self.assertEqual(safe_name('a/b:c d'), 'a_b_c_d')


class RefusalTests(unittest.TestCase):
    def test_values_that_cannot_be_stored(self):
        with work_dir() as root:
            for value in (object(), Foreign(), np.array([None, 1], dtype=object), {'k': lambda: 0},
                          b'raw'):
                with self.assertRaises(SerializeError, msg=repr(value)):
                    write_value(root / 'x', value)

    def test_unknown_version_and_format(self):
        with work_dir() as root:
            write_value(root / 'v', {'a': 1})
            payload = json.loads((root / 'v.json').read_text())
            for change in ({'version': 99}, {'format': 'other'}):
                (root / 'v.json').write_text(json.dumps({**payload, **change}))
                with self.assertRaises(SerializeError):
                    read_value(root / 'v')

    def test_only_this_packages_dataclasses_are_rebuilt(self):
        with work_dir() as root:
            for target in ('os:system', 'builtins:dict', 'tests.x:Foreign'):
                (root / 'v.json').write_text(json.dumps({
                    'format': 'upxo.gbconformant.checkpoint', 'version': 1,
                    'value': {'__dataclass__': target, 'fields': {}}}))
                with self.assertRaises(SerializeError, msg=target):
                    read_value(root / 'v')
            (root / 'v.json').write_text(json.dumps({
                'format': 'upxo.gbconformant.checkpoint', 'version': 1,
                'value': {'__dataclass__': 'upxo.meshing.gbconformant.d3v2p0.paths:run_directory',
                          'fields': {}}}))
            with self.assertRaises(SerializeError):
                read_value(root / 'v')


if __name__ == '__main__':
    unittest.main()
