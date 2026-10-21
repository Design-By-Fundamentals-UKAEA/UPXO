import unittest
from upxo.meshing.gbconformant.fm_steel_abaqus import constants as C


class ConstantsTests(unittest.TestCase):
    def test_unit_scales(self):
        self.assertEqual(C.unit_scale('mm', 'microns'), 1e3)
        self.assertEqual(C.unit_scale('microns', 'm'), 1e-6)
        self.assertEqual(C.unit_scale('m', 'm'), 1.0)
        with self.assertRaises(ValueError):
            C.unit_scale('mm', 'km')

    def test_matches_voxel_exporter(self):
        try:
            from upxo.pxtal.fm_steel_3d import mesh_exporter_3d as voxel
        except Exception as error:  # optional heavy dependencies
            self.skipTest(f'voxel exporter unavailable: {error}')
        self.assertEqual(C.MATERIAL_FORMATS, voxel.MATERIAL_FORMATS)
        self.assertEqual(C.LOAD_AXES, voxel.LOAD_AXES)
        self.assertEqual(C.PART_NAME, voxel.ABAQUS_PART_NAME)
        self.assertEqual(C.INSTANCE_NAME, voxel.ABAQUS_INSTANCE_NAME)
        self.assertEqual(C.FACE_NSET_NAMES, voxel.FACE_NSET_NAMES)
        self.assertEqual(C.REFERENCE_UMAT_TAIL, voxel.REFERENCE_UMAT_TAIL)
        self.assertEqual(C.REFERENCE_N_DEPVAR, voxel.REFERENCE_N_DEPVAR)
        self.assertEqual(C.HEADER, voxel._UPXO_HEADER)
        for src in C.UNITS:
            for dst in C.UNITS:
                self.assertEqual(C.unit_scale(src, dst), voxel.MeshExporter3D._unit_scale(src, dst))
        args = (80.0, 0.01, 1e-8, 1.0, 10000, 0.5, 'le, s', ('U',))
        self.assertEqual(C.validate_step_controls(*args), voxel._validate_step_controls(*args))


if __name__ == '__main__':
    unittest.main()
