import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np
from upxo.meshing.gbconformant.d3v2p0.abaqus_export import export_tets_abaqus
from contextlib import contextmanager
import uuid
import shutil


@contextmanager
def export_directory():
    # Avoid tempfile's mode=0o700 ACL handling on managed Windows hosts.
    root = Path.cwd() / ('abaqus_test_' + uuid.uuid4().hex)
    root.mkdir()
    try:
        yield root
    finally:
        assert root.parent == Path.cwd() and root.name.startswith('abaqus_test_')
        shutil.rmtree(root)


class ExportTests(unittest.TestCase):
    def mesh(self):
        return SimpleNamespace(points=np.array([[0.,0,0],[1.,0,0],[0.,1,0],[0.,0,1],[0.,0,-1]]),
            tetrahedra=np.array([[0,1,2,3],[0,2,1,4]]), grain_ids=np.array([9,3]),
            quality=np.array([.8,.7]), report={'ready': False})

    def test_round_trip(self):
        with export_directory() as root:
            mesh = self.mesh()
            out = export_tets_abaqus(mesh, Path(root)/'mesh', expected_grains=[3,9])
            nodes=np.loadtxt(out/'01_nodes.inp', skiprows=1, delimiter=',')
            tets=np.loadtxt(out/'02_elements.inp', skiprows=1, delimiter=',', dtype=int)
            np.testing.assert_array_equal(nodes[:,1:], mesh.points)
            np.testing.assert_array_equal(tets[:,1:]-1, mesh.tetrahedra)
            self.assertEqual((out/'03a_elsets_grains.inp').read_text(),
                             '*ELSET, ELSET=grn_3\n2\n*ELSET, ELSET=grn_9\n1\n')
            self.assertTrue((out/'README.md').exists())
            self.assertIn('C3D4', (out/'02_elements.inp').read_text())
            with self.assertRaises(FileExistsError): export_tets_abaqus(mesh,out)

    def test_reject_invalid_before_writing(self):
        with export_directory() as root:
            for mode in ('inverted','ids','prefix','index'):
                mesh=self.mesh(); kw={}
                if mode=='inverted': mesh.tetrahedra[0]=[0,2,1,3]
                if mode=='ids': kw['expected_grains']=[1]
                if mode=='prefix': kw['prefix']='bad prefix'
                if mode=='index': mesh.tetrahedra[0,0]=99
                out=Path(root)/mode
                with self.assertRaises(ValueError): export_tets_abaqus(mesh,out,**kw)
                self.assertFalse(out.exists())
