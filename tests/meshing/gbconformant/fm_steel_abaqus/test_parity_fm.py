"""Compare a deck of structured tetrahedra with the voxel FM-steel export of the same structure.

Runs only when a raw export folder and the voxel deck made from it are on
disk (``data/conformalMeshing3DData/raw`` and
``data/FmSteels/fm_steel_blk_001/ABQInputFiles/instance7``); skipped otherwise.
"""
from pathlib import Path
import unittest
import numpy as np
from upxo._sup.data_dir import default_data_dir
from upxo.meshing.gbconformant.fm_steel_abaqus import (
    DeckSettings, load_hierarchy, resolve_identities, write_deck)
from .helpers import read_elsets, voxel_tets, work_dir

DATA = default_data_dir()
RAW = DATA / 'conformalMeshing3DData' / 'raw'
VOXEL_DECK = DATA / 'FmSteels' / 'fm_steel_blk_001' / 'ABQInputFiles' / 'instance7'


def _available():
    master = VOXEL_DECK / 'model_master.inp'
    return (RAW / 'raw_export.pkl').is_file() and master.is_file() and \
        'C3D8' in master.read_text(errors='replace') and \
        'Grid: 50 x 75 x 100' in master.read_text(errors='replace')


@unittest.skipUnless(_available(), 'reference export folders are not on disk')
class VoxelDeckParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hier = load_hierarchy(RAW, 'blk')
        pts, tets, grains = voxel_tets(cls.hier.lgi)
        settings = DeckSettings(
            applied_displacement=0.02, step_time=50.0, initial_inc=0.01, min_inc=1e-8,
            max_inc=0.5, max_increments=10000, output_interval=2.5,
            element_outputs=('LE', 'S', 'SDV'), node_outputs=('U', 'RF'), output_units='mm')
        cls.ctx = work_dir()
        root = cls.ctx.__enter__()
        cls.result = write_deck(root / 'deck', pts, tets, grains, cls.hier,
                                resolve_identities(cls.hier.voxel_counts), settings)
        cls.deck = cls.result['directory']

    @classmethod
    def tearDownClass(cls):
        cls.ctx.__exit__(None, None, None)

    def text(self, folder, name):
        return (folder / name).read_text(errors='replace')

    def test_materials_are_identical(self):
        self.assertEqual(self.text(self.deck, '05_materials.inp'),
                         self.text(VOXEL_DECK, '05_materials.inp'))

    def test_sections_are_identical(self):
        mine = self.text(self.deck, '06_sections.inp').splitlines()[1:]
        theirs = self.text(VOXEL_DECK, '06_sections.inp').splitlines()[1:]
        self.assertEqual(mine, theirs)

    def test_step_is_identical(self):
        self.assertEqual(self.text(self.deck, '08_steps_output.inp'),
                         self.text(VOXEL_DECK, '08_steps_output.inp'))

    def test_nodes_and_face_sets_are_identical(self):
        self.assertEqual(self.text(self.deck, '01_nodes.inp'), self.text(VOXEL_DECK, '01_nodes.inp'))
        self.assertEqual(self.text(self.deck, '04_nsets_bc.inp'), self.text(VOXEL_DECK, '04_nsets_bc.inp'))

    def test_element_sets_cover_the_same_voxels(self):
        for name in ('03a_elsets_pag.inp', '03b_elsets_pck.inp', '03c_elsets_blk.inp'):
            mine = read_elsets(self.deck / name)
            theirs = read_elsets(VOXEL_DECK / name)
            self.assertEqual(set(mine), set(theirs), name)
            for elset, tets in mine.items():
                voxels = np.unique((np.asarray(tets) - 1) // 6 + 1)
                self.assertTrue(np.array_equal(voxels, np.sort(theirs[elset])), f'{name} {elset}')
                self.assertEqual(len(tets), 6 * len(theirs[elset]))

    def test_every_block_is_kept_and_audit_is_clean(self):
        self.assertEqual(self.result['info']['features_without_elements'], [])
        self.assertEqual(self.result['audit']['violations'], [])
        self.assertEqual(self.result['audit']['volume_over_own_raw_label'], 1.0)
        self.assertEqual(self.result['info']['tetrahedra'], 6 * 375000)


if __name__ == '__main__':
    unittest.main()
