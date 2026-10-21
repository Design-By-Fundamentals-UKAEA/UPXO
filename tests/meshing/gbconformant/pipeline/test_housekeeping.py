"""Disk use, dropping history and deleting runs; counts of the hierarchy in a crop."""
import os
import stat
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from upxo.meshing.gbconformant.pipeline import results, run
from upxo.meshing.gbconformant.pipeline.checkpoints import CheckpointStore
from upxo.meshing.gbconformant.pipeline.run import open_run
from upxo.meshing.gbconformant.pipeline.runfiles import current_run_dir
from upxo.meshing.gbconformant.pipeline.runs import (
    RunDeletionError, delete_run, describe_run, drop_run_history, folder_size, is_run_folder, list_runs)
from upxo.meshing.gbconformant.pipeline.stage_input import crop_hierarchy_report, handle_subset_extract

from ..fm_steel_abaqus.helpers import work_dir, write_raw_folder, write_twinned_folder

FAST = dict(
    STAGE_FREEZE_THIN=False, REQUIRE_CLEAN_TOPOLOGY=False, UPSCALE_FACTOR=1, LAPLACE_ITERATIONS=2,
    JUNCTION_ITERATIONS=1, SPIKE_MAX_PASSES=5, PINCH_MAX_PASSES=2, TOPOLOGY_MAX_PASSES=2, GRAIN_ID_SEED=3)


class RunFolderTests(unittest.TestCase):
    def setUp(self):
        context = work_dir()
        self.root = context.__enter__()
        self.addCleanup(context.__exit__, None, None, None)
        patcher = mock.patch.dict(os.environ, {'UPXO_CONFORMAL_DATA': str(self.root / 'data')})
        patcher.start()
        self.addCleanup(patcher.stop)
        raw = self.root / 'raw'
        raw.mkdir()
        write_raw_folder(raw)
        self.raw = raw
        self.runs = self.root / 'data' / 'runs'

    def make_run(self, name='one', **settings):
        result = run(dict(FAST, RUN_NAME=name, RAW_FOLDER=str(self.raw), **settings), through='pinches')
        self.assertTrue(result.ok, result.error)
        return result.run_folder

    # --- disk use -------------------------------------------------------------------------------------------
    def test_a_run_reports_its_disk_use_and_its_history(self):
        folder = self.make_run()
        info = describe_run(folder)
        self.assertEqual(info.disk_bytes, folder_size(folder))
        self.assertGreater(info.disk_bytes, info.saved_bytes // 2)
        self.assertGreater(info.history_bytes, 0)                 # several stages kept a copy of the labels
        self.assertIn('history', info.disk_text)
        self.assertEqual(info.folder_text.split()[1], 'KB' if info.disk_bytes < 1024 ** 2 else 'MB')

    def test_the_run_list_carries_the_disk_use(self):
        self.make_run('a')
        self.make_run('b')
        found = {i.name: i for i in list_runs(self.runs)}
        self.assertTrue(all(i.disk_bytes > 0 for i in found.values()))

    def test_a_run_without_history_shows_none(self):
        folder = self.make_run('nohistory', CHECKPOINT_KEEP_HISTORY=False)
        self.assertEqual(describe_run(folder).history_bytes, 0)
        self.assertNotIn('history', describe_run(folder).disk_text)

    # --- dropping the history -------------------------------------------------------------------------------
    def test_dropping_the_history_frees_what_was_counted_and_keeps_the_run_whole(self):
        folder = self.make_run()
        before = describe_run(folder)
        labels_before = open_run(folder).get_artefact('labels')
        statuses_before = {k: v for k, v in open_run(folder).status.items()}
        freed = drop_run_history(folder)
        self.assertEqual(freed, before.history_bytes)
        after = describe_run(folder)
        self.assertEqual(after.history_bytes, 0)
        shrunk = before.disk_bytes - after.disk_bytes
        self.assertGreaterEqual(shrunk, freed)                       # the notes about what was dropped are smaller too
        self.assertLess(shrunk - freed, 0.2 * freed + 4000)
        reopened = open_run(folder)
        np.testing.assert_array_equal(reopened.get_artefact('labels'), labels_before)
        self.assertEqual({k: v for k, v in reopened.status.items()}, statuses_before)
        self.assertEqual(drop_run_history(folder), 0)                # nothing left to drop

    def test_the_newest_copy_of_each_result_is_the_one_kept(self):
        folder = self.make_run()
        store = CheckpointStore(folder)
        from upxo.meshing.gbconformant.pipeline.stages import stage_ids
        drop_run_history(folder)
        holders = {}
        for stage in store.stages():
            for key in (store.meta(stage) or {}).get('artefacts', {}):
                holders.setdefault(key, []).append(stage)
        order = stage_ids()
        for key, stages in holders.items():
            self.assertEqual(len(stages), 1, key)
        self.assertEqual(holders['labels'], ['pinches'])
        self.assertEqual(max(holders['labels'], key=order.index), 'pinches')

    # --- deleting -------------------------------------------------------------------------------------------
    def test_a_run_is_deleted_and_its_size_returned(self):
        folder = self.make_run('doomed')
        size = folder_size(folder)
        self.assertEqual(delete_run(folder, root=self.runs), size)
        self.assertFalse(folder.exists())

    def test_read_only_files_do_not_stop_a_deletion(self):
        folder = self.make_run('readonly')
        for path in folder.rglob('*.json'):
            path.chmod(stat.S_IREAD)
        delete_run(folder, root=self.runs)
        self.assertFalse(folder.exists())

    def test_the_runs_folder_itself_is_never_deleted(self):
        self.make_run('keepme')
        with self.assertRaisesRegex(RunDeletionError, 'runs folder itself'):
            delete_run(self.runs, root=self.runs)
        self.assertTrue(self.runs.exists())

    def test_a_folder_outside_the_runs_folder_is_refused(self):
        outside = self.root / 'precious'
        outside.mkdir()
        (outside / 'run_manifest.json').write_text('{}')
        with self.assertRaisesRegex(RunDeletionError, 'not a run folder inside'):
            delete_run(outside, root=self.runs)
        self.assertTrue(outside.exists())

    def test_a_folder_inside_a_run_is_refused(self):
        folder = self.make_run('nested')
        with self.assertRaisesRegex(RunDeletionError, 'not a run folder inside'):
            delete_run(folder / 'checkpoints', root=self.runs)
        self.assertTrue((folder / 'checkpoints').exists())

    def test_the_loaded_run_is_refused(self):
        folder = self.make_run('loaded')
        with self.assertRaisesRegex(RunDeletionError, 'loaded'):
            delete_run(folder, root=self.runs, keep=folder)
        self.assertTrue(folder.exists())

    def test_a_folder_that_is_not_a_run_is_refused(self):
        self.runs.mkdir(parents=True, exist_ok=True)
        other = self.runs / 'notes'
        other.mkdir()
        (other / 'readme.txt').write_text('mine')
        with self.assertRaisesRegex(RunDeletionError, 'does not look like a run'):
            delete_run(other, root=self.runs)
        self.assertTrue((other / 'readme.txt').exists())
        self.assertFalse(is_run_folder(other))

    def test_the_run_folder_of_the_settings_is_found_without_creating_it(self):
        folder = current_run_dir({'RUN_NAME': 'nothing_yet'})
        self.assertEqual(folder.name, 'nothing_yet')
        self.assertFalse(folder.exists())
        self.assertEqual(current_run_dir({'_OPEN_FOLDER': str(self.root)}), self.root)


class OlderExportTests(unittest.TestCase):
    def make(self, root, layout):
        folder = root / 'runs' / 'r'
        for relative, size in layout.items():
            path = folder / relative
            path.mkdir(parents=True)
            (path / 'file.bin').write_bytes(b'x' * size)
        (folder / 'run_manifest.json').write_text('{}')
        return folder

    def test_the_newest_export_of_each_kind_is_kept(self):
        from upxo.meshing.gbconformant.pipeline.runs import older_exports
        with work_dir() as root:
            folder = self.make(root, {'abaqus/001_fm': 100, 'abaqus/001_mesh': 10, 'abaqus/002_fm': 100,
                                      'abaqus/003_fm': 100, 'abaqus/003_mesh': 10, 'surfaces/001': 5,
                                      'surfaces/002': 5, 'volume/001': 3})
            names = sorted(f'{p.parent.name}/{p.name}' for p in older_exports(folder))
            self.assertEqual(names, ['abaqus/001_fm', 'abaqus/001_mesh', 'abaqus/002_fm', 'surfaces/001'])

    def test_time_stamped_exports_sort_by_time(self):
        from upxo.meshing.gbconformant.pipeline.runs import older_exports
        with work_dir() as root:
            folder = self.make(root, {'abaqus/20261019_101000_000001_fm': 1, 'abaqus/20261019_101500_000002_fm': 1,
                                      'abaqus/20261019_093000_000003_fm': 1})
            names = sorted(p.name for p in older_exports(folder))
            self.assertEqual(names, ['20261019_093000_000003_fm', '20261019_101000_000001_fm'])

    def test_one_export_or_none_means_nothing_is_older(self):
        from upxo.meshing.gbconformant.pipeline.runs import older_exports
        with work_dir() as root:
            folder = self.make(root, {'abaqus/001_fm': 1})
            self.assertEqual(older_exports(folder), [])
            self.assertEqual(older_exports(root / 'nowhere'), [])

    def test_deleting_removes_only_the_older_exports_and_reports_the_bytes(self):
        from upxo.meshing.gbconformant.pipeline.runs import delete_older_exports, describe_run, older_exports
        with work_dir() as root:
            folder = self.make(root, {'abaqus/001_fm': 100, 'abaqus/002_fm': 100, 'abaqus/003_fm': 100,
                                      'surfaces/001': 7, 'surfaces/002': 7})
            (folder / 'abaqus' / 'notes.txt').write_text('keep me')
            info = describe_run(folder)
            self.assertEqual(info.older_export_bytes, 100 + 100 + 7)
            self.assertIn('older exports', info.disk_text)
            self.assertEqual(delete_older_exports(folder), 207)
            self.assertEqual(sorted(p.name for p in (folder / 'abaqus').iterdir()), ['003_fm', 'notes.txt'])
            self.assertEqual(sorted(p.name for p in (folder / 'surfaces').iterdir()), ['002'])
            self.assertEqual(older_exports(folder), [])
            self.assertEqual(delete_older_exports(folder), 0)
            self.assertEqual(describe_run(folder).older_export_bytes, 0)


class CropCountTests(unittest.TestCase):
    def setUp(self):
        context = work_dir()
        self.root = context.__enter__()
        self.addCleanup(context.__exit__, None, None, None)
        patcher = mock.patch.dict(os.environ, {'UPXO_CONFORMAL_DATA': str(self.root / 'data')})
        patcher.start()
        self.addCleanup(patcher.stop)
        (self.root / 'raw').mkdir()
        write_raw_folder(self.root / 'raw')

    def hierarchy(self, level='blk'):
        from upxo.meshing.gbconformant.fm_steel_abaqus import load_hierarchy
        return load_hierarchy(self.root / 'raw', level)

    def test_the_levels_of_the_structure_are_counted(self):
        from upxo.meshing.gbconformant.fm_steel_abaqus.subset import hierarchy_counts
        self.assertEqual(hierarchy_counts(self.hierarchy()), {'pag': 2, 'packet': 3, 'block': 5})
        self.assertEqual(hierarchy_counts(self.hierarchy('pck')), {'pag': 2, 'packet': 3})
        self.assertEqual(hierarchy_counts(self.hierarchy('pag')), {'pag': 2})

    def test_a_crop_counts_only_what_is_in_it(self):
        from upxo.meshing.gbconformant.fm_steel_abaqus.subset import hierarchy_counts
        hier = self.hierarchy()
        self.assertEqual(hierarchy_counts(hier, [1, 2]), {'pag': 1, 'packet': 1, 'block': 2})
        self.assertEqual(hierarchy_counts(hier, [3, 4]), {'pag': 2, 'packet': 2, 'block': 2})
        self.assertEqual(hierarchy_counts(hier, [99]), {'pag': 0, 'packet': 0, 'block': 0})

    def test_a_twinned_structure_counts_grains_and_host_families(self):
        from upxo.meshing.gbconformant.fm_steel_abaqus.subset import hierarchy_counts
        from upxo.meshing.gbconformant.fm_steel_abaqus.twinned import load_twinned
        folder = self.root / 'twinned'
        folder.mkdir()
        write_twinned_folder(folder)
        hier = load_twinned(folder)
        self.assertEqual(hierarchy_counts(hier), {'grains': 5, 'families': 2})
        self.assertEqual(hierarchy_counts(hier, [2, 3]), {'grains': 2, 'families': 1})
        self.assertEqual(hierarchy_counts(hier, [6]), {'grains': 1, 'families': 0})

    def test_warnings_only_for_the_levels_with_a_minimum(self):
        from upxo.meshing.gbconformant.fm_steel_abaqus.subset import crop_adequacy
        counts = {'pag': 1, 'packet': 4, 'block': 9}
        self.assertEqual(crop_adequacy(counts, {}), [])
        self.assertEqual(crop_adequacy(counts, {'pag': None, 'packet': 0, 'block': None}), [])
        self.assertEqual(crop_adequacy(counts, {'pag': 3}), ['Only 1 PAGs in the crop; at least 3 were asked for.'])
        self.assertEqual(crop_adequacy(counts, {'pag': 1, 'packet': 4, 'block': 9}), [])
        self.assertEqual(len(crop_adequacy(counts, {'pag': 2, 'packet': 5, 'block': 10})), 3)
        self.assertEqual(crop_adequacy({'grains': 1, 'families': 0}, {'pag': 5}), [])      # not a level of this structure

    def test_the_subset_stage_reports_counts_and_warnings(self):
        result = run(dict(FAST, RUN_NAME='crop', RAW_FOLDER=str(self.root / 'raw'), STAGE_SUBSET=True,
                          SUBSET_MODE='index', SUBSET_LO=[0, 0, 0], SUBSET_HI=[2, 4, 4], RVE_MIN_PAGS=2,
                          RVE_MIN_BLOCKS=1), through='subset_extract')
        self.assertTrue(result.ok, result.error)
        report = result.state.reports['subset_extract']
        self.assertEqual(report['counts']['crop'], {'pag': 1, 'packet': 1, 'block': 2})
        self.assertEqual(report['counts']['parent'], {'pag': 2, 'packet': 3, 'block': 5})
        self.assertEqual(report['warnings'], ['Only 1 PAGs in the crop; at least 2 were asked for.'])
        rows = dict((r[0], r[1]) for r in results.subset_block(result.state).rows)
        self.assertEqual(rows['PAGs in the subset'], '1 of 2')
        self.assertEqual(rows['Blocks in the subset'], '2 of 5')
        self.assertIn('Only 1 PAGs', rows['Warnings about the crop'])
        record = (result.run_folder / 'subset.json').read_text()
        self.assertIn('"counts"', record)

    def test_without_minimums_there_are_no_warnings(self):
        report = crop_hierarchy_report({}, self.hierarchy(), [1])
        self.assertEqual(report['warnings'], [])
        self.assertEqual(report['counts']['crop'], {'pag': 1, 'packet': 1, 'block': 1})
        self.assertEqual(crop_hierarchy_report({}, None, [1]), {'counts': None, 'warnings': []})

    def test_the_minimum_settings_are_empty_by_default(self):
        from upxo.meshing.gbconformant.pipeline.settings import default_config
        config = default_config()
        self.assertEqual((config['RVE_MIN_PAGS'], config['RVE_MIN_PACKETS'], config['RVE_MIN_BLOCKS']),
                         (None, None, None))


if __name__ == '__main__':
    unittest.main()
