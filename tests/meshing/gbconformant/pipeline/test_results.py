import contextlib
import csv
import io
import json
import os
import unittest
from types import SimpleNamespace
from unittest import mock

import matplotlib
import numpy as np

matplotlib.use('Agg')

from upxo.meshing.gbconformant.pipeline import build_config, results, run
from upxo.meshing.gbconformant.pipeline.cli import EXIT_OK, EXIT_USAGE, main
from upxo.meshing.gbconformant.pipeline.run import new_state, open_run
from upxo.meshing.gbconformant.pipeline.runner import PipelineRunner
from upxo.meshing.gbconformant.pipeline.stage_output import (
    COMPARISON_KEY, handle_compare_before_after, handle_export_abaqus)
from upxo.reporting import ReportSession, render_html

from pathlib import Path

from ..fm_steel_abaqus.helpers import voxel_tets, work_dir, write_raw_folder

FAST = dict(
    STAGE_FREEZE_THIN=False, REQUIRE_CLEAN_TOPOLOGY=False, UPSCALE_FACTOR=1, LAPLACE_ITERATIONS=2,
    JUNCTION_ITERATIONS=1, SPIKE_MAX_PASSES=5, PINCH_MAX_PASSES=2, TOPOLOGY_MAX_PASSES=2,
    GRAIN_ID_SEED=3)


def meshed_state(root, name='results'):
    """A state that has loaded the helper structure and holds a structured tetrahedral mesh of it."""
    raw = root / 'raw'
    raw.mkdir(exist_ok=True)
    write_raw_folder(raw)
    state = new_state(build_config(dict(FAST, RUN_NAME=name, RAW_FOLDER=str(raw),
                                        STAGE_GRAIN_ID_SHUFFLE=False)))
    PipelineRunner(state).run_stage('load_lgi')
    labels = np.asarray(state.get_artefact('labels'))
    points, tets, grains = voxel_tets(labels)
    state.store_artefact('labels_meshing', labels.copy())
    state.store_artefact('verified_tets', SimpleNamespace(
        points=points, tetrahedra=tets, grain_ids=grains, quality=np.linspace(0.3, 0.9, len(tets)),
        report={'ready': True, 'status': 'TET_MESH_VERIFIED', 'grains': 5, 'total_volume': 96.0,
                'minimum_quality': 0.3, 'quality_metric': 'Gmsh minSICN',
                'dihedral_angles': {'minimum': 45.0, 'maximum': 90.0}}))
    state.set_status('tet_mesh', results.StageStatus.OK)
    state.reports['tet_mesh'] = {'duration_s': 1.5, 'minimum_quality': 0.3, 'n_tets': int(len(tets)),
                                 'dihedral_angles': {'minimum': 45.0, 'maximum': 90.0}}
    handle_compare_before_after(state)
    state.set_status('compare_before_after', results.StageStatus.OK)
    return state


class ResultsTests(unittest.TestCase):
    def setUp(self):
        context = work_dir()
        self.root = context.__enter__()
        self.addCleanup(context.__exit__, None, None, None)
        patcher = mock.patch.dict(os.environ, {'UPXO_CONFORMAL_DATA': str(self.root / 'data')})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.state = meshed_state(self.root)

    def test_summary_tables_describe_the_input_and_the_mesh(self):
        blocks = {b.title: b for b in results.summary_blocks(self.state)}
        self.assertEqual(dict(map(tuple, blocks['Input'].rows))['Hierarchy level'], 'blk')
        self.assertEqual(dict(map(tuple, blocks['Input'].rows))['Grains'], 5)
        mesh = dict(map(tuple, blocks['Tetrahedral mesh'].rows))
        self.assertEqual(mesh['Tetrahedra'], int(len(self.state.get_artefact('verified_tets').tetrahedra)))
        self.assertIs(mesh['Meets the quality targets'], True)
        self.assertEqual(mesh['Smallest dihedral angle (deg)'], 45.0)
        self.assertIn('Tetrahedral mesh after each stage', blocks)
        self.assertNotIn('Subset of the input', blocks)
        text = results.summary_text(self.state)
        self.assertIn('Hierarchy level', text)
        self.assertEqual(results.summary_text(new_state(build_config())), 'Nothing has been run yet.')

    def test_stage_rows_and_timings(self):
        rows = {row[0]: row for row in results.stage_rows(self.state)}
        self.assertEqual(rows['load_lgi'][2], 'OK')
        self.assertEqual(rows['spikes'][2], 'Not run')
        timing = results.timing_block(self.state)
        self.assertEqual(timing.rows[-1][0], 'Total')
        self.assertGreater(timing.rows[-1][2], 1.5)

    def test_quality_tables_and_figures(self):
        blocks = results.quality_blocks(self.state)
        titles = ' | '.join(b.title for b in blocks)
        self.assertIn('Tetrahedra:', titles)
        self.assertNotIn('Surface triangles', titles)          # no surface in this state
        figures = results.quality_figures(self.state)
        self.assertEqual(len(figures), 3)
        self.assertTrue(all(f.title.startswith('Tetrahedra:') for f in figures))
        self.assertEqual(len({f.key for f in figures}), 3)

    def test_fidelity_tables_and_every_plot(self):
        self.state.config['COMPARE_PLOT_UNITS'] = 'mm'
        blocks = {b.title: b for b in results.fidelity_blocks(self.state)}
        facts = dict(map(tuple, blocks['Comparison'].rows))
        self.assertEqual(facts['Grains in the tetrahedral mesh'], 5)
        self.assertEqual((facts['Before'], facts['After']), ('Voxel structure', 'Tetrahedral mesh'))
        self.assertIn('Junction line length (per line) (mm)', blocks)
        self.assertEqual(blocks['Junction line length (per line) (mm)'].columns,
                         ['Statistic', 'Voxel structure', 'Tetrahedral mesh'])
        for which in results.FIDELITY_PLOTS:
            picture = results.fidelity_figure(self.state, which)
            self.assertTrue(picture.title and picture.figure.axes, which)
        self.assertEqual(len(results.fidelity_figures(self.state)), len(results.FIDELITY_PLOTS))
        with self.assertRaises(KeyError):
            results.fidelity_figure(self.state, 'nope')
        self.state.artefacts.pop(COMPARISON_KEY)
        titles = [b.title for b in results.fidelity_blocks(self.state)]
        self.assertNotIn('Comparison', titles)
        self.assertIn('Features compared one by one', titles)       # the per-feature tables do not need it
        self.assertEqual(len(results.fidelity_figures(self.state)), len(results.GRAIN_PLOTS))
        with self.assertRaises(RuntimeError):
            results.fidelity_figure(self.state, 'diameter')

    def test_deliverables_list_what_exists(self):
        rows = {d['label']: d for d in results.deliverables(self.state)}
        self.assertTrue(rows['Run folder']['exists'])
        self.assertTrue(rows['Saved stage results']['exists'])
        self.assertFalse(rows['Run manifest']['exists'])
        self.assertGreater(rows['Saved stage results']['bytes'], 0)
        block = results.deliverables_block(self.state)
        self.assertTrue(any(row[2] == 'not written' for row in block.rows))
        self.assertEqual(results.human_size(900), '900 B')
        self.assertEqual(results.human_size(3 * 1024 ** 2), '3.0 MB')

    def test_settings_table_lists_only_changes(self):
        block = results.settings_block(self.state)
        changed = {row[0] for row in block.rows}
        self.assertIn('SPIKE_MAX_PASSES', changed)
        self.assertNotIn('MIN_GRAIN_VOXELS', changed)
        self.assertFalse(any(name.startswith('_') for name in changed))

    def test_report_gets_the_standard_sections_once(self):
        session = results.report_session(self.state)
        added = results.populate_report(session, self.state)
        self.assertEqual(added, list(results.SECTIONS))
        count = len(session)
        sections = [e.title for e in session.entries if e.kind == 'section']
        self.assertEqual(sections, [results.SECTION_TITLES[s] for s in results.SECTIONS])
        images = [e for e in session.entries if e.kind == 'image']
        self.assertEqual(len(images), 3 + len(results.FIDELITY_PLOTS))
        for image in images:
            self.assertTrue((session.reports_dir / image.image_path).is_file())
        self.assertTrue(all(e.source_page and e.source_page.startswith('results:')
                            for e in session.entries if e.kind in ('table', 'image')))
        again = results.populate_report(session, self.state, ['summary', 'fidelity'])
        self.assertEqual(again, ['summary', 'fidelity'])
        self.assertEqual(len(session), count)                     # rebuilt, not repeated
        html_path = render_html(session)
        html = html_path.read_text(encoding='utf-8')
        self.assertIn('Run summary', html)
        self.assertIn('assets/', html)
        self.assertEqual(results.populate_report(session, self.state, ['quality'], figures=False), ['quality'])
        with self.assertRaises(KeyError):
            results.populate_report(session, self.state, ['nope'])

    def test_report_continues_what_a_person_added(self):
        session = results.report_session(self.state)
        session.add_text('by hand', 'Note', section='Extra notes')
        session.save()
        results.populate_report(ReportSession.load(session.reports_dir), self.state, ['summary'])
        reloaded = ReportSession.load(session.reports_dir)
        self.assertIn('Note', [e.title for e in reloaded.entries])


def merged_state(root, name='merged'):
    """A meshed state in which feature 4 was absorbed into feature 3 (as the small-grain stage records it)."""
    state = meshed_state(root, name)
    labels = np.asarray(state.get_artefact('labels')).copy()
    labels[labels == 4] = 3
    points, tets, grains = voxel_tets(labels)
    state.store_artefact('labels', labels)
    state.store_artefact('labels_meshing', labels.copy())
    state.store_artefact('small_grains.survivor_map', {1: 1, 2: 2, 3: 3, 4: 3, 5: 5})
    state.store_artefact('verified_tets', SimpleNamespace(
        points=points, tetrahedra=tets, grain_ids=grains, quality=np.full(len(tets), 0.5),
        report={'ready': True}))
    return state


class PhaseFiveTests(unittest.TestCase):
    def setUp(self):
        context = work_dir()
        self.root = context.__enter__()
        self.addCleanup(context.__exit__, None, None, None)
        patcher = mock.patch.dict(os.environ, {'UPXO_CONFORMAL_DATA': str(self.root / 'data')})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_quality_tables_name_the_worst_grains_and_list_every_grain(self):
        state = meshed_state(self.root)
        blocks = {b.title: b for b in results.quality_blocks(state)}
        limits = next(b for t, b in blocks.items() if t.startswith('Tetrahedra against the usual limits'))
        self.assertEqual(limits.columns, ['Tetrahedra with', 'Count', 'Share'])
        self.assertTrue(any(r[0] == 'smallest dihedral below 30 deg' for r in limits.rows))
        worst = next(b for t, b in blocks.items() if 'smallest dihedral angle' in t and t.startswith('The '))
        self.assertEqual(len(worst.rows), 5)                        # five grains, fewer than the ten asked for
        self.assertTrue(all(str(r[1]).startswith('es_blk_') for r in worst.rows))
        self.assertIn('Aspect ratio of the tetrahedra (1 is regular)', blocks)
        table = results.quality_grain_table(state)
        self.assertEqual(len(table.rows), 5)
        self.assertIn('max_aspect', table.columns)
        path = results.write_csv(table, self.root / 'q.csv')
        with path.open(newline='') as handle:
            rows = list(csv.reader(handle))
        self.assertEqual(rows[0], table.columns)
        self.assertEqual(len(rows), 6)
        self.assertIsNone(results.quality_grain_table(new_state(build_config())))

    def test_per_feature_fidelity_of_a_faithful_mesh(self):
        state = meshed_state(self.root)
        blocks = {b.title: b for b in results.grain_blocks(state)}
        facts = dict(map(tuple, blocks['Features compared one by one'].rows))
        self.assertEqual(facts['Features'], 5)
        self.assertEqual(facts['Features with no mesh'], 'none')
        self.assertEqual(facts['Volume change: median'], '0 %')
        table = results.grain_table(state)
        self.assertEqual(table.columns[:3], ['label', 'name', 'volume_before'])
        self.assertEqual(len(table.rows), 5)
        self.assertTrue(all(row[1].startswith('es_blk_') for row in table.rows))

    def test_a_shrunk_feature_is_flagged_and_the_limit_is_a_setting(self):
        state = meshed_state(self.root)
        tets = state.get_artefact('verified_tets')
        grain_2 = np.flatnonzero(np.asarray(tets.grain_ids) == 2)
        keep = np.ones(len(tets.tetrahedra), dtype=bool)
        keep[grain_2[:len(grain_2) // 2]] = False
        state.store_artefact('verified_tets', SimpleNamespace(
            points=tets.points, tetrahedra=tets.tetrahedra[keep], grain_ids=np.asarray(tets.grain_ids)[keep],
            quality=np.asarray(tets.quality)[keep], report={}))
        self.assertEqual(results.grain_data(state)['flagged'], [2])
        state.config['FIDELITY_SHRINK_LIMIT'] = 0.9
        self.assertEqual(results.grain_data(state)['flagged'], [])

    def test_merged_grains_get_a_merge_log_in_the_tables_and_beside_the_deck(self):
        state = merged_state(self.root)
        self.assertIsNone(results.merge_data(meshed_state(self.root, 'plain')))
        blocks = {b.title: b for b in results.merge_blocks(state)}
        facts = dict(map(tuple, blocks['Merged grains: orientation substitution'].rows))
        self.assertEqual(facts['Features absorbed'], 1)
        self.assertEqual(facts['Merged grains'], 1)
        self.assertIsNotNone(facts['Misorientation to the winner: largest (deg)'])
        listing = next(b for t, b in blocks.items() if t.startswith('Absorbed features'))
        self.assertEqual(len(listing.rows), 1)
        report = handle_export_abaqus(state)
        folder = Path(report['fm_deck']['folder'])
        self.assertEqual(report['fm_deck']['merged_features'], 1)
        log = json.loads((folder / 'merge_log.json').read_text())
        self.assertEqual(log['summary']['absorbed_features'], 1)
        self.assertTrue((folder / 'merge_log.csv').is_file())
        state.reports['export_abaqus'] = report
        block = next(b for b in results.summary_blocks(state) if b.title == 'Abaqus export')
        self.assertEqual(dict(map(tuple, block.rows))['Merged features in the merge log'], 1)
        plain = handle_export_abaqus(meshed_state(self.root, 'plain2'))
        self.assertEqual(plain['fm_deck']['merged_features'], 0)
        self.assertFalse((Path(plain['fm_deck']['folder']) / 'merge_log.json').exists())

    def test_non_cubic_symmetry_is_said_not_failed(self):
        state = merged_state(self.root)
        state.config['TEXTURE_CRYSTAL_SYMMETRY'] = 'hexagonal'
        blocks = results.merge_blocks(state)
        self.assertEqual(blocks[0].title, 'Merged grains')
        self.assertIn('cubic', str(blocks[0].rows))
        report = handle_export_abaqus(state)
        self.assertIsNone(report['fm_deck']['merged_features'])

    def test_every_grain_plot_can_be_drawn_without_the_comparison_stage(self):
        state = meshed_state(self.root)
        state.artefacts.pop(COMPARISON_KEY)
        for which in results.GRAIN_PLOTS:
            picture = results.fidelity_figure(state, which)
            self.assertTrue(picture.figure.axes, which)
        self.assertEqual(len(results.fidelity_figures(state)), len(results.GRAIN_PLOTS))
        with self.assertRaises(RuntimeError):
            results.fidelity_figure(state, 'diameter')
        self.assertTrue(results.fidelity_blocks(state))              # the per-feature tables remain

    def test_the_report_gets_the_spreadsheets_beside_it(self):
        state = merged_state(self.root, 'sheets')
        session = results.report_session(state)
        results.populate_report(session, state, ['quality', 'fidelity'], figures=False)
        for name in ('quality_per_grain.csv', 'fidelity_per_feature.csv', 'merge_log.csv'):
            self.assertTrue((session.reports_dir / name).is_file(), name)
        notes = [e.text for e in session.entries if e.kind == 'text']
        self.assertEqual(len(notes), 2)
        self.assertIn('quality_per_grain.csv', notes[0])
        self.assertIn('merge_log.csv', notes[1])
        results.populate_report(session, state, ['quality', 'fidelity'], figures=False)
        self.assertEqual(len([e for e in session.entries if e.kind == 'text']), 2)    # rebuilt, not repeated


class OpenRunTests(unittest.TestCase):
    def test_a_finished_run_folder_opens_with_its_results_and_gives_a_report(self):
        with work_dir() as root:
            raw = root / 'raw'
            raw.mkdir()
            write_raw_folder(raw)
            with mock.patch.dict(os.environ, {'UPXO_CONFORMAL_DATA': str(root / 'data')}):
                made = run(dict(FAST, RUN_NAME='opened', RAW_FOLDER=str(raw)), through='spikes')
                self.assertTrue(made.ok, made.error)
                state = open_run(made.run_folder)
                self.assertEqual(state.status['spikes'].value, 'OK')
                self.assertEqual(state.config['_OPEN_FOLDER'], str(made.run_folder))
                np.testing.assert_array_equal(state.get_artefact('labels'),
                                              made.state.get_artefact('labels'))
                self.assertIn('Cleaning of the voxel structure',
                              [b.title for b in results.summary_blocks(state)])
                out, err = io.StringIO(), io.StringIO()
                with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                    code = main(['report', str(made.run_folder), '--no-figures'])
                self.assertEqual(code, EXIT_OK, err.getvalue())
                self.assertIn('sections: summary, settings, deliverables', out.getvalue())
                self.assertTrue((made.run_folder / 'report' / 'report.html').is_file())
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(main(['report', str(root / 'nowhere')]), EXIT_USAGE)
                    self.assertEqual(main(['report', str(made.run_folder), '--sections', 'nope']), EXIT_USAGE)
                with self.assertRaises(FileNotFoundError):
                    open_run(root / 'raw')


if __name__ == '__main__':
    unittest.main()
