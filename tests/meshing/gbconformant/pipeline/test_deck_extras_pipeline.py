"""The deck extras through the export stage: report, summary, deliverables and the blocking switch."""
import json
import unittest
from pathlib import Path
from unittest import mock

from upxo.meshing.gbconformant.pipeline import advice, results
from upxo.meshing.gbconformant.pipeline.stage_output import handle_export_abaqus

from .test_results import meshed_state
from ..fm_steel_abaqus.helpers import work_dir


class ExportExtrasTests(unittest.TestCase):
    def setUp(self):
        context = work_dir()
        self.root = context.__enter__()
        self.addCleanup(context.__exit__, None, None, None)
        patcher = mock.patch.dict('os.environ', {'UPXO_CONFORMAL_DATA': str(self.root / 'data')})
        patcher.start()
        self.addCleanup(patcher.stop)

    def export(self, name='extras', **settings):
        state = meshed_state(self.root, name)
        state.config.update(settings)
        return state, handle_export_abaqus(state)

    def test_the_default_export_checks_itself_and_writes_the_table_and_layer_sets(self):
        state, report = self.export()
        deck = Path(report['fm_deck']['folder'])
        names = {p.name for p in deck.iterdir()}
        for expected in ('features.csv', 'deck_check.json', '03f_elsets_layers.inp'):
            self.assertIn(expected, names)
        check = report['fm_deck']['deck_check']
        self.assertTrue(check['ok'], check['failed'])
        self.assertEqual(report['fm_deck']['feature_table'], 'features.csv')
        layers = report['fm_deck']['layer_sets']
        self.assertGreater(layers['gb_layer_elements'], 0)

    def test_the_summary_and_the_deliverables_show_them(self):
        state, report = self.export('summary')
        state.reports['export_abaqus'] = report
        block = results.export_block(state)
        rows = dict((row[0], row[1]) for row in block.rows)
        self.assertEqual(rows['Deck check'], 'all checks passed')
        self.assertIn('%', rows['Elements next to a grain boundary'])
        labels = [item['label'] for item in results.deliverables(state)]
        self.assertIn('Feature table (CSV)', labels)
        self.assertIn('Deck check', labels)

    def test_the_boundaries_are_classed_and_summarised(self):
        state, report = self.export('classes', BOUNDARY_LOW_ANGLE_DEGREES=20.0)
        deck = Path(report['fm_deck']['folder'])
        names = {p.name for p in deck.iterdir()}
        for expected in ('boundary_classes.csv', 'fm_interfaces.vtp', '04c_nsets_boundary_classes.inp'):
            self.assertIn(expected, names)
        summary = report['fm_deck']['boundary_classes']
        self.assertEqual(summary['low_angle_degrees'], 20.0)
        self.assertEqual(list(summary['by_kind']), ['block', 'packet', 'pag'])
        state.reports['export_abaqus'] = report
        block = results.boundary_block(state)
        self.assertIn('below 20 degrees', block.title)
        self.assertEqual([row[0] for row in block.rows], ['block', 'packet', 'pag'])
        self.assertAlmostEqual(sum(row[3] for row in block.rows), 100.0, places=1)
        self.assertIn(block.title, [b.title for b in results.summary_blocks(state)])
        labels = [item['label'] for item in results.deliverables(state)]
        self.assertIn('Grain boundaries by kind (CSV)', labels)
        self.assertIn('Grain boundary surface (VTK)', labels)

    def test_the_symmetry_setting_reaches_the_classes(self):
        state, report = self.export('hex', TEXTURE_CRYSTAL_SYMMETRY='hexagonal')
        self.assertFalse(report['fm_deck']['boundary_classes']['misorientation_available'])

    def test_a_strain_load_reaches_the_deck_and_the_summary(self):
        state, report = self.export('strain', FM_LOAD_MODE='strain', FM_TARGET_STRAIN=0.02, FM_STRAIN_RATE=0.001,
                                    FM_LOAD_SENSE='compression', FM_HOLD_TIME=10.0)
        deck = Path(report['fm_deck']['folder'])
        text = (deck / '08_steps_output.inp').read_text()
        self.assertEqual(text.count('*Step,'), 2)
        summary = report['fm_deck']['load_case']
        self.assertEqual((summary['case'], summary['sense'], summary['total_time']), ('monotonic', 'compression', 30.0))
        self.assertGreater(report['fm_deck']['output_estimate']['frames'], 1)
        state.reports['export_abaqus'] = report
        rows = dict((row[0], row[1]) for row in results.export_block(state).rows)
        self.assertIn('Monotonic compression', rows['Load case'])
        self.assertIn('field output (rough)', rows['Analysis output'])
        self.assertTrue(json.loads((deck / 'deck_info.json').read_text())['load_case']['total_time'] == 30.0)

    def test_a_cyclic_load_makes_a_deck_that_checks_out(self):
        state, report = self.export('cyclic', FM_LOAD_MODE='strain', FM_LOAD_CASE='cyclic', FM_CYCLES=3,
                                    FM_CYCLE_DEFINITION='amplitude_r', FM_STRAIN_AMPLITUDE=0.005, FM_R_RATIO=0.1)
        self.assertTrue(report['fm_deck']['deck_check']['ok'], report['fm_deck']['deck_check']['failed'])
        text = (Path(report['fm_deck']['folder']) / '08_steps_output.inp').read_text()
        self.assertIn('*Amplitude, name=LOAD-HISTORY', text)

    def test_an_output_set_the_deck_does_not_have_stops_the_export(self):
        with self.assertRaisesRegex(ValueError, "output_elset 'es_missing'"):
            self.export('badset', FM_OUTPUT_ELSET='es_missing')

    def test_each_extra_has_its_switch(self):
        state, report = self.export('off', EXPORT_FEATURE_TABLE=False, EXPORT_BOUNDARY_LAYER=False,
                                    EXPORT_DECK_CHECK=False, EXPORT_BOUNDARY_CLASSES=False)
        deck = Path(report['fm_deck']['folder'])
        names = {p.name for p in deck.iterdir()}
        self.assertFalse(names & {'features.csv', 'deck_check.json', '03f_elsets_layers.inp', 'boundary_classes.csv'})
        self.assertIsNone(report['fm_deck']['boundary_classes'])
        self.assertIsNone(report['fm_deck']['deck_check'])
        self.assertIsNone(report['fm_deck']['layer_sets'])

    def failing(self):
        return {'ok': False, 'checks': [{'name': 'nodes numbered 1..N', 'ok': False, 'detail': 'x'}],
                'counts': {}, 'failed': ['nodes numbered 1..N']}

    def test_a_failed_check_is_reported_and_the_deck_kept_by_default(self):
        with mock.patch('upxo.meshing.gbconformant.fm_steel_abaqus.export.check_deck', return_value=self.failing()):
            state, report = self.export('soft')
        self.assertFalse(report['fm_deck']['deck_check']['ok'])
        self.assertTrue(Path(report['fm_deck']['folder']).is_dir())
        state.reports['export_abaqus'] = report
        self.assertIn('FAILED', dict((r[0], r[1]) for r in results.export_block(state).rows)['Deck check'])

    def test_a_failed_check_stops_the_stage_when_asked_to(self):
        with mock.patch('upxo.meshing.gbconformant.fm_steel_abaqus.export.check_deck', return_value=self.failing()):
            with self.assertRaisesRegex(RuntimeError, 'Deck check failed: nodes numbered'):
                self.export('hard', REQUIRE_DECK_CHECK=True)

    def test_the_advice_for_that_failure_is_to_stop_blocking(self):
        items = advice.suggest('export_abaqus', 'Deck check failed: nodes numbered 1..N.',
                               {'REQUIRE_DECK_CHECK': True})
        self.assertEqual([(i.setting, i.value) for i in items], [('REQUIRE_DECK_CHECK', False)])


if __name__ == '__main__':
    unittest.main()
