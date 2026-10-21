"""Load cases (strain, strain rate, compression, hold, cycles) and the output size estimate."""
import re
import shutil
import unittest
import numpy as np
from upxo.meshing.gbconformant.fm_steel_abaqus import DeckSettings, applied_strain
from upxo.meshing.gbconformant.fm_steel_abaqus import loading, outputs
from upxo.meshing.gbconformant.fm_steel_abaqus.deckcheck import check_deck
from .helpers import read_elsets, work_dir
from .test_deck import build

LENGTH = 6.0   # mm along x of the test structure at voxel size 0.001 mm


def strain_settings(**kwargs):
    base = dict(load_mode='strain', strain_rate=0.001, target_strain=0.02, load_axis='z')
    base.update(kwargs)
    return DeckSettings(**base)


class MonotonicTests(unittest.TestCase):
    def test_displacement_mode_is_one_step_with_the_given_values(self):
        plan = loading.plan_load(DeckSettings(applied_displacement=0.3, step_time=40.0), LENGTH)
        self.assertEqual(len(plan.steps), 1)
        step = plan.steps[0]
        self.assertEqual((step.time, step.displacement, step.initial_inc, step.max_inc, step.amplitude),
                         (40.0, 0.3, 0.01, 1.0, None))
        self.assertAlmostEqual(plan.summary['strain'], 0.3 / LENGTH)

    def test_strain_and_rate_give_the_time_and_the_displacement(self):
        plan = loading.plan_load(strain_settings(target_strain=0.02, strain_rate=0.001), 10.0)
        step = plan.steps[0]
        self.assertAlmostEqual(step.time, 20.0)
        self.assertAlmostEqual(step.displacement, 0.2)
        self.assertEqual(plan.summary['case'], 'monotonic')
        self.assertAlmostEqual(plan.summary['total_time'], 20.0)

    def test_compression_moves_the_face_the_other_way(self):
        plan = loading.plan_load(strain_settings(load_sense='compression'), 10.0)
        self.assertLess(plan.steps[0].displacement, 0)
        self.assertAlmostEqual(plan.steps[0].displacement, -0.2)
        self.assertAlmostEqual(plan.summary['strain'], -0.02)

    def test_increments_are_cut_to_fit_a_short_step(self):
        plan = loading.plan_load(strain_settings(target_strain=0.001, strain_rate=0.01, max_inc=1.0,
                                                 initial_inc=0.5, output_interval=2.0), 10.0)
        step = plan.steps[0]
        self.assertAlmostEqual(step.time, 0.1)
        self.assertLessEqual(step.max_inc, step.time)
        self.assertLessEqual(step.initial_inc, step.max_inc)
        self.assertLessEqual(step.output_interval, step.time)
        self.assertTrue(any('output interval cut' in note for note in plan.summary['notes']))

    def test_a_hold_is_a_second_step_at_the_final_displacement(self):
        plan = loading.plan_load(strain_settings(hold_time=30.0), 10.0)
        self.assertEqual([s.name for s in plan.steps], ['Step-1', 'Hold'])
        hold = plan.steps[1]
        self.assertEqual((hold.kind, hold.time, hold.amplitude), ('hold', 30.0, None))
        self.assertAlmostEqual(hold.displacement, plan.steps[0].displacement)
        self.assertAlmostEqual(plan.summary['total_time'], 20.0 + 30.0)

    def test_a_hold_after_a_displacement_step_works_too(self):
        plan = loading.plan_load(DeckSettings(hold_time=5.0), LENGTH)
        self.assertEqual(len(plan.steps), 2)
        self.assertAlmostEqual(loading.total_time(DeckSettings(hold_time=5.0)), 85.0)

    def test_time_alone_needs_no_length(self):
        self.assertAlmostEqual(loading.total_time(strain_settings(target_strain=0.05, strain_rate=0.002)), 25.0)

    def test_bad_settings_are_refused(self):
        for bad in (dict(load_mode='nope'), dict(strain_rate=0), dict(target_strain=0), dict(load_case='nope'),
                    dict(load_sense='sideways'), dict(hold_time=-1), dict(load_case='cyclic', cycles=0),
                    dict(load_case='cyclic', strain_min=0.01, strain_max=0.01),
                    dict(load_case='cyclic', strain_min=0.0, strain_max=0.0),
                    dict(load_case='cyclic', cycle_definition='amplitude_r', r_ratio=1.0),
                    dict(load_case='cyclic', cycle_definition='amplitude_r', strain_amplitude=0),
                    dict(load_case='cyclic', cycle_definition='other')):
            with self.assertRaises(ValueError, msg=str(bad)):
                strain_settings(**bad).validate()


class CyclicTests(unittest.TestCase):
    def cyclic(self, **kwargs):
        return strain_settings(load_case='cyclic', strain_min=-0.01, strain_max=0.01, strain_rate=0.001, cycles=2,
                               **kwargs)

    def test_the_history_goes_up_then_down_and_up_for_each_cycle(self):
        vertices = loading.strain_vertices(self.cyclic())
        strains = [s for _t, s in vertices]
        self.assertEqual(strains, [0.0, 0.01, -0.01, 0.01, -0.01, 0.01])
        times = [t for t, _s in vertices]
        np.testing.assert_allclose(times, [0, 10, 30, 50, 70, 90])

    def test_the_amplitude_table_is_signed_fractions_of_the_peak(self):
        plan = loading.plan_load(self.cyclic(), 10.0)
        table = plan.amplitudes[loading.AMPLITUDE_NAME]
        self.assertEqual([round(f, 12) for _t, f in table], [0, 1, -1, 1, -1, 1])
        step = plan.steps[0]
        self.assertEqual(step.amplitude, loading.AMPLITUDE_NAME)
        self.assertAlmostEqual(step.displacement, 0.1)             # the peak strain times the length
        self.assertAlmostEqual(step.time, 90.0)

    def test_the_longest_increment_reaches_the_peaks(self):
        plan = loading.plan_load(self.cyclic(max_inc=100.0, initial_inc=100.0), 10.0)
        self.assertAlmostEqual(plan.steps[0].max_inc, 1.0)          # a tenth of the shortest segment, 10
        self.assertTrue(any('peaks are reached' in note for note in plan.summary['notes']))

    def test_asymmetric_limits_and_tension_tension(self):
        settings = strain_settings(load_case='cyclic', strain_min=0.002, strain_max=0.01, strain_rate=0.001, cycles=1)
        self.assertEqual([s for _t, s in loading.strain_vertices(settings)], [0.0, 0.01, 0.002, 0.01])
        plan = loading.plan_load(settings, 10.0)
        self.assertEqual(min(f for _t, f in plan.amplitudes[loading.AMPLITUDE_NAME]), 0.0)

    def test_compression_first_limits(self):
        settings = strain_settings(load_case='cyclic', strain_min=-0.02, strain_max=0.0, strain_rate=0.001, cycles=1)
        self.assertEqual([s for _t, s in loading.strain_vertices(settings)], [0.0, -0.02, 0.0])
        plan = loading.plan_load(settings, 10.0)
        self.assertEqual(plan.steps[0].displacement, 0.2)
        self.assertEqual([round(f, 9) for _t, f in plan.amplitudes[loading.AMPLITUDE_NAME]], [0.0, -1.0, 0.0])

    def test_amplitude_and_r_give_the_same_cycle_as_the_limits(self):
        fully_reversed = strain_settings(load_case='cyclic', cycle_definition='amplitude_r', strain_amplitude=0.01,
                                         r_ratio=-1.0)
        low, high = loading.cycle_limits(fully_reversed)
        self.assertAlmostEqual(low, -0.01)
        self.assertAlmostEqual(high, 0.01)
        tension = strain_settings(load_case='cyclic', cycle_definition='amplitude_r', strain_amplitude=0.004,
                                  r_ratio=0.1)
        low, high = loading.cycle_limits(tension)
        self.assertAlmostEqual(high, 2 * 0.004 / 0.9)
        self.assertAlmostEqual(low, 0.1 * high)
        self.assertAlmostEqual((high - low) / 2, 0.004)
        self.assertAlmostEqual(low / high, 0.1)

    def test_r_above_one_is_compression_compression(self):
        settings = strain_settings(load_case='cyclic', cycle_definition='amplitude_r', strain_amplitude=0.002, r_ratio=2.0)
        low, high = loading.cycle_limits(settings)
        self.assertLess(high, 0)
        self.assertLess(low, high)
        self.assertAlmostEqual((high - low) / 2, 0.002)
        self.assertAlmostEqual(low / high, 2.0)

    def test_a_hold_after_cycles_keeps_the_last_displacement(self):
        plan = loading.plan_load(self.cyclic(hold_time=20.0), 10.0)
        self.assertEqual([s.name for s in plan.steps], ['Step-1', 'Hold'])
        self.assertAlmostEqual(plan.steps[1].displacement, 0.1)       # the history ends at the maximum strain
        self.assertAlmostEqual(plan.summary['total_time'], 110.0)

    def test_the_description_names_the_case(self):
        text = '\n'.join(loading.describe(loading.plan_load(self.cyclic(), 10.0), 'mm', 'z'))
        self.assertIn('Cyclic between strains -0.01 and 0.01', text)
        self.assertIn('2 cycle(s)', text)
        text = '\n'.join(loading.describe(loading.plan_load(strain_settings(hold_time=3.0), 10.0), 'mm', 'z'))
        self.assertIn('Monotonic tension', text)
        self.assertIn('hold of 3', text)


class DeckStepTests(unittest.TestCase):
    def steps_text(self, **kwargs):
        with work_dir() as root:
            _, _, _, res = build(root, settings=DeckSettings(**kwargs))
            return (res['directory'] / '08_steps_output.inp').read_text(), res

    def test_the_default_step_text_is_unchanged(self):
        text, _ = self.steps_text()
        self.assertEqual(text, (
            '** STEP: Step-1\n*Step, name=Step-1, nlgeom=YES, inc=10000\n*Static\n0.01, 80, 1e-08, 1\n'
            '*Boundary\nns_zplus, 3, 3, 0.2\n*Boundary\nns_xminus, 1, 1\n*Boundary\nns_yminus, 2, 2\n'
            '*Boundary\nns_zminus, 3, 3\n*Restart, write, frequency=0\n*Output, field, time interval=0.5\n'
            '*Element Output, directions=YES\nLE, NE, S, SDV\n*Node Output\nU,\n'
            '*Output, history, variable=PRESELECT\n*End Step\n'))

    def test_a_monotonic_strain_step(self):
        text, res = self.steps_text(load_mode='strain', target_strain=0.02, strain_rate=0.001, load_sense='compression')
        self.assertIn('*Static\n0.01, 20, 1e-08, 1\n', text)
        length_mm = 6 * 0.001                                  # six voxels of 0.001 mm along x; z is four voxels
        z_length = 4 * 0.001
        self.assertIn('ns_zplus, 3, 3, %.10g' % (-0.02 * z_length), text)
        self.assertNotIn('*Amplitude', text)
        self.assertEqual(res['info']['load_case']['case'], 'monotonic')

    def test_a_cyclic_deck_defines_the_amplitude_before_its_step(self):
        text, res = self.steps_text(load_mode='strain', load_case='cyclic', strain_min=-0.01, strain_max=0.01,
                                    strain_rate=0.001, cycles=2)
        self.assertLess(text.index('*Amplitude, name=LOAD-HISTORY, definition=TABULAR'), text.index('*Step'))
        table = text.split('definition=TABULAR\n')[1].split('** STEP')[0].strip().splitlines()
        self.assertEqual(len(table), 2)                          # six pairs, four to a line
        self.assertTrue(table[0].startswith('0, 0, 10, 1, 30, -1'))
        self.assertIn('*Boundary, amplitude=LOAD-HISTORY\nns_zplus, 3, 3, ', text)
        self.assertEqual(text.count('*Step,'), 1)
        self.assertEqual(res['info']['load_case']['cycles'], 2)

    def test_a_cyclic_step_ends_with_its_amplitude_table(self):
        text, res = self.steps_text(load_mode='strain', load_case='cyclic', cycle_definition='amplitude_r',
                                    strain_amplitude=0.004, r_ratio=0.1, strain_rate=0.001, cycles=2)
        table = text.split('definition=TABULAR\n')[1].split('** STEP')[0].replace('\n', ', ').split(', ')
        last_time = float(table[-3])                  # the table ends with the pair "time, fraction" and a blank
        period = float(re.search(r'\*Static\n[^,]+, ([^,]+),', text).group(1))
        self.assertEqual(period, last_time)

    def test_a_hold_writes_a_second_step_that_restates_the_displacement(self):
        text, res = self.steps_text(load_mode='strain', target_strain=0.02, strain_rate=0.001, hold_time=30.0)
        self.assertEqual(text.count('*Step,'), 2)
        hold = text.split('** STEP: Hold')[1]
        self.assertIn('*Static\n0.01, 30, 1e-08, 1\n', hold)
        self.assertIn('ns_zplus, 3, 3, %.10g' % (0.02 * 4 * 0.001), hold)
        self.assertNotIn('ns_xminus', hold)                       # the fixed faces are stated once
        self.assertIn('*Output, field, time interval=', hold)
        self.assertEqual(text.count('*Restart'), 1)

    def test_the_length_scale_multiplies_the_displacement(self):
        text, _ = self.steps_text(load_mode='strain', target_strain=0.02, strain_rate=0.001, length_scale=1000.0)
        self.assertIn('ns_zplus, 3, 3, %.10g' % (0.02 * 4 * 0.001 * 1000.0), text)

    def test_the_deck_check_accepts_these_decks(self):
        for kwargs in (dict(load_mode='strain', load_case='cyclic', cycles=3, hold_time=5.0),
                       dict(load_mode='strain', hold_time=5.0), dict()):
            with work_dir() as root:
                _, _, _, res = build(root, settings=DeckSettings(**kwargs))
                result = check_deck(res['directory'])
                self.assertTrue(result['ok'], result['failed'])

    def test_the_deck_check_finds_an_amplitude_that_is_not_defined(self):
        with work_dir() as root:
            _, _, _, res = build(root, settings=DeckSettings(load_mode='strain', load_case='cyclic'))
            copy = root / 'copy'
            shutil.copytree(res['directory'], copy)
            path = copy / '08_steps_output.inp'
            path.write_text(path.read_text().replace('*Amplitude, name=LOAD-HISTORY', '*Amplitude, name=OTHER'))
            self.assertIn('boundary conditions use defined amplitudes', check_deck(copy)['failed'])

    def test_applied_strain_follows_the_mode(self):
        with work_dir() as root:
            hier, _, _, _ = build(root)
            z = 4 * 0.001
            self.assertAlmostEqual(applied_strain(DeckSettings(applied_displacement=0.0004), hier), 0.0004 / z)
            self.assertAlmostEqual(applied_strain(strain_settings(target_strain=0.03), hier), 0.03)
            self.assertAlmostEqual(applied_strain(strain_settings(target_strain=0.03, load_sense='compression'), hier), -0.03)
            cyc = strain_settings(load_case='cyclic', strain_min=-0.02, strain_max=0.01)
            self.assertAlmostEqual(applied_strain(cyc, hier), 0.02)


class OutputSetTests(unittest.TestCase):
    def test_output_can_be_limited_to_an_element_set(self):
        with work_dir() as root:
            _, _, (pts, tets, grains), res = build(root, settings=DeckSettings(layer_sets=True, boundary_sets=True,
                                                                              output_elset='es_gb_layer'))
            text = (res['directory'] / '08_steps_output.inp').read_text()
            self.assertIn('*Element Output, elset=es_gb_layer, directions=YES', text)
            layer = read_elsets(res['directory'] / '03f_elsets_layers.inp')['es_gb_layer']
            self.assertEqual(res['info']['output_estimate']['elements_in_output'], len(layer))

    def test_an_unknown_set_is_refused_with_examples(self):
        with work_dir() as root:
            with self.assertRaisesRegex(ValueError, "output_elset 'es_nothing' is not an element set"):
                build(root, settings=DeckSettings(output_elset='es_nothing'))


class SizeEstimateTests(unittest.TestCase):
    def test_the_figures_follow_the_variables_nodes_elements_and_frames(self):
        figures = outputs.estimate_output(('S', 'SDV'), ('U',), n_nodes=100, n_elements=500, n_depvar=2,
                                          step_times=[10.0], intervals=[1.0])
        per_frame = 100 * (4 * 3 + 4) + 500 * (4 * 6 + 4) + 500 * (4 * 2 + 4)
        self.assertEqual(figures['bytes_per_frame'], per_frame)
        self.assertEqual(figures['frames'], 11)
        self.assertAlmostEqual(figures['megabytes'], per_frame * 11 / 2 ** 20)
        self.assertEqual(figures['dofs'], 300)
        self.assertTrue(figures['rough'])

    def test_frames_count_every_step(self):
        self.assertEqual(outputs.frame_count([10.0, 5.0], [1.0, 2.0]), 1 + 10 + 3)

    def test_an_unknown_variable_is_listed_not_hidden(self):
        figures = outputs.estimate_output(('MYVAR',), ('U',), 10, 10, 1, [1.0], [1.0])
        self.assertEqual(figures['unknown_variables'], ['MYVAR'])

    def test_a_smaller_output_set_gives_a_smaller_file(self):
        full = outputs.estimate_output(('S',), (), 100, 500, 1, [10.0], [1.0])
        part = outputs.estimate_output(('S',), (), 100, 500, 1, [10.0], [1.0], n_output_elements=50)
        self.assertAlmostEqual(part['megabytes'] * 10, full['megabytes'])

    def test_the_deck_reports_the_estimate_for_its_own_plan(self):
        with work_dir() as root:
            _, _, (pts, tets, grains), res = build(root, settings=DeckSettings(
                load_mode='strain', target_strain=0.02, strain_rate=0.001, output_interval=2.0))
            figures = res['info']['output_estimate']
            self.assertEqual(figures['frames'], 1 + 10)
            self.assertEqual(figures['dofs'], 3 * res['info']['nodes'])
            self.assertIn('rough', outputs.text(figures))

    def test_the_text_uses_gb_for_big_files(self):
        figures = {'dofs': 3, 'frames': 2, 'megabytes': 3000.0}
        self.assertIn('2.9 GB', outputs.text(figures))


if __name__ == '__main__':
    unittest.main()
