import unittest

from upxo.meshing.gbconformant.pipeline import (
    CONFIG_VERSION, SCHEMA, SETTINGS, STAGES, SettingsError, build_config, default_config,
    load_versioned, parse_assignment, stage_ids)
from upxo.meshing.gbconformant.pipeline.handlers import default_handlers
from upxo.meshing.gbconformant.pipeline.settings import (
    coerce_config, coerce_value, differences, migrate_config, parse_value, settings_of_stage)


class TableTests(unittest.TestCase):
    def test_every_default_passes_its_own_checks(self):
        config, problems = coerce_config(default_config())
        self.assertEqual(problems, [])
        self.assertEqual(config, default_config())

    def test_defaults_do_not_share_mutable_values(self):
        first, second = default_config(), default_config()
        first['SUBSET_LO'][0] = 99
        self.assertEqual(second['SUBSET_LO'], [0, 0, 0])

    def test_stage_names_in_the_table_are_stages(self):
        known = set(stage_ids())
        for setting in SETTINGS:
            self.assertLessEqual(set(setting.stages), known, setting.key)

    def test_stage_switches_are_settings(self):
        for stage in STAGES:
            if stage.optional_flag:
                self.assertIn(stage.optional_flag, SCHEMA, stage.id)

    def test_every_stage_has_a_handler_and_every_handler_a_stage(self):
        self.assertEqual(set(default_handlers()), set(stage_ids()))

    def test_stage_settings_lookup(self):
        self.assertIn('MIN_GRAIN_VOXELS', settings_of_stage('small_grains'))
        self.assertIn('SPIKE_MAX_PASSES', settings_of_stage('spikes'))
        self.assertNotIn('SPIKE_MAX_PASSES', settings_of_stage('pinches'))
        for key in ('FM_OUTPUT_UNITS', 'IDENTITY_RULE'):
            self.assertIn('compare_before_after', SCHEMA[key].stages)
            self.assertIn('export_abaqus', SCHEMA[key].stages)

    def test_limits_hold_for_the_defaults(self):
        for setting in SETTINGS:
            if setting.kind in ('int', 'float') and setting.default is not None:
                if setting.minimum is not None:
                    self.assertGreaterEqual(setting.default, setting.minimum, setting.key)
                if setting.maximum is not None:
                    self.assertLessEqual(setting.default, setting.maximum, setting.key)


class CoercionTests(unittest.TestCase):
    def test_values_are_brought_to_their_types(self):
        config, problems = coerce_config({
            'MIN_GRAIN_VOXELS': '6', 'LAPLACE_RELAXATION': '0.5', 'REMOVE_SMALL_GRAINS': 'false',
            'UPSCALE_FACTOR': 2.0, 'SPACING': [1, 2, 3], 'SUBSET_HI': ['5', 6, 7.0],
            'GMSH_ALGORITHM': '5', 'FM_HEADER_MESSAGE': 12, 'PROTECTED_GRAIN_IDS': [3, '4']})
        self.assertEqual(problems, [])
        self.assertEqual(config['MIN_GRAIN_VOXELS'], 6)
        self.assertIsInstance(config['MIN_GRAIN_VOXELS'], int)
        self.assertEqual(config['LAPLACE_RELAXATION'], 0.5)
        self.assertIs(config['REMOVE_SMALL_GRAINS'], False)
        self.assertEqual(config['UPSCALE_FACTOR'], 2)
        self.assertEqual(config['SPACING'], (1.0, 2.0, 3.0))
        self.assertEqual(config['SUBSET_HI'], [5.0, 6.0, 7.0])
        self.assertEqual(config['GMSH_ALGORITHM'], 5)
        self.assertEqual(config['FM_HEADER_MESSAGE'], '12')
        self.assertEqual(config['PROTECTED_GRAIN_IDS'], (3, 4))

    def test_unusable_values_fall_back_and_are_listed(self):
        defaults = default_config()
        bad = {'MIN_GRAIN_VOXELS': 'many', 'LAPLACE_RELAXATION': 99, 'UPSCALE_FACTOR': 2.5,
               'REMOVE_SMALL_GRAINS': 'maybe', 'FM_LOAD_AXIS': 'w', 'DATASET_TYPE': 'other',
               'SUBSET_MODE': 'cube', 'SPACING': [1, 2], 'SUBSET_LO': 'abc',
               'HIERARCHY_LEVEL': 'lath', 'PROTECTED_GRAIN_IDS': 'x'}
        config, problems = coerce_config({**bad, 'MADE_UP': 4, '_RUN_FOLDER': 'x'})
        for key in bad:
            self.assertEqual(config[key], defaults[key], key)
            self.assertTrue(any(line.startswith(key + ':') for line in problems), key)
        self.assertEqual(config['MADE_UP'], 4)
        self.assertTrue(any('MADE_UP' in line for line in problems))
        self.assertEqual(config['_RUN_FOLDER'], 'x')
        self.assertFalse(any('_RUN_FOLDER' in line for line in problems))

    def test_automatic_values_stay_automatic_only_where_allowed(self):
        config, problems = coerce_config({'GMSH_SIZE': None, 'TET_MESH_SIZE': 0.5})
        self.assertEqual(problems, [])
        self.assertIsNone(config['GMSH_SIZE'])
        config, problems = coerce_config({'MIN_GRAIN_VOXELS': None})
        self.assertEqual(config['MIN_GRAIN_VOXELS'], default_config()['MIN_GRAIN_VOXELS'])
        self.assertTrue(problems)

    def test_coerce_value_reports_the_limits(self):
        value, problem = coerce_value(SCHEMA['UPSCALE_FACTOR'], 7)
        self.assertEqual(value, 1)
        self.assertIn('outside [1, 3]', problem)


class TypedEntryTests(unittest.TestCase):
    def test_assignments(self):
        self.assertEqual(parse_assignment('MIN_GRAIN_VOXELS=7'), ('MIN_GRAIN_VOXELS', 7))
        self.assertEqual(parse_assignment('STAGE_SPIKES=no'), ('STAGE_SPIKES', False))
        self.assertEqual(parse_assignment('SUBSET_LO=1, 2 ,3'), ('SUBSET_LO', [1, 2, 3]))
        self.assertEqual(parse_assignment('SPACING=1;1;2'), ('SPACING', (1.0, 1.0, 2.0)))
        self.assertEqual(parse_assignment('GMSH_SIZE=auto'), ('GMSH_SIZE', None))
        self.assertEqual(parse_assignment('GMSH_ALGORITHM=5'), ('GMSH_ALGORITHM', 5))
        self.assertEqual(parse_assignment('PROTECTED_GRAIN_IDS=4 9'), ('PROTECTED_GRAIN_IDS', (4, 9)))
        self.assertEqual(parse_assignment('HIERARCHY_LEVEL='), ('HIERARCHY_LEVEL', ''))
        self.assertEqual(parse_assignment('TEXTURE_GRID=auto'), ('TEXTURE_GRID', 'auto'))
        self.assertEqual(parse_assignment('FM_HEADER_MESSAGE=a=b'), ('FM_HEADER_MESSAGE', 'a=b'))

    def test_bad_assignments_are_refused(self):
        for text in ('MIN_GRAIN_VOXELS', 'NOT_A_SETTING=1', 'MIN_GRAIN_VOXELS=abc',
                     'MIN_GRAIN_VOXELS=0', 'FM_LOAD_AXIS=w', 'SPACING=1,2', 'MIN_GRAIN_VOXELS=auto'):
            with self.assertRaises(SettingsError, msg=text):
                parse_assignment(text)

    def test_parse_value_of_a_json_setting(self):
        self.assertEqual(parse_value(SCHEMA['SMOOTH_ITERATIONS_BY_PAIR'], '{"1-2": 5}'), {'1-2': 5})
        with self.assertRaises(SettingsError):
            parse_value(SCHEMA['SMOOTH_ITERATIONS_BY_PAIR'], '{')

    def test_build_config_is_strict(self):
        config = build_config({'MIN_GRAIN_VOXELS': 9}, SPIKE_MAX_PASSES='12')
        self.assertEqual((config['MIN_GRAIN_VOXELS'], config['SPIKE_MAX_PASSES']), (9, 12))
        self.assertEqual(config['RUN_NAME'], default_config()['RUN_NAME'])
        with self.assertRaises(SettingsError):
            build_config(NOT_A_SETTING=1)
        with self.assertRaises(SettingsError):
            build_config(UPSCALE_FACTOR=5)
        self.assertEqual(build_config(_RUN_FOLDER='x')['_RUN_FOLDER'], 'x')


class VersionTests(unittest.TestCase):
    def test_old_settings_with_a_sample_path_load_a_labels_image(self):
        old = {'SAMPLE_PATH': 'a.npy', 'MIN_GRAIN_VOXELS': 4}
        self.assertEqual(migrate_config(old, 1)['DATASET_TYPE'], 'labels')
        self.assertNotIn('DATASET_TYPE', old)
        self.assertEqual(load_versioned(old, 1)[0]['DATASET_TYPE'], 'labels')
        self.assertEqual(load_versioned({'DATASET_TYPE': 'fm_steel', 'SAMPLE_PATH': 'x'}, 1)[0]['DATASET_TYPE'],
                         'fm_steel')
        self.assertEqual(load_versioned({}, 1)[0]['DATASET_TYPE'], 'fm_steel')
        self.assertEqual(migrate_config(old, CONFIG_VERSION), old)

    def test_newer_settings_are_refused(self):
        with self.assertRaises(ValueError):
            migrate_config({}, CONFIG_VERSION + 1)


class DifferenceTests(unittest.TestCase):
    def test_only_changed_settings_are_listed(self):
        defaults = default_config()
        current = dict(defaults, MIN_GRAIN_VOXELS=9, SPACING=(1.0, 1.0, 2.0), EXTRA=1)
        changed = {row['key']: row for row in differences(defaults, current)}
        self.assertEqual(set(changed), {'MIN_GRAIN_VOXELS', 'SPACING', 'EXTRA'})
        self.assertEqual(changed['MIN_GRAIN_VOXELS']['default'], 4)
        self.assertIsNone(changed['EXTRA']['default'])
        self.assertEqual(differences(defaults, dict(defaults, LAPLACE_RELAXATION=0.75 + 1e-15)), [])


class SignatureOfALateSettingTests(unittest.TestCase):
    """GMSH_RETRY_FAILED_ONLY is signed only when it is changed, so results saved before it existed stay valid."""

    def test_the_setting_exists_off_and_names_its_stages(self):
        setting = SCHEMA['GMSH_RETRY_FAILED_ONLY']
        self.assertIs(setting.default, False)
        self.assertEqual(setting.stages, ('gmsh_interfaces', 'joint_remesh'))
        self.assertTrue(setting.signed_when_changed)

    def test_at_its_default_the_signatures_are_those_of_a_config_without_it(self):
        from upxo.meshing.gbconformant.pipeline import saved_results
        config = default_config()
        older = {k: v for k, v in config.items() if k != 'GMSH_RETRY_FAILED_ONLY'}
        self.assertEqual(saved_results.signatures(config), saved_results.signatures(older))
        self.assertEqual(saved_results.signatures(config), saved_results.signatures(dict(older, GMSH_RETRY_FAILED_ONLY=None)))

    def test_turning_it_on_changes_the_two_stages_and_those_after_them(self):
        from upxo.meshing.gbconformant.pipeline import saved_results
        config = default_config()
        before = saved_results.signatures(config)
        after = saved_results.signatures(dict(config, GMSH_RETRY_FAILED_ONLY=True))
        order = stage_ids()
        first = order.index('gmsh_interfaces')
        for index, stage in enumerate(order):
            self.assertEqual(before[stage] == after[stage], index < first, stage)

    def test_a_setting_that_is_always_signed_is_unaffected(self):
        from upxo.meshing.gbconformant.pipeline import saved_results
        config = default_config()
        changed = saved_results.signatures(dict(config, GMSH_ALGORITHM=5))
        self.assertNotEqual(saved_results.signatures(config)['gmsh_interfaces'], changed['gmsh_interfaces'])


if __name__ == '__main__':
    unittest.main()
