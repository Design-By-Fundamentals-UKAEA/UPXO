"""Advice after a failed stage: proposals only, grounded in the settings table."""
import unittest

from upxo.meshing.gbconformant.pipeline import advice
from upxo.meshing.gbconformant.pipeline.settings import SCHEMA, default_config
from upxo.meshing.gbconformant.pipeline.stages import stage_ids


class RuleTests(unittest.TestCase):
    def test_every_rule_names_a_real_stage_and_real_settings(self):
        stages = set(stage_ids())
        for stage, pattern, changes in advice.RULES:
            self.assertTrue(stage is None or stage in stages, stage)
            for setting, _value, why in changes:
                self.assertIn(setting, SCHEMA, setting)
                self.assertTrue(why)

    def test_every_proposed_value_passes_the_settings_checks(self):
        config = default_config()
        for stage in stage_ids():
            for text in ("Unresolved voxel boundary defects", "exhausted passes", "exceeds max_correction",
                         "intersection check failed", "minimum triangle angle", "not resolved all surface defects",
                         "below quality thresholds", "frozen", "anything else"):
                for item in advice.suggest(stage, text, config):
                    if item.setting is None:
                        continue
                    from upxo.meshing.gbconformant.pipeline.settings import coerce_value
                    _checked, problem = coerce_value(SCHEMA[item.setting], item.value)
                    self.assertIsNone(problem, f"{stage}/{text}/{item.setting}")


class SuggestTests(unittest.TestCase):
    def test_a_known_failure_gets_settings_that_bear_on_it(self):
        items = advice.suggest("surface_gate", "Surface minimum triangle angle 12.0 is below 30", default_config())
        names = [i.setting for i in items]
        self.assertIn("SURFACE_ANGLE_MAX_PASSES", names)
        self.assertIn("APPROVE_TET_MESHING_WITH_POOR_SURFACE", names)

    def test_a_proposal_equal_to_the_current_value_is_dropped(self):
        config = default_config()
        config["APPROVE_TET_MESHING_WITH_POOR_SURFACE"] = True
        items = advice.suggest("surface_gate", "minimum triangle angle", config)
        self.assertNotIn("APPROVE_TET_MESHING_WITH_POOR_SURFACE", [i.setting for i in items])

    def test_increases_scale_the_current_value_and_round_floats(self):
        config = default_config()
        config["TOPOLOGY_MAX_PASSES"] = 10
        config["TOPOLOGY_MAX_VOLUME_FRACTION"] = 0.2
        by_name = {i.setting: i for i in advice.suggest("voxel_topology",
                                                       "Unresolved voxel boundary defects: 3", config)}
        self.assertEqual(by_name["TOPOLOGY_MAX_PASSES"].value, 20)
        self.assertEqual(by_name["TOPOLOGY_MAX_VOLUME_FRACTION"].value, 0.3)

    def test_nothing_is_proposed_for_an_unknown_failure(self):
        self.assertEqual(advice.suggest("load_lgi", "disk on fire", default_config()), [])

    def test_a_stage_typo_is_an_error(self):
        with self.assertRaises(KeyError):
            advice.suggest("no_such_stage", "x", default_config())

    def test_the_stage_a_suggestion_restarts_from_is_known(self):
        item = next(i for i in advice.suggest("surface_gate", "minimum triangle angle", default_config())
                    if i.setting == "SURFACE_ANGLE_MAX_PASSES")
        self.assertIn(item.stage, stage_ids())

    def test_apply_changes_only_the_named_setting(self):
        config = default_config()
        item = advice.Suggestion("TET_OPTIMIZE_NETGEN", False, "x")
        before = dict(config)
        advice.apply_suggestion(config, item)
        changed = {k for k in config if config[k] != before[k]}
        self.assertEqual(changed, {"TET_OPTIMIZE_NETGEN"} if before["TET_OPTIMIZE_NETGEN"] else set())

    def test_a_hint_that_is_not_a_setting_cannot_be_applied(self):
        with self.assertRaises(ValueError):
            advice.apply_suggestion({}, advice.Suggestion(None, None, "run something first"))


if __name__ == "__main__":
    unittest.main()
