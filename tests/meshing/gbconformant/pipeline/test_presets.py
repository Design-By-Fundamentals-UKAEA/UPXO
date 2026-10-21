"""Built-in presets: real settings, switchable, and recognised again."""
import json
import unittest

from upxo.meshing.gbconformant.pipeline import presets
from upxo.meshing.gbconformant.pipeline.settings import SCHEMA, default_config
from upxo.meshing.gbconformant.pipeline.stages import stage_ids
from .test_cli import call as run_cli


class PresetTests(unittest.TestCase):
    def test_every_preset_uses_real_settings_with_valid_values(self):
        presets.validate_presets()
        for preset in presets.PRESETS.values():
            self.assertTrue(preset.title and preset.summary)
            self.assertIn(preset.material, presets.MATERIALS)
            self.assertTrue(all(name in SCHEMA for name in preset.settings))

    def test_the_default_preset_changes_nothing_on_a_default_config(self):
        self.assertEqual(presets.changes(default_config(), "default"), [])
        self.assertEqual(presets.which_preset(default_config()), "default")

    def test_applying_sets_the_preset_values_and_nothing_else(self):
        config = default_config()
        config["GRAIN_ID_SEED"] = 7
        made = presets.apply_preset(config, "quick_look")
        self.assertEqual([name for name, _old, _new in made], ["GMSH_SIZE_FACTOR"])
        self.assertEqual(config["GMSH_SIZE_FACTOR"], 1.5)
        self.assertEqual(config["GRAIN_ID_SEED"], 7)
        self.assertEqual(presets.which_preset(config), "quick_look")

    def test_presets_can_be_switched_between_without_leftovers(self):
        config = default_config()
        presets.apply_preset(config, "upscaled")
        self.assertEqual(config["UPSCALE_FACTOR"], 2)
        presets.apply_preset(config, "quick_look")
        self.assertEqual(config["UPSCALE_FACTOR"], 1)
        self.assertEqual(config["REQUIRE_CLEAN_TOPOLOGY"], False)
        presets.apply_preset(config, "default")
        self.assertEqual(config, default_config())

    def test_a_hand_edited_config_matches_no_preset(self):
        config = default_config()
        config["GMSH_SIZE_FACTOR"] = 1.1
        self.assertIsNone(presets.which_preset(config))

    def test_no_preset_touches_the_seed_or_the_input(self):
        for name in ("GRAIN_ID_SEED", "RAW_FOLDER", "SAMPLE_PATH", "DATASET_TYPE", "RUN_NAME"):
            self.assertNotIn(name, presets.controlled_keys())

    def test_unknown_names_are_errors(self):
        with self.assertRaises(KeyError):
            presets.get_preset("nope")

    def test_first_stage_of_every_controlled_setting_is_a_real_stage(self):
        stages = set(stage_ids())
        for name in presets.controlled_keys():
            for stage in SCHEMA[name].stages:
                self.assertIn(stage, stages)

    def test_size_classes_and_recommendation(self):
        self.assertEqual(presets.size_class(8_800), "small")
        self.assertEqual(presets.size_class(27_000), "medium")
        self.assertEqual(presets.size_class(150_000), "large")
        self.assertEqual(presets.recommend(150_000)[0].key, "large_crop")
        self.assertEqual(presets.recommend(5_000)[0].size_class, "small")
        self.assertNotIn("large_crop", [p.key for p in presets.recommend(5_000)])

    def test_describe_names_the_settings_and_the_source(self):
        text = presets.get_preset("strict").describe()
        self.assertIn("REQUIRE_CLEAN_TOPOLOGY = True", text)
        self.assertIn("development notebook", text)


class PresetCommandTests(unittest.TestCase):
    def test_the_list_names_every_preset(self):
        code, out, _err = run_cli("presets")
        self.assertEqual(code, 0)
        for key in presets.PRESETS:
            self.assertIn(key, out)

    def test_json_describes_one_preset(self):
        code, out, _err = run_cli("presets", "quick_look", "--json")
        self.assertEqual(code, 0)
        data = json.loads(out)
        self.assertEqual(data[0]["settings"], {"GMSH_SIZE_FACTOR": 1.5})

    def test_an_unknown_name_is_a_usage_error(self):
        code, _out, err = run_cli("presets", "nope")
        self.assertEqual(code, 2)
        self.assertIn("not a preset", err)


if __name__ == "__main__":
    unittest.main()
