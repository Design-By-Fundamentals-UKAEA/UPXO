"""The forecast of size, time and memory, and the earlier runs it learns from."""
import json
import unittest

from upxo.meshing.gbconformant.pipeline import estimate
from upxo.meshing.gbconformant.pipeline.estimate import Observation, forecast, observation_from_folder
from upxo.meshing.gbconformant.pipeline.settings import default_config

from .test_cli import call
from ..fm_steel_abaqus.helpers import work_dir


def seen(name, voxels, tets, seconds=100.0, memory=800.0, **settings):
    return Observation(name, voxels, tets=tets, nodes=int(tets * 0.2), seconds=seconds, peak_memory_mb=memory,
                       tet_size_factor=settings.get("tet", 1.5), gmsh_size_factor=settings.get("gmsh", 0.75),
                       complete=True)


class ModelTests(unittest.TestCase):
    def test_the_reference_crop_gives_the_reference_numbers(self):
        result = forecast(estimate.REFERENCE_VOXELS)
        self.assertAlmostEqual(result.tets / estimate.REFERENCE_TETS, 1.0, places=2)
        self.assertAlmostEqual(result.seconds / estimate.REFERENCE_SECONDS, 1.0, places=2)

    def test_more_voxels_mean_more_of_everything(self):
        small, large = forecast(10_000), forecast(40_000)
        self.assertGreater(large.tets, small.tets)
        self.assertGreater(large.seconds, small.seconds)
        self.assertGreater(large.memory_mb, small.memory_mb)

    def test_the_measured_crops_are_reproduced(self):
        self.assertAlmostEqual(forecast(27_000).tets, 308_481, delta=0.1 * 308_481)
        for gmsh, tets in ((0.5, 271_545), (1.0, 88_302), (1.5, 82_995)):
            config = default_config()
            config["GMSH_SIZE_FACTOR"] = gmsh
            self.assertAlmostEqual(forecast(8800, config).tets, tets, delta=0.01 * tets)
        config = default_config()
        config["TET_MESH_SIZE_FACTOR"] = 0.6
        self.assertAlmostEqual(forecast(8800, config).tets, 170_529, delta=0.35 * 170_529)

    def test_a_coarser_surface_mesh_gives_fewer_tetrahedra(self):
        config = default_config()
        config["GMSH_SIZE_FACTOR"] = 1.5
        self.assertLess(forecast(20_000, config).tets, forecast(20_000).tets)

    def test_a_volume_size_below_the_surface_size_gives_more_tetrahedra_otherwise_the_same(self):
        base = forecast(20_000).tets
        finer = default_config()
        finer["TET_MESH_SIZE_FACTOR"] = 0.5
        same = default_config()
        same["TET_MESH_SIZE_FACTOR"] = 1.0
        coarser = default_config()
        coarser["TET_MESH_SIZE_FACTOR"] = 2.5
        self.assertGreater(forecast(20_000, finer).tets, base)
        self.assertEqual(forecast(20_000, same).tets, base)
        self.assertEqual(forecast(20_000, coarser).tets, base)

    def test_a_surface_factor_outside_the_measured_range_is_noted(self):
        config = default_config()
        config["GMSH_SIZE_FACTOR"] = 0.3
        self.assertTrue(any("outside the measured range" in note for note in forecast(5000, config).notes))
        self.assertEqual(forecast(5000).notes, [])

    def test_the_ranges_bracket_the_figures(self):
        result = forecast(15_000)
        for value, (low, high) in ((result.tets, result.tets_range), (result.seconds, result.seconds_range),
                                   (result.memory_mb, result.memory_range)):
            self.assertLess(low, value)
            self.assertGreater(high, value)

    def test_an_explicit_mesh_size_is_noted(self):
        config = default_config()
        config["GMSH_SIZE"] = 0.5
        self.assertTrue(any("explicit mesh size" in note for note in forecast(5000, config).notes))

    def test_nodes_follow_the_tetrahedra(self):
        result = forecast(9000)
        self.assertAlmostEqual(result.nodes / result.tets, estimate.NODES_PER_TET, places=2)


class HistoryTests(unittest.TestCase):
    def test_earlier_runs_scale_the_forecast_to_this_computer(self):
        model_tets = int(estimate.model_tets(8800))
        model_seconds = estimate.model_seconds(8800)
        history = [seen("a", 8800, model_tets, seconds=2 * model_seconds)]
        result = forecast(8800, history=history)
        self.assertAlmostEqual(result.seconds / model_seconds, 2.0, places=1)
        self.assertEqual(result.runs_used, 1)
        self.assertIn("1 earlier run", result.basis)

    def test_unfinished_runs_are_not_used(self):
        run = seen("a", 8800, 100_000)
        run.complete = False
        result = forecast(8800, history=[run])
        self.assertEqual(result.runs_used, 0)
        self.assertIn("built-in", result.basis)

    def test_runs_with_other_size_factors_are_compared_with_their_own_model(self):
        coarse = estimate.model_tets(8800, None, 1.5)
        history = [seen("a", 8800, int(coarse), gmsh=1.5)]
        result = forecast(8800, history=history)
        self.assertAlmostEqual(result.tets / estimate.REFERENCE_TETS, 1.0, places=1)

    def test_more_scatter_between_runs_widens_the_range(self):
        base = int(estimate.model_tets(8800))
        tight = forecast(8800, history=[seen("a", 8800, base), seen("b", 8800, int(base * 1.02))])
        loose = forecast(8800, history=[seen("a", 8800, int(base * 0.5)), seen("b", 8800, int(base * 2))])
        self.assertGreater(loose.tets_range[1] / loose.tets, tight.tets_range[1] / tight.tets)

    def test_a_single_run_never_gives_a_narrow_range(self):
        base = int(estimate.model_tets(8800))
        result = forecast(8800, history=[seen("a", 8800, base)])
        self.assertGreaterEqual(result.tets_range[1] / result.tets, 1.4)


class ObservationFileTests(unittest.TestCase):
    def write_run(self, root, name, manifest, verified=None):
        folder = root / name
        folder.mkdir()
        (folder / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf8")
        if verified is not None:
            (folder / "CM04_dev_verified_tets.json").write_text(json.dumps(verified), encoding="utf8")
        return folder

    def test_a_manifest_with_results_is_read_as_it_is(self):
        with work_dir() as root:
            folder = self.write_run(root, "r1", {
                "config": {"TET_MESH_SIZE_FACTOR": 2.0, "GMSH_SIZE_FACTOR": 1.0},
                "results": {"voxels": 8800, "tets": 90_000, "nodes": 18_000, "seconds": 150.0, "peak_memory_mb": 900.0},
                "saved_results": {"stages": {"insert_nodes": {}}}})
            found = observation_from_folder(folder)
            self.assertEqual((found.voxels, found.tets, found.seconds, found.peak_memory_mb), (8800, 90_000, 150.0, 900.0))
            self.assertEqual((found.tet_size_factor, found.gmsh_size_factor), (2.0, 1.0))
            self.assertTrue(found.complete)

    def test_an_older_manifest_is_read_from_its_box_timings_and_tet_report(self):
        with work_dir() as root:
            folder = self.write_run(root, "r2", {
                "config": {}, "subset": {"box": {"voxels": 8800}},
                "stage_timings": [{"stage_id": "a", "duration_s": 10.0}, {"stage_id": "b", "duration_s": 5.5}],
                "saved_results": {"stages": {"dihedral_gate": {}}}}, verified={"tetrahedra": 111628})
            found = observation_from_folder(folder)
            self.assertEqual((found.voxels, found.tets, found.seconds), (8800, 111628, 15.5))

    def test_a_run_stopped_early_is_not_complete(self):
        with work_dir() as root:
            folder = self.write_run(root, "r3", {"config": {}, "results": {"voxels": 100, "tets": 50},
                                                 "saved_results": {"stages": {"gmsh_interfaces": {}}}})
            self.assertFalse(observation_from_folder(folder).complete)

    def test_a_folder_without_a_manifest_or_voxel_count_is_skipped(self):
        with work_dir() as root:
            (root / "empty").mkdir()
            self.assertIsNone(observation_from_folder(root / "empty"))
            folder = self.write_run(root, "novox", {"config": {}})
            self.assertIsNone(observation_from_folder(folder))

    def test_load_observations_reads_a_runs_folder(self):
        with work_dir() as root:
            runs = root / "runs"
            runs.mkdir()
            self.write_run(runs, "a", {"config": {}, "results": {"voxels": 10, "tets": 5}})
            self.write_run(runs, "b", {"config": {}})
            found = estimate.load_observations(runs)
            self.assertEqual([o.name for o in found], ["a"])
            self.assertEqual(estimate.load_observations(root / "nowhere"), [])


class TextAndCommandTests(unittest.TestCase):
    def test_text_has_one_line_each_for_size_time_memory_output_and_basis(self):
        text = forecast(8800).text()
        self.assertEqual(len(text.splitlines()), 5)
        self.assertIn('field output', text)
        self.assertIn("tetrahedra", text)
        self.assertIn("memory", text)

    def test_durations_and_memory_read_naturally(self):
        self.assertEqual(estimate.duration_text(45), "45 s")
        self.assertEqual(estimate.duration_text(600), "10 min")
        self.assertEqual(estimate.duration_text(7200), "2.0 h")
        self.assertEqual(estimate.memory_text(512), "512 MB")
        self.assertEqual(estimate.memory_text(2048), "2.0 GB")

    def test_the_command_prints_the_forecast(self):
        with work_dir() as root:
            code, out, _err = call("estimate", "--voxels", "8800", "--data-dir", str(root))
            self.assertEqual(code, 0)
            self.assertIn("8,800 voxels", out)
            code, out, _err = call("estimate", "--shape", "20", "20", "20", "--json", "--data-dir", str(root))
            self.assertEqual(json.loads(out)["voxels"], 8000)

    def test_the_output_size_follows_the_load_and_the_tetrahedra(self):
        base = forecast(8800)
        self.assertGreater(base.output['megabytes'], 0)
        self.assertLess(base.output_range[0], base.output['megabytes'])
        finer = default_config()
        finer['FM_OUTPUT_INTERVAL'] = 0.25
        self.assertGreater(forecast(8800, finer).output['frames'], base.output['frames'])
        strain = default_config()
        strain.update(FM_LOAD_MODE='strain', FM_TARGET_STRAIN=0.04, FM_STRAIN_RATE=0.0004)
        self.assertGreater(forecast(8800, strain).output['frames'], base.output['frames'])
        self.assertGreater(forecast(20000).output['megabytes'], base.output['megabytes'])

    def test_an_unusable_load_is_noted_and_the_rest_still_forecast(self):
        config = default_config()
        config.update(FM_LOAD_MODE='strain', FM_LOAD_CASE='cyclic', FM_STRAIN_MIN=0.02, FM_STRAIN_MAX=0.01)
        result = forecast(8800, config)
        self.assertIsNone(result.output)
        self.assertTrue(any('output size is not estimated' in note for note in result.notes))
        self.assertGreater(result.tets, 0)

    def test_the_command_follows_the_preset_and_settings(self):
        with work_dir() as root:
            base = json.loads(call("estimate", "--voxels", "20000", "--json", "--data-dir", str(root))[1])
            quick = json.loads(call("estimate", "--voxels", "20000", "--preset", "quick_look", "--json",
                                    "--data-dir", str(root))[1])
            self.assertLess(quick["tets"], base["tets"])

    def test_the_command_needs_a_size(self):
        code, _out, err = call("estimate")
        self.assertEqual(code, 2)
        self.assertIn("--voxels", err)


if __name__ == "__main__":
    unittest.main()
