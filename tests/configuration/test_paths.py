from pathlib import Path

from upxo.configuration import ConfigStore, UPXOPaths


def test_run_paths_create_expected_tree(tmp_path: Path):
    config = UPXOPaths(output_root=tmp_path)
    paths = config.run_paths("fm_steel", "run-001", create=True)
    assert paths.abaqus.is_dir()
    assert paths.distributions.is_dir()
    assert paths.manifest.name == "manifest.json"


def test_config_round_trip(tmp_path: Path):
    location = tmp_path / "paths.json"
    original = UPXOPaths(output_root=tmp_path / "exports")
    ConfigStore.save(original, location)
    loaded = ConfigStore.load(location)
    assert loaded.output_root == (tmp_path / "exports").resolve()
    assert loaded.default_pipeline == "fm_steel"


def test_missing_config_returns_defaults(tmp_path: Path):
    loaded = ConfigStore.load(tmp_path / "missing.json")
    assert loaded.output_root == (Path.home() / "UPXO").resolve()
