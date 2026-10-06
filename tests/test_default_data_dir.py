"""Pipelines write under <checkout>/data or ./data, never inside the Python environment."""
import subprocess
import sys
from pathlib import Path

from upxo._sup import data_dir


def test_checkout_uses_the_checkout_data_folder():
    root = Path(data_dir.__file__).resolve().parents[3]
    if (root / 'pyproject.toml').is_file():
        assert data_dir.default_data_dir() == root / 'data'


def test_installed_layout_uses_the_working_directory(tmp_path, monkeypatch):
    fake = tmp_path / 'env' / 'Lib' / 'site-packages' / 'upxo' / '_sup'
    fake.mkdir(parents=True)
    (fake / 'data_dir.py').write_text(Path(data_dir.__file__).read_text())
    work = tmp_path / 'work'
    work.mkdir()
    out = subprocess.run([sys.executable, '-c', 'import runpy,sys; print(runpy.run_path(sys.argv[1])["default_data_dir"]())',
                          str(fake / 'data_dir.py')], cwd=work, capture_output=True, text=True, check=True).stdout.strip()
    assert Path(out) == work.resolve() / 'data'


def test_every_pipeline_default_uses_the_helper():
    from upxo.pxtal.fm_steel_3d.mesh_exporter_3d import _DEFAULT_OUTPUT_BASE
    from upxo.pxtal.fm_steel_3d.steps.steps_start import DEFAULT_OUTPUT_DIR as fm
    from upxo.pxtal.twinned_simple_3d.abaqus_exporter_3d import DEFAULT_ABQ_OUT_DIR
    from upxo.pxtal.twinned_simple_3d.steps.steps_start import DEFAULT_OUTPUT_DIR as tw
    base = data_dir.default_data_dir()
    assert Path(fm) == base and Path(tw) == base
    assert Path(_DEFAULT_OUTPUT_BASE) == base / 'ABQInputFiles'
    assert Path(DEFAULT_ABQ_OUT_DIR) == base / 'ABQInputFiles' / 'ofhcCu'
