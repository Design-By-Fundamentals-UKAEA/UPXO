"""No tracked module uses an API that the supported matplotlib, NumPy or SciPy releases have removed."""
import ast
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PYPLOT_NAMES = {'plt', 'pyplot', '_mcm'}                        # pyplot keeps get_cmap; matplotlib.cm lost it in 3.9
NUMPY_REMOVED = {'NaN', 'Inf', 'infty', 'float_', 'complex_', 'product', 'cumproduct', 'sometrue', 'alltrue', 'asfarray',
                 'unicode_', 'string_', 'cast', 'bool8', 'in1d', 'trapz', 'row_stack', 'msort', 'find_common_type'}
ANY_RECEIVER_REMOVED = {'tostring_rgb', 'register_cmap', 'cmap_d', 'simps', 'cumtrapz'}


def _tracked():
    out = subprocess.run(['git', 'ls-files', 'src/upxo'], cwd=ROOT, capture_output=True, text=True).stdout.split()
    return [f for f in out if f.endswith('.py') and '/gui/' not in f and not f.startswith('src/upxo/demos/')]


def _findings(path):
    tree = ast.parse((ROOT / path).read_bytes().decode('utf-8-sig'))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        receiver = node.value.id if isinstance(node.value, ast.Name) else None
        if node.attr == 'get_cmap' and receiver not in PYPLOT_NAMES:
            yield node.lineno, f'{receiver or "?"}.get_cmap (use plt.get_cmap)'
        elif node.attr in NUMPY_REMOVED and receiver in ('np', 'numpy'):
            yield node.lineno, f'{receiver}.{node.attr}'
        elif node.attr in ANY_RECEIVER_REMOVED:
            yield node.lineno, node.attr


@pytest.mark.parametrize('path', _tracked())
def test_no_removed_api(path):
    found = list(_findings(path))
    assert not found, f'{path}: {found}'


def test_the_scan_sees_a_removed_call(tmp_path, monkeypatch):
    sample = tmp_path / 'm.py'
    sample.write_text('import numpy as np\nimport matplotlib.cm as cm\nx = np.NaN\ny = cm.get_cmap("viridis")\nz = plt.get_cmap("viridis")\n')
    monkeypatch.setattr(f'{__name__}.ROOT', tmp_path)
    assert [w for _, w in _findings('m.py')] == ['np.NaN', 'cm.get_cmap (use plt.get_cmap)']
