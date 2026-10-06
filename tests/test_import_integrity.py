"""Exhaustive import-integrity checks for the tracked UPXO sources.

Catches the bugs a plain "import upxo" misses:

* a module that fails to import, or prints when it is imported
* an ``import`` / ``from X import name`` anywhere (also inside functions) whose
  module or name does not exist, or that needs a package no installation level
  declares
* ``importlib.import_module('upxo....')`` strings that point nowhere
* dependency lists that disagree (pyproject.toml, setup.py, requirements.txt)
* entry points, Sphinx listings and tracked notebooks that import something missing

Modules are imported in fresh interpreters (one per subpackage), so an import
that only works after another module was imported first is reported too.
Packages from a declared optional extra may be missing (``[viz]``, ``[mesh]``,
``[ebsd]``); anything else may not.
"""
import ast
import json
import os
import re
import subprocess
import sys
import tomllib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
STD = set(sys.stdlib_module_names)
# distribution name -> import name, where they differ
ALIAS = {'scikit-image': 'skimage', 'scikit-learn': 'sklearn', 'pillow': 'PIL', 'connected-components-3d': 'cc3d'}
# installed with the declared dependencies, not listed themselves
TRANSITIVE = {'vtk', 'IPython', 'mpl_toolkits', 'packaging', 'PIL', 'pkg_resources', 'setuptools', 'typing_extensions'}
SKIP_PARTS = ('.gui', '__main__')


def _tracked_files():
    try:
        out = subprocess.run(['git', '-C', str(ROOT), 'ls-files'], capture_output=True, text=True, check=True).stdout
        files = [f for f in out.split('\n') if f]
        if any(f.startswith('src/upxo/') for f in files):
            return files
    except (OSError, subprocess.CalledProcessError):
        pass
    return sorted(str(p.relative_to(ROOT)).replace(os.sep, '/') for p in (ROOT / 'src' / 'upxo').rglob('*') if p.is_file())


FILES = _tracked_files()
MODULES = {}
for _f in FILES:
    if _f.startswith('src/upxo/') and _f.endswith('.py'):
        _name = _f[4:-3].replace('/', '.')
        MODULES[_name[:-9] if _name.endswith('.__init__') else _name] = _f
PACKAGES = {m for m, f in MODULES.items() if f.endswith('__init__.py')}
CHECKED = sorted(m for m in MODULES if not any(s in m for s in SKIP_PARTS))


def _import_name(requirement):
    dist = re.split(r'[\[<>=!~ ;]', requirement, maxsplit=1)[0].strip().lower()
    return ALIAS.get(dist, dist.replace('-', '_'))


PYPROJECT = tomllib.loads((ROOT / 'pyproject.toml').read_text(encoding='utf-8'))
BASE = {_import_name(r) for r in PYPROJECT['project']['dependencies']}
EXTRAS = {k: {_import_name(r) for r in v if not r.startswith('upxo')}
          for k, v in PYPROJECT['project']['optional-dependencies'].items()}
OPTIONAL = set().union(*EXTRAS.values())


class _Statements(ast.NodeVisitor):
    """Every import statement, with whether a try/except ImportError guards it."""

    def __init__(self):
        self.items, self.guard = [], 0

    def visit_Try(self, node):
        catches = any(h.type is None or any(isinstance(n, ast.Name) and n.id in ('ImportError', 'ModuleNotFoundError', 'Exception')
                                          for n in ast.walk(h.type)) for h in node.handlers)
        for part in (node.body, *[h.body for h in node.handlers]):
            self.guard += catches
            for n in part:
                self.visit(n)
            self.guard -= catches
        for part in (node.orelse, node.finalbody):
            for n in part:
                self.visit(n)

    visit_TryStar = visit_Try

    def visit_Import(self, node):
        self.items.append(('import', node.lineno, [a.name for a in node.names], None, 0, self.guard > 0))

    def visit_ImportFrom(self, node):
        self.items.append(('from', node.lineno, [a.name for a in node.names], node.module, node.level, self.guard > 0))

    def visit_Call(self, node):
        f = node.func
        name = f.attr if isinstance(f, ast.Attribute) else getattr(f, 'id', '')
        if name in ('import_module', '__import__') and node.args and isinstance(node.args[0], ast.Constant) \
                and isinstance(node.args[0].value, str):
            self.items.append(('call', node.lineno, [node.args[0].value], None, 0, self.guard > 0))
        self.generic_visit(node)


def _statements(source):
    visitor = _Statements()
    visitor.visit(ast.parse(source))
    return visitor.items


def _absolute(module_name, is_package, module, level):
    if not level:
        return module or ''
    parts = (module_name if is_package else module_name.rpartition('.')[0]).split('.')
    return '.'.join(parts[:len(parts) - (level - 1)] + ([module] if module else []))


def _resolve(where, items, module_name=None, is_package=False, notebook=False):
    """Static findings and the (where, line, module, name) pairs to check dynamically."""
    findings, names = [], []
    for kind, line, targets, module, level, guarded in items:
        if kind == 'call':
            if targets[0].split('.')[0] == 'upxo' and targets[0] not in MODULES and targets[0] not in PACKAGES:
                findings.append(f'{where}:{line}: import_module({targets[0]!r}) does not exist')
            continue
        pairs = [(t, None) for t in targets] if kind == 'import' else \
            [(_absolute(module_name, is_package, module, level) if module_name else (module or ''), targets)]
        for mod, from_names in pairs:
            top = mod.split('.')[0]
            if top == 'upxo':
                if mod not in MODULES and mod not in PACKAGES:
                    findings.append(f'{where}:{line}: module {mod} does not exist')
                    continue
                for n in from_names or []:
                    if n != '*' and f'{mod}.{n}' not in MODULES and f'{mod}.{n}' not in PACKAGES:
                        names.append((where, line, mod, n))
            elif top in STD or not top or top in BASE or top in OPTIONAL or top in TRANSITIVE:
                continue
            elif not guarded and not notebook:
                findings.append(f'{where}:{line}: imports {mod}, which no installation level declares '
                                f'(declare it, or guard the import with try/except ImportError)')
    return findings, names


def _scan_sources():
    findings, names = [], []
    for name in CHECKED:
        text = (ROOT / MODULES[name]).read_text(encoding='utf-8-sig')
        f, n = _resolve(MODULES[name], _statements(text), name, MODULES[name].endswith('__init__.py'))
        findings += f
        names += n
    return findings, names


def _scan_notebooks():
    findings, names = [], []
    for f in FILES:
        if not f.endswith('.ipynb') or not f.startswith('src/upxo/'):
            continue
        try:
            cells = json.loads((ROOT / f).read_text(encoding='utf-8')).get('cells', [])
        except ValueError:
            continue
        for i, cell in enumerate(cells):
            if cell.get('cell_type') != 'code':
                continue
            code = '\n'.join(l for l in ''.join(cell['source']).split('\n') if not l.lstrip().startswith(('%', '!', '?')))
            try:
                items = _statements(code)
            except SyntaxError:
                continue
            fi, na = _resolve(f'{f} [cell {i}]', items, notebook=True)
            findings += fi
            names += na
    return findings, names


WORKER = r'''
import contextlib, importlib, io, json, sys, warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
payload = json.load(sys.stdin)
result = {'imports': {}, 'names': []}
for m in payload['modules']:
    buffer, error, missing = io.StringIO(), None, None
    try:
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
            importlib.import_module(m)
    except BaseException as e:
        error = type(e).__name__ + ': ' + str(e)[:300]
        if isinstance(e, ModuleNotFoundError):
            missing = (e.name or '').split('.')[0]
    result['imports'][m] = {'error': error, 'missing': missing, 'output': buffer.getvalue()[:300]}
for where, line, mod, name in payload['names']:
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            module = importlib.import_module(mod)
    except BaseException:
        continue
    if not hasattr(module, name):
        result['names'].append(f'{where}:{line}: from {mod} import {name} -- no such name')
sys.stdout.write('@@RESULT@@' + json.dumps(result))
'''


def _group(module):
    parts = module.split('.')
    return parts[1] if len(parts) > 2 else 'upxo'


@pytest.fixture(scope='session')
def sweep():
    """Import every tracked module in fresh interpreters, one per subpackage."""
    source_findings, source_names = _scan_sources()
    nb_findings, nb_names = _scan_notebooks()
    groups = {}
    for m in CHECKED:
        groups.setdefault(_group(m), []).append(m)
    names_by_group = {}
    for where, line, mod, name in source_names + nb_names:
        key = next((g for g, mods in groups.items() if any(MODULES.get(m) == where for m in mods)), None) or 'notebooks'
        names_by_group.setdefault(key, []).append([where, line, mod, name])
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(ROOT / 'src'), os.environ.get('PYTHONPATH', '')]).rstrip(os.pathsep))

    def run(key):
        payload = {'modules': sorted(groups.get(key, [])), 'names': names_by_group.get(key, [])}
        done = subprocess.run([sys.executable, '-c', WORKER], input=json.dumps(payload), capture_output=True, text=True,
                              env=env, timeout=900)
        text = done.stdout.split('@@RESULT@@')[-1] if '@@RESULT@@' in done.stdout else ''
        if not text:
            raise RuntimeError(f'worker for {key} produced no result: {done.stderr[-500:]}')
        return json.loads(text)

    keys = sorted(set(groups) | set(names_by_group))
    with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 2)) as pool:
        results = list(pool.map(run, keys))
    imports, names = {}, []
    for r in results:
        imports.update(r['imports'])
        names += r['names']
    return dict(imports=imports, names=names, static=source_findings + nb_findings)


@pytest.mark.parametrize('module', CHECKED)
def test_module_imports(sweep, module):
    entry = sweep['imports'][module]
    if entry['error'] is None:
        return
    assert entry['missing'] in OPTIONAL, f"{module} fails to import: {entry['error']}"


@pytest.mark.parametrize('module', CHECKED)
def test_module_import_is_silent(sweep, module):
    assert not sweep['imports'][module]['output'].strip(), \
        f"importing {module} prints: {sweep['imports'][module]['output'][:120]!r}"


def test_every_import_statement_resolves(sweep):
    """Static (module exists, package declared) and dynamic (name exists) results."""
    problems = sweep['static'] + sweep['names']
    assert not problems, '\n' + '\n'.join(sorted(set(problems)))


def _setup_py():
    tree = ast.parse((ROOT / 'setup.py').read_text(encoding='utf-8'))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, 'id', '') == 'setup':
            kw = {k.arg: ast.literal_eval(k.value) for k in node.keywords if k.arg in ('install_requires', 'extras_require')}
            return kw
    return {}


def _norm(requirement):
    return re.sub(r'\s+', '', requirement).lower()


def test_dependency_lists_agree():
    py = {_norm(r) for r in PYPROJECT['project']['dependencies']}
    setup = {_norm(r) for r in _setup_py()['install_requires']}
    lines = (ROOT / 'requirements.txt').read_text(encoding='utf-8').splitlines()
    stop = next(i for i, l in enumerate(lines) if l.startswith('# --- Optional'))
    txt = {_norm(l) for l in lines[:stop] if l.strip() and not l.lstrip().startswith('#')}
    assert py == setup, f'pyproject.toml vs setup.py: {sorted(py ^ setup)}'
    assert py == txt, f'pyproject.toml vs requirements.txt: {sorted(py ^ txt)}'
    extras_py = {k: sorted(_norm(r) for r in v) for k, v in PYPROJECT['project']['optional-dependencies'].items()}
    extras_setup = {k: sorted(_norm(r) for r in v) for k, v in _setup_py()['extras_require'].items()}
    assert extras_py == extras_setup, f'extras differ: {extras_py} vs {extras_setup}'


def test_entry_points_resolve():
    for section in ('scripts', 'gui-scripts'):
        for name, target in PYPROJECT['project'].get(section, {}).items():
            assert target.split(':')[0] in MODULES, f'entry point {name} -> {target} does not exist'


def test_sphinx_listings_resolve():
    stale = []
    for f in FILES:
        if f.startswith('docs/') and f.endswith('.rst'):
            for i, line in enumerate((ROOT / f).read_text(encoding='utf-8', errors='ignore').splitlines(), 1):
                m = re.match(r'^\s*(?:\.\. (?:automodule|autoclass|autofunction)::\s*)?(upxo(?:\.\w+)+)\s*$', line)
                if m and m.group(1) not in MODULES and m.group(1) not in PACKAGES:
                    stale.append(f'{f}:{i}: {m.group(1)}')
    assert not stale, '\n' + '\n'.join(stale)
