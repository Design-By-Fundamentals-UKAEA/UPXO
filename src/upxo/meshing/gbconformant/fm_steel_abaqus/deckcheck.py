"""Checks of a written deck, made on the files and not on the arrays they came from.

``check_deck(folder)`` reads the node, element, set, material, section and step files back and
reports, check by check, whether the deck is consistent:

* nodes and elements are numbered 1..N without gaps, every connectivity entry names a node,
  and every node is used;
* every element (C3D4 tetrahedron or C3D8 hexahedron) has a positive volume;
* the element sets that sections point at cover each element exactly once;
* every section names an existing element set and material, and every material is used;
* every element set and node set is non-empty and names only existing elements or nodes;
* no set name is defined twice (ignoring case, as Abaqus does) and every name follows the naming rules;
* every surface names existing elements and faces those elements have;
* a step's element output set exists, named through the instance;
* every set the step applies a boundary condition to exists;
* every file the master file includes exists.

A deck that passes is consistent as a file; it has not been run in Abaqus.
"""
from itertools import islice
from pathlib import Path
import re
import numpy as np

from .elements import element_volumes

_BLOCK = 200_000


def _ids(lines):
    """Parsed as float and rounded to int64, not read directly as int64: numpy raises ValueError
    (not just a DeprecationWarning) from 2.4 onward for an unmatched/float-formatted value (e.g.
    "1.0") in an integer-dtype np.fromstring parse -- same failure mode as _read_table above and
    the CTF BC-column bug in upxo.interfaces.defdap.ebsd_reader."""
    text = ','.join(line.replace(' ', '').strip(',') for line in lines if line.strip())
    if not text:
        return np.zeros(0, dtype=np.int64)
    return np.round(np.fromstring(text, dtype=np.float64, sep=',')).astype(np.int64)


def read_sets(path, keyword):
    """``{name: ids}`` of the ``*Elset`` or ``*Nset`` blocks of an include file."""
    sets, name, buffer = {}, None, []
    pattern = re.compile(rf'^\*{keyword},.*?{keyword.lower()}=([^,\s]+)', re.IGNORECASE)

    def close():
        if name is not None:
            sets[name] = _ids(buffer)

    with Path(path).open(encoding='ascii') as handle:
        for line in handle:
            line = line.rstrip('\n')
            match = pattern.match(line)
            if match:
                close()
                name, buffer = match.group(1), []
            elif line.startswith('*'):
                close()
                name, buffer = None, []
            elif name is not None and line.strip() and not line.startswith('**'):
                buffer.append(line)
    close()
    return sets


def set_names(path, keyword):
    """Every ``*Elset``/``*Nset``/``*Surface`` name of an include file, in file order (repeats kept)."""
    pattern = re.compile(rf'^\*{keyword},.*?\b{"name" if keyword == "Surface" else keyword.lower()}=([^,\s]+)',
                         re.IGNORECASE)
    return [m.group(1) for line in Path(path).read_text(encoding='ascii').splitlines()
            if (m := pattern.match(line))]


def read_surfaces(path):
    """``{name: (element ids, face numbers)}`` of the ``*Surface, type=ELEMENT`` blocks of an include file."""
    out, name, rows = {}, None, []

    def close():
        if name is not None:
            out[name] = (np.array([r[0] for r in rows], dtype=np.int64), np.array([r[1] for r in rows], dtype=np.int64))

    for line in Path(path).read_text(encoding='ascii').splitlines():
        match = re.match(r'^\*Surface,.*?\bname=([^,\s]+)', line, re.IGNORECASE)
        if match:
            close()
            name, rows = match.group(1), []
        elif line.startswith('*'):
            close()
            name, rows = None, []
        elif name is not None and line.strip() and not line.startswith('**'):
            element, face = (v.strip() for v in line.split(',')[:2])
            rows.append((int(element), int(face.upper().lstrip('S'))))
    close()
    return out


def _read_table(path, columns, dtype=float):
    """The numbers under the keyword line of a node or element file, read a block of lines at a time.

    Always parsed as float and cast to *dtype* afterwards, rather than handing an integer dtype
    straight to np.loadtxt: numpy raises ValueError (not just a DeprecationWarning) from 2.4 onward
    if a value happens to be written with a decimal point (e.g. "1.0") -- the same failure mode as
    the CTF BC-column bug in upxo.interfaces.defdap.ebsd_reader. Our own exporter always writes
    clean integers, but this keeps deck-checking robust to any writer that doesn't.
    """
    parts = []
    with Path(path).open(encoding='ascii') as handle:
        handle.readline()
        while True:
            lines = list(islice(handle, _BLOCK))
            if not lines:
                break
            rows = [line for line in lines if line.strip() and not line.startswith('*')]
            if rows:
                parts.append(np.loadtxt(rows, delimiter=',', ndmin=2, dtype=float))
    out = np.concatenate(parts) if parts else np.zeros((0, columns), dtype=float)
    return out.astype(dtype) if dtype is not float else out


def _check(name, ok, detail=''):
    return {'name': name, 'ok': bool(ok), 'detail': detail}


def check_deck(folder):
    """Read the deck in ``folder`` back and return ``{'ok', 'checks': [{name, ok, detail}], 'counts'}``."""
    folder = Path(folder)
    checks = []

    def need(name):
        path = folder / name
        if not path.is_file():
            checks.append(_check(f'{name} present', False, 'the file is missing'))
            return None
        return path

    master = need('model_master.inp')
    included = re.findall(r'\*INCLUDE,\s*INPUT=(\S+)', master.read_text(encoding='ascii', errors='replace')) \
        if master else []
    missing = [name for name in included if not (folder / name).is_file()]
    checks.append(_check('included files exist', not missing, ', '.join(missing) if missing else f'{len(included)} files'))

    nodes_path, elements_path = need('01_nodes.inp'), need('02_elements.inp')
    if nodes_path is None or elements_path is None:
        return _finish(checks, {})
    nodes = _read_table(nodes_path, 4)
    elements = _read_table(elements_path, 5, dtype=np.int32 if len(nodes) < 2 ** 31 - 1 else np.int64)
    n_nodes, n_elements = len(nodes), len(elements)
    counts = {'nodes': n_nodes, 'elements': n_elements}

    node_ids = nodes[:, 0].astype(np.int64) if n_nodes else np.zeros(0, dtype=np.int64)
    element_ids = elements[:, 0].astype(np.int64) if n_elements else np.zeros(0, dtype=np.int64)
    checks.append(_check('nodes numbered 1..N', n_nodes > 0 and np.array_equal(node_ids, np.arange(1, n_nodes + 1)),
                         f'{n_nodes} nodes'))
    checks.append(_check('elements numbered 1..M', n_elements > 0 and np.array_equal(element_ids, np.arange(1, n_elements + 1)),
                         f'{n_elements} elements'))
    connectivity = elements[:, 1:] if n_elements else np.zeros((0, 4), dtype=np.int64)
    noun = 'hexahedron' if connectivity.shape[1] == 8 else 'tetrahedron'
    counts['element_type'] = 'C3D8' if connectivity.shape[1] == 8 else 'C3D4'
    in_range = bool(connectivity.size) and connectivity.min() >= 1 and connectivity.max() <= n_nodes
    checks.append(_check('connectivity names existing nodes', in_range,
                         '' if in_range else 'an element refers to a node that does not exist'))
    if in_range:
        seen = np.zeros(n_nodes + 1, dtype=bool)
        coordinates = nodes[:, 1:]
        bad, smallest, largest = 0, np.inf, -np.inf
        for start in range(0, n_elements, _BLOCK):
            block = connectivity[start:start + _BLOCK]
            seen[block.ravel()] = True
            volume = element_volumes(coordinates, block - 1)
            bad += int((~(volume > 0)).sum())
            smallest, largest = min(smallest, float(volume.min())), max(largest, float(volume.max()))
        unused = n_nodes - int(seen[1:].sum())
        checks.append(_check('every node is used', unused == 0, f'{unused} unused' if unused else ''))
        counts['smallest_volume'] = smallest
        counts['largest_volume'] = largest
        checks.append(_check(f'every {noun} has a positive volume', bad == 0,
                             f'{bad} not positive' if bad else f'smallest {smallest:.4g}'))

    elsets = {}
    for path in sorted(folder.glob('03*.inp')):
        elsets.update(read_sets(path, 'Elset'))
    nsets = {}
    for path in sorted(folder.glob('04*.inp')):
        nsets.update(read_sets(path, 'Nset'))
    counts.update(element_sets=len(elsets), node_sets=len(nsets))

    sections = []
    if (folder / '06_sections.inp').is_file():
        sections = re.findall(r'\*Solid Section,\s*elset=(\S+?),\s*material=(\S+)',
                              (folder / '06_sections.inp').read_text(encoding='ascii'))
    materials = set()
    if (folder / '05_materials.inp').is_file():
        materials = set(re.findall(r'^\*Material,\s*name=(\S+)', (folder / '05_materials.inp').read_text(encoding='ascii'),
                                   re.MULTILINE))
    counts.update(sections=len(sections), materials=len(materials))
    missing_sets = sorted({e for e, _m in sections if e not in elsets})
    missing_materials = sorted({m for _e, m in sections if m not in materials})
    checks.append(_check('sections name existing element sets', bool(sections) and not missing_sets,
                         ', '.join(missing_sets[:5]) if missing_sets else f'{len(sections)} sections'))
    checks.append(_check('sections name existing materials', not missing_materials,
                         ', '.join(missing_materials[:5]) if missing_materials else ''))
    unused = sorted(materials - {m for _e, m in sections})
    checks.append(_check('every material is used', not unused, ', '.join(unused[:5]) if unused else ''))

    covered = [elsets[e] for e, _m in sections if e in elsets]
    joined = np.concatenate(covered) if covered else np.zeros(0, dtype=np.int64)
    if len(joined) and n_elements:
        times = (np.bincount(joined, minlength=n_elements + 1)[1:n_elements + 1]
                 if joined.min() >= 1 and joined.max() <= n_elements else None)
        if times is None:
            checks.append(_check('each element is in exactly one section set', False, 'a set names a missing element'))
        else:
            none, many = int((times == 0).sum()), int((times > 1).sum())
            checks.append(_check('each element is in exactly one section set', none == 0 and many == 0,
                                 f'{none} in none, {many} in more than one' if none or many else ''))
    else:
        checks.append(_check('each element is in exactly one section set', False, 'no section sets to check'))

    from .setnames import name_problem

    for keyword, pattern in (('Elset', '03*.inp'), ('Nset', '04*.inp'), ('Surface', '03*.inp')):
        names = [n for path in sorted(folder.glob(pattern)) for n in set_names(path, keyword)]
        lowered = [n.lower() for n in names]
        twice = sorted({n for n in names if lowered.count(n.lower()) > 1})
        checks.append(_check(f'no {keyword.lower()} name defined twice', not twice,
                             ', '.join(twice[:5]) if twice else f'{len(names)} names'))
        bad = sorted({n for n in names if name_problem(n)})
        checks.append(_check(f'{keyword.lower()} names follow the naming rules', not bad, ', '.join(bad[:5])))
    surfaces = {}
    for path in sorted(folder.glob('03*.inp')):
        surfaces.update(read_surfaces(path))
    if surfaces:
        width = connectivity.shape[1] if n_elements else 4
        n_faces = 6 if width == 8 else 4
        wrong = sorted(name for name, (ids, numbers) in surfaces.items()
                       if not len(ids) or ids.min() < 1 or ids.max() > n_elements
                       or numbers.min() < 1 or numbers.max() > n_faces)
        checks.append(_check('surfaces name existing elements and faces', not wrong,
                             ', '.join(wrong[:5]) if wrong else f'{len(surfaces)} surfaces'))
        counts['surfaces'] = len(surfaces)
    empty = sorted([n for n, ids in elsets.items() if not len(ids)] + [n for n, ids in nsets.items() if not len(ids)])
    checks.append(_check('no empty sets', not empty, ', '.join(empty[:5]) if empty else ''))
    out_of_range = sorted([n for n, ids in elsets.items() if len(ids) and (ids.min() < 1 or ids.max() > n_elements)]
                          + [n for n, ids in nsets.items() if len(ids) and (ids.min() < 1 or ids.max() > n_nodes)])
    checks.append(_check('sets name existing elements and nodes', not out_of_range,
                         ', '.join(out_of_range[:5]) if out_of_range else ''))

    steps = folder / '08_steps_output.inp'
    if steps.is_file():
        used = _boundary_sets(steps.read_text(encoding='ascii').splitlines())
        known = {n.lower() for n in nsets}
        absent = sorted(n for n in used if n.lower() not in known)
        checks.append(_check('step boundary conditions use existing node sets', not absent,
                             ', '.join(absent) if absent else (f'{len(used)} sets' if used else 'no step written')))
        text = steps.read_text(encoding='ascii')
        defined = {m.lower() for m in re.findall(r'^\*Amplitude,[^\n]*?name=([^,\s]+)', text, re.MULTILINE | re.IGNORECASE)}
        wanted = {m for m in re.findall(r'^\*Boundary,[^\n]*?amplitude=([^,\s]+)', text, re.MULTILINE | re.IGNORECASE)}
        missing = sorted(m for m in wanted if m.lower() not in defined)
        counts['steps'] = len(re.findall(r'^\*Step,', text, re.MULTILINE))
        outputs = re.findall(r'^\*Element Output,[^\n]*?elset=([^,\s]+)', text, re.MULTILINE | re.IGNORECASE)
        known_sets = {n.lower() for n in elsets}
        unknown = sorted({o for o in outputs if '.' not in o or o.split('.', 1)[1].lower() not in known_sets})
        checks.append(_check('element output sets exist (named through the instance)', not unknown,
                             ', '.join(unknown) if unknown else (f'{len(outputs)} used' if outputs else 'whole model')))
        checks.append(_check('boundary conditions use defined amplitudes', not missing,
                             ', '.join(missing) if missing else (f'{len(wanted)} used' if wanted else 'no amplitude used')))
    return _finish(checks, counts)


def _boundary_sets(lines):
    """Names a step's ``*Boundary`` blocks point at (the first field of each data line that is not a node number)."""
    names, inside = set(), False
    for line in lines:
        if line.startswith('*'):
            inside = line.lower().startswith('*boundary')
        elif inside and line.strip() and not line.startswith('**'):
            first = line.split(',')[0].strip()
            if first and not first.isdigit():
                names.add(first)
    return names


def _finish(checks, counts):
    return {'ok': all(c['ok'] for c in checks), 'checks': checks, 'counts': counts,
            'failed': [c['name'] for c in checks if not c['ok']]}


def format_check(result):
    """The check as short text lines, one per check."""
    lines = []
    for item in result['checks']:
        lines.append(f"[{'ok' if item['ok'] else 'FAILED'}] {item['name']}" + (f" - {item['detail']}" if item['detail'] else ''))
    return '\n'.join(lines)
