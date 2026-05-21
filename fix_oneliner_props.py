"""Fix single-line property defs that have no docstring in polycrystal.py."""
import re, sys

def infer_doc(name):
    n = name
    if n.startswith('get_'):
        attr = n[4:].replace('_', ' ')
        return f'Return ``{n[4:]}``.'
    return f'{name.replace("_", " ").capitalize()}.'

def fix(fpath):
    with open(fpath, encoding='utf-8') as f:
        lines = f.readlines()
    result = []
    fixes = 0
    for line in lines:
        # Match: [spaces]def name(...): [body]  (single-line def with body on same line)
        m = re.match(r'^(\s+)(def (\w+)\([^)]*\):\s*)(.+)$', line)
        if m:
            indent, def_part, name, body = m.group(1), m.group(2), m.group(3), m.group(4).strip()
            body_indent = indent + '    '
            doc = infer_doc(name)
            result.append(f'{indent}{def_part}\n')
            result.append(f'{body_indent}"""{doc}"""\n')
            result.append(f'{body_indent}{body}\n')
            fixes += 1
        else:
            result.append(line)
    with open(fpath, 'w', encoding='utf-8') as f:
        f.writelines(result)
    print(f'{fpath.split(chr(92))[-1]}: fixed {fixes} one-liner defs')

for fpath in sys.argv[1:]:
    fix(fpath)
