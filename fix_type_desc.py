"""
Batch-fix TYPE/DESCRIPTION placeholders in UPXO docstrings.
Rules:
  - 'param : TYPE, optional' where param name gives a hint → replace TYPE
  - Generic fallback: keep TYPE as 'object' when we can't determine
  - DESCRIPTION lines → keep as short placeholder text based on param name
"""
import re
import sys

# Type heuristics based on parameter name patterns
TYPE_HINTS = {
    # booleans
    'saa': 'bool', 'throw': 'bool', 'update': 'bool', 'closed': 'bool',
    'ordered': 'bool', 'verbose': 'bool', 'print_': 'bool', 'debug': 'bool',
    'make_mp': 'bool', 'make_emp': 'bool', 'show': 'bool', 'display': 'bool',
    'return_new': 'bool', 'inplace': 'bool', 'normalize': 'bool',
    # integers
    'n': 'int', 'npts': 'int', 'niterations': 'int', 'count': 'int',
    'ndim': 'int', 'dim': 'int', 'iterations': 'int', 'k': 'int',
    # floats
    'tol': 'float', 'tol_ang': 'float', 'tdist': 'float', 'eps': 'float',
    'length': 'float', 'angle': 'float', 'ang': 'float', 'gradient': 'float',
    'intercept': 'float', 'ratio': 'float', 'zloc': 'float', 'dpi': 'int',
    'r': 'float', 'dist': 'float', 'threshold': 'float', 'weight': 'float',
    # strings
    'method': 'str', 'plane': 'str', 'otype': 'str', 'lean': 'str',
    'return_type': 'str', 'edge_type': 'str', 'distribution': 'str',
    'return_format': 'str', 'mode': 'str', 'spacing': 'str', 'label': 'str',
    # lists/arrays
    'coeff': 'list', 'weights': 'list', 'constraints': 'list',
    'coords': 'list', 'points': 'list', 'edges': 'list', 'clist': 'list',
    'vector': 'list', 'xlist': 'list', 'ylist': 'list', 'cpairs': 'list',
}

def infer_type(param_name):
    name = param_name.lower().strip()
    for key, val in TYPE_HINTS.items():
        if name == key or name.startswith(key + '_') or name.endswith('_' + key):
            return val
    if 'pnt' in name or 'point' in name:
        return 'Point2d'
    if 'edge' in name:
        return 'edge2d'
    if 'line' in name:
        return 'Sline2d'
    if name in ('x', 'y', 'z', 'x0', 'y0', 'x1', 'y1', 'xa', 'ya', 'xb', 'yb'):
        return 'float'
    if name.startswith('n_') or name.endswith('_n'):
        return 'int'
    if name.endswith('s') and name[:-1] in TYPE_HINTS:
        return 'list'
    return 'object'

def fix_type_desc(content, filename):
    lines = content.split('\n')
    result = []
    i = 0
    fixes = 0
    while i < len(lines):
        line = lines[i]
        # Match "        param : TYPE" or "        param : TYPE, optional"
        m = re.match(r'^(\s+)(\w+)\s*:\s*TYPE(,?\s*optional)?\s*$', line)
        if m:
            indent, param, opt = m.group(1), m.group(2), m.group(3) or ''
            inferred = infer_type(param)
            result.append(f'{indent}{param} : {inferred}{opt}')
            fixes += 1
        # Match "            DESCRIPTION." (description line after TYPE)
        elif re.match(r'^\s+DESCRIPTION\.?\s*$', line):
            # Get context from previous param line
            prev_param = None
            for prev in reversed(result):
                m2 = re.match(r'^\s+(\w+)\s*:', prev)
                if m2:
                    prev_param = m2.group(1)
                    break
            if prev_param:
                result.append(line.replace('DESCRIPTION.', f'{prev_param} value.').replace('DESCRIPTION', f'{prev_param}'))
            else:
                result.append(line)
            fixes += 1
        else:
            result.append(line)
        i += 1
    print(f'{filename}: fixed {fixes} TYPE/DESCRIPTION placeholders')
    return '\n'.join(result)

files = sys.argv[1:]
for fpath in files:
    with open(fpath, encoding='utf-8') as f:
        content = f.read()
    new_content = fix_type_desc(content, fpath.split('\\')[-1])
    with open(fpath, 'w', encoding='utf-8') as f:
        f.write(new_content)
