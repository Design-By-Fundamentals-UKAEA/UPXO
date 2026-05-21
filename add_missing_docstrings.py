"""
Batch-add minimal one-liner docstrings to all undocumented functions/methods
in specified Python files. Only adds to functions that have NO docstring at all.
"""
import ast
import re
import sys

def get_missing(src):
    tree = ast.parse(src)
    missing = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            has_doc = (node.body and
                       isinstance(node.body[0], ast.Expr) and
                       isinstance(node.body[0].value, ast.Constant))
            if not has_doc:
                missing.add(node.lineno)
    return missing

def infer_docstring(name):
    """Generate a minimal one-liner based on the function name."""
    n = name.lower()
    if name.startswith('__') and name.endswith('__'):
        dunder_map = {
            '__init__': 'Initialise the instance.',
            '__repr__': 'Return a string representation of this instance.',
            '__str__': 'Return the string form of this instance.',
            '__len__': 'Return the number of items in this instance.',
            '__iter__': 'Return an iterator over this instance.',
            '__next__': 'Return the next item from this iterator.',
            '__eq__': 'Check equality with another object.',
            '__ne__': 'Check inequality with another object.',
            '__lt__': 'Less-than comparison.',
            '__le__': 'Less-than-or-equal comparison.',
            '__gt__': 'Greater-than comparison.',
            '__ge__': 'Greater-than-or-equal comparison.',
            '__contains__': 'Test membership.',
            '__getitem__': 'Return item at the given index or key.',
            '__setitem__': 'Set item at the given index or key.',
            '__delitem__': 'Delete item at the given index or key.',
            '__call__': 'Call this instance as a function.',
            '__add__': 'Add another object to this instance.',
            '__sub__': 'Subtract another object from this instance.',
            '__mul__': 'Multiply this instance by a scalar.',
            '__truediv__': 'Divide this instance.',
            '__hash__': 'Return a hash of this instance.',
            '__copy__': 'Return a shallow copy.',
            '__deepcopy__': 'Return a deep copy.',
            '__post_init__': 'Post-initialisation hook for dataclass fields.',
            '__att__': 'Return a string listing of all attributes.',
            '__slots__': 'Slot definition.',
        }
        return dunder_map.get(name, f'{name} special method.')
    # Property-like names
    if n.startswith('get_') or n.startswith('calc_') or n.startswith('compute_'):
        return f'Return the {name[4:].replace("_", " ")}.'
    if n.startswith('set_') or n.startswith('update_'):
        return f'Set or update {name[4:].replace("_", " ")}.'
    if n.startswith('make_') or n.startswith('build_') or n.startswith('create_'):
        return f'Build and return {name[5:].replace("_", " ")}.'
    if n.startswith('find_') or n.startswith('search_'):
        return f'Find {name[5:].replace("_", " ")}.'
    if n.startswith('check_') or n.startswith('validate_') or n.startswith('is_') or n.startswith('has_'):
        return f'Check or validate {name.replace("_", " ")}.'
    if n.startswith('plot_') or n.startswith('show_') or n.startswith('vis_') or n.startswith('draw_'):
        return f'Visualise {name[5:].replace("_", " ")} using Matplotlib or PyVista.'
    if n.startswith('from_'):
        return f'Construct this instance from {name[5:].replace("_", " ")}.'
    if n.startswith('to_') or n.startswith('export_') or n.startswith('write_'):
        return f'Export or convert to {name[3:].replace("_", " ")}.'
    if n.startswith('load_') or n.startswith('read_') or n.startswith('import_'):
        return f'Load or import {name[5:].replace("_", " ")}.'
    if n.startswith('reset_') or n.startswith('clear_') or n.startswith('delete_'):
        return f'Reset or clear {name[6:].replace("_", " ")}.'
    if n.startswith('add_') or n.startswith('insert_') or n.startswith('append_'):
        return f'Add or insert {name[4:].replace("_", " ")}.'
    if n.startswith('remove_') or n.startswith('pop_') or n.startswith('delete_'):
        return f'Remove {name[7:].replace("_", " ")}.'
    if n.startswith('apply_') or n.startswith('run_') or n.startswith('execute_'):
        return f'Apply or execute {name[6:].replace("_", " ")}.'
    if n.startswith('print_') or n.startswith('display_'):
        return f'Print or display {name[6:].replace("_", " ")}.'
    # Default
    words = name.replace('_', ' ')
    return f'{words.capitalize()}.'

def add_docstrings(fpath):
    with open(fpath, encoding='utf-8') as f:
        lines = f.readlines()
    src = ''.join(lines)
    try:
        missing_lines = get_missing(src)
    except SyntaxError as e:
        print(f'  SKIP (syntax error): {e}')
        return 0

    if not missing_lines:
        return 0

    # Parse again to get name→lineno mapping
    tree = ast.parse(src)
    line_to_name = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.lineno in missing_lines:
                line_to_name[node.lineno] = node.name

    # Find the last line of each def signature (the line ending with ':')
    # We'll insert the docstring after the signature closes
    # Strategy: scan forward from the def line until we find a line ending ':' (ignoring comments)
    result = []
    i = 0
    fixes = 0
    while i < len(lines):
        line = lines[i]
        lineno = i + 1  # 1-based

        if lineno in line_to_name:
            func_name = line_to_name[lineno]
            # Find the end of the signature (line ending with ':' not inside parens)
            j = i
            paren_depth = 0
            sig_end = i
            while j < len(lines):
                for ch in lines[j]:
                    if ch == '(':
                        paren_depth += 1
                    elif ch == ')':
                        paren_depth -= 1
                stripped = lines[j].rstrip()
                if paren_depth == 0 and stripped.endswith(':'):
                    sig_end = j
                    break
                j += 1
            # Add all signature lines to result
            for k in range(i, sig_end + 1):
                result.append(lines[k])
            # Determine indentation from the body (first non-blank line after sig)
            body_line_idx = sig_end + 1
            while body_line_idx < len(lines) and lines[body_line_idx].strip() == '':
                body_line_idx += 1
            if body_line_idx < len(lines):
                body_indent = len(lines[body_line_idx]) - len(lines[body_line_idx].lstrip())
            else:
                # Fallback: def indent + 4
                def_indent = len(lines[i]) - len(lines[i].lstrip())
                body_indent = def_indent + 4
            indent_str = ' ' * body_indent
            doc = infer_docstring(func_name)
            result.append(f'{indent_str}"""{doc}"""\n')
            fixes += 1
            i = sig_end + 1
            continue

        result.append(line)
        i += 1

    with open(fpath, 'w', encoding='utf-8') as f:
        f.writelines(result)
    return fixes

files = sys.argv[1:]
total = 0
for fpath in files:
    fname = fpath.split('\\')[-1].split('/')[-1]
    n = add_docstrings(fpath)
    print(f'{fname}: added {n} docstrings')
    total += n
print(f'Total added: {total}')
