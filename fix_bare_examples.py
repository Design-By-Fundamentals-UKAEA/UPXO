"""
Add >>> prefix to bare code lines in Example/Examples sections of docstrings.
Fixes: singular "Example" -> "Examples", normalises dashes to "--------".

Usage: python fix_bare_examples.py file1.py file2.py ...
"""
import re
import ast
import sys


def fix_example_sections(src):
    lines = src.splitlines(keepends=True)
    result = []
    i = 0
    changes = 0

    while i < len(lines):
        line = lines[i]
        stripped = line.rstrip()

        m = re.match(r'^(\s+)(Examples?)\s*$', stripped)
        if m:
            indent = m.group(1)
            indent_len = len(indent)

            # Find next non-blank line – should be the dashes underline
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1

            if j < len(lines) and re.match(r'^\s+-+\s*$', lines[j]):
                # Emit normalised header
                result.append(indent + 'Examples\n')
                result.append(indent + '--------\n')
                i = j + 1

                # Preserve blank lines between header and first code line
                while i < len(lines) and not lines[i].strip():
                    result.append(lines[i])
                    i += 1

                # Process code lines
                code_changed = False
                while i < len(lines):
                    cl = lines[i]
                    cl_stripped = cl.strip()

                    # Blank line – preserve and continue (may be between examples)
                    if not cl_stripped:
                        result.append(cl)
                        i += 1
                        continue

                    cl_indent = len(cl) - len(cl.lstrip())

                    # Stop: dedented past section header
                    if cl_indent < indent_len:
                        break

                    # Stop: closing triple-quote at or below section indent
                    if cl_stripped in ('"""', "'''"):
                        break

                    # Stop: a new NumPy section header (same indent + next line is dashes)
                    if (cl_indent == indent_len and
                            i + 1 < len(lines) and
                            re.match(r'^\s+-+\s*$', lines[i + 1])):
                        break

                    # Emit with >>> prefix if bare
                    if cl_stripped.startswith('>>>') or cl_stripped.startswith('.. ') or cl_stripped == '..':
                        result.append(indent + cl.lstrip())
                    else:
                        result.append(indent + '>>> ' + cl.lstrip())
                        code_changed = True

                    i += 1

                if code_changed:
                    changes += 1
                continue  # outer while

        result.append(line)
        i += 1

    return ''.join(result), changes


for fpath in sys.argv[1:]:
    with open(fpath, encoding='utf-8') as f:
        src = f.read()
    new_src, changes = fix_example_sections(src)
    with open(fpath, 'w', encoding='utf-8') as f:
        f.write(new_src)
    fname = fpath.replace('\\', '/').split('/')[-1]
    print(f'{fname}: {changes} example sections fixed')
    try:
        ast.parse(new_src)
        print('  Parse OK')
    except SyntaxError as e:
        print(f'  Parse ERROR: {e}')
