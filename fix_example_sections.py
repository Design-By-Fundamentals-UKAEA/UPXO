"""
Fix bare-code Example sections in Python docstrings.
Wraps them in '.. code-block:: python' directives.
"""
import re, ast, sys

def fix_example_section(src):
    lines = src.splitlines(keepends=True)
    result = []
    i = 0
    changes = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.rstrip()
        # Detect an Example(s) section header
        m = re.match(r'^(\s+)Examples?\s*$', stripped)
        if m:
            indent = len(m.group(1))
            indent_str = ' ' * indent
            # Check next line is a dashes underline
            if i + 1 < len(lines) and re.match(r'^\s+-+\s*$', lines[i + 1]):
                # Look ahead to first non-blank line after the header
                j = i + 2
                while j < len(lines) and not lines[j].strip():
                    j += 1
                if j < len(lines):
                    first_code = lines[j]
                    first_stripped = first_code.lstrip()
                    # Only rewrite if code is bare (not >>>, not .. directive)
                    is_bare = (
                        first_code.startswith(indent_str + ' ')
                        and not first_stripped.startswith('>>>')
                        and not first_stripped.startswith('..')
                    )
                    if is_bare:
                        # Emit updated section
                        result.append(indent_str + 'Examples\n')
                        result.append(indent_str + '--------\n')
                        result.append(indent_str + '.. code-block:: python\n')
                        result.append('\n')
                        i += 2  # skip original header + dashes
                        # Skip blank lines between header and code
                        while i < len(lines) and not lines[i].strip():
                            i += 1
                        # Emit code lines indented 4 spaces more than section indent
                        code_indent = indent + 4
                        while i < len(lines):
                            cl = lines[i]
                            cl_stripped_inner = cl.strip()
                            if not cl_stripped_inner:
                                result.append(cl)
                                i += 1
                                continue
                            cl_indent = len(cl) - len(cl.lstrip())
                            # Stop when we hit content at or below section indent level
                            # (but not closing triple-quote which may be at base indent)
                            if cl_indent <= indent and cl_stripped_inner not in ('"""', "'''"):
                                break
                            result.append(' ' * code_indent + cl.lstrip())
                            i += 1
                        changes += 1
                        continue
        result.append(line)
        i += 1
    return ''.join(result), changes


for fpath in sys.argv[1:]:
    with open(fpath, encoding='utf-8') as f:
        src = f.read()
    new_src, changes = fix_example_section(src)
    with open(fpath, 'w', encoding='utf-8') as f:
        f.write(new_src)
    fname = fpath.replace('\\', '/').split('/')[-1]
    print(f'{fname}: {changes} example sections fixed')
    try:
        ast.parse(new_src)
        print(f'  Parse OK')
    except SyntaxError as e:
        print(f'  Parse ERROR: {e}')
