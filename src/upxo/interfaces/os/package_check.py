"""
package_check.py
=================
Small standalone script that checks whether an optional dependency
(named by ``package_name``, e.g. ``pyvista``) is importable in the
current environment and prints the result.
"""

import importlib.util

package_name = "pyvista"  # Replace 'package_name' with the name of the package you're checking for

package_spec = importlib.util.find_spec(package_name)

if package_spec is not None:
    print(f"{package_name} is installed.")
else:
    print(f"{package_name} is not installed.")