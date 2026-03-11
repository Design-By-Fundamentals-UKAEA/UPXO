import importlib.util

package_name = "pyvista"  # Replace 'package_name' with the name of the package you're checking for

package_spec = importlib.util.find_spec(package_name)

if package_spec is not None:
    print(f"{package_name} is installed.")
else:
    print(f"{package_name} is not installed.")