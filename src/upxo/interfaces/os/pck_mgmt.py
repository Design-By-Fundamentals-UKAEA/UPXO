import importlib.metadata
import sys

def check_and_install_packages(required_packages, confirmation_message):
    """
    Checks if the given packages with specific versions are installed. If not, asks for user confirmation to install them.

    Args:
        required_packages (list): A list of tuples with package names and versions.
        confirmation_message (str): A message to ask for user confirmation.
    """
    from packaging.specifiers import SpecifierSet
    for package, version_spec in required_packages:
        try:
            installed_version = importlib.metadata.version(package)
            if version_spec and installed_version not in SpecifierSet(version_spec):
                print(f"Package '{package}' version '{installed_version}' does not meet '{version_spec}'.")
                if input(f"{confirmation_message} to meet version requirement (y/n): ").lower() == 'y':
                    install_package(f"{package}{version_spec}")
            else:
                print(f"Package '{package}' version '{installed_version}' is already installed.")
        except importlib.metadata.PackageNotFoundError:
            print(f"Package '{package}' not found.")
            if input(f"{confirmation_message} (y/n): ").lower() == 'y':
                install_package(f"{package}{version_spec}")

def install_package(package_spec):
    """
    Installs the given package using pip, including the version.

    Args:
        package_spec (str): The name and version specifier of the package to install.
    """
    try:
        import subprocess
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package_spec])
        print(f"Package '{package_spec}' installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to install {package_spec}: {e}")
