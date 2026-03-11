import importlib
import sys
import pkg_resources

def check_and_install_packages(required_packages, confirmation_message):
    """
    Checks if the given packages with specific versions are installed. If not, asks for user confirmation to install them.

    Args:
        required_packages (list): A list of tuples with package names and versions.
        confirmation_message (str): A message to ask for user confirmation.
    """
    for package, version in required_packages:
        try:
            # Check if the package and version are already installed
            pkg_resources.require(f"{package}{version}")
            print(f"Package '{package}' with version '{version}' is already installed.")
        except pkg_resources.DistributionNotFound:
            print(f"Package '{package}' not found.")
            if input(f"{confirmation_message} (y/n): ").lower() == 'y':
                install_package(f"{package}{version}")
        except pkg_resources.VersionConflict as e:
            print(f"Package '{package}' is installed but does not meet the version requirement: {e}")
            if input(f"{confirmation_message} to meet version requirement (y/n): ").lower() == 'y':
                install_package(f"{package}{version}")

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
