"""Imports of optional packages that fail with a message the user can act on."""
import importlib

_GMSH_SYSTEM_LIBRARIES = (
    "gmsh needs system libraries that pip does not install. On Debian or Ubuntu:\n"
    "    sudo apt-get install -y libglu1-mesa libxcursor1 libxinerama1 libxft2\n"
    "On Fedora or RHEL: sudo dnf install -y mesa-libGLU libXcursor libXinerama libXft"
)


def import_gmsh():
    """Return the gmsh module.

    The gmsh shared library links against GLU and X11 libraries. When they are missing,
    `import gmsh` raises OSError (for example "libGLU.so.1: cannot open shared object file"),
    which is turned into an ImportError that names the packages to install.
    """
    try:
        return importlib.import_module('gmsh')
    except OSError as error:
        raise ImportError(f"gmsh could not be loaded ({error}).\n{_GMSH_SYSTEM_LIBRARIES}") from error
