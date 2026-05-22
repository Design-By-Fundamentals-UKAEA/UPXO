import os
import sys

# Point Sphinx at the src/ layout so autodoc can import upxo without
# needing the package to be pip-installed in the build environment.
sys.path.insert(0, os.path.abspath('../src'))

# General project information
project = 'UPXO: UKAEA Poly-XTAL Operations'
copyright = '2024, UKAEA'
author = 'Dr. Sunil Anandatheertha'
version = '1.0.0'
release = '1.0.0'

extensions = [
    'sphinx_rtd_theme',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
]

# Mock heavy C-extension / optional dependencies so autodoc does not fail
# when they are absent in the ReadTheDocs build environment.
autodoc_mock_imports = [
    'numpy',
    'scipy',
    'matplotlib',
    'mpl_toolkits',
    'pandas',
    'numba',
    'networkx',
    'pyvista',
    'vtk',
    'skimage',
    'sklearn',
    'shapely',
    'tqdm',
    'seaborn',
    'defdap',
    'rasterio',
    'colorama',
    'pyvoro',
    'tetgen',
    'cc3d',
    'xlrd',
    'termcolor',
    'plotly',
    'openpyxl',
    'netlsd',
    'meshio',
    'pygmsh',
    'gmsh',
]

napoleon_google_docstring = True
napoleon_numpy_docstring = True

html_theme = 'sphinx_rtd_theme'

source_suffix = '.rst'
master_doc = 'index'

html_static_path = []

# Replaces the deprecated autodoc_default_flags dict.
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
    'member-order': 'bysource',
}

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable', None),
}

# Suppress duplicate object description warnings that arise when class
# Attributes sections and autodoc property pages both document the same name.
suppress_warnings = ['py', 'py.duplicate_obj', 'autodoc.duplicate_id']

# Exclude demo scripts — they contain top-level simulation code that executes
# on import and would stall the Sphinx build indefinitely.
exclude_patterns = [
    '_build',
    'docs/srcAPIdocs/upxo.demos*',
    'docs/srcAPIdocs/upxo.scripts*',
]
