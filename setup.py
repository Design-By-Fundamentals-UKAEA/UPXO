"""Backward-compatible setup entry.

Canonical packaging metadata lives in ``pyproject.toml``. Prefer::

    pip install .
    python -m build

This file mirrors the current project version and core dependencies so
legacy ``python setup.py`` / older tooling stays consistent.
"""
from setuptools import setup, find_packages

setup(
    name="upxo",
    version="1.1.0.post5",
    author="Dr. Sunil Anandatheertha",
    author_email="vaasu.anandatheertha@ukaea.uk",
    description=(
        "An open-source Python package for generation, analysis, assessment, "
        "visualisation, meshing, and export of representative polycrystalline "
        "grain structures."
    ),
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Design-By-Fundamentals-UKAEA/UPXO",
    python_requires=">=3.13",
    install_requires=[
        "numpy>=2.2.6",
        "scipy>=1.16.2",
        "matplotlib>=3.10.6",
        "pandas>=2.3.3",
        "numba>=0.62.1",
        "networkx>=3.5",
        "pyvista[jupyter]>=0.46.3",
        "scikit-image>=0.25.2",
        "scikit-learn>=1.7.2",
        "shapely>=2.1.1",
        "tqdm>=4.67.1",
        "seaborn>=0.13.2",
        "colorama>=0.4.6",
        "termcolor>=2.4.0",
        "connected-components-3d>=3.26.1",
        "xlrd>=2.0.1",
        "openpyxl>=3.1.0",
        "ipywidgets>=8.0.0",
        "Pillow>=10.0.0",
    ],
    extras_require={
        "viz": ["plotly>=5.0.0"],
        "mesh": ["pyvoro>=1.3.2", "tetgen>=0.8.2"],
        "io": ["rasterio>=1.4.3"],
        "ebsd": ["defdap==0.93.6"],
        "all": ["upxo[viz,mesh,io,ebsd]"],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.13",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    include_package_data=True,
)
