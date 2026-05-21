from setuptools import setup, find_packages

classifiers = [
    'Development Status :: 3 - Alpha',  # If package is still in early development
    #'Development Status :: 4 - Beta',  # If package that's getting closer to a stable release
    #'Development Status :: 5 - Production/Stable',  # If stable package
    "Intended Audience :: Science/Research",
    "Topic :: Scientific/Engineering :: Physics",
    "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    "Programming Language :: Python :: 3.13",
]
setup(
    name="upxo",
    version="1.0.0",
    author="Dr. Sunil Anandatheertha",
    author_email="vaasu.anandatheertha@ukaea.uk",
    description="An open-source Python package for generation, analysis, assessment, visualisation, meshing, and export of representative polycrystalline grain structures.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Design-By-Fundamentals-UKAEA/UPXO",
    install_requires=[
        "numpy>=2.2.6",
        "scipy>=1.16.2",
        "matplotlib>=3.10.6",
        "pandas>=2.3.3",
        "numba>=0.62.1",
        "networkx>=3.5",
        "pyvista>=0.46.3",
        "vtk>=9.5.2",
        "scikit-image>=0.25.2",
        "scikit-learn>=1.7.2",
        "shapely>=2.1.1",
        "tqdm>=4.67.1",
        "seaborn>=0.13.2",
        "defdap==0.93.6",
    ],
    python_requires=">=3.13",
    classifiers=classifiers,
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    include_package_data=True,
)