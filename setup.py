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
    version="1.0",
    author="Dr. Sunil Anandatheertha",
    author_email="vaasu.anandatheertha@ukaea.uk",
    description="An open-source Python package for generation, analysis, assessment, visualisation, meshing, and export of representative polycrystalline grain structures.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Design-By-Fundamentals-UKAEA/UPXO",
    install_requires=[],
    python_requires=">=3.13",
    classifiers=classifiers,
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    include_package_data=True,
)