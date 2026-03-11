from setuptools import setup, find_packages

classifiers = [
    'Development Status :: 3 - Alpha',  # If package is still in early development
    #'Development Status :: 4 - Beta',  # If package that's getting closer to a stable release
    #'Development Status :: 5 - Production/Stable',  # If stable package

    # Specify the audience and topic of your package
    'Intended Audience :: Science/Research',
    'Topic :: Scientific/Engineering :: Physics',

    # Specify the license
    'License :: OSI Approved :: MIT License',
    # 'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',

    # Specify supported Python versions
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3.10',
]

setup(
    name='upxo',
    version='10.0',
    author='Dr. Sunil Anandatheertha',
    author_email='vaasu.anandatheertha@ukaea.uk',
    description='A package for grain structure generation, analysis, and export to FE simulation software',
    long_description=open('readme.md').read(),
    url='https://github.com/SunilAnandatheertha/upxo_private/',
    download_url='https://github.com/SunilAnandatheertha/upxo_private/',
    project_urls= {'Source Code': 'https://github.com/SunilAnandatheertha/upxo_private/tree/upxo.v.1.26.1/src',
                   'Documentation': 'https://github.com/SunilAnandatheertha/upxo_private/tree/upxo.v.1.26.1/docs',
                   },
    install_requires='',
    python_requires='>=3.6',
    classifiers=classifiers,
    package_dir={'': 'src'},  # packages are under 'src'
    packages=find_packages(where='src'),  
)
