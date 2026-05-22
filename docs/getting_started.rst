Getting Started
===============

.. contents:: On this page
   :local:
   :depth: 2

----

System Requirements
-------------------

- **Python:** >= 3.13
- **Operating System:** Windows *(tested by the development team)*. Linux and macOS are untested — community feedback welcome.
- **Recommended:** Use a virtual environment (conda, miniforge, or venv).

----

Installation
------------

Quick install (recommended for most users):

.. code-block:: bash

   pip install upxo

Install with all optional extras:

.. code-block:: bash

   pip install upxo[all]

Install specific extras only:

.. code-block:: bash

   pip install upxo[viz]    # Plotly interactive plots
   pip install upxo[mesh]   # FE meshing (pyvoro, tetgen)
   pip install upxo[io]     # Raster I/O (rasterio)
   pip install upxo[ebsd]   # EBSD data import (DefDAP)

Developer install (editable, from source):

.. code-block:: bash

   git clone https://github.com/Design-By-Fundamentals-UKAEA/UPXO.git
   cd UPXO
   pip install -e .

See the `Dependencies wiki page <https://github.com/Design-By-Fundamentals-UKAEA/UPXO/wiki/Dependencies>`_
for a full list of what each extra provides.

----

Verify Installation
-------------------

Once installed, confirm UPXO is working:

.. code-block:: python

   import upxo
   print(upxo.__version__)

For a full end-to-end validation, run the ``gschar1.ipynb`` notebook described below.

----

Getting the Demo Notebooks
--------------------------

Demo notebooks are not shipped with the pip package. Clone the repository to get them:

.. code-block:: bash

   git clone https://github.com/Design-By-Fundamentals-UKAEA/UPXO.git --depth=1

Navigate to ``src/upxo/demos/`` — notebooks are organised by topic.

----

Start Here — Demo Notebooks
----------------------------

gschar1.ipynb — MCGS2D (start here)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Location:** ``src/upxo/demos/gschar/gschar1.ipynb``

`View on GitHub <https://github.com/Design-By-Fundamentals-UKAEA/UPXO/blob/main/src/upxo/demos/gschar/gschar1.ipynb>`_

This is the recommended entry point. It covers the complete 2D Monte Carlo Grain Structure
(MCGS2D) workflow and validates your installation.

You will learn to:

- Initiate and run a 2D MCGS simulation from the dashboard
- Understand time slices (``gsid`` / ``tslice``) and grain structure evolution
- Detect grains and understand state/grain division principles
- Calculate primary shape metrics: area, aspect ratio, solidity, axis lengths
- Access grain metadata and property tables (``pxt``, ``pxt.gs[gsid]``, ``lgi``, ``prop``)
- Visualise grain maps with labels

gschar2.ipynb — MCGS3D
~~~~~~~~~~~~~~~~~~~~~~~~

**Location:** ``src/upxo/demos/gschar/gschar2.ipynb``

`View on GitHub <https://github.com/Design-By-Fundamentals-UKAEA/UPXO/blob/main/src/upxo/demos/gschar/gschar2.ipynb>`_

Introduces 3D Monte Carlo Grain Structure generation and characterisation.

You will learn to:

- Generate 3D grain structures and navigate the data structure
- Calculate intercept grain size and morphological properties (voxel count, aspect ratio, solidity)
- Study surface-to-subsurface morphological property relationships
- Characterise a 2D slice of a 3D grain structure
- Extract subsets and clean grain structures using erosion
- Visualise 3D microstructures with pyvista

----

Python Script Demonstrations
-----------------------------

For applications that are difficult to explain step-by-step, UPXO includes Python script
demonstrations under ``src/upxo/scripts/``. These are useful as testing grounds or templates
for building your own workflows.

----

Video Tutorials
---------------

UPXO has an official YouTube channel with introductory videos and walkthroughs:

`UPXO on YouTube <https://www.youtube.com/@UPXODBF>`_

----

Next Steps
----------

Once you have run ``gschar1.ipynb`` successfully, explore:

- :doc:`concepts` — key data structures and how MCGS2D works
- :doc:`workflows` — annotated end-to-end code examples
- `2D Grain Structure Generation wiki <https://github.com/Design-By-Fundamentals-UKAEA/UPXO/wiki/2D-Grain-Structure-Generation>`_
- `3D Grain Structure Generation wiki <https://github.com/Design-By-Fundamentals-UKAEA/UPXO/wiki/3D-Grain-Structure-Generation>`_
- `API Reference <https://design-by-fundamentals-ukaea.github.io/UPXO/>`_

----

Need Help?
----------

If you are from an academic institution and feel UPXO could be useful to your project,
contact Dr. Sunil Anandatheertha — happy to help you set up and use UPXO.

📧 vaasu.anandatheertha@ukaea.uk
