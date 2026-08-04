"""
UPXO — UKAEA Poly-XTAL Operations
===================================

UPXO is an open-source Python framework for generating, analysing, manipulating,
meshing, visualising, and exporting representative polycrystalline grain structures
for materials science.

It is primarily developed for multi-scale computational studies of nuclear structural
materials but applies equally to aerospace, automotive, and data-driven materials
research.

----

Quick Start
-----------

UPXO grain structure simulations are driven by an Excel dashboard file that holds
all simulation parameters. The minimal workflow is:

.. code-block:: python

    from upxo.ggrowth.mcgs import mcgs

    pxt = mcgs(input_dashboard='path/to/input_dashboard.xls')
    pxt.simulate()
    pxt.detect_grains()

    tslice = pxt.m[-1]          # last saved Monte-Carlo step
    gs = pxt.gs[tslice]         # grain structure at that step

    gs.char_morph_2d(
        use_version=2,
        npixels=True,
        aspect_ratio=True,
        solidity=True,
        make_skim_prop=True,
    )
    print(gs.prop.head())       # pandas DataFrame of grain properties

See the `Getting Started <../../getting_started.html>`_ and
`Workflows <../../workflows.html>`_ pages for annotated step-by-step examples.

----

Package Structure
-----------------

The UPXO package is organised into the following sub-packages:

**Grain structure generation**

- :mod:`upxo.ggrowth` — Monte-Carlo grain growth simulation (Potts model, 2D and 3D)
- :mod:`upxo.pxtal` — Core grain structure classes and temporal-slice data model
- :mod:`upxo.grids` — Grid construction and management
- :mod:`upxo.algorithms` — Low-level MC iteration algorithms

**Grain characterisation and operations**

- :mod:`upxo.gsdataops` — Grid and grain-ID operations on labelled images
- :mod:`upxo.charops` — Morphological characterisation routines
- :mod:`upxo.pxtalops` — Grain structure operations: smoothing, merging, detection
- :mod:`upxo.connops` — Connectivity and neighbourhood operations
- :mod:`upxo.topOps` — Topological analysis
- :mod:`upxo.netops` — Grain network / graph operations
- :mod:`upxo.gbops` — Grain boundary operations

**Crystal structure and orientation**

- :mod:`upxo.xtal` — Grain object definitions (2D and 3D)
- :mod:`upxo.xtalops` — Crystal operations
- :mod:`upxo.xtalphy` — Crystal physics: orientation, texture, slip systems
- :mod:`upxo.material` — Material definitions

**Meshing**

- :mod:`upxo.meshing` — Conformant and non-conformant FE mesh generation

**Visualisation**

- :mod:`upxo.viz` — 2D and 3D plotting (matplotlib, pyvista)

**Data interfaces**

- :mod:`upxo.interfaces` — Import/export: Abaqus, DAMASK, MOOSE, Dream.3D, EBSD, Excel, VTK
- :mod:`upxo.statops` — Statistical operations and sampling
- :mod:`upxo.repqual` — Representativeness quality assessment

**Geometry and spatial entities**

- :mod:`upxo.geoEntities` — Points, lines, polygons, surfaces (2D and 3D)

**Supporting utilities**

- :mod:`upxo._sup` — Internal data handlers, type validators, and utility functions

----

Key Concepts
------------

**Labelled Feature Image (LFI / lgi)**
    A NumPy integer array where every element holds the integer ID of the grain it
    belongs to. This is the fundamental data representation of a grain structure in UPXO.
    Grain IDs are 1-based; 0 is reserved for background.

**Time slice (tslice / gsid)**
    A single saved snapshot of the grain structure at a given Monte-Carlo step.
    ``pxt.m`` is the list of saved step indices; ``pxt.gs[tslice]`` retrieves the
    corresponding :class:`~upxo.pxtal.mcgs2_temporal_slice.mcgs2_grain_structure` object.

**mcgs2_grain_structure**
    The per-time-slice container. Holds the labelled grain image (``gs.lgi``), the
    number of detected grains (``gs.n``), per-grain morphological properties
    (``gs.prop``), and neighbour data (``gs.neigh_gid``).

See the `Key Concepts <../../concepts.html>`_ page for a full explanation.

----

Further Reading
---------------

- `Getting Started <../../getting_started.html>`_
- `Key Concepts <../../concepts.html>`_
- `Workflows <../../workflows.html>`_
- `GitHub repository <https://github.com/Design-By-Fundamentals-UKAEA/UPXO>`_
- `Wiki <https://github.com/Design-By-Fundamentals-UKAEA/UPXO/wiki>`_
- `YouTube channel <https://www.youtube.com/@UPXODBF>`_
"""

from importlib.metadata import PackageNotFoundError, version as _version

try:
    __version__ = _version("upxo")
except PackageNotFoundError:
    __version__ = "unknown"
