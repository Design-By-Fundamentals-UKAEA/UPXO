upxo package
============

**UPXO (UKAEA Poly-XTAL Operations)** is an open-source Python framework for generating,
analysing, manipulating, meshing, visualising, and exporting representative polycrystalline
grain structures for materials science.

----

Quick Start
-----------

UPXO simulations are driven by an Excel dashboard file that holds all parameters.
The minimal workflow is:

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
    print(gs.prop.head())       # pandas DataFrame — one row per grain

For step-by-step walkthroughs see :doc:`../../getting_started` and :doc:`../../workflows`.

----

Key Concepts
------------

**Labelled Feature Image (LFI / lgi)**
    A NumPy integer array where each element holds the integer ID of the grain it belongs to.
    Grain IDs are 1-based; 0 is reserved for background.

**Time slice (tslice / gsid)**
    A saved snapshot of the grain structure at a given Monte-Carlo step.
    ``pxt.m`` is the list of saved step indices; ``pxt.gs[tslice]`` retrieves the
    corresponding :class:`~upxo.pxtal.mcgs2_temporal_slice.mcgs2_grain_structure` object.

**mcgs2_grain_structure**
    The per-time-slice container. Holds the labelled grain image (``gs.lgi``), grain count
    (``gs.n``), morphological properties (``gs.prop``), and neighbour map (``gs.neigh_gid``).

See :doc:`../../concepts` for a full explanation of the data model.

----

Package Structure
-----------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Sub-package
     - Purpose
   * - :mod:`upxo.ggrowth`
     - Monte-Carlo grain growth simulation (Potts model, 2D and 3D)
   * - :mod:`upxo.pxtal`
     - Core grain structure classes and temporal-slice data model
   * - :mod:`upxo.gsdataops`
     - Grid and grain-ID operations on labelled images
   * - :mod:`upxo.charops`
     - Morphological characterisation routines
   * - :mod:`upxo.pxtalops`
     - Grain structure operations: smoothing, merging, detection
   * - :mod:`upxo.xtal`
     - Grain object definitions (2D and 3D)
   * - :mod:`upxo.xtalphy`
     - Crystal physics: orientation, texture, slip systems
   * - :mod:`upxo.gbops`
     - Grain boundary operations
   * - :mod:`upxo.meshing`
     - Conformant and non-conformant FE mesh generation
   * - :mod:`upxo.viz`
     - 2D and 3D plotting (matplotlib, pyvista)
   * - :mod:`upxo.interfaces`
     - Import/export: Abaqus, DAMASK, MOOSE, Dream.3D, EBSD, VTK
   * - :mod:`upxo.statops`
     - Statistical operations and sampling
   * - :mod:`upxo.geoEntities`
     - Points, lines, polygons, surfaces (2D and 3D)
   * - :mod:`upxo._sup`
     - Internal data handlers and utility functions

----

Subpackages
-----------

.. toctree::
   :maxdepth: 4

   upxo.algorithms
   upxo.analysis
   upxo.charops
   upxo.connops
   upxo.dclasses
   upxo.external
   upxo.flags_and_controls
   upxo.gbops
   upxo.geoEntities
   upxo.ggrowth
   upxo.grids
   upxo.gsContainters
   upxo.gsdataops
   upxo.heirGs
   upxo.imageOps
   upxo.interfaces
   upxo.jpops
   upxo.material
   upxo.mechanics
   upxo.meshing
   upxo.misc
   upxo.netops
   upxo.parswep
   upxo.profiling
   upxo.propOps
   upxo.pxtal
   upxo.pxtalops
   upxo.repgen
   upxo.repqual
   upxo.scripts
   upxo.statops
   upxo.surrModelOps
   upxo.tempops
   upxo.tests_and_benchMarks
   upxo.topOps
   upxo.tutorials
   upxo.uiOps
   upxo.viz
   upxo.xtal
   upxo.xtalops
   upxo.xtalphy

Submodules
----------

upxo.initialize module
~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: upxo.initialize
   :members:
   :show-inheritance:
   :undoc-members:

Module contents
---------------

.. automodule:: upxo
   :members:
   :show-inheritance:
   :undoc-members:
