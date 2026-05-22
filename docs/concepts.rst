Key Concepts
============

.. contents:: On this page
   :local:
   :depth: 2

This page explains the core ideas and data structures in UPXO. Reading it before
working with the API will save considerable time.

----

Grain Structure Generation Methods
------------------------------------

UPXO supports two families of grain structure generation:

**Monte-Carlo Grain Structures (MCGS)**
   Physics-based simulation using the Potts model. A lattice of sites evolves
   stochastically over time — sites adopt the spin (orientation) of their neighbours
   with a probability governed by the Boltzmann criterion. Grain boundaries emerge
   as regions where neighbouring sites carry different spins. Both 2D (MCGS2D)
   and 3D (MCGS3D) variants are available.

**Voronoi Tessellation Grain Structures (VTGS)**
   Geometry-based construction. Seed points are placed in the domain, and each
   point in space is assigned to its nearest seed. The result is a partition of
   the domain into convex grains whose boundaries are straight lines (2D) or
   planar facets (3D). VTGS produces grain structures instantly but without
   growth physics.

Most users start with MCGS2D.

----

The Potts Model — What Happens During Simulation
--------------------------------------------------

The Potts model drives MCGS simulations. A brief description:

1. **Initialise** — Assign a random *spin* value (1 to *Q*) to every site on the lattice.
   Each unique spin represents a distinct grain orientation.
2. **Iterate** — Repeatedly select a random site. Propose a new spin drawn from the *Q*
   states. Calculate the energy change ``ΔE`` from switching: a switch is favourable if
   it reduces the number of unlike-spin neighbour pairs (i.e. removes grain boundary).
   Accept the switch with probability ``min(1, exp(−ΔE / kT))``.
3. **Time** — One Monte-Carlo Step (MCS) is defined as *N* attempted spin flips, where *N*
   is the number of lattice sites. As MCS increases, grains coarsen and the mean grain
   size grows.

Key parameters:

- **Q** — number of allowed spin states (orientations). Typical range: 100–1000.
- **J** — interaction energy (grain boundary energy per unit length).
- **T** — effective temperature (controls acceptance of unfavourable flips).
- **Domain size** — lattice dimensions (rows × columns for 2D).

----

Time Slices (tslice / gsid)
----------------------------

A single MCGS simulation produces a grain structure at every saved MCS step.
Each saved step is called a **time slice** and is identified by ``gsid`` (grain structure ID)
— an integer index into the simulation's time history.

.. code-block:: python

   pxt = ...           # polycrystal object, produced by the dashboard
   tslice = 10         # the 10th saved time step
   gs = pxt.gs[tslice] # the grain structure at that step

Examining multiple ``gsid`` values lets you study grain growth kinetics: how grain size,
morphology, and topology evolve with simulated time.

----

The Labelled Feature Image (LFI / lgi)
----------------------------------------

The fundamental data representation of a grain structure in UPXO is a **Labelled Feature Image**
(LFI), stored as a 2D (or 3D) NumPy integer array. Each pixel (voxel) holds the integer ID of
the grain it belongs to.

.. code-block:: text

   [ 1  1  1  2  2 ]
   [ 1  1  2  2  2 ]   ← Each number is a grain ID
   [ 1  3  3  2  2 ]
   [ 3  3  3  4  4 ]
   [ 3  3  4  4  4 ]

Within the API you will see two names for the same concept:

- ``lfi`` — used in lower-level modules (``gsdataops``, ``gid_ops``, etc.)
- ``lgi`` — "Labelled Grain Image", used inside grain structure objects (``mcgs2_grain_structure``)

They are the same NumPy array; the naming reflects which layer of the code you are in.

----

The Polycrystal Object (pxt)
------------------------------

The polycrystal object (conventionally named ``pxt``) is the top-level container produced
by running an MCGS simulation through the dashboard. It holds:

- ``pxt.gs`` — dictionary of time slices, keyed by ``gsid``
- ``pxt.gs[gsid]`` — a :class:`~upxo.pxtal.mcgs2_temporal_slice.mcgs2_grain_structure` object
- ``pxt.gs[gsid].lgi`` — the labelled grain image for that time slice
- ``pxt.gs[gsid].n`` — number of detected grains at that time slice

.. code-block:: python

   # Access the 5th time slice
   gs5 = pxt.gs[5]

   # How many grains?
   print(gs5.n)

   # The labelled image
   print(gs5.lgi.shape)

----

Grain Structure Object (mcgs2_grain_structure)
-----------------------------------------------

Each time slice (``pxt.gs[gsid]``) is an instance of
:class:`~upxo.pxtal.mcgs2_temporal_slice.mcgs2_grain_structure`. It provides:

**Grain detection**
   ``gs.char_morph_2d()`` — detects grains via connected components and computes
   morphological properties (area, aspect ratio, solidity, axis lengths, perimeter, etc.).

**Property access**
   ``gs.prop`` — a pandas DataFrame with one row per grain and one column per property.

**Neighbourhood**
   ``gs.find_neigh()`` — computes first-order grain neighbours.
   ``gs.neigh_gid`` — dictionary mapping each grain ID to its list of neighbour IDs.

**Boundary**
   ``gs.char_gb=True`` inside ``char_morph_2d`` — also detects grain boundary pixels.

----

Grain IDs (gid / fid)
-----------------------

Grain IDs are 1-based integers assigned by the connected-component labelling step.
In most public APIs they appear as ``gid`` (grain ID) or ``fid`` (feature ID) — the terms
are interchangeable. Grain ID 0 is reserved for background or unlabelled pixels and is
excluded from all per-grain operations.

----

Key Modules at a Glance
------------------------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Module
     - Purpose
   * - ``upxo.pxtal.mcgs2_temporal_slice``
     - Core grain structure class for 2D MCGS time slices
   * - ``upxo.pxtal.mcgs3_temporal_slice``
     - Core grain structure class for 3D MCGS time slices
   * - ``upxo.gsdataops.gid_ops``
     - Grain-ID–level queries: neighbours, boundary grains, small grains
   * - ``upxo.gsdataops.grid_ops``
     - Grid operations: slicing, resampling, rescaling
   * - ``upxo.pxtalops.gssmooth2d``
     - Grain structure smoothing and small-grain merging
   * - ``upxo.viz.gsviz``
     - Grain structure visualisation (matplotlib)
   * - ``upxo.meshing``
     - Conformant and non-conformant FE mesh generation
   * - ``upxo.interfaces``
     - Export to Abaqus, DAMASK, MOOSE, Dream.3D, and others
