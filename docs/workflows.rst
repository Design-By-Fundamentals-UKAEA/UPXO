Workflows
=========

.. contents:: On this page
   :local:
   :depth: 2

This page shows complete, annotated code examples for common UPXO tasks.
Each example can be copied into a Jupyter notebook or Python script and run directly.

----

Workflow 1 — MCGS2D from Dashboard to Grain Properties
--------------------------------------------------------

This is the standard starting workflow. It mirrors what ``gschar1.ipynb`` demonstrates.

**Step 1: Run the simulation**

The dashboard is a dictionary of simulation parameters. Pass it to the polycrystal
constructor to run the Potts-model MC simulation and produce a ``pxt`` object.

.. code-block:: python

   from upxo.ggrowth.mcgs import mcgs2d

   # Minimal dashboard — adjust domain size and steps to your needs
   pxt = mcgs2d.create(
       domain = (100, 100),   # lattice size: rows x columns
       Q      = 200,          # number of grain orientations
       T      = 0.5,          # effective temperature
       J      = 1.0,          # grain boundary energy
       n_mcs  = 50,           # number of Monte-Carlo steps to run
       n_tslice = 10,         # save every 10th step
   )

After this call, ``pxt.gs`` contains one entry per saved time slice.

**Step 2: Pick a time slice**

.. code-block:: python

   gsid = 5              # the 5th saved time step
   gs = pxt.gs[gsid]     # mcgs2_grain_structure object for that slice

**Step 3: Detect grains and compute morphological properties**

.. code-block:: python

   gs.char_morph_2d(char_gb=True)   # connected-component detection + shape metrics

   print(f"Number of grains: {gs.n}")
   print(gs.prop.head())            # pandas DataFrame — one row per grain

Available columns in ``gs.prop`` include: ``area``, ``aspect_ratio``, ``solidity``,
``major_axis_length``, ``minor_axis_length``, ``perimeter``, ``equivalent_diameter``.

**Step 4: Access the labelled grain image**

.. code-block:: python

   import matplotlib.pyplot as plt

   plt.imshow(gs.lgi, cmap='tab20')
   plt.colorbar(label='Grain ID')
   plt.title(f'MCGS2D — tslice {gsid}, {gs.n} grains')
   plt.show()

**Step 5: Query grain neighbours**

.. code-block:: python

   gs.find_neigh(include_central_grain=False)

   # Neighbours of grain 10
   print(gs.neigh_gid[10])

----

Workflow 2 — Grain Size Distribution
--------------------------------------

Once ``char_morph_2d`` has been called, the area of every grain is in ``gs.prop``.

.. code-block:: python

   import matplotlib.pyplot as plt

   areas = gs.prop['area'].values   # pixel counts

   plt.figure()
   plt.hist(areas, bins=20, edgecolor='k')
   plt.xlabel('Grain area (pixels)')
   plt.ylabel('Count')
   plt.title('Grain size distribution')
   plt.tight_layout()
   plt.show()

----

Workflow 3 — Finding Small Grains and Boundary Grains
-------------------------------------------------------

Use the ``gid_ops`` module to query the labelled image directly.

.. code-block:: python

   import upxo.gsdataops.gid_ops as gidOps

   lfi = gs.lgi

   # Grains with fewer than 5 pixels
   small_grains = gidOps.find_small_fids(lfi, threshold=5)
   print("Small grain IDs:", small_grains)

   # Grains touching the domain boundary
   boundary_grains = gidOps.find_boundary_fids2d(lfi)
   print("Boundary grain IDs:", boundary_grains)

----

Workflow 4 — Resampling a Grain Structure Grid
------------------------------------------------

Use ``grid_ops`` to downsample or rescale the labelled image, for example
before exporting to a coarser FE mesh.

.. code-block:: python

   import upxo.gsdataops.grid_ops as gridOps

   lfi = gs.lgi

   # Downsample to half resolution (sf=0.5 means every 2nd pixel)
   resampled, x_new, y_new, xinc_new, yinc_new = gridOps.resample_grid_2d(
       data=lfi, uigrid=None, sf=0.5
   )
   print("Original shape:", lfi.shape)
   print("Resampled shape:", resampled.shape)

   # Scale by a factor of 2 in both dimensions
   scaled = gridOps.rescale_grid_2d(lfi, scale_factor=2.0)
   print("Scaled shape:", scaled.shape)

----

Workflow 5 — Merging Small Grains
-----------------------------------

Small grains (single-pixel artefacts from the MC evolution) can be absorbed
into their largest neighbour before downstream analysis.

.. code-block:: python

   from upxo.pxtalops.gssmooth2d import _merge_small_grains

   lfi = gs.lgi
   lfi_clean = _merge_small_grains(lfi, area_threshold=3)

   print("Unique grains before:", len(set(lfi.ravel())))
   print("Unique grains after :", len(set(lfi_clean.ravel())))

----

Workflow 6 — Comparing Morphology Across Time Slices
------------------------------------------------------

Iterate over multiple time slices to track how mean grain area evolves with MCS.

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt

   gsids      = sorted(pxt.gs.keys())
   mean_areas = []

   for gsid in gsids:
       gs = pxt.gs[gsid]
       gs.char_morph_2d()
       mean_areas.append(gs.prop['area'].mean())

   plt.plot(gsids, mean_areas, marker='o')
   plt.xlabel('Time slice (gsid)')
   plt.ylabel('Mean grain area (pixels)')
   plt.title('Grain growth kinetics')
   plt.tight_layout()
   plt.show()

----

Next Steps
----------

- :doc:`concepts` — understand the data model behind these examples
- `API Reference <https://design-by-fundamentals-ukaea.github.io/UPXO/>`_ — full module and class documentation
- `Grain Characterisation wiki <https://github.com/Design-By-Fundamentals-UKAEA/UPXO/wiki/Grain-Characterisation>`_ — extended characterisation workflows
- `Visualisation wiki <https://github.com/Design-By-Fundamentals-UKAEA/UPXO/wiki/Visualisation>`_ — 2D and 3D plotting
- `Meshing wiki <https://github.com/Design-By-Fundamentals-UKAEA/UPXO/wiki/Meshing>`_ — FE mesh generation
