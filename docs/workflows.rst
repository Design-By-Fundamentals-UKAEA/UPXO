Workflows
=========

.. contents:: On this page
   :local:
   :depth: 2

This page shows complete, annotated code examples for common UPXO tasks.
All examples follow patterns taken directly from the demo notebooks in
``src/upxo/demos/``.

.. note::

   UPXO reads simulation parameters from an Excel dashboard file
   (``input_dashboard.xls``). The examples below use a placeholder path — replace
   it with the path to your own dashboard file. Template dashboards are provided
   under ``src/upxo/interfaces/user_inputs/``.

----

Workflow 1 — MCGS2D: Simulate, Detect Grains, Characterise
------------------------------------------------------------

This is the standard MCGS2D pipeline, equivalent to what ``gschar1.ipynb`` demonstrates.

.. code-block:: python

   from upxo.ggrowth.mcgs import mcgs

   # Step 1 — Load dashboard and run the MC simulation
   pxt = mcgs(input_dashboard='path/to/input_dashboard.xls')
   pxt.simulate()

   # Step 2 — Detect grains at every saved time slice
   pxt.detect_grains()

   # Step 3 — Pick a time slice
   # pxt.m is the list of saved MCS step indices
   tslice = pxt.m[-1]        # last saved step
   gs = pxt.gs[tslice]       # mcgs2_grain_structure object

   # Step 4 — Characterise: request exactly the properties you need
   gs.char_morph_2d(
       use_version=2,
       npixels=True,
       area=True,
       aspect_ratio=True,
       solidity=True,
       circularity=True,
       char_gb=False,
       make_skim_prop=True,
       get_grain_coords=True,
   )

   # Step 5 — Inspect results
   print(f"Number of grains: {gs.n}")
   print(gs.prop.columns.tolist())   # shows which columns were computed
   print(gs.prop.head())

Properties are only present in ``gs.prop`` if the matching flag was set to ``True``
in the ``char_morph_2d`` call. Available flags include: ``npixels``, ``area``,
``aspect_ratio``, ``solidity``, ``circularity``, ``eccentricity``,
``major_axis_length``, ``minor_axis_length``, ``perimeter``, ``eq_diameter``,
``compactness``, ``morph_ori``, ``euler_number``.

----

Workflow 2 — Visualise the Labelled Grain Image
-------------------------------------------------

After ``detect_grains()``, the labelled grain image is stored in ``gs.lgi``.

.. code-block:: python

   import matplotlib.pyplot as plt

   tslice = pxt.m[-1]
   gs = pxt.gs[tslice]

   plt.figure()
   plt.imshow(gs.lgi, cmap='tab20')
   plt.colorbar(label='Grain ID')
   plt.title(f'MCGS2D — tslice {tslice}, {gs.n} grains')
   plt.axis('off')
   plt.tight_layout()
   plt.show()

To visualise the raw MC spin state (before grain detection):

.. code-block:: python

   plt.imshow(pxt.S, cmap='nipy_spectral')
   plt.title('MC spin state (final)')
   plt.show()

----

Workflow 3 — Grain Size Distribution
--------------------------------------

Request ``npixels=True`` (pixel count per grain) when calling ``char_morph_2d``,
then plot the distribution.

.. code-block:: python

   import matplotlib.pyplot as plt

   gs.char_morph_2d(use_version=2, npixels=True, make_skim_prop=True)

   pixel_counts = gs.prop['npixels'].values

   plt.figure()
   plt.hist(pixel_counts, bins=20, edgecolor='k')
   plt.xlabel('Grain size (pixels)')
   plt.ylabel('Count')
   plt.title('Grain size distribution')
   plt.tight_layout()
   plt.show()

----

Workflow 4 — Grain Neighbourhood
----------------------------------

.. code-block:: python

   tslice = pxt.m[-1]
   gs = pxt.gs[tslice]

   # Characterise first so bounding boxes exist for the neighbour search
   gs.char_morph_2d(use_version=2, bbox=True, bbox_ex=True, make_skim_prop=True)

   # Compute neighbours for every grain
   gs.find_neigh(include_central_grain=False, print_msg=True, use_numba=True)

   # Neighbours of grain with ID 10
   print(gs.neigh_gid[10])

.. note::

   A known bug in some builds causes the central grain to appear in its own
   neighbour list when ``include_central_grain=False``. The workaround used in
   the demo notebooks is:

   .. code-block:: python

      for gid in gs.neigh_gid.keys():
          if gid in gs.neigh_gid[gid]:
              gs.neigh_gid[gid].remove(gid)

----

Workflow 5 — Finding Small Grains and Boundary Grains
-------------------------------------------------------

Use the ``gid_ops`` module to query the labelled image directly.

.. code-block:: python

   import upxo.gsdataops.gid_ops as gidOps

   lfi = gs.lgi

   # Grains with 5 pixels or fewer
   small_grains = gidOps.find_small_fids(lfi, threshold=5)
   print("Small grain IDs:", small_grains)

   # Grains whose pixels touch the domain boundary
   boundary_grains = gidOps.find_boundary_fids2d(lfi)
   print("Boundary grain IDs:", boundary_grains)

----

Workflow 6 — Resampling and Rescaling the Grid
------------------------------------------------

Use ``grid_ops`` to change the resolution of the state array or labelled image.

.. code-block:: python

   from upxo.gsdataops.grid_ops import resample_grid_2d, rescale_grid_2d

   # Downsample by factor 0.25 using the simulation's own grid object
   resampled, x_new, y_new, xinc_new, yinc_new = resample_grid_2d(
       pxt.S, pxt.uigrid, sf=0.25, method='nearest'
   )
   print("Original shape:", pxt.S.shape)
   print("Resampled shape:", resampled.shape)

   # Rescale to twice the resolution
   scaled = rescale_grid_2d(pxt.S, scale_factor=2, method='nearest')
   print("Scaled shape:", scaled.shape)

----

Workflow 7 — Merging Small Grains
-----------------------------------

Single-pixel or sub-threshold grains can be absorbed into their largest neighbour
before downstream analysis.

.. code-block:: python

   import numpy as np
   from upxo.pxtalops.gssmooth2d import _merge_small_grains

   lfi = gs.lgi
   lfi_clean = _merge_small_grains(lfi, area_threshold=3)

   print("Unique grains before:", len(np.unique(lfi)))
   print("Unique grains after :", len(np.unique(lfi_clean)))

----

Workflow 8 — Comparing Grain Size Across Time Slices
------------------------------------------------------

Iterate over saved time slices to track how mean grain size evolves.

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt

   mean_sizes = []

   for tslice in pxt.m:
       gs = pxt.gs[tslice]
       gs.char_morph_2d(use_version=2, npixels=True, make_skim_prop=True)
       mean_sizes.append(gs.prop['npixels'].mean())

   plt.figure()
   plt.plot(pxt.m, mean_sizes, marker='o')
   plt.xlabel('Monte-Carlo step (tslice)')
   plt.ylabel('Mean grain size (pixels)')
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
