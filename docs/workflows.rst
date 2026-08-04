Workflows
=========

.. contents:: On this page
   :local:
   :depth: 2

This page shows complete, annotated code examples for common UPXO tasks covering all major capabilities:
2D and 3D grain generation (MCGS, Voronoi), hierarchical microstructures (FM Steels), twinned microstructures (FCC),
EBSD integration, meshing, and visualization.

All examples reference real, importable classes and functions, verified against the current source
(not just plausible-looking API guesses). Where a full pipeline needs inputs this page can't fabricate
(real EBSD scan data, an active gmsh session, etc.), the example shows the correct real call pattern and
points to the actual tested demo notebook under ``src/upxo/demos/`` for the complete runnable reference.

.. note::

   Some workflows reference Excel dashboard files (``input_dashboard.xls``). These use placeholder paths — replace
   them with paths to your own dashboard files. Template dashboards are provided under ``src/upxo/interfaces/user_inputs/``.

----

Part 1: 2D Grain Structures
===========================

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

Workflow 2 — Visualise the Labelled Grain Image (2D)
-----------------------------------------------------

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

Workflow 3 — Grain Size Distribution (2D)
-------------------------------------------

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

Workflow 4 — Grain Neighbourhood (2D)
--------------------------------------

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

Workflow 5 — Finding Small and Boundary Grains (2D)
----------------------------------------------------

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
----------------------------------

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

Part 2: 3D Grain Structures
===========================

Workflow 9 — MCGS3D: Monte Carlo 3D Grain Growth
-------------------------------------------------

3D Monte Carlo grain growth uses the **same** ``mcgs`` class as the 2D workflow above —
dimensionality is read from the input dashboard (``dim: 3``), not a separate 3D class.
This is equivalent to ``gschar2.ipynb``.

.. code-block:: python

   from upxo.ggrowth.mcgs import mcgs

   # Step 1 — Load a 3D dashboard (dim=3) and run the MC simulation
   pxt = mcgs(input_dashboard='path/to/input_dashboard_3d.xls')
   pxt.simulate()

   # Step 2 — Detect grains at every saved time slice
   pxt.detect_grains()

   # Step 3 — Pick a time slice
   tslice = pxt.m[-1]
   gs = pxt.gs[tslice]

   # Step 4 — Access the 3D labelled grain image
   lgi_3d = gs.lgi  # 3D array of grain IDs

   # Step 5 — Basic characterisation
   import numpy as np
   print(f"Domain size: {lgi_3d.shape}")
   print(f"Number of grains detected: {len(np.unique(lgi_3d)) - 1}")  # exclude background (0)

   # Step 6 — Extract a 2D slice for visualization
   import matplotlib.pyplot as plt
   mid_z = lgi_3d.shape[2] // 2
   plt.figure()
   plt.imshow(lgi_3d[:, :, mid_z], cmap='tab20')
   plt.title(f'3D MCGS — XY slice at Z={mid_z}')
   plt.colorbar(label='Grain ID')
   plt.show()

See ``src/upxo/demos/gschar/gschar2.ipynb`` for the complete reference, including
morphological characterisation and 3D visualisation with PyVista.

----

Workflow 10 — 3D Voronoi Tessellation
--------------------------------------

3D Voronoi tessellation is built via ``gtess3d`` (in ``upxo.pxtal.vortess3d``) —
**not** a class named ``Voronoi3D``, which does not exist in this codebase.
``bounds`` must be shape ``(3, 2)`` — ``[[xmin, xmax], [ymin, ymax], [zmin, zmax]]`` —
a flat 6-tuple will raise ``ValueError``.

.. code-block:: python

   import numpy as np
   from upxo.pxtal.vortess3d import gtess3d

   # Step 1 — Define seed points and domain bounds
   np.random.seed(42)
   seed_points = np.random.uniform(0, 100, size=(50, 3))
   bounds = [[0, 100], [0, 100], [0, 100]]

   # Step 2 — Build the tessellation directly from seed coordinates
   vor3d = gtess3d.from_seed_points(seed_points, bounds=bounds)

   # Step 3 — Access results (instance 1; gtess3d supports multi-instance ensembles)
   print(f"Number of grains: {vor3d.tprop[1]['ncells']}")
   print(f"Bounds: {vor3d.bounds['bbox']}")
   print(f"Seed coordinates: {vor3d.sp['coords'][0].shape}")

Periodic boundaries on selected axes:

.. code-block:: python

   vor3d_periodic = gtess3d.from_seed_points(
       seed_points, bounds=bounds, periodic=(True, True, False)
   )

Alternative constructors on the same class: ``gtess3d.from_mpoint3d`` (from an
``MPoint3d`` seed object with richer metadata), ``gtess3d.from_regular_lattice``
(structured seed lattices), and ``gtess3d.from_seed_point_random``.

----

Workflow 11 — 3D Grain Characterisation
----------------------------------------

Compute morphological properties directly from a 3D labelled grain image
(works the same whether ``lgi_3d`` came from MCGS3D or ``gtess3d``).

.. code-block:: python

   import numpy as np

   lgi_3d = gs.lgi  # from Workflow 9, or vor3d.pxtals[...] voxel data from Workflow 10

   # Step 1 — Compute volume of each grain
   unique_ids = np.unique(lgi_3d)
   grain_volumes = {}
   for gid in unique_ids:
       if gid != 0:  # skip background
           grain_volumes[gid] = np.sum(lgi_3d == gid)

   # Step 2 — Compute center of mass
   grain_centers = {}
   for gid in unique_ids:
       if gid != 0:
           mask = (lgi_3d == gid)
           coords = np.argwhere(mask)
           grain_centers[gid] = coords.mean(axis=0)

   # Step 3 — Compute principal moments (inertia tensor)
   # This gives aspect ratios and orientation
   grain_moments = {}
   for gid in unique_ids:
       if gid != 0:
           mask = (lgi_3d == gid)
           coords = np.argwhere(mask) - grain_centers[gid]
           # Inertia tensor
           Ixx = np.sum(coords[:, 1]**2 + coords[:, 2]**2)
           Iyy = np.sum(coords[:, 0]**2 + coords[:, 2]**2)
           Izz = np.sum(coords[:, 0]**2 + coords[:, 1]**2)
           grain_moments[gid] = {'Ixx': Ixx, 'Iyy': Iyy, 'Izz': Izz}

   # Step 4 — Summary statistics
   volumes = np.array(list(grain_volumes.values()))
   print(f"Grain volume stats:")
   print(f"  Mean: {volumes.mean():.1f} voxels")
   print(f"  Std: {volumes.std():.1f} voxels")
   print(f"  Min: {volumes.min()} voxels")
   print(f"  Max: {volumes.max()} voxels")

----

Part 3: Specialized Microstructures
====================================

Workflow 12 — Ferritic-Martensitic Steel Hierarchical Microstructure
---------------------------------------------------------------------

Generate a hierarchical lath-based microstructure for FM steels (Eurofer, F82H, T91)
using the real ``fm_steel_3d`` chainable pipeline. Each stage returns a **new**
object of the next class in the chain (``FMSteel3DBase`` → ``...WithPAGs`` →
``...WithBlocks`` → ``...WithOrientations`` → ``...WithSubBlocks``) — it does not
mutate in place. This exact sequence is verified to run end-to-end.

.. code-block:: python

   import numpy as np
   from upxo.pxtal.fm_steel_3d import FMSteel3DBase

   # Step 1 — Start from an existing labelled grain image (LFI/LGI), e.g. from
   # a Voronoi or MCGS3D base structure (Workflows 9-10), with physical dimensions
   lfi = np.random.randint(1, 100, (50, 50, 50))   # replace with a real base structure
   fm = FMSteel3DBase.from_lfi(lfi, physical_dimensions=(100, 100, 100))

   # Step 2 — Partition grains into Prior Austenite Grain (PAG) clusters
   fm_pag = fm.generate_pag_clusters(
       pag_size_distribution={'sizes': [3, 4, 6], 'probs': [0.25, 0.5, 0.25]},
       pag_grain_fraction=0.8,
       random_seed=42,
   )

   # Step 3 — Assign PAG orientations, then generate martensitic blocks
   fm_pag.assign_pag_orientations(pag_ori_mode='random', random_seed=42)
   fm_blk = fm_pag.generate_blocks(block_thickness_range=(2.0, 5.0), random_seed=42)

   # Step 4 — Assign Kurdjumov-Sachs block orientations
   fm_ori = fm_blk.assign_orientations(
       ks_variant_selection='random_per_block', random_seed=42
   )

   # Step 5 — Optional: sub-block (lath) generation with intra-block scatter
   fm_sub = fm_ori.generate_subblocks(
       subblock_thickness_range=(0.5, 1.5), random_seed=42
   )

   # Step 6 — Inspect results
   print(f"Grains: {fm_ori.n_grains}   Blocks: {fm_ori.n_blocks}")
   stats = fm_ori.get_full_hierarchy_statistics()
   print(stats)

   # Step 7 — Visualise
   fm_ori.visualize_block_ipf_map()
   fm_ori.plot_pag_map_pyvista()

**Real, verified parameters** (from the module's own usage example in
``upxo/pxtal/fm_steel_3d/__init__.py``, confirmed by running the pipeline above):

- ``pag_size_distribution``: dict with ``sizes`` (grains-per-PAG options) and matching ``probs``.
- ``pag_grain_fraction``: fraction of base grains absorbed into PAG clusters (0.0–1.0).
- ``block_thickness_range``: tuple, physical block thickness bounds.
- ``ks_variant_selection``: variant-assignment strategy, e.g. ``'random_per_block'``.

See ``src/upxo/demos/FMSteel3D/block_level_01.ipynb`` for the complete reference
notebook, including retained austenite and mesh export.

----

Workflow 13 — Twinned FCC Microstructure Generation
-----------------------------------------------------

Generate a microstructure with Sigma-3 twin lamellae in FCC materials (Cu, CuCrZr, OFHC-Cu)
using the real ``twinned_simple_3d`` package. Host-grain setup below is verified to run;
the full physical twin-introduction step (``TwinGenerator3D.introduce_primary_twins``)
additionally requires EBSD-derived twin-thickness and volume-fraction targets, which
this snippet doesn't fabricate — see the note and demo notebook below for that part.

.. code-block:: python

   import numpy as np
   from upxo.pxtal.vortess3d import gtess3d
   from upxo.pxtal.twinned_simple_3d import TwinnedSimple3DBase, TwinGenerator3D

   # Step 1 — Generate base (host) grain structure, e.g. via Voronoi (Workflow 10)
   seed_points = np.random.uniform(0, 100, size=(40, 3))
   vor3d = gtess3d.from_seed_points(seed_points, bounds=[[0, 100], [0, 100], [0, 100]])

   # Step 2 — Wrap a labelled grain image (lgi) as a TwinnedSimple3DBase
   # (voxel_size is required; units default to 'microns')
   twin_base = TwinnedSimple3DBase(lgi=lgi_3d, voxel_size=1.0, rng_seed=42)

   # Step 3 — Allocate twin-host grains, spatial-dispersal-aware (MIS algorithm)
   # avoids selecting adjacent grains as hosts
   twin_base.allocate_twin_hosts_spatial(
       target_hosting_fraction=0.3, seed=42
   )
   print(f"Host grains: {len(twin_base.host_grain_ids)}")

   # Step 4 — Configure the twin generator (real constructor, all kwargs shown
   # have defaults except `base`)
   twin_gen = TwinGenerator3D(
       base=twin_base,
       n_lamellae_per_host=2,
       twin_nucleation_site='random_gb',
       meshing_route='conformal',
       rng_seed=42,
   )

.. note::

   ``TwinGenerator3D.introduce_primary_twins(host_orientations, twin_thickness, tvf)``
   requires ``twin_thickness`` and ``tvf`` dicts derived from real EBSD measurements
   (via a representativeness/registry object's ``compute_mc_twin_thickness`` and
   ``compute_ebsd_tvf`` methods) — these encode the target twin volume fraction and
   thickness distribution to match. Fabricating placeholder values here would produce
   a misleadingly "complete-looking" example that doesn't reflect how the physics is
   actually constrained. For the full, real, working pipeline (EBSD import through
   twin generation, cleaning, and export), see:

   - ``src/upxo/demos/Twinned3D/repOFHCCu3d_ebsdVf.1.0.ipynb`` — EBSD-measured volume fraction
   - ``src/upxo/demos/Twinned3D/repOFHCCu3d_probVf.1.0.ipynb`` — probability-weighted volume fraction
   - ``src/upxo/demos/twins/mcgs3d02.ipynb`` — end-to-end MC-grown host + twin workflow

After twin generation, ``StructureCleaner3D.clean(...)`` removes voxel spikes and
splits disconnected lobes; ``RepresentativenessValidator3D`` validates the result
against multi-axis 2D-slice misorientation distributions. See the notebooks above
for both in context.

----

Workflow 14 — EBSD-Guided Microstructure Generation
-----------------------------------------------------

Import experimental EBSD data and use it to guide synthetic microstructure generation.

.. code-block:: python

   from upxo.interfaces.defdap.ebsd_reader import EBSDReader

   # Step 1 — Load and grain-detect an EBSD scan in one call
   # Supported formats depend on the underlying DefDAP reader (e.g. .cif/.ctf, .crc)
   ebsd = EBSDReader.from_file('path/to/ebsd_scan.cif', min_grain_size=10)

   # Step 2 — Inspect the scan
   print(f"Scan dimensions: {ebsd.nx} x {ebsd.ny}")
   print(f"Number of grains: {ebsd.n_grains}")

   # Step 3 — Visualise
   ebsd.plot_grain_map()
   ebsd.plot_euler_maps()
   ebsd.plot_grain_size_histogram()

   # Step 4 — Re-characterise the grain-labelled image if needed
   # (e.g. after cropping to a region of interest)
   ebsd_cropped = ebsd.crop(region=(0, 200, 0, 200))
   ebsd_cropped.rechar_lfi(connectivity=4)

Building a texture profile from EBSD-measured orientations, for use in synthetic
generation, uses ``TextureComponentProfile`` (in ``upxo.material.texture``) — a
dataclass of ``crystal_family`` plus a ``component_fractions`` dict (component name
→ volume fraction), fitted from measured data rather than constructed by hand for
anything beyond a quick baseline. See ``src/upxo/demos/Twinned3D/repOFHCCu3d_ebsdVf.1.0.ipynb``
for the complete EBSD-to-synthetic-microstructure pipeline in context, including
how the fitted texture feeds into host-grain orientation assignment (Workflow 13).

----

Part 4: Meshing and Export
===========================

Workflow 15 — Conformal Tetrahedral Meshing (3D)
-------------------------------------------------

Conformal (grain-boundary-aligned) tetrahedral meshing is a **5-stage functional
pipeline** in ``upxo.meshing.confMesh3d`` — not a single mesher class. Each stage
is a plain function taking the previous stage's result.

.. code-block:: python

   from upxo.meshing.confMesh3d import (
       run_surface_nets, build_conformal_surface_complex,
       validate_surface_complex, fix_winding,
       generate_conformal_tet_mesh, export_conformal_mesh,
   )

   voxel_size = 1.0  # microns per voxel

   # Stage 1 — Extract the multi-label grain-boundary surface (marching-cubes-like)
   sn_result = run_surface_nets(lgi_3d, voxel_size)

   # Stage 2 — Build the shared-vertex conformal surface complex
   complex_ = build_conformal_surface_complex(sn_result)

   # Stage 3 — Validate (watertight, volume, bounds) and fix triangle winding
   report = validate_surface_complex(complex_, lgi_3d)
   print(report)
   complex_ = fix_winding(complex_)

   # Stage 4 — Generate the conformal tet mesh via gmsh
   # Must be called before gmsh.finalize()
   gmsh_result = generate_conformal_tet_mesh(complex_)

   # Stage 5 — Export to Abaqus .inp (+ optional meshio formats)
   # all_quats: {grain_id: quaternion(4,)} — per-grain crystal orientation
   export_conformal_mesh(gmsh_result, all_quats, voxel_size=voxel_size)

See ``src/upxo/demos/confMesh/confMesh3d1.ipynb`` (and the numbered notebooks
``confMesh3d2.ipynb`` through ``confMesh3d12.ipynb``, each covering a specific
aspect of the pipeline) for complete, runnable references.

----

Workflow 16 — Export FM Steel Structure to Abaqus
---------------------------------------------------

FM Steel mesh export uses ``MeshExporter3D`` (in ``upxo.pxtal.fm_steel_3d``),
operating on the state object produced by the pipeline in Workflow 12
(``fm_ori`` or ``fm_sub``), via element-type-specific ``export_c3d8`` /
``export_c3d4`` / ``export_c3d20`` / ``export_c3d10`` methods.

.. code-block:: python

   from upxo.pxtal.fm_steel_3d import MeshExporter3D

   exporter = MeshExporter3D(verbosity=1)

   # fm_state is the pipeline object from Workflow 12 (fm_ori or fm_sub)
   exporter.export_c3d8(fm_state, folder_name='fm_steel_export')

.. note::

   The exported ``07_interactions.inp`` and ``08_steps_output.inp`` files contain
   only placeholder ``** TODO`` content — real interaction (periodic BCs, contact)
   and ``*Step``/``*Output`` definitions require simulation-specific choices this
   exporter cannot infer, and calling this now emits a ``warnings.warn`` making that
   explicit. Edit those two files before submitting the job.

----

Workflow 17 — Export Twinned FCC Structure to Abaqus
-----------------------------------------------------

Twinned FCC export uses ``AbaqusExporter3D`` (in ``upxo.pxtal.twinned_simple_3d``).
``twin_role``, ``twin_parent_of``, and ``all_quats`` are all required — they are the
real outputs of the twin-generation pipeline in Workflow 13, not optional extras.

.. code-block:: python

   from upxo.pxtal.twinned_simple_3d.abaqus_exporter_3d import AbaqusExporter3D

   # lgi, twin_role, twin_parent_of, all_quats come from the Workflow 13 pipeline
   exporter = AbaqusExporter3D(
       lgi=lgi_with_twins,
       twin_role=twin_role,             # {gid: 'host' | 'primary_twin' | 'secondary_twin' | 'non_host'}
       twin_parent_of=twin_parent_of,   # {child_gid: parent_gid}
       all_quats=all_quats,             # {gid: quaternion(4,)}
       twinmake=twin_gen,                # the TwinGenerator3D instance, for variant ELSETs
       voxel_size_um=1.0,
   )
   exporter.write(out_dir='twinned_fcc_export')

.. note::

   ``es_variant_ptwin_<v>`` ELSETs currently group primary-twin grains by a
   round-robin placeholder, not each grain's real Sigma-3 {111} variant (that
   index is computed during generation but not yet persisted per grain) — do
   not assign variant-specific material behaviour from this grouping. As with
   Workflow 16, ``07_interactions.inp``/``08_steps_output.inp`` are placeholders;
   both limitations now emit a ``warnings.warn`` at export time.

See the notebooks listed in Workflow 13 for the complete pipeline through to export.

----

Part 5: Advanced Visualization
===============================

Workflow 18 — 3D Visualization with PyVista
---------------------------------------------

UPXO provides ready-made PyVista grid helpers in ``upxo.viz.gsviz`` for
visualising a 3D labelled grain image — use these rather than building a
PyVista grid by hand.

.. code-block:: python

   from upxo.viz import gsviz

   # Step 1 — Build a PyVista ImageData grid from the labelled grain image
   pvgrid = gsviz.make_pvgrid(lgi_3d, scalar_name='lgi')

   # Step 2 — Plot interactively
   gsviz.plot_pvgrid(
       pvgrid, scalar_name='lgi', show_edges=False,
       cmap='nipy_spectral', title='3D grain structure',
   )

   # Step 3 — Export to VTK format for ParaView, using PyVista directly
   pvgrid.save('microstructure.vti')

``gsviz`` also provides ``grain_viewer(lfi)`` for a quick interactive viewer,
and ``view_selected_grain_boundary_voxels(lfi, grain_ids, ...)`` to highlight
specific grains' boundary voxels.

----

Workflow 19 — Grain Boundary Visualization
--------------------------------------------

Highlight and visualize grain boundaries using edge detection.

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from scipy import ndimage

   # Assume lgi_3d is a 3D labelled grain image

   # Step 1 — Detect grain boundaries (edges where label changes)
   gb_edges = np.zeros_like(lgi_3d, dtype=bool)

   for i in range(lgi_3d.shape[0] - 1):
       gb_edges[i, :, :] |= (lgi_3d[i, :, :] != lgi_3d[i + 1, :, :])
   for j in range(lgi_3d.shape[1] - 1):
       gb_edges[:, j, :] |= (lgi_3d[:, j, :] != lgi_3d[:, j + 1, :])
   for k in range(lgi_3d.shape[2] - 1):
       gb_edges[:, :, k] |= (lgi_3d[:, :, k] != lgi_3d[:, :, k + 1])

   # Step 2 — Visualise slice with GB overlay
   slice_idx = lgi_3d.shape[0] // 2
   slice_data = lgi_3d[slice_idx, :, :]
   slice_gb = gb_edges[slice_idx, :, :]

   fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

   # Show grain IDs
   ax1.imshow(slice_data, cmap='tab20')
   ax1.set_title('Grain IDs')

   # Show grain boundaries
   ax2.imshow(slice_data, cmap='gray', alpha=0.5)
   ax2.imshow(slice_gb, cmap='Reds', alpha=0.7)
   ax2.set_title('Grain Boundaries')

   plt.tight_layout()
   plt.show()

   # Step 3 — Compute GB network statistics
   num_gb_voxels = np.sum(gb_edges)
   total_voxels = gb_edges.size
   gb_fraction = num_gb_voxels / total_voxels
   print(f"Grain boundary voxel fraction: {gb_fraction:.4f}")

----

Workflow 20 — Texture Visualization (Pole Figure)
---------------------------------------------------

Plot crystallographic pole figures directly from a 3D labelled grain image and
per-grain orientations, using the real ``plot_pole_figure_from_3d`` function
(in ``upxo.viz.xphy.pole_figure``).

.. code-block:: python

   import matplotlib.pyplot as plt
   from upxo.viz.xphy.pole_figure import plot_pole_figure_from_3d

   # quats_dict: {grain_id: quaternion(4,)} — e.g. from the Workflow 12/13 pipelines
   fig, ax = plot_pole_figure_from_3d(
       lgi_3d, quats_dict,
       axis=2,                 # slice normal: 0=X, 1=Y, 2=Z
       pole_family='111',      # or '100', '110', or an explicit (h, k, l) tuple
       plot_type='density',    # 'scatter' or 'density'
   )
   plt.show()

The same module's ``PoleFigure`` class, ``plot_components()``, and ``plot_variants()``
support multi-pole-figure layouts and KS/Sigma3-variant-coloured pole figures — see
``src/upxo/viz/xphy/pole_figure.py`` for the full function list, and
``src/upxo/demos/Twinned3D/repOFHCCu3d_ebsdVf.1.0.ipynb`` for pole figures used
alongside real EBSD-fitted texture.

----

Next Steps
==========

- :doc:`concepts` — understand the data model behind these examples
- `API Reference <https://design-by-fundamentals-ukaea.github.io/UPXO/>`_ — full module and class documentation
- `Grain Characterisation wiki <https://github.com/Design-By-Fundamentals-UKAEA/UPXO/wiki/Grain-Characterisation>`_ — extended characterisation workflows
- `Use Cases wiki <https://github.com/Design-By-Fundamentals-UKAEA/UPXO/wiki/Use-Cases>`_ — material-specific applications
- `Meshing wiki <https://github.com/Design-By-Fundamentals-UKAEA/UPXO/wiki/Meshing>`_ — detailed meshing documentation
