Workflows
=========

.. contents:: On this page
   :local:
   :depth: 2

This page shows complete, annotated code examples for common UPXO tasks covering all major capabilities:
2D and 3D grain generation (MCGS, Voronoi), hierarchical microstructures (FM Steels), twinned microstructures (FCC),
EBSD integration, meshing, and visualization.

All examples follow patterns taken directly from the demo notebooks in ``src/upxo/demos/`` and the module documentation.

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

Generate a 3D grain structure using the Potts model Monte Carlo algorithm,
equivalent to ``gschar2.ipynb``.

.. code-block:: python

   from upxo.ggrowth.make3d import make_mcgs_3d

   # Step 1 — Create a 3D MCGS object with specified parameters
   mc3d = make_mcgs_3d(
       NX=100,              # voxels in X
       NY=100,              # voxels in Y
       NZ=100,              # voxels in Z
       Q=100,               # number of grain orientations
       kT=0.5,              # temperature
       num_mcs=50,          # Monte Carlo steps
       random_seed=42,
       sampling_interval=10 # save state every 10 MCS
   )

   # Step 2 — Run the simulation
   mc3d.simulate()

   # Step 3 — Detect grains in the final state
   mc3d.detect_grains()

   # Step 4 — Access the labelled grain image
   lgi_3d = mc3d.lgi  # 3D array of grain IDs

   # Step 5 — Basic characterisation
   print(f"Domain size: {mc3d.lgi.shape}")
   print(f"Number of grains detected: {len(np.unique(mc3d.lgi)) - 1}")  # exclude background (0)

   # Step 6 — Extract a 2D slice for visualization
   slice_z = mc3d.lgi[:, :, 50]  # mid-slice in Z
   plt.figure()
   plt.imshow(slice_z, cmap='tab20')
   plt.title('3D MCGS — XY slice at Z=50')
   plt.colorbar(label='Grain ID')
   plt.show()

----

Workflow 10 — 3D Voronoi Tessellation
--------------------------------------

Generate a 3D grain structure using Voronoi tessellation from seed points.

.. code-block:: python

   import numpy as np
   from upxo.pxtal.vortess3d import Voronoi3D

   # Step 1 — Define domain
   NX, NY, NZ = 100, 100, 100
   num_seeds = 50

   # Step 2 — Create Voronoi structure
   vor3d = Voronoi3D(
       NX=NX, NY=NY, NZ=NZ,
       num_seeds=num_seeds,
       random_seed=42,
       connectivity='26-connectivity'  # or '6-connectivity' for simpler adjacency
   )

   # Step 3 — Generate tessellation
   lgi_3d = vor3d.lgi  # 3D labelled grain image

   # Step 4 — Characterise grain volumes
   unique_ids = np.unique(lgi_3d)
   grain_volumes = np.array([np.sum(lgi_3d == gid) for gid in unique_ids if gid != 0])

   print(f"Number of grains: {len(grain_volumes)}")
   print(f"Mean grain volume: {grain_volumes.mean():.1f} voxels")
   print(f"Volume std dev: {grain_volumes.std():.1f} voxels")

   # Step 5 — Visualise a slice
   slice_z = lgi_3d[:, :, NZ//2]
   plt.figure()
   plt.imshow(slice_z, cmap='tab20')
   plt.title(f'3D Voronoi — XY slice at Z={NZ//2}')
   plt.colorbar(label='Grain ID')
   plt.show()

----

Workflow 11 — 3D Grain Characterisation
----------------------------------------

Compute morphological properties of 3D grains.

.. code-block:: python

   import numpy as np
   from scipy import ndimage

   lgi_3d = mc3d.lgi  # or vor3d.lgi

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

Generate a hierarchical lath-based microstructure for FM steels (Eurofer, F82H, T91).
This workflow covers prior austenite grain (PAG) generation, packet formation, block/sub-block subdivision,
and KS variant assignment.

.. code-block:: python

   import numpy as np
   from upxo.pxtal.fm_steel_3d import FMSteelGenerator

   # Step 1 — Initialize the generator with domain and PAG parameters
   fm_gen = FMSteelGenerator(
       NX=150, NY=150, NZ=150,        # domain size (voxels)
       base_grain_method='voronoi',   # or 'monte_carlo'
       num_pags=30,                   # number of prior austenite grains
       retained_austenite_vf=0.05,    # retained austenite volume fraction
       random_seed=42
   )

   # Step 2 — Generate PAGs (prior austenite grains)
   lgi_pag = fm_gen.generate_pags()
   print(f"Generated {np.unique(lgi_pag).max()} PAGs")

   # Step 3 — Assign PAG orientations
   # Option A: Random orientations
   fm_gen.assign_pag_orientations(method='random')

   # Option B: Texture-guided (rolling texture with spreads)
   # from upxo.material.texture import TextureComponentProfile
   # texture = TextureComponentProfile(
   #     components=['Copper', 'Brass', 'S'],
   #     spreads=[15.0, 15.0, 15.0]  # spread in degrees
   # )
   # fm_gen.assign_pag_orientations(method='texture', texture=texture)

   # Step 4 — Perform PAG clustering (optional: group PAGs for packet generation)
   lgi_clustered = fm_gen.cluster_pags(max_packets_per_pag=4, method='bfs')

   # Step 5 — Generate hierarchical structure (packets → blocks → sub-blocks)
   lgi_hierarchical = fm_gen.generate_hierarchical_structure(
       block_thickness_range=(2.0, 5.0),      # µm
       subblock_thickness_range=(0.5, 1.5),   # µm
       enable_subblocks=True,
       ks_variant_mode='random_adjacency_aware'  # avoid same variant in neighbours
   )

   # Step 6 — Inspect the result
   print(f"Total voxels: {lgi_hierarchical.size}")
   unique_ids = np.unique(lgi_hierarchical)
   print(f"Total features (all hierarchy levels): {len(unique_ids) - 1}")

   # Step 7 — Extract role information (which phase, which hierarchy level)
   roles = fm_gen.get_feature_roles(lgi_hierarchical)
   # roles dict: 'martensite' | 'austenite', and sub-keys for hierarchy level
   print(f"Martensite volume fraction: {roles['martensite']['total_vf']:.3f}")
   print(f"Austenite volume fraction: {roles['austenite']['total_vf']:.3f}")

   # Step 8 — Visualise a slice
   slice_z = lgi_hierarchical[:, :, 75]
   plt.figure(figsize=(10, 10))
   plt.imshow(slice_z, cmap='nipy_spectral')
   plt.title('FM Steel Hierarchical Microstructure (XY slice)')
   plt.colorbar(label='Feature ID')
   plt.show()

**Key Parameters for FM Steel Generation:**

- ``block_thickness_range``: Controls lath thickness (µm). Typical values: 0.5–5 µm.
- ``subblock_thickness_range``: Finer subdivision within blocks. Typical: 0.1–1 µm.
- ``ks_variant_mode``: ``'random'`` for random KS variants, ``'deterministic'`` for reproducibility, ``'random_adjacency_aware'`` to avoid same variant in adjacent blocks.
- ``retained_austenite_vf``: Fraction of PAGs left untransformed (0.0–1.0). Typical: 0.0–0.15.

----

Workflow 13 — Twinned FCC Microstructure Generation
-----------------------------------------------------

Generate a microstructure with Sigma-3 twin lamellae in FCC materials (Cu, CuCrZr, OFHC-Cu).
This workflow covers host grain allocation, texture-guided orientation, twin embedding, and cleaning.

.. code-block:: python

   import numpy as np
   from upxo.pxtal.twinned_simple_3d import TwinnedSimple3DBase

   # Step 1 — Generate base grain structure (host grains)
   from upxo.pxtal.vortess3d import Voronoi3D

   vor3d = Voronoi3D(
       NX=100, NY=100, NZ=100,
       num_seeds=40,
       random_seed=42
   )
   lgi_hosts = vor3d.lgi

   # Step 2 — Initialize twinned structure generator
   twin_gen = TwinnedSimple3DBase(
       lgi=lgi_hosts,
       material='OFHC-Cu',  # or 'CuCrZr'
       random_seed=42
   )

   # Step 3 — Assign orientations to host grains
   # Option A: Random orientations
   twin_gen.assign_orientations(method='random')

   # Option B: Texture-guided using EBSD ODF
   # from upxo.material.texture import TextureComponentProfile
   # texture = TextureComponentProfile(
   #     components=['Copper', 'Brass'],
   #     spreads=[20.0, 20.0]
   # )
   # twin_gen.assign_orientations(method='texture', texture_profile=texture)

   # Step 4 — Allocate twin host grains
   # Spatial-dispersal-aware: select non-adjacent grains using MIS algorithm
   host_gids = twin_gen.allocate_twin_hosts(
       method='spatial_dispersal',  # or 'random'
       target_vf=0.3,               # aim for 30% of grains as hosts
       use_mis=True                 # Maximal Independent Set for non-adjacency
   )
   print(f"Selected {len(host_gids)} grains as twin hosts")

   # Step 5 — Pre-compute host coordinates (performance optimization)
   host_coords_map = {}
   for gid in host_gids:
       host_coords = np.argwhere(lgi_hosts == gid)
       host_coords_map[gid] = host_coords

   # Step 6 — Generate twin lamellae
   lgi_with_twins = lgi_hosts.copy()
   twin_count = 0

   for host_gid in host_gids:
       # Generate 1-3 Sigma-3 twins per host
       num_twins = np.random.randint(1, 4)
       for _ in range(num_twins):
           twin_coords = twin_gen.introduce_twin_lamella_3d(
               lgi=lgi_with_twins,
               host_gid=host_gid,
               next_gid=int(np.unique(lgi_with_twins).max()) + 1,
               abrupt=False,  # smooth twin boundaries
               host_coords=host_coords_map[host_gid]  # use pre-computed coords
           )
           if twin_coords is not None:
               twin_count += 1

   print(f"Successfully embedded {twin_count} twin lamellae")

   # Step 7 — Clean up artifacts (remove sub-threshold twins)
   from upxo.pxtal.twinned_simple_3d import Cleaner3D

   cleaner = Cleaner3D(lgi_with_twins)
   lgi_clean = cleaner.remove_subthreshold_twins(
       thickness_threshold=2,  # voxels
       minimum_aspect_ratio=0.1
   )

   # Step 8 — Compute twin statistics
   unique_ids = np.unique(lgi_clean)
   print(f"Final grain count: {len(unique_ids) - 1}")  # exclude background
   twin_volume_fraction = np.sum(lgi_clean > lgi_hosts.max()) / lgi_clean.size
   print(f"Twin volume fraction: {twin_volume_fraction:.3f}")

   # Step 9 — Visualise
   slice_z = lgi_clean[:, :, 50]
   plt.figure(figsize=(10, 10))
   plt.imshow(slice_z, cmap='tab20')
   plt.title('Twinned FCC Microstructure (XY slice)')
   plt.colorbar(label='Grain/Twin ID')
   plt.show()

**Key Parameters for Twinned Generation:**

- ``method``: ``'spatial_dispersal'`` uses MIS for non-adjacent host placement (recommended); ``'random'`` places hosts randomly.
- ``target_vf``: Target volume fraction of grains to use as twin hosts (0.0–1.0).
- ``use_mis``: Enable Maximal Independent Set algorithm for spatial awareness.
- ``thickness_threshold``: Minimum thickness for twins to survive cleaning (voxels).
- ``num_variants``: Number of Sigma-3 variants to embed per host (typically 1–3).

----

Workflow 14 — EBSD-Guided Microstructure Generation
-----------------------------------------------------

Import experimental EBSD data and use it to guide synthetic microstructure generation.

.. code-block:: python

   import numpy as np
   from upxo.interfaces.defdap import EBSDReader

   # Step 1 — Load EBSD data from file
   # Supported formats: .cif (Channel), .hdf5 (EDAX/TSL)
   ebsd_reader = EBSDReader(filepath='path/to/ebsd_scan.cif')
   ebsd_data = ebsd_reader.read()

   # ebsd_data contains:
   #  - orientations: (N, 3) array of Euler angles or quaternions
   #  - grain_map: (H, W) labelled grain ID map
   #  - x, y: pixel coordinates
   #  - phase_map: (H, W) phase IDs (if multi-phase)

   print(f"EBSD scan dimensions: {ebsd_data['grain_map'].shape}")
   print(f"Number of grains in scan: {len(np.unique(ebsd_data['grain_map']))}")

   # Step 2 — Build texture profile from EBSD data
   from upxo.material.texture import TextureComponentProfile

   # Extract Euler angles and compute ODF
   euler_angles = ebsd_data['orientations']
   texture = TextureComponentProfile.from_euler_angles(
       euler_angles,
       num_components=3,  # fit 3 texture components
       resolution=5  # degree resolution
   )

   # Step 3 — Generate synthetic microstructure guided by EBSD texture
   from upxo.pxtal.vortess3d import Voronoi3D

   vor3d = Voronoi3D(
       NX=100, NY=100, NZ=100,
       num_seeds=50,  # aim for similar grain count to EBSD
       random_seed=42
   )
   lgi_synthetic = vor3d.lgi

   # Step 4 — Assign orientations using EBSD texture
   from upxo.pxtal.twinned_simple_3d import OrientationAssigner3D

   ori_assigner = OrientationAssigner3D()
   orientations = ori_assigner.assign_from_texture(
       num_grains=np.unique(lgi_synthetic).max(),
       texture_profile=texture,
       method='odf'  # use Orientation Density Function
   )

   # Step 5 — Optionally add twinning based on EBSD twin observations
   # (Detect twins in EBSD and embed similar density in synthetic structure)
   twin_fraction_in_ebsd = 0.25  # example
   print(f"EBSD twin fraction: {twin_fraction_in_ebsd:.2%}")

   # Generate twinned structure with matching density
   # ... (see Workflow 13 for twinning steps)

   # Step 6 — Export merged structure for FE simulation
   # (see Workflow 16 below)

----

Part 4: Meshing and Export
===========================

Workflow 15 — Conformal Tetrahedral Meshing (3D)
-------------------------------------------------

Generate a conformal (grain-boundary-aligned) tetrahedral mesh suitable for FE simulation.
This workflow uses gmsh via a conforming surface extraction pipeline.

.. code-block:: python

   import numpy as np
   from upxo.meshing.confMesh3d import ConformalMesher3D
   from upxo.pxtal.vortess3d import Voronoi3D

   # Step 1 — Generate or load 3D grain structure
   vor3d = Voronoi3D(NX=80, NY=80, NZ=80, num_seeds=30, random_seed=42)
   lgi = vor3d.lgi

   # Step 2 — Initialize conformal mesher
   mesher = ConformalMesher3D(
       lgi=lgi,
       voxel_size=1.0,  # µm per voxel
       mesh_target_size=2.0  # target element edge length (µm)
   )

   # Step 3 — Extract grain boundary surfaces
   surfaces = mesher.extract_surfaces()
   print(f"Extracted {len(surfaces)} grain boundary surfaces")

   # Step 4 — Build conformal mesh
   mesh = mesher.build_mesh(
       use_gmsh=True,
       gmsh_exe_path=None,  # auto-detect gmsh
       refinement_levels=1,
       mesh_algorithm='frontal'  # or 'delaunay'
   )

   # Step 5 — Inspect mesh statistics
   num_nodes = mesh.n_points
   num_elements = mesh.n_cells
   print(f"Mesh statistics:")
   print(f"  Nodes: {num_nodes}")
   print(f"  Elements: {num_elements}")
   print(f"  Aspect ratio range: {mesh.cell_data.get('aspect_ratio', [1]).min():.2f} — {mesh.cell_data.get('aspect_ratio', [1]).max():.2f}")

   # Step 6 — Validate mesh quality
   quality_metrics = mesher.validate_mesh(mesh)
   print(f"Mesh quality:")
   for metric, value in quality_metrics.items():
       print(f"  {metric}: {value:.3f}")

   # Step 7 — Export to Abaqus format
   mesh.write('microstructure_mesh.inp')
   print("Exported to microstructure_mesh.inp")

   # Step 8 — Optional: Visualise mesh
   import pyvista
   mesh.plot()

----

Workflow 16 — Export FM Steel Structure to Abaqus
---------------------------------------------------

Export a hierarchical FM steel microstructure to Abaqus input file format,
with element sets for each phase/hierarchy level.

.. code-block:: python

   import numpy as np
   from upxo.pxtal.fm_steel_3d import FMSteelExporter

   # Assume lgi_hierarchical from Workflow 12
   # lgi_hierarchical is the labelled grain image with hierarchy encoded in ID ranges

   # Step 1 — Initialize exporter
   exporter = FMSteelExporter(
       lgi=lgi_hierarchical,
       voxel_size=1.0,  # µm
       material_name='Eurofer97'
   )

   # Step 2 — Define element sets for different phases
   element_sets = {
       'MARTENSITE': exporter.get_elements_by_phase('martensite'),
       'AUSTENITE': exporter.get_elements_by_phase('austenite'),
       'BLOCK_REGIONS': exporter.get_elements_by_hierarchy_level('block'),
       'LATH_REGIONS': exporter.get_elements_by_hierarchy_level('lath')
   }

   # Step 3 — Define material properties
   material_props = {
       'martensite': {
           'E': 210e3,      # MPa
           'nu': 0.3,
           'rho': 7.85e-9,  # kg/mm³
           'alpha': 12e-6   # /K
       },
       'austenite': {
           'E': 200e3,
           'nu': 0.31,
           'rho': 7.85e-9,
           'alpha': 15e-6
       }
   }

   # Step 4 — Build mesh
   from upxo.meshing.confMesh3d import ConformalMesher3D

   mesher = ConformalMesher3D(
       lgi=lgi_hierarchical,
       voxel_size=1.0,
       mesh_target_size=2.0
   )
   mesh = mesher.build_mesh(use_gmsh=True)

   # Step 5 — Write Abaqus input file
   exporter.write_abaqus_inp(
       filepath='fm_steel_hierarchical.inp',
       mesh=mesh,
       element_sets=element_sets,
       material_properties=material_props,
       include_orientation_data=True  # embed crystal orientations in ABAQUS ORINAME cards
   )
   print("Exported: fm_steel_hierarchical.inp")

   # Step 6 — Optional: Create a separate file for orientation data
   exporter.write_orientation_file(
       filepath='fm_steel_orientations.txt',
       format='rodrigues'  # or 'euler', 'quaternion'
   )

----

Workflow 17 — Export Twinned FCC Structure to Abaqus
-----------------------------------------------------

Export a twinned FCC microstructure to Abaqus with twin and matrix element sets.

.. code-block:: python

   import numpy as np
   from upxo.pxtal.twinned_simple_3d import TwinExporter3D

   # Assume lgi_with_twins from Workflow 13

   # Step 1 — Initialize exporter
   exporter = TwinExporter3D(
       lgi=lgi_with_twins,
       lgi_hosts=lgi_hosts,  # original host grain image
       voxel_size=1.0,
       material_name='OFHC-Cu'
   )

   # Step 2 — Classify elements by role
   element_sets = {
       'MATRIX': exporter.get_matrix_elements(),
       'TWINS': exporter.get_twin_elements(),
       'INTERFACES': exporter.get_twin_interface_elements(interface_width=2)  # voxels
   }

   # Step 3 — Get twin statistics
   twin_stats = exporter.get_twin_statistics()
   print(f"Twin statistics:")
   print(f"  Total twins: {twin_stats['num_twins']}")
   print(f"  Avg thickness: {twin_stats['mean_thickness']:.2f} voxels")
   print(f"  Volume fraction: {twin_stats['volume_fraction']:.3f}")

   # Step 4 — Build mesh
   from upxo.meshing.confMesh3d import ConformalMesher3D

   mesher = ConformalMesher3D(
       lgi=lgi_with_twins,
       voxel_size=1.0,
       mesh_target_size=2.0
   )
   mesh = mesher.build_mesh(use_gmsh=True)

   # Step 5 — Write Abaqus input file
   material_props = {
       'matrix': {'E': 130e3, 'nu': 0.34, 'rho': 8.96e-9},
       'twin': {'E': 130e3, 'nu': 0.34, 'rho': 8.96e-9},  # same for Cu twins
       'interface': {'K_normal': 1e5, 'K_shear': 1e4}  # cohesive zone
   }

   exporter.write_abaqus_inp(
       filepath='twinned_fcc.inp',
       mesh=mesh,
       element_sets=element_sets,
       material_properties=material_props,
       include_orientation_data=True
   )
   print("Exported: twinned_fcc.inp")

----

Part 5: Advanced Visualization
===============================

Workflow 18 — 3D Visualization with PyVista
---------------------------------------------

Create interactive 3D visualizations of grain structures using PyVista (VTK backend).

.. code-block:: python

   import numpy as np
   import pyvista as pv

   # Assume lgi_3d is a 3D labelled grain image

   # Step 1 — Convert voxel array to PyVista mesh
   mesh = pv.voxelize(lgi_3d, scalars=lgi_3d.flatten())

   # Step 2 — Create interactive plot
   plotter = pv.Plotter(shape=(1, 2))

   # Left plot: Render by grain ID with colourmap
   plotter.subplot(0, 0)
   plotter.add_mesh(mesh, scalars=lgi_3d.flatten(), cmap='nipy_spectral', show_edges=False)
   plotter.set_title('Grain IDs')

   # Right plot: Render subset of grains
   grain_subset = (lgi_3d > 0) & (lgi_3d <= 20)  # first 20 grains only
   mesh_subset = pv.voxelize(lgi_3d * grain_subset)
   plotter.subplot(0, 1)
   plotter.add_mesh(mesh_subset, scalars='scalars', cmap='tab20')
   plotter.set_title('Grain subset (1–20)')

   plotter.show()

   # Step 3 — Export to VTK format for ParaView
   mesh.save('microstructure.vti')  # VTK ImageData format
   mesh.save('microstructure.vtu')  # VTK UnstructuredGrid format

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

Plot crystallographic pole figures to visualize texture/orientation distribution.

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from mpl_toolkits.mplot3d import Axes3D

   # Assume orientations is an (N, 3) array of Euler angles (degrees)
   # or an (N, 4) array of quaternions

   from upxo.material.texture import pole_figure

   # Step 1 — Plot (100), (110), (111) pole figures
   fig = plt.figure(figsize=(15, 5))

   hkl_list = [(1, 0, 0), (1, 1, 0), (1, 1, 1)]

   for idx, hkl in enumerate(hkl_list):
       ax = fig.add_subplot(1, 3, idx + 1, projection='stereographic')
       pole_figure(
           orientations=orientations,
           hkl=hkl,
           ax=ax,
           contours=True,
           num_contours=10
       )
       ax.set_title(f'{hkl[0]}{hkl[1]}{hkl[2]} Pole Figure')

   plt.tight_layout()
   plt.show()

   # Step 2 — Compute texture strength (Orientation Density Function peak)
   from upxo.material.texture import compute_odf

   odf = compute_odf(orientations, resolution=5)
   max_odf = odf.max()
   random_odf = 1.0 / odf.size
   texture_strength = max_odf / random_odf

   print(f"Texture strength (MRD): {texture_strength:.2f}")
   print(f"  Random texture = 1.0 MRD")
   print(f"  This sample: {texture_strength:.2f}× random")

----

Next Steps
==========

- :doc:`concepts` — understand the data model behind these examples
- `API Reference <https://design-by-fundamentals-ukaea.github.io/UPXO/>`_ — full module and class documentation
- `Grain Characterisation wiki <https://github.com/Design-By-Fundamentals-UKAEA/UPXO/wiki/Grain-Characterisation>`_ — extended characterisation workflows
- `Use Cases wiki <https://github.com/Design-By-Fundamentals-UKAEA/UPXO/wiki/Use-Cases>`_ — material-specific applications
- `Meshing wiki <https://github.com/Design-By-Fundamentals-UKAEA/UPXO/wiki/Meshing>`_ — detailed meshing documentation
