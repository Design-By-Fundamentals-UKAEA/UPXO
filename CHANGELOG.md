# CHANGELOG

All notable changes to UPXO are documented in this file.

## [1.2.0] — 2026-09-11

### Added

#### 2D Voronoi Tessellation Engine
- **`pxtal/voronoi_tessellation_2d/`**: Standalone high-fidelity 2D Voronoi geometry engine — periodic boundary conditions, Laguerre/power diagrams (weighted Voronoi), centroidal Voronoi tessellation (CVT) via Lloyd relaxation, and grain-boundary interface perturbation
- **`pxtal/vortess2d.gtess2d`**: Wired onto the new engine — `periodic`, `weights`, `cvt_iterations`, `perturb_factor` exposed as constructor keywords; new `from_mpoint2d` and working `from_shapely_mulpolygon` constructors; `bounds`/`info`/`seeds` properties; `__len__`
- **`pxtal/geotess.geotess2d`**: Completed previously-stubbed container/topology methods — `__getitem__`/`__setitem__`/`__repr__`, `make_seeds_random`/`make_seeds_pdisc` (Bridson sampling), first/second nearest-neighbour topology, boundary/internal grain filtering; `perturb_grain_boundaries` now delegates to the new engine
- **Tests**: `tests/pxtal/test_vortess2d.py` covering the engine, `gtess2d`, and `geotess2d` integration

#### Twinned FCC 3D Demo Automation
- **`pxtal/twinned_simple_3d/steps/`**: Automation-ready wrapper package for the Twinned FCC pipeline, mirroring `fm_steel_3d/steps/`
- **Demos**: `twinned_fcc_bas0.ipynb`/`bas1.ipynb` (minimal, annotated), `twinned_fcc_int0.ipynb`/`int1.ipynb` (full pipeline with validation/diagnostics), `twinned_fcc_adv0.ipynb`/`adv1.ipynb` (deeper tuning, mapping optimization, slice sweep) under `src/upxo/demos/Twinned3D/`

#### Conformal Meshing Extensions
- **`meshing/confMesh3d/`**: Optional surface remeshing stage (`surface_remesh.py`) and a TetGen tet-meshing backend (`tetgen_mesh.py`)

#### 2D conformal meshing (raw Gmsh)
- **`confMesh2dGMSH`**: Shared grain-boundary line tags, island/void interiors, Gmsh physical-group ELSETs, `from_geometric_pxtal`, safer `try`/`finally` Gmsh sessions, vectorised mesh extract
- **`viz.meshviz.plot_conformal_2d_by_grain`**: Grain-ELSET fill with optional GB overlay and face NSETs (used by `confMesh2dGMSH.plot_by_grain` and `gsmesh2d.visualize_gs_mesh`)
- **`writer_ABQ.export_confmesh2d_inp`**: Compact 1..N nodes; CPS3/CPS6/CPS4/CPS8 or CPE3/CPE6/CPE4/CPE8 from connectivity width; `GRAIN_*` ELSETs, `NS_LEFT`/`RIGHT`/`TOP`/`BOTTOM`/`GB` NSETs, optional dummy sections (`summarize_inp` helper)
- **2D mesh fidelity**: `prepare_grain_polygons` (snap / orient); Threshold `dist_min`/`dist_max` actually applied; optional Laplace2D optimize; clockwise elements reversed after extract; `fidelity_report` (mesh vs Shapely area/GB length) and `quality_report` (aspect ratio, min angle)
- **Demos** (force-tracked under `src/upxo/demos/confMesh/`): `confMesh2d_gmsh.ipynb` (canonical), `confMesh2d_export.ipynb` (plot + Abaqus INP), `confMesh2d_mcgs.ipynb` (MCGS teaching path), `confMesh2d_geometrify.ipynb` (polygonise then `mesh_gs`), `confMesh2d_technique_b.ipynb` (Technique B then `mesh_gs`); capability notebooks `confMesh2d_tri_vs_quad`, `confMesh2d_islands_voids`, `confMesh2d_voronoi`, `confMesh2d_size_field`, `confMesh2d_quadratic`, `confMesh2d_quality`, `confMesh2d_nsets`
- **Tests**: `tests/meshing/test_confmesh2d_gmsh.py`

#### Centralized Reporting
- **`reporting/`**: Session/entries/HTML-rendering core for building structured pipeline reports
- **`pxtal/fm_steel_3d/raw_export.py`**: Raw pipeline-state export across PAG/Packet/Block/Sub-block hierarchy levels
- Reporting integration wired into `fm_steel_3d` and `twinned_simple_3d`

#### Self-Representativeness Assessment (twinned_simple_3d)
- `selfrepr_morphology.py`: Per-grain morphological parameters for EBSD self-representativeness studies
- `representativeness_metrics.py`: Registry of two-sample distribution-comparison similarity metrics
- `selfrepr_qualification.py`: Threshold-based qualification of representativeness results
- `subsetting_2d.py`: 2D rectangular sub-domain extraction (sibling of the existing 3D `subsetting.py`)
- `stride_study.py`: Moving-window stride/tiling study support
- `mc_qualification.py`: Monte-Carlo candidate qualification
- `base_3d.percentile_trim`: Percentile-of-range outlier trimming, alongside the existing IQR-based trim
- `crystal_orientation.compute_grain_csl_participation`: Per-grain parent/twin role tally across all CSL types

#### FM Steel Packet-Level Tooling
- `orientation_mean_3d.py`: Crystallographically-correct packet mean orientation
- `slice_metrics_2d.py`: Fast numpy-only 2D slice geometric metrics

#### FM Steel 3D Demo Automation
- **`pxtal/fm_steel_3d/steps/`**: New automation-ready wrapper package, one thin module per pipeline stage (start, base grain structure, cleaning, transformations, PAG clustering, block generation, sub-block generation, visualization, raw export, mesh export, ensemble seed config); lives inside the installed package (unlike Twinned3D's own `steps/`, which sits under `demos/`) so it is directly importable by automation scripts, not just notebooks
- `geom_metrics_3d.feature_aspect_ratio_bbox`: Free-function per-feature 3D bounding-box aspect ratio, generalizing the existing 2D/instance-bound versions
- `viz/grain_structure_viz_3d.GrainStructureViz3D.build_distinguishable_cmap`: Shuffled large-N colormap so per-feature ID coloring stays distinguishable past `tab20`'s 20-color limit
- `viz/grain_structure_viz_3d.GrainStructureViz3D.lfi_to_polydata`: New `jupyter_backend` param to embed an interactive PyVista widget (e.g. `'trame'`) in a notebook cell instead of a blocking native window
- **Demos**: `fm_steel_3d_bas0.ipynb` (concise, full pipeline through mesh export/ensemble config), `fm_steel_3d_bas1.ipynb` (annotated, + Quick-Demo Mode toggle) under `src/upxo/demos/FMSteel3D/`

#### Visualization
- Side-by-side voxel-grid comparison render
- IPF triangle key plot and side-by-side LFI comparison render
- Pole-figure size-colored scatter overlay with its own colorbar
- Histogram-with-slice-band plotting (`vizDistr.plot_hist_with_slice_band`)
- Per-axes property-distribution plotting (`vizDistr.plot_property_distribution_on_axes`)
- Grain-role-aware EBSD visualization for CSL parent/twin analysis

#### Other Core Additions
- `charops.boundary_grain_fraction`: Fraction of grains touching the domain boundary, for 2D labelled images
- `gsdataops/grid_ops`: Majority-vote-safe downsampling and anisotropic stretch
- `pxtal/gridops`: Raw per-crossing lengths exposed from `axis_intercept_grain_size`

#### Testing
- Reporting integration tests (fm_steel_3d, twinned_simple_3d)
- FM steel raw-export and packet-orientation-mean tests
- GB-conformant meshing (`meshing/gbconformant/d3v2p0`) test suite

### Changed

- **Twinned FCC orientation assignment**: `max_retries` exposed; MDF bin width/angle range now derived from `mdf_ref` instead of a hardcoded 65°; `neighbour_frac` added to Pool B host allocation with do/do-not fallback; MRF Gibbs/MAP initialization reuses `self.fallback_quats` instead of drawing a fresh unseeded sample
- **`xtalphy/texops.py`**: Fixed the same Bunge ZXZ Euler-decomposition bug present in `crystal_orientation.matrix_to_euler_bunge`; added texture-component detection; ported GUI-only quaternion helpers into core
- **Packaging**: `requirements.txt` enables `tetgen`; `setup.py`/`pyproject.toml` exclude the local-only `upxo.gui` package from the built distribution; optional extra `[mesh]` now also installs `gmsh>=4.13`
- **GUI applications removed from version control**: `upxo_gui_launcher.py` and the GUI test suites are untracked (kept local-only); all GUI source under `src/upxo/gui/`, `fm_steel_3d/gui/`, and `twinned_simple_3d/gui/` remains local-only and out of this release
- **`confMesh2d` (pygmsh)** is deprecated; new 2D conformal work must use `confMesh2dGMSH` / `gsmesh2d.mesh_gs`
- Removed redundant tracked demos `confMesh2d9.ipynb` and `mesh2d01.ipynb`
- The voxel-lattice cleaving mesh generator (`meshing/cleaving/`) is untracked pending relocation to `meshing/gbconformant/cleaving3dV1P0/` (WIP, not part of this release)

### Not Included (Deferred)

- MC Metropolis-acceptance refinement (permanently out of scope per architecture decision)
- GB-energy/crystallography coupling for block selection (deferred to future versions)
- Orientation-mode GUI exposure in Twinned FCC (deferred for UX refinement)
- Multi-CSL twin registry beyond Sigma-3 Path A (CSL Path B deferred post-Path A validation)
- Cleaving mesh generator migration to `meshing/gbconformant/cleaving3dV1P0/` (WIP, untracked)

---

## [1.1.0] — 2026-08-04

**First official PyPI release of UPXO.**

### Added

#### GUI Applications
- **FM Steel GUI**: Interactive Tkinter/CustomTkinter wizard for hierarchical Ferritic-Martensitic steel microstructure generation
  - PAG generation (Voronoi or Monte Carlo)
  - PAG clustering and packet subdivision
  - Block and sub-block (lath) generation with configurable thickness ranges
  - KS (Kurdjumov–Sachs) variant assignment (random, adjacency-aware, or deterministic)
  - Retained austenite modeling
  - Texture-guided PAG orientation assignment
  - Real-time visualization and feasibility validation
  - Abaqus `.inp` and VTK export

- **Twinned FCC GUI**: Interactive wizard for twinned grain generation in Cu, CuCrZr, OFHC-Cu
  - Host grain allocation with spatial-dispersal-aware MIS algorithm
  - Texture-guided orientation assignment via EBSD ODF
  - EBSD microstructure import (CIF, HDF5 formats)
  - Sigma-3 twin lamella embedding with configurable density and thickness
  - Twin artifact removal and cleaning
  - Real-time visualization of grain and twin distributions
  - Abaqus and VTK export

#### Core Pipeline Modules
- **`pxtal/fm_steel_3d/`**: Hierarchical microstructure generation for FM steels
  - `base_3d.py`: Core FMSteelGenerator with PAG generation and clustering
  - `orientation_3d.py`: Texture-guided crystallographic orientation assignment
  - `cleaning_3d.py`: Twin and artifact removal
  - `feature_props_3d.py`: Role-stratified property queries (martensite/austenite)
  - `repr_validator_3d.py`: Post-generation validation metrics
  - `viz_3d.py`: Role-colored visualization and refinement helpers

- **`pxtal/twinned_simple_3d/`**: Twinned FCC microstructure generation
  - `base_3d.py`: Core TwinnedSimple3DBase with spatial-dispersal host allocation
  - `orientation_3d.py`: Texture-guided orientation with Rodrigues calculations
  - `twin_generator_3d.py`: Multi-path Sigma-3 twin embedding
  - `feature_props_3d.py`: Twin and matrix property queries
  - `cleaning_3d.py`: Twin-specific artifact removal
  - `repr_validator_3d.py`: Post-twin validation metrics
  - `viz_3d.py`: Twin-colored visualization

- **Conformal Meshing (`meshing/confMesh3d/`)**: Five-stage pipeline for grain-boundary-aligned tetrahedral meshing
  - Surface extraction via marching cubes
  - Complex builder for grain boundary geometry
  - Mesh validation and quality checks
  - Volume mesh generation via gmsh
  - Abaqus and VTK export

#### Material Module Refactoring
- **`material/`**: Type-safe material registry and properties
  - `identity.py`: MaterialIdentity with name, alloy, composition fields
  - `processing.py`: ProcessingRoute and ProcessingStep with deformation type classification
  - `provenance.py`: Provenance tracking (source, method, parameters, timestamp)
  - `registry.py`: MaterialRegistry with typed instance storage and soft validation
  - `texture.py`: TextureComponentProfile for crystal-family-generic texture modeling

#### Operations Helpers (New)
- **`gbops/gbpoint_ops.py`**: Grain-boundary point extraction and analysis
- **`netops/neighops.py`**: Neighbor graph connectivity utilities
- **`propOps/morphops.py`**: Morphological property helpers (volumes, binning)
- **`imageOps/labelops.py`**: Label reindexing and manipulation
- **`fdbOps/fdbops.py`**: Feature database entry helpers

#### EBSD Integration
- Support for EBSD data import (CIF, HDF5 via DefDAP)
- Texture ODF (Orientation Density Function) profiling from EBSD scans
- EBSD-guided synthetic microstructure generation
- Crystal orientation validation against experimental data

#### Documentation & Examples
- Comprehensive wiki (18+ pages) covering all capabilities
- Detailed workflow examples (20 workflows covering 2D/3D generation, meshing, visualization)
- Updated README with all new capabilities highlighted
- Updated Sphinx documentation with GUI application exposure
- Demo notebooks for FM Steel and Twinned FCC pipelines

#### Testing
- 12 new test modules with 1,280 lines of test code
- Material registry tests (initialization, ingestion, provenance, validation)
- Operations helper tests (gb_ops, net_ops, prop_ops, image_ops, fdb_ops)
- Visualization smoke tests (ebsdviz, vizDistr)
- Grain structure and Voronoi constructor tests
- 82/87 tests passing (93.1%)

### Changed

- **`pxtalops/twin3d.py`**: Added optional `host_coords` parameter to `introduce_twin_lamella_3d()` for coordinate caching optimization
- **README.md**: Expanded Core Capabilities and Microstructures Supported sections; fixed typos
- **Sphinx documentation**: Updated introduction and getting_started sections with GUI application exposure
- **Workflows documentation**: Expanded from 8 to 20 workflows covering all major capabilities

### Fixed

- Typos in README: "pertaining multi-scale" → "pertaining to multi-scale", "teknology" → "technology", "powerpoint" → "power plant", "visibility" → "viability"
- Test suite: Resolved 26 initial test failures through systematic debugging and API alignment

### Infrastructure

- Added `.gitignore` entries for legacy GUI folder and generated output directories
- Organized data, sessions, and generated output structures
- Sphinx build workflow configured for automated API documentation generation

### Not Included (Deferred)

- MC Metropolis-acceptance refinement (permanently out of scope per architecture decision)
- GB-energy/crystallography coupling for block selection (deferred to future versions)
- Orientation-mode GUI exposure in Twinned FCC (deferred for UX refinement)
- Multi-CSL twin registry beyond Sigma-3 Path A (CSL Path B deferred post-Path A validation)

---

## [1.0.0] — Development Only (Never Published)

Initial development version. Features and APIs evolved significantly before first PyPI release.
