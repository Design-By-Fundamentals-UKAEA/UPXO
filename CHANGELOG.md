# CHANGELOG

All notable changes to UPXO are documented in this file.

## [Unreleased]

### Added

- **EBSD → smoothed mesh → Abaqus (2D)**: end-to-end pipeline from a real EBSD `.ctf` map to a conformal 2D mesh with per-grain, EBSD-measured Bunge-Euler orientations in the exported Abaqus `.inp`.
  - `interfaces/defdap/ebsd_reader.py`: `EBSDReader.split_disconnected_grains(connectivity=4)` — a grain id spread across spatially disconnected pixel regions (most commonly from `crop()` slicing an irregular grain in two) is relabelled so every id is one connected region; `euler_ebsd`/`quat_ebsd` untouched. `EBSDReader.grain_average_euler_deg()` — per-grain Bunge-Euler angles (degrees) from a mean of `quat_ebsd` over each grain's pixels (renormalised, positive-hemisphere), exposed as a standalone public method (the same averaging approach `rechar_lfi` already used internally).
  - `meshing/writer_ABQ.py`: `export_confmesh2d_inp` gains `material_format='bunge_euler'` — writes one `*Material` + `*User Material, constants=3` (the grain's Bunge-Euler angles) + `*Depvar` per grain, from a `grain_euler_deg` dict, mirroring the convention already used by the 3D exporter (`twinned_simple_3d.abaqus_exporter_3d.AbaqusExporter3D`). Default behaviour (`material_format='isotropic'`) unchanged; also gains an `elastic_constants` override for that path.
  - Demo: `src/upxo/demos/ebsdOps/ebsd_to_abaqus_2d.ipynb` — EBSD read → crop → characterise → split disconnected grains → grain-averaged orientation → Technique A geometrification/smoothing → conformal mesh → element-quality statistics → Abaqus export → interactive PyVista mesh-vs-grain-boundary view.
  - Wiki: new [EBSD to Abaqus (2D)](https://github.com/Design-By-Fundamentals-UKAEA/UPXO/wiki/EBSD-to-Abaqus-2D) page, cross-linked from Data I/O, Meshing, and Use Cases.
  - Tests: `tests/interfaces/defdap/test_ebsd_reader.py`, `tests/meshing/test_writer_abq.py` — synthetic-array fixtures, no DefDAP / `.ctf` file / Gmsh dependency.
- **Editable 2D grain polygons**: grain boundaries built from UPXO geometry objects, convertible to and from Shapely, with per-grain scalar properties.
  - `geoEntities/polygon2d.py`: `Polygon2d` (one grain, built on `ring2d`) and `NestedPolygon2d` (a host with hole polygons, nestable). `edit_segment` replaces the interior nodes of one boundary segment in place, keeping its two end nodes; `subdivide_segment(seg_index, n=None, at_fractions=None)` inserts interior points by arc-length fraction. `polygons_to_prop_dataframe` / `apply_prop_dataframe` move per-grain scalars to and from a DataFrame.
  - `geoEntities/polygon2d_from_shapely.py`: `polygon_collection_from_shapely(cells, gid_key=None, tol=1e-6)` converts `{gid: Polygon | MultiPolygon | GeometryCollection}` (for example `GrainManifold2D.cells`, or a Voronoi tessellation) into `{gid: Polygon2d | NestedPolygon2d}`. Neighbouring grains reference the same `MSline2d` object for a shared wall, so an edit made through one grain is seen by the other, the boundary stays gap-free and total area is conserved. For a `MultiPolygon` cell the largest part is the value and the rest are in `props['extra_parts']`.
  - `pxtal/geometrification.py`: `polygonised_grain_structure.construct_geometric_xtals_from_gbcoords(..., dtype='upxo')` returns `Polygon2d` / `NestedPolygon2d` built directly from `self.GB`, without copying.
  - `geoEntities/mulsline2d.py`: `ring2d.translate(dx, dy, seen=None)`.
  - Demos: `src/upxo/demos/geom/poly2d01.ipynb` to `poly2d06.ipynb` — `Polygon2d` basics, the Shapely converter, `subdivide_segment` with a hand-rolled boundary-roughening preview, and conversion of Voronoi, Monte Carlo (Technique B) and Technique A grain structures.
  - Tests: `tests/geoEntities/test_polygon2d*.py`, `tests/pxtal/test_geometrification_polygon2d.py`, `tests/pxtal/test_polygon2d_from_shapely_grainmanifold.py`.

### Changed

- **`pxtal/geometrification.py`, Technique A with island grains**: `self.GB` and `self.GBCoords` now include island grains, keyed by original grain id (in 1.2.1 an island id had no entry). A new `self.GB_holes` maps a host grain id to the `ring2d` of each island it directly encloses; an island nested inside another island is a hole of its immediate parent. An island's `ring2d` is the same object in its own `self.GB` entry and in its host's `self.GB_holes` list. Tests: `tests/pxtal/test_geometrification_islands.py`.
- `pxtal/_gb_topology.py` (new): `junction_points_from_polygons` and `assemble_ring_from_wall_segments`, extracted from `polygonised_grain_structure`; `get_junction_points_from_grain_intersections` and `flip_segments_to_reorder_GBS` call them. Behaviour is unchanged.

### Fixed

- **`GrainManifold2D.smooth_interfaces`**: cells could extend past the label-image domain after smoothing. `_laplacian_step` holds only vertices lying exactly on the domain edge, and the negative-`mu` Taubin pass can move other boundary-adjacent vertices outward. `smooth_interfaces` now ends with `trim_to_rve(bounds=(0, 0, width, height))`. `trim_to_rve(bounds)` remains available for cropping to a sub-window. Tests: `tests/pxtal/test_grainmanifold2d_domain_clip.py`.

## [1.2.1] — 2026-09-22

### Fixed

- **`pxtal/geometrification.py`**: Technique A (`polygonised_grain_structure`) now handles island grains (a grain fully enclosed by another) correctly. `pix_to_geom` splits the structure into a hole-filled version plus one geometrification pass per island cluster (recursing for nested islands), then cuts each island out of its host with a Shapely `difference`. Previously any structure containing an island grain raised `IndexError` or `AttributeError`, or produced a host with no interior ring.
- Junction-point extraction (`get_junction_points_from_grain_intersections`) recorded only the first end of a `LineString` intersection; a grain with a four-grain corner could lose a junction point and raise `A linearring requires at least 4 coordinates`. Both ends are now recorded.
- `set_grain_loc_ids` classified domain corners with an `elif` chain, so a grain spanning two corners (e.g. wrapping a one-pixel grain on the domain edge) lost the second corner's boundary segment; its ring then failed to close and the polygon collapsed to zero area. Corner tests are now independent; the boundary-segment consolidation step removes the resulting duplicate segments and adds any domain-edge segment still missing.
- Island-containing structures: `self.GB` / `self.GBCoords` were keyed by the filled sub-structure's local relabelled ids rather than original grain ids, so indexing by `gid` could silently return a different grain's boundary ring. Now remapped to original ids; an island grain has no entry (its boundary is a hole in its host, not a same-level ring) rather than returning wrong data.
- `set_grain_centroids_raw`'s vectorised fast path could raise `IndexError`, or silently return a wrong centroid, for a caller-supplied `gid` absent from the label image; it now returns `NaN` for that case, matching the original per-grain loop.
- `ring2d.clone()` copied `coords`/`conn0`/`conn1` by reference despite documenting an independent copy; now copied by value.
- **`GrainManifold2D`** (Technique B): single-pixel and other small grains, including islands, could collapse to a sliver under repeated Taubin smoothing passes — small-grain vertices are now frozen by default (`thin_grain_px`). Mirrored ghost seeds in `_generate_clipped_polygons` could carve spurious voids near the domain edge into the raw (untrimmed) manifold — guard-seed placement fixed. `generate_constrained_hybrid_seeds` could miss a single-pixel or one-pixel-wide grain's boundary entirely (a central-difference gradient blind spot) — a 4-neighbour-difference pass, plus a final per-label seed guarantee independent of the boundary-sampling stride, close both gaps.
- **`viz/gsviz.plot_multipolygon_geometric`**: interior rings (holes) were drawn as opaque white polygons, hiding an island grain drawn earlier in iteration order. Each polygon is now one compound path so holes are true cut-outs.
- **`repqual/`**: `determine_distr_type` used target values for every sample; `mc2repr` threshold setters could store invalid values and loop forever; NLSD results overwrote `rkf['ed']` instead of their own key.
- **`pxtal/`**: zero-twin case in `remove_overlaps_in_twins` handled; `repgen3d` missing `__slots__` prevented construction; undefined `np` reference in `geoEntities`' abstract coords body removed.

### Performance

- Technique A (`polygonised_grain_structure.pix_to_geom` + `smooth_gbsegs`): roughly 15x faster (930 grains: 54 s → 3.7 s, then further to ~1.9 s in a second pass; scaling improved from roughly quadratic to roughly linear in grain count). All-pairs geometry tests replaced with Shapely `STRtree` candidate queries; array scans replaced with hash lookups; one method removed as dead work (its result was overwritten before being read); per-segment properties cached instead of recomputed for every neighbour pair; `deepcopy` in `smooth_gbsegs` replaced with new `MSline2d.clone()` / `ring2d.clone()`.
- Technique B (`GrainManifold2D.smooth_interfaces` + `_generate_clipped_polygons`): the vertex-adjacency graph is now built once per smoothing run instead of once per Taubin pass; Voronoi cells that do not cross the RVE boundary skip the clipping step. Roughly 20-30% faster depending on grain-structure regularity. Neither technique is reliably faster than the other across all grain-structure types — see `confMesh2d_compare_techniques.ipynb` for current, measured numbers on a given case.

### Removed

- `rasterio` is no longer a UPXO dependency. `pxtal/geometrification.py`'s `polygonize()` and `pxtal_ori_map_2d.py`'s `polygonize_voronoi_grid` now use the new pure NumPy/Shapely `_sup/raster_polygonize.py`, matching `rasterio.features.shapes`'s output structure. The `[io]` extra is removed from `pyproject.toml` / `setup.py` / `requirements.txt`; `docs/conf.py`'s mock-imports list and the README / docs install instructions updated to match.

### Added

- **`_sup/raster_polygonize.py`**: pure NumPy/Shapely label-to-polygon tracer (4-connected, hole-aware).
- **`_sup/raster_islands.py`**: detects grains fully enclosed by another grain (islands) ahead of polygonisation.
- **Demo**: `confMesh2d_compare_techniques.ipynb` — Technique A vs Technique B on the same label image, with geometry, mesh-quality and timing comparisons.
- Wiki: new 2D Grain Structure Geometrification page documenting both techniques, island handling, and how to choose between them.

### Infrastructure

- CI: Sphinx docs now also deploy on pushes to `dev` (previously `main` only); tests now also run on pushes to `main` (previously `dev` only).

---

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
- **`xtalphy/texops.py`**: Fixed the same Bunge ZXZ Euler-decomposition bug present in `crystal_orientation.matrix_to_euler_bunge`; added texture-component detection; ported previously interactive-only quaternion helpers into core
- **Packaging**: `requirements.txt` enables `tetgen`; optional extra `[mesh]` now also installs `gmsh>=4.13`
- **`confMesh2d` (pygmsh)** is deprecated; new 2D conformal work must use `confMesh2dGMSH` / `gsmesh2d.mesh_gs`
- Removed redundant tracked demos `confMesh2d9.ipynb` and `mesh2d01.ipynb`
- The voxel-lattice cleaving mesh generator (`meshing/cleaving/`) is untracked pending relocation to `meshing/gbconformant/cleaving3dV1P0/` (WIP, not part of this release)

### Not Included (Deferred)

- MC Metropolis-acceptance refinement (permanently out of scope per architecture decision)
- GB-energy/crystallography coupling for block selection (deferred to future versions)
- Multi-CSL twin registry beyond Sigma-3 Path A (CSL Path B deferred post-Path A validation)
- Cleaving mesh generator migration to `meshing/gbconformant/cleaving3dV1P0/` (WIP, untracked)

---

## [1.1.0] — 2026-08-04

**First official TestPyPI release of UPXO.**

### Added

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
- **Workflows documentation**: Expanded from 8 to 20 workflows covering all major capabilities

### Fixed

- Typos in README: "pertaining multi-scale" → "pertaining to multi-scale", "teknology" → "technology", "powerpoint" → "power plant", "visibility" → "viability"
- Test suite: Resolved 26 initial test failures through systematic debugging and API alignment

### Infrastructure

- Added `.gitignore` entries for legacy local-only folders and generated output directories
- Organized data, sessions, and generated output structures
- Sphinx build workflow configured for automated API documentation generation

### Not Included (Deferred)

- MC Metropolis-acceptance refinement (permanently out of scope per architecture decision)
- GB-energy/crystallography coupling for block selection (deferred to future versions)
- Multi-CSL twin registry beyond Sigma-3 Path A (CSL Path B deferred post-Path A validation)

---

## [1.0.0] — Development Only (Never Published)

Initial development version. Features and APIs evolved significantly before first TestPyPI release.
