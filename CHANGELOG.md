# CHANGELOG

All notable changes to UPXO are documented in this file.

## [1.1.0] — 2026-08-04

**First official PyPI release of UPXO.**

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

- Added `.gitignore` entries for legacy local-only application folders and generated output directories
- Organized data, sessions, and generated output structures
- Sphinx build workflow configured for automated API documentation generation

### Not Included (Deferred)

- MC Metropolis-acceptance refinement (permanently out of scope per architecture decision)
- GB-energy/crystallography coupling for block selection (deferred to future versions)
- Multi-CSL twin registry beyond Sigma-3 Path A (CSL Path B deferred post-Path A validation)

---

## [1.0.0] — Development Only (Never Published)

Initial development version. Features and APIs evolved significantly before first PyPI release.
