# Changelog

All notable changes to UPXO will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0] - 2026-05-22

### Added
- 2D and 3D Monte Carlo grain structure (MCGS) generation engine
- Temporal grain structure analysis and visualisation (`mcgs2_temporal_slice`, `mcgs3_temporal_slice`)
- Grain characterisation: area, perimeter, shape descriptors, boundary detection
- Grain boundary operations (`gbops`) including misorientation and network analysis
- Polycrystal container classes (`gsContainers`) for managing grain structure ensembles
- Geometry entities: multi-point, polygon, Voronoi cell representations
- FEM mesh export via `tetgen` (optional) and VTK output via `pyvista`
- EBSD data import and mapping via `defdap` (optional extra)
- Excel-based user input interface (`interfaces/user_inputs`)
- Statistical distribution fitting and grain size analysis (`statops`)
- Image operations for microstructure data (`imageOps`)
- Grid resampling, stretching, and interpolation utilities (`gsdataops/grid_ops`)
- ReadTheDocs documentation infrastructure
- Optional dependency extras: `[viz]`, `[mesh]`, `[io]`, `[ebsd]`, `[all]`

---

## How to maintain this file (developer notes)

Before each release:
1. Add a new section at the **top** (below this header) in the form:
   ```
   ## [x.y.z] - YYYY-MM-DD
   ### Added / Changed / Fixed / Removed
   - ...
   ```
2. Use user-facing language — describe *what changed*, not *how*.
3. Bump the version in **three places** together:
   - `pyproject.toml` → `version = "x.y.z"`
   - `setup.py` → `version="x.y.z"`
   - `src/upxo/__init__.py` → `__version__ = "x.y.z"`
4. Tag the commit: `git tag v x.y.z && git push --tags`
   - Tags matching `v*.*.*` trigger the PyPI publish workflow automatically.
