# UPXO Copilot Instructions

## Project Overview

**UPXO** (Universal PolyXtal Operations) is a comprehensive materials science framework for generating and analyzing polycrystalline grain structures using Monte-Carlo simulations. The codebase bridges crystallography, computational geometry, and meshing for microstructure analysis.

**Core Purpose**: Generate synthetic 2D/3D grain structures via Monte-Carlo algorithms, characterize grain properties (morphology, orientation, network topology), and export to FEA/simulation formats.

## Architecture

### Layered Organization

```
upxo/
├── ggrowth/           # CORE: Monte-Carlo grain growth engine
│   └── mcgs.py       # Main classes: grid (base), mcgs (Monte-Carlo)
├── algorithms/        # Algorithm implementations (alg200-302)
├── pxtal/            # Physical crystal/grain representations
│   └── mcgs2_temporal_slice.py  # Temporal snapshots of grain structure
├── geoEntities/       # Geometric primitives (point2d, edge2d, polygon3d, etc.)
├── interfaces/        # Integration points (abaqus, dream3d, matlab, etc.)
├── xtalphy/          # Crystallography (orientations, Euler angles)
├── xtalops/          # Crystal operations
├── grids/            # Grid generation and discretization
├── meshing/          # FE mesh generation
├── analysis/         # Microstructure analysis (2D/3D)
├── statops/          # Statistical analysis (distributions, KDE)
├── netops/           # Network analysis (grain boundary graphs, k-nearest neighbors)
└── _sup/             # Support utilities (data handlers, validation, I/O)
```

### Key Data Flow

1. **User Input** (`interfaces/user_inputs/`) → Excel dashboard configuration
2. **Grid Initialization** (`grids/gridder.py`) → Discretized domain (x/y/z)
3. **Monte-Carlo Simulation** (`ggrowth/mcgs.py`) → Sequential state evolution
4. **Grain Detection** (`detect_grains()`) → Label connected regions
5. **Temporal Snapshots** (`pxtal/mcgs2_temporal_slice.py`) → Per-timestep grain data
6. **Characterization** (`char_morph_2d()`, etc.) → Compute grain properties
7. **Export** (`interfaces/*/`) → VTK, ABAQUS, HDF5, etc.

### Critical Classes

| Class | Module | Purpose |
|-------|--------|---------|
| `grid` | `ggrowth/mcgs.py` | Base class: grid setup, algorithm dispatch, state matrices |
| `mcgs` | `ggrowth/mcgs.py` | Monte-Carlo wrapper: `initiate()`, `simulate()`, algorithm routing |
| `mcgs2_grain_structure` | `pxtal/mcgs2_temporal_slice.py` | Temporal snapshot: grains, properties, orientations |
| `gsan2d` | `analysis/analysis2d.py` | 2D spatial analysis: neighbor graphs, clusters |
| `point2d`, `edge2d`, `polygon3d` | `geoEntities/` | Geometric primitives with UPXO/Shapely/VTK compatibility |

## Developer Workflows

### Monte-Carlo Simulation Pipeline

```python
from upxo.ggrowth.mcgs import mcgs

# 1. Initialize with Excel dashboard
gs = mcgs(study='independent', input_dashboard='config.xls')

# 2. Setup grid, state matrix, algorithms
gs.initiate(AR_teevrate=0, consider_NLM_b=False)

# 3. Run Monte-Carlo steps
gs.simulate(rsfso=2, LIWM=user_weights)

# 4. Detect grains at temporal slices
gs.detect_grains(mcsteps=[0, 100, 500], library='scikit-image')

# 5. Characterize morphology
gs.char_morph_2d(area=True, perimeter=True, aspect_ratio=True)

# 6. Export results
gs.write_to_vtk(tslice=100, filename='result.vtk')
```

### Key Method Patterns

- **Algorithm dispatch**: `start_algo2d_without_hops()`, `start_algo3d_without_hops()` select algorithms (alg200, alg201, alg202 for 2D; alg300a/b, alg301, alg302 for 3D) via `self.uisim.mcalg` string ID.
- **Temporal iteration**: Use `self.tslices` list to iterate over saved Monte-Carlo steps.
- **Property access**: Grain properties stored in `gs[tslice].g[grain_id].prop` dict; spatial data in `gs[tslice].gb` (boundaries), `gs[tslice].positions` (centroids).

## Project-Specific Conventions

### Naming & Abbreviations

| Abbreviation | Meaning |
|--------------|---------|
| `mcgs` | Monte-Carlo Grain Structure |
| `tslice` | Temporal slice (Monte-Carlo step snapshot) |
| `gid` | Grain ID (numeric identifier) |
| `gb` | Grain boundary |
| `px_size` / `vox_size` | Pixel/voxel size (2D/3D) |
| `S` | Number of orientation states |
| `LIWM` | Local interaction weight matrix |
| `NLM` / `NLM_b`, `NLM_d` | Non-locality matrix (boundary/distance-based) |
| `EAPGLB` | Euler angle (Primary Global) |
| `ui*` | User input (e.g., `uigrid`, `uisim`, `uiint`) |

### Configuration via Excel Dashboard

The project uses an Excel file (default: `input_dashboard.xls`) with sheets:
- **Grid**: `xmin`, `xmax`, `xinc`, `ymin`, `ymax`, `yinc`, `dim` (2 or 3)
- **Simulation**: `S` (states), `mcalg` (algorithm ID), `mcsteps` (iterations)
- **Interaction**: `mcint_save_at_mcstep_interval` (snapshot frequency)

### State Matrix Convention

- **State matrix `S`**: 2D array (xgr.shape[0], ygr.shape[1]) or 3D, values are **integer state IDs** (0 to S-1).
- **Grain detection**: Connected regions of same state ID = one grain.
- **Orientation mapping**: Each state has an assigned Euler angle orientation (via `EAPGLB` dict).

### Data Type Flexibility

Geometric operations accept multiple input types (unified via `_sup/dataTypeHandlers.py`):
- UPXO types: `point2d`, `edge2d`, `polygon3d`
- Standard: NumPy arrays, Python lists, tuples
- External: Shapely, VTK, PyVista objects

### Validation Pattern

Before using data, check via `_sup/dataTypeHandlers.py`:
```python
from upxo._sup import dataTypeHandlers as dth
assert isinstance(data, dth.dt.ITERABLES)  # or dth.dt.NUMBERS
```

## Key Algorithms

### Monte-Carlo Algorithms (Numbered IDs)

| ID | Dimension | Algorithm |
|----|-----------|-----------|
| `200`, `200.0` | 2D | Base MC growth |
| `201`, `201.0` | 2D | Weighted MC growth |
| `202` | 2D | Additional weighted variant |
| `300a`, `300b`, `301`, `302` | 3D | 3D MC variants |

Each algorithm module (`alg*.py`) exports a `run()` or `mc_iterations_*()` function accepting:
- State arrays: `xgr`, `ygr`, `zgr` (or indices for 3D)
- Parameters: `S`, `px_size`/`vox_size`, `LIWM`, `rsfso` (radius scaling)
- User inputs: `uidata`, `uigrid`, `uisim`, `uiint`

### Grain Detection

Supports:
- `library='scikit-image'`: Uses `skimage.measure.label` (default, faster)
- `library='opencv'`: Uses OpenCV for 2D labeling
- 3D: SciPy `ndimage.label`

Parameters: `kernel_order` (connectivity), `store_state_ng` (save new grain states).

## Common Integration Points

### Excel I/O (`interfaces/user_inputs/`)
- Read config → `uigrid`, `uisim`, `uiint` dataclass instances
- Example: `from upxo.interfaces.user_inputs import read_dashboard`

### Export Formats (`interfaces/`)
- **VTK/VTU**: Unstructured mesh export
- **ABAQUS**: FE model with grain element sets
- **DREAM3D**: HDF5 with grain data
- **MATLAB**: MAT files with state matrices

### Crystallography (`xtalphy/`)
- Orientation management: `grainoris` class stores grain orientations
- Euler angle transformations: `eamo` module
- Crystal symmetry: `xtal/` module

## Testing & Validation Patterns

- **Validation**: `_sup/validation_values.py` defines legal ranges (e.g., valid algorithm IDs).
- **Data templates**: `_sup/data_templates.py` provides dict/DataFrame schema templates.
- **Logging**: Use `display_messages` flag in `mcgs.__init__()` to enable/disable verbose output.

## Performance Considerations

- **State matrices**: Large grids (>1000³) → memory-intensive; consider upsampling/downsampling via `finer()`/`coarser()`.
- **Algorithm selection**: `alg200` (simplest) vs. `alg300a/b/301/302` (3D, more complex); choose based on dimensionality.
- **Spatial indexing**: Network analysis uses `cKDTree` (in `netops/`) for fast neighbor queries.
- **Grain characterization**: Heavy computation; use `char_morph_2d(..., find_neigh=False)` to skip neighbor detection if not needed.

## File Structure Rules

- **Module imports**: Always use absolute imports (`from upxo.module import ...`), never relative imports.
- **Backup files**: Outdated code in `__pycache__/`, `*_old.py`, `*_bu.py` → ignore when modifying.
- **Data files**: Input XLS in workspace root; outputs go to `_written_data/` and `_writer_data/` subdirectories.

## Debugging Tips

1. **State matrix inspection**: `gs.S` (state array), `gs.m` (Monte-Carlo steps) → verify shape and values.
2. **Grain detection failures**: Check `library` parameter, `kernel_order` connectivity, state matrix sparsity.
3. **Missing properties**: Call `char_morph_2d()` first; verify `gs[tslice].prop_flag['area']` is True.
4. **Orientation assignment**: Ensure `EAPGLB` dict populated before exporting; check `__ori_assign_status_stack__`.

---

**Last Updated**: December 2024 | **Version**: 1.26.1

## Analysis 2D (Active Development)

- **Location**: [analysis/analysis2d.py](analysis/analysis2d.py)
- **Core classes**: `gsan2d` (grain-set analysis), `kmodel` (NetworkX-based graph analysis), `principle_component_analysis` (PCA holder).
- **Single-slice setup**: Use `gsan2d.from_mcgs2d_single(gstslice, ...)` after `mcgs.simulate()` and `detect_grains()` as needed. It calls `char_morph_2d()` with toggles from class dicts.
- **Temporal setup**: Use `gsan2d.from_gsstack_temporal(gsstack, gsids=[...], ...)` where `gsstack` maps time slice → `mcgs2_grain_structure`.
- **Property toggles**: Control characterization via `gsan2d.defmp` (properties) and `gsan2d.chctrl` (controls). Enabling `aspect_ratio=True` auto-enables `major_axis_length` and `minor_axis_length`.
- **Neighbor detection**: `find_neigh_v2()` is invoked separately to avoid Jupyter kernel crashes; set `find_neigh=True` with parameters `find_neigh_p`, `find_neigh_include_central_feat`, `find_neigh_throw_numba_dict` to compute neighbors post-characterization.
- **Extraction to DataFrame**: `extract_props_pxtal_single()` builds per-slice DataFrames; `moments_hu` is expanded to columns `mhu_1..mhu_n`; `aspect_ratio` computed from major/minor axes.
- **Statistics & Correlation**: `compute_statistics()` populates `stts`; `correlate()` and `correlate_temporal()` compute correlation matrices—use `see_correlation()` for seaborn heatmaps and `see_correlation_temporal()` for animated Plotly heatmaps.
- **PCA workflow**: `pcanalyis(gsids, pnames, ...)` standardizes (`StandardScaler`) then runs PCA (`sklearn.decomposition.PCA`), storing `scores` and `explained_variance_ratio_`; generates scree and cumulative variance plots.
- **Graph analysis**: `initiate_kmodel(gsids, ...)` builds graphs from `neigh_gid` (`mcgs2_temporal_slice.make_graph(...)`) and characterizes `num_nodes`, `num_edges`, `density`, `avg_clustering_coeff`, `degree_assortativity`, and optional eccentricity-based metrics for connected graphs < 5000 nodes. Subgraph helpers: `extract_subgraph_connected_neighbors()` (ego graphs) and `extract_largest_connected_component()`.
- **Quick example**:
	```python
	from upxo.ggrowth.mcgs import mcgs
	from upxo.analysis.analysis2d import gsan2d
	gs = mcgs(input_dashboard='interfaces/user_inputs/input_dashboard.xls')
	gs.initiate(); gs.simulate(); gs.detect_grains([0])
	gsan = gsan2d.from_mcgs2d_single(gs[0], aspect_ratio=True, moments_hu=True,
																	 find_neigh=True, find_neigh_p=1.0)
	gsan.extract_props(); gsan.compute_statistics(); gsan.see_correlation([1], ['area','aspect_ratio'])
	gsan.initiate_kmodel(gsids=[1], k_char_level='full')
	```
