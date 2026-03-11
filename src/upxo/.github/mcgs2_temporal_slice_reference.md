# mcgs2_temporal_slice.py - Internal Reference Guide

**Purpose**: Deep dive into the temporal snapshot class that holds grain structure data at specific Monte-Carlo timesteps. This is the workhorse for grain characterization, neighbor detection, visualization, and graph construction.

**Location**: [pxtal/mcgs2_temporal_slice.py](../pxtal/mcgs2_temporal_slice.py)

---

## 1. Class Structure & Architecture

### Memory-Optimized Design
- **54 `__slots__`** for memory efficiency (critical for large datasets)
- Maximum grid size threshold: `__maxGridSizeToIgnoreStoringGrids = 1000000` pixels (1M)

### Key Attributes

| Attribute | Type | Purpose |
|-----------|------|---------|
| `lgi` | `np.ndarray` (uint8) | **Labeled Grain Image** - core data structure where pixel value = grain ID |
| `g` | `dict` | Grain objects: `{gid: {'s': state, 'grain': grain2d()}}` (v1) or `{gid: grain2d()}` (v2) |
| `gb` | `dict` | Grain boundary data structure |
| `neigh_gid` | `dict` | Neighbor map: `{gid: [list of neighbor gids]}` |
| `s` | `np.ndarray` | State matrix (orientation states) |
| `xgr`, `ygr` | `np.ndarray` | Grid coordinate arrays (x and y directions) |
| `n` | `int` | Total number of grains |
| `gid` | `list` | List of all grain IDs |
| `S` | `int` | Total number of orientation states |
| `px_size` | `float` | Physical size of one pixel (e.g., microns/pixel) |
| `positions` | `dict` | Grain centroid positions |
| `species` | `dict` | Species/phase partitioning data |
| `EAPGLB` | `dict` | Euler angles (Primary Global) - statewise orientation map |
| `prop_flag` | `dict` | Flags indicating which properties have been computed |
| `_char_fx_version_` | `int` | Characterization version: 1 (legacy) or 2 (feature-based) |

### Valid Property Names
```python
valid_morpho_props = ['area', 'area_bbox', 'area_convex', 'axis_major_length', 
                      'axis_minor_length', 'bbox', 'centroid', 'eccentricity', 
                      'equivalent_diameter_area', 'euler_number', 'feret_diameter_max',
                      'moments_hu', 'orientation', 'perimeter', 'perimeter_crofton', 
                      'solidity', ...]

valid_topo_props = ['euler_characteristic', 'avg_nneigh', 'nneigh', 'n_vertices', 
                    'n_boundaries', 'ntjp', 'nqp', 'nneigh_dist_P_n', 
                    'aboav_weaire_params']
```

---

## 2. Grain Characterization (char_morph_2d)

### Two Implementations

#### Version 1: `char_morph_2d()` (Legacy, Grain-Centric)
- Sets `self._char_fx_version_ = 1`
- Iterates over states → grains, using `cv2.connectedComponents`
- Stores grains as: `self.g[gid] = {'s': state, 'grain': grain2d()}`
- Uses scikit-image `regionprops` stored in `grain.skprop`

#### Version 2: `char_morph_2d_v2()` (Feature-Based)
- Sets `self._char_fx_version_ = 2`
- Uses `detect_features_in_image()` for connected component labeling per state
- Stores grains directly: `self.g[gid] = grain2d()` (no nested dict)
- Supports species assignment and feature detection

### Key Parameters (char_morph_2d)

```python
def char_morph_2d(self, bbox=True, bbox_ex=True, npixels=False,
                  npixels_gb=False, area=False, eq_diameter=False,
                  perimeter=False, perimeter_crofton=False,
                  compactness=False, gb_length_px=False, aspect_ratio=False,
                  solidity=False, morph_ori=False, circularity=False,
                  eccentricity=False, feret_diameter=False,
                  major_axis_length=False, minor_axis_length=False,
                  euler_number=False, moments_hu=True, 
                  append=False, saa=True, throw=False,
                  char_grain_positions=False, find_neigh=False,
                  char_gb=False, make_skim_prop=False,
                  get_grain_coords=True):
```

### Property Definitions

| Property | Formula/Definition |
|----------|-------------------|
| `area` | `npixels * px_size^2` |
| `eq_diameter` | Diameter of circle with same area |
| `perimeter` | Total length of boundary |
| `compactness` | `px_area / (Area of circle with perimeter P)` |
| `aspect_ratio` | `major_axis_length / minor_axis_length` |
| `solidity` | `npixels / convex_hull_pixels` |
| `morph_ori` | Morphological orientation (-π/2 to +π/2, counter-clockwise from x-axis) |
| `circularity` | Closeness to circular shape |
| `eccentricity` | `focal_distance / major_axis_length` |
| `feret_diameter` | Caliper diameter (max perpendicular distance) |
| `euler_number` | Topological invariant (1 for grains without holes) |

### Critical Workflow Pattern

```python
# ALWAYS call char_morph_2d FIRST before accessing grain properties
gs[tslice].char_morph_2d(area=True, perimeter=True, moments_hu=True)

# Then properties are available:
grain_area = gs[tslice].g[gid]['grain'].skprop.area  # v1
grain_area = gs[tslice].g[gid].skprop.area          # v2
```

---

## 3. Neighbor Detection

### Three Methods (Critical: Avoid Jupyter Kernel Crashes)

#### Method 1: `find_neigh()` - Original Implementation
```python
def find_neigh(self, include_central_grain=False, print_msg=True,
               user_defined_bbox_ex_bounds=False, bbox_ex_bounds=None,
               update_grain_object=True):
```
- Uses extended bounding boxes (`bbox_ex_bounds`)
- Pixel-level neighbor search via masked arrays
- Safe but slower for large structures
- Stores results in `self.neigh_gid`

#### Method 2: `find_neigh_v2()` - **Numba-Accelerated (RECOMMENDED)**
```python
def find_neigh_v2(self, p=1.0, include_central_grain=False,
                  throw_numba_dict=False, verbosity_nfids=1000):
```
- **CRITICAL**: Call **separately** from `char_morph_2d()` to avoid Jupyter kernel crashes
- Uses `GidOps.find_O1_neigh_2d()` (Numba-compiled)
- Parameters:
  - `p`: Probability/threshold (0.0 to 1.0) for neighbor inclusion
  - `include_central_grain`: Include grain itself in neighbor list
  - `verbosity_nfids`: Print progress every N grains

**Safe Pattern**:
```python
# Do NOT do this (will crash in Jupyter):
gs[tslice].char_morph_2d(find_neigh=True)  # ❌ AVOID

# Do this instead:
gs[tslice].char_morph_2d(find_neigh=False)  # ✅ Safe
gs[tslice].find_neigh_v2(p=1.0)             # ✅ Call separately
```

#### Method 3: `find_neigh_gid(gid)` - Per-Grain Detection
```python
def find_neigh_gid(self, gid, include_central_grain=False, throw=False, 
                   update_grain_object=True, 
                   user_defined_bbox_ex_bounds=False, bbox_ex_bounds_fid=None):
```
- Finds neighbors for a single grain
- Returns `np.ndarray` of neighbor IDs when `throw=True`
- Stores boundary segments in `grain.gbsegs_pre`

### Higher-Order Neighbors

#### Up-to-Nth Order
```python
def get_upto_nth_order_neighbors(self, grain_id, neigh_order,
                                fast_estimate=False,
                                include_parent=True,
                                output_type='list'):
```
- Computes all neighbors up to order `n` (1st + 2nd + ... + nth)
- Returns: `list`, `nparray`, or `set`

#### Exactly Nth Order
```python
def get_nth_order_neighbors(self, grain_id, neigh_order,
                            fast_estimate=False,
                            recalculate=False,
                            include_parent=True):
```
- Computes only nth-order neighbors (excludes lower orders)
- Example: 2nd-order = neighbors of neighbors (excluding 1st-order)

### Probabilistic Neighbor Selection
```python
def get_upto_nth_order_neighbors_all_grains_prob(self, neigh_order,
                                                 recalculate=False,
                                                 include_parent=False,
                                                 print_msg=False,
                                                 _int_approx_=0.05):
```
- Allows float values for `neigh_order` (e.g., 2.5 → probabilistic between 2 and 3)

---

## 4. Graph Construction

### Primary Method: `make_graph()`
```python
def make_graph(self, neigh_gid):
    """Create NetworkX graph from neighbor dictionary."""
    return kmake.make_gid_net_from_neighlist(neigh_gid)
```

**Usage**:
```python
# Step 1: Detect neighbors
gs[tslice].find_neigh_v2()

# Step 2: Build graph
G = gs[tslice].make_graph(gs[tslice].neigh_gid)

# G is now a NetworkX graph where:
# - Nodes = grain IDs
# - Edges = grain adjacency (shared boundary)
```

### Helper: `build_grain_pairs()`
```python
def build_grain_pairs(self, neigh_gid):
    """Extract unique grain pairs (avoid duplicates by using gid1 < gid2)."""
    grain_pairs = []
    for grain_id, neighbors in neigh_gid.items():
        for neighbor_id in neighbors:
            if neighbor_id > grain_id:
                grain_pairs.append((grain_id, neighbor_id))
    return grain_pairs
```

### Boundary Pixel Mapping: `identify_grain_boundary_pixels()`
```python
def identify_grain_boundary_pixels(self, grain_pairs):
    """
    Returns: dict {(gid1, gid2): Nx2 array of boundary coordinates}
    Maps grain pairs → exact pixel locations of shared boundaries.
    """
```
- Performs horizontal and vertical passes on `lgi`
- Detects transitions: `left != right` (vertical boundaries), `top != bottom` (horizontal)
- Returns sorted pairs as keys with coordinate arrays as values

---

## 5. Visualization (plot_grains)

### Main Plotting Method
```python
def plot_grains(self, gids, hide_non_actors=True,
                default_cmap='jet', title="user grains",
                throw_plt_object=False, figsize=(6, 6), dpi=120):
```

**Key Features**:
- `gids`: List/iterable of grain IDs to display
- `hide_non_actors=True`: Masks non-selected grains (sets to white)
- `default_cmap`: Colormap for grain visualization
- `throw_plt_object=False`: Return `plt` object for further customization

**Masking Pattern**:
```python
# Create mask for selected grains
lgi_masked = np.sum([gid*(self.lgi == gid) for gid in gids], axis=0)
lgi_masked[lgi_masked == 0] = -10  # Non-actors set to -10
cmap.set_under('white')  # Display -10 as white
```

### Specialized Variants

#### By Grain IDs
```python
def plot_grains_gids(self, gids, gclr='color', title="user grains",
                    cmap_name='CMRmap_r', ...):
```

#### By Property Range
```python
def plot_grains_prop_range(self, PROP_NAME, prop_min, prop_max, ...):
```

#### By Position
```python
def plot_grains_at_position(self, position='corner', overlay_centroids=True,
                            markersize=6):
```
- `position`: 'corner', 'boundary', 'triple_point'
- Overlays centroids on plot

### Grain Boundary Visualization
```python
# Extract grain boundary locations from grain object
gb = gs[tslice].g[gid]['grain'].gbloc  # Nx2 array (row, col)

# Plot grain boundaries
plt.plot(*np.roll(gb, 1, axis=1).T, 'k.', markersize=2)  # Roll to swap x/y
```

---

## 6. Species & Features

### Species Assignment
```python
def assign_species(self, method='mc state partitioned global combined',
                   ignore_vf=True, vf={}, spid=1, combineids=[], ninstances=10,
                   detect_features=True, bso=1, characterise_features=True, 
                   make_feature_skprops=True, extract_feature_coords=True,
                   throw_feature_bounding_box=True):
```

**Methods**:
- `'mc state partitioned global'`: All states with equal volume fractions
- `'mc state partitioned global combined'`: Combine multiple states into species
- `'mc state partitioned local'`: Local state-based partitioning
- `'mc state partitioned local vf'`: Local with volume fractions

**Combine States Example**:
```python
# Combine states [1,2] into species 1, states [3,4] into species 2
gs[tslice].assign_species(method='mc state partitioned global combined',
                          combineids=[[1,2], [3,4]], 
                          ninstances=5,
                          spid=1)
```

**Data Structure**:
```python
gs[tslice].species[spid] = {
    'inst_1': combined_state_array,
    'inst_1_img': labeled_features,
    'inst_1_orig_to_labels': {original_state: [feature_ids]},
    'inst_1_labels_to_orig': {feature_id: original_state},
    'inst_1_feature_skprops': {fid: regionprop},
    'inst_1_feature_bbox_limits': {fid: [rmin, rmax, cmin, cmax]},
    'inst_1_feature_coords': {fid: Nx2_coords},
}
```

### Feature Detection
```python
def detect_features_in_image(self, image_data, binary_structure_order=2):
    """
    Label connected components per state value.
    Returns: labeled_image, original_to_labels, labels_to_original
    """
```
- Uses `scipy.ndimage.label` with custom binary structure
- Each state value → multiple features (connected components)
- Returns bidirectional mappings (state ↔ feature IDs)

### Partition Combining
```python
def combine_partitions(self, image_data, combinations):
    """
    Merge multiple partition IDs into groups.
    Example: [[1, 2], [3, 4]] → merge 1&2 into single ID, 3&4 into another.
    """
```

---

## 7. Data Access Patterns

### Grain Object Access

#### Version 1 (Legacy)
```python
grain = gs[tslice].g[gid]['grain']  # Nested dict structure
state = gs[tslice].g[gid]['s']      # Grain's orientation state
```

#### Version 2 (Feature-Based)
```python
grain = gs[tslice].g[gid]           # Direct grain object
```

### Property Access
```python
# Scikit-image properties
area = grain.skprop.area
perimeter = grain.skprop.perimeter
centroid = grain.skprop.centroid  # (row, col)
moments_hu = grain.skprop.moments_hu  # 7-element array

# UPXO-specific attributes
grain.gid          # Grain ID
grain.loc          # Nx2 array of pixel indices (row, col)
grain.npixels      # Number of pixels
grain.coords       # Nx2 array of physical coordinates (x, y)
grain.bbox_bounds  # [rmin, rmax, cmin, cmax] - tight bounding box
grain.bbox_ex_bounds  # Extended by 1 pixel on each side
grain.gbloc        # Grain boundary pixel locations (Nx2)
grain.neigh        # Tuple of neighbor grain IDs
grain.s            # Orientation state ID
```

### Bounding Box Usage
```python
# Tight bounding box (exact grain extent)
bounds = grain.bbox_bounds  # [rmin, rmax, cmin, cmax]
grain_img = self.lgi[bounds[0]:bounds[1], bounds[2]:bounds[3]]

# Extended bounding box (includes 1-pixel buffer for neighbor search)
bounds_ex = grain.bbox_ex_bounds
grain_img_ex = self.lgi[bounds_ex[0]:bounds_ex[1], bounds_ex[2]:bounds_ex[3]]
```

### Grid Coordinates
```python
# Convert pixel indices to physical coordinates
x_coord = gs[tslice].xgr[row, col]
y_coord = gs[tslice].ygr[row, col]

# Pre-computed in grain object:
coords = grain.coords  # Nx2 array: [[x1, y1], [x2, y2], ...]
```

### Neighbor Data
```python
# Global neighbor map (all grains)
neigh_dict = gs[tslice].neigh_gid  # {gid: [list of neighbor gids]}

# Per-grain neighbor access (if updated during find_neigh)
neighbors = grain.neigh  # Tuple of neighbor IDs

# Neighbor property extraction
mprop = {'area': np.array([grain_areas for all grains])}
neigh_props = gs[tslice].extract_neigh_props(gids=[10, 20], mprop=mprop)
# Returns: {'gids': {10: [11, 12], 20: [21, 22]}, 
#           'vals': {10: [area11, area12], 20: [area21, area22]}}
```

---

## 8. Performance Considerations

### Memory Management
- **Large grids** (> 1M pixels): Grid storage skipped to save memory
- **`__slots__`**: Reduces per-instance memory overhead (~50% savings)
- **uint8 for lgi**: Supports up to 255 grains; use uint16/uint32 for larger structures

### Numba Optimization
- `find_neigh_v2()` uses Numba-compiled `GidOps.find_O1_neigh_2d()`
- **10-100x faster** than pure Python implementation
- **Caveat**: Crashes Jupyter kernels when called inside `char_morph_2d()` → always call separately

### Extended Bounding Boxes
- Minimize full-grid scans by searching only within `bbox_ex`
- Adds 1-pixel buffer around each grain for neighbor detection
- Trade-off: Small memory increase vs. large speed gain

### Characterization Overhead
- **Heavy computation**: `char_morph_2d()` with all properties enabled
- **Optimization**: Toggle only needed properties (set others to `False`)
- Example: Skip `perimeter_crofton`, `feret_diameter` if not needed

### Grid Resolution Scaling
```python
# Upsampling/downsampling methods exist in mcgs.py:
gs.finer(Grid_Data, ParentStateMatrix, Factor, InterpMethod)
gs.coarser(Grid_Data, ParentStateMatrix, Factor, InterpMethod)
```

### Connected Component Performance
- `char_morph_2d_v2()` uses `scipy.ndimage.label` (faster for large structures)
- `char_morph_2d()` uses `cv2.connectedComponents` (legacy, grain-by-grain)

---

## Typical Workflow Chains

### Basic Characterization
```python
from upxo.ggrowth.mcgs import mcgs

# 1. Setup and simulate
pxt = mcgs(input_dashboard='config.xls')
pxt.simulate()

# 2. Detect grains
pxt.detect_grains(mcsteps=[0, 100, 500])

# 3. Characterize (with Numba neighbor detection separate)
tslice = 100
pxt.gs[tslice].char_morph_2d(area=True, aspect_ratio=True, 
                              moments_hu=True, find_neigh=False)
pxt.gs[tslice].find_neigh_v2(p=1.0)

# 4. Build graph
G = pxt.gs[tslice].make_graph(pxt.gs[tslice].neigh_gid)

# 5. Visualize
pxt.gs[tslice].plot_grains([1, 5, 10, 15], hide_non_actors=True)
```

### Analysis Integration
```python
from upxo.analysis.analysis2d import gsan2d, kmodel

# 1. Build analysis object
gsan = gsan2d.from_mcgs2d_single(pxt.gs[tslice], 
                                 aspect_ratio=True, 
                                 moments_hu=True,
                                 find_neigh=True, 
                                 find_neigh_p=1.0)

# 2. Extract properties to DataFrame
gsan.extract_props()

# 3. Compute statistics
gsan.compute_statistics()

# 4. Network analysis
gsan.initiate_kmodel(gsids=[1], k_char_level='full')
gsan.K[1].characterize_graph()
```

### Species Assignment & Feature Analysis
```python
# 1. Assign species by combining states
pxt.gs[tslice].assign_species(method='mc state partitioned global combined',
                              combineids=[[1, 2, 3], [4, 5, 6]],
                              ninstances=10, spid=1)

# 2. Access species data
species_img = pxt.gs[tslice].species[1]['inst_1_img']
feature_props = pxt.gs[tslice].species[1]['inst_1_feature_skprops']

# 3. Analyze features
for fid, prop in feature_props.items():
    print(f"Feature {fid}: area = {prop.area}, eccentricity = {prop.eccentricity}")
```

### Property-Bounded Grain Selection
```python
# 1. Get property arrays
mprops = gsan.gsstack[gsid].get_mprops(['area', 'aspect_ratio', 'solidity'],
                                       set_missing_mprop=True)

# 2. Define thresholds
thresholds = {
    'area': [10, None],          # area <= 10
    'aspect_ratio': [2.0, None], # aspect_ratio >= 2.0
    'solidity': [0.8, None]      # solidity >= 0.8
}

# 3. Extract bounded grains
bounded_gids = pxt.gs[tslice].get_property_bounded_grains(
    pnames=['area', 'aspect_ratio', 'solidity'],
    mprops=mprops,
    pvalue_thresholds=thresholds
)

# 4. Visualize subset
pxt.gs[tslice].plot_grains(bounded_gids['aspect_ratio'], hide_non_actors=True)
```

---

## Common Pitfalls & Solutions

### 1. Jupyter Kernel Crashes
**Problem**: Calling `char_morph_2d(find_neigh=True)` crashes kernel.
**Solution**: Always call `find_neigh_v2()` separately:
```python
gs[tslice].char_morph_2d(find_neigh=False)  # Safe
gs[tslice].find_neigh_v2()                  # Call after
```

### 2. Missing Properties
**Problem**: Accessing `grain.skprop.area` before characterization → `AttributeError`.
**Solution**: Verify characterization:
```python
if not gs[tslice].are_properties_available:
    gs[tslice].char_morph_2d(area=True)
```

### 3. Version Confusion (v1 vs v2)
**Problem**: Accessing `gs[tslice].g[gid].skprop` when using v1 (nested dict).
**Solution**: Check version:
```python
if gs[tslice]._char_fx_version_ == 1:
    grain = gs[tslice].g[gid]['grain']
elif gs[tslice]._char_fx_version_ == 2:
    grain = gs[tslice].g[gid]
```

### 4. Neighbor Data Not Populated
**Problem**: `neigh_gid` is `None` → graph construction fails.
**Solution**: Always call `find_neigh()` or `find_neigh_v2()` first:
```python
if gs[tslice].neigh_gid is None:
    gs[tslice].find_neigh_v2()
```

### 5. Aspect Ratio Infinity
**Problem**: Division by zero when `minor_axis_length = 0`.
**Solution**: Handle infinities:
```python
df['aspect_ratio'] = df['major_axis_length'] / df['minor_axis_length']
df['aspect_ratio'].replace([np.inf, -np.inf], np.nan, inplace=True)
```

---

## Quick Reference Cheat Sheet

```python
# === SETUP === #
from upxo.ggrowth.mcgs import mcgs
pxt = mcgs(input_dashboard='config.xls')
pxt.simulate()
pxt.detect_grains([0])
gs = pxt.gs[0]  # Get temporal slice

# === CHARACTERIZATION === #
gs.char_morph_2d(area=True, aspect_ratio=True, moments_hu=True, 
                 find_neigh=False)  # Safe: neighbor detection OFF
gs.find_neigh_v2(p=1.0)             # Numba-accelerated neighbor detection

# === DATA ACCESS === #
grain = gs.g[gid]['grain']  # v1
grain = gs.g[gid]           # v2
area = grain.skprop.area
coords = grain.coords
neighbors = gs.neigh_gid[gid]

# === GRAPH === #
G = gs.make_graph(gs.neigh_gid)

# === VISUALIZATION === #
gs.plot_grains([1, 5, 10], hide_non_actors=True, default_cmap='jet')

# === HIGHER-ORDER NEIGHBORS === #
neigh_2nd = gs.get_nth_order_neighbors(gid=10, neigh_order=2)
neigh_upto_3rd = gs.get_upto_nth_order_neighbors(gid=10, neigh_order=3)

# === SPECIES ASSIGNMENT === #
gs.assign_species(method='mc state partitioned global combined',
                  combineids=[[1,2], [3,4]], ninstances=5)
```

---

**Last Updated**: December 2025  
**For**: Internal UPXO collaboration and AI agent reference
