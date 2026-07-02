# repgen3mcgs1.ipynb — Part C Design Specification

**Part C: 3D MC Grain Structure Generation**

**Location in notebook:** After Part B (§16), before Part D  
**Status:** Design specification (ready for implementation)  
**Date:** June 2026

---

## Overview

Part C generates 3D Monte Carlo grain structures from the `twinned3d-1.xls` input dashboard, producing temporal slices that will be analyzed in Part D for representativeness against the experimental 2D EBSD reference.

---

## Part C Structure (5 Sections)

### §17: Configuration & Setup

**Purpose:** Define MC simulation parameters and prepare for generation

```python
# ── 3D MC Configuration ────────────────────────────────────────────────────
INPUT_DASHBOARD = r'C:\Development\UPXO\upxo_library\src\upxo\demos\Twinned3D\twinned3d-1.xls'
MC_TIME_START    = 2      # Start time slice index
MC_TIME_END      = -1     # End time slice index (-1 = all)
MC_TIME_STEP     = 2      # Sample every N time steps
CHARACTERIZE     = True   # Compute grain properties
MIN_GRAIN_SIZE   = 4      # Minimum grain voxel count for characterization

print("Configuration loaded.")
```

**Expected output:**
```
Configuration loaded.
INPUT_DASHBOARD: C:\Development\UPXO\upxo_library\src\upxo\demos\Twinned3D\twinned3d-1.xls
MC_TIME_START: 2, MC_TIME_END: -1, MC_TIME_STEP: 2
```

---

### §18: Instantiate MCGS and Load Dashboard

**Purpose:** Create MCGS object and inspect configuration

```python
from upxo.ggrowth.mcgs import mcgs

# Instantiate with independent study mode
pxt = mcgs(study='independent', input_dashboard=INPUT_DASHBOARD)

# Display configuration
print("MCGS Configuration loaded:")
print(pxt)
print("\nGrid parameters:")
print(pxt.uigrid)
print("\nSimulation parameters:")
print(pxt.uisim)
```

**Expected output:**
```
MCGS Configuration loaded:
UPXO 3D.MCGS
(A: GRID)::   x:(0.0,60.0,1.0),   y:(0.0,60.0,1.0),   z:(0.0,60.0,1.0)
(B: SIMPAR)::   nstates: 16  mcsteps: 100  algorithms: (('300a', 100),)
(C: MESHPAR)::   GB Conformity: Non-Conformal

Grid parameters:
Attribues of gridding definitions: 
     TYPE: square
     DIMENSIONALITY: 3
     X: (0.0, 60.0, 1.0)
     Y: (0.0, 60.0, 1.0)
     Z: (0.0, 60.0, 1.0)
     PIXEL SIZE: 1.0
     TRANSFORMATION: none

Simulation parameters:
Attributes of Simulation parameters:
     MCSTEPS: 100
     STATE SAMPLING SCHEME: rejection
     ...
```

---

### §19: Run MC Simulation

**Purpose:** Generate 3D grain structures at all time steps

```python
# Run Monte Carlo simulation
print("Starting 3D MC grain structure simulation...")
pxt.simulate(verbose=False)

print(f"\nSimulation complete!")
print(f"Number of temporal slices: {len(pxt.m)}")
print(f"Available time slice indices: {pxt.m}")
```

**Expected output:**
```
Starting 3D MC grain structure simulation...
 Initiating Monte-Carlo simulation
     xmin, xmax, xinc: 0.0, 60.0, 1.0
     ymin, ymax, yinc: 0.0, 60.0, 1.0
     zmin, zmax, zinc: 0.0, 60.0, 1.0
     No. of states: 16
     Dimensionality: 3
Using ALG-300a
////////////////////////////////
Initiating grain growth
----------------------------------------
GS temporal slice 0 stored
GS temporal slice 1 stored
GS temporal slice 2 stored
... (many more time steps)
|--------------- MC SIM RUN COMPLETED on: ALG310---------------|
Number of gs tslices: 100

Simulation complete!
Number of temporal slices: 100
Available time slice indices: [0, 1, 2, 3, ..., 99]
```

---

### §20: Select Representative Temporal Slice(s)

**Purpose:** Choose which time step(s) to analyze in Part D

```python
# Select specific time slice for characterization
# We'll focus on later time steps (more mature grain growth)
tslice_for_analysis = pxt.m[-1]  # Latest time step (most grains)

# Alternatively, sample multiple time steps
tslices_for_analysis = pxt.m[MC_TIME_START::MC_TIME_STEP]

print(f"Time slice(s) selected for analysis: {tslices_for_analysis}")

# Get grain structure at selected time slice
gstslice = pxt.gs[tslice_for_analysis]
print(f"\nGrain structure at time slice {tslice_for_analysis}:")
print(gstslice)
```

**Expected output:**
```
Time slice(s) selected for analysis: [2, 4, 6, 8, ..., 98]
Grain structure at time slice 98:
UPXO. gs-tslice.3d. 2264384898032
```

---

### §21: Characterize 3D Grain Structure

**Purpose:** Compute morphological properties for representativeness analysis

```python
# Characterize grain morphology
gstslice.char_morphology_of_grains(
    label_str_order=1,
    find_grain_voxel_locs=True,
    find_spatial_bounds_of_grains=True,
    force_compute=True
)

# Set morphological properties
gstslice.set_mprops(
    volnv=True,           # Volume (voxel count)
    eqdia=True,           # Equivalent diameter
    eqdia_base_size_spec='volnv',
    arbbox=False,
    arellfit=False,
    solidity=False,
    sanv=False            # Surface area (defer for now)
)

print(f"Grain characterization complete!")
print(f"Number of grains: {gstslice.n}")
print(f"Available properties: {list(gstslice.mprop.keys())}")
```

**Expected output:**
```
---------------------------------------- 
Finding grains.
No. of grains detected = 2294
---------------------------------------- 
Setting PyVista grid.
...
Grain characterization complete!
Number of grains: 2294
Available properties: ['volnv', 'eqdia', ...]
```

---

### §22: Summary Statistics

**Purpose:** Display overview of 3D grain structure

```python
# Get grain statistics
n_grains = gstslice.n
print(f"3D MC Grain Structure Summary (Time Slice {tslice_for_analysis})")
print(f"{'─' * 60}")
print(f"Domain size: 60 × 60 × 60 voxels (60 × 60 × 60 μm)")
print(f"Number of grains: {n_grains}")
print(f"Voxel size: {pxt.vox_size} μm")

# Volume statistics
volnv = gstslice.mprop['volnv']
print(f"\nVolume statistics (voxel-based):")
print(f"  Mean: {np.mean(volnv):.1f} voxels")
print(f"  Median: {np.median(volnv):.1f} voxels")
print(f"  Std Dev: {np.std(volnv):.1f} voxels")
print(f"  Min: {np.min(volnv):.1f} voxels")
print(f"  Max: {np.max(volnv):.1f} voxels")

# Equivalent diameter statistics
eqdia = gstslice.mprop['eqdia']
print(f"\nEquivalent diameter statistics:")
print(f"  Mean: {np.mean(eqdia):.2f} μm")
print(f"  Median: {np.median(eqdia):.2f} μm")
print(f"  Std Dev: {np.std(eqdia):.2f} μm")
print(f"  Min: {np.min(eqdia):.2f} μm")
print(f"  Max: {np.max(eqdia):.2f} μm")

print(f"\nReady for Part D: 3D Representativeness Assessment")
```

**Expected output:**
```
3D MC Grain Structure Summary (Time Slice 98)
────────────────────────────────────────────────────────────
Domain size: 60 × 60 × 60 voxels (60 × 60 × 60 μm)
Number of grains: 2294
Voxel size: 1.0 μm

Volume statistics (voxel-based):
  Mean: 157.3 voxels
  Median: 89.5 voxels
  Std Dev: 312.4 voxels
  Min: 4.0 voxels
  Max: 4521.0 voxels

Equivalent diameter statistics:
  Mean: 7.23 μm
  Median: 5.84 μm
  Std Dev: 6.51 μm
  Min: 1.63 μm
  Max: 37.42 μm

Ready for Part D: 3D Representativeness Assessment
```

---

## Key Design Decisions

### 1. Multi-Slice Sampling

**Decision:** Sample time steps at intervals (e.g., every 2 time steps) rather than analyzing all 100

**Rationale:**
- Computational efficiency for Part D (fewer slices to compare)
- Captures grain growth progression
- Typical sampling: `MC_TIME_START=2, MC_TIME_STEP=2` → 50 slices

### 2. Defer Surface Area Calculation

**Decision:** Skip surface area (`sanv=False`) in initial characterization

**Rationale:**
- Surface area computation is slower
- Not needed for initial 3D representativeness assessment
- Can be added in Part D if required for detailed analysis
- Morphological volume/diameter sufficient for grain size distribution

### 3. Store Temporal Slices

**Decision:** Keep `pxt.gs` dictionary in memory (all time steps)

**Rationale:**
- Needed for Part D multi-axis slice extraction
- Allows flexible time step selection
- Memory footprint acceptable (~3D structures × 8 GB RAM typical)

---

## Expected File Sizes & Counts

| Parameter | Typical Value | Notes |
|---|---|---|
| Domain | 60×60×60 voxels | From twinned3d-1.xls |
| Total voxels | 216,000 | 60³ |
| Temporal slices | 100 | MC time steps |
| Grains per slice (early) | ~50 | t=0 |
| Grains per slice (late) | ~2500 | t=100 |
| Memory per slice (LGI) | ~1 MB | 60×60×60 × int32 |
| Memory per slice (orientations) | ~100 KB | Per grain dict |
| Total memory (all slices) | ~150–200 MB | Entire simulation |

---

## Integration with Part D

### Output from Part C for Part D

```python
# Part D will use:
pxt.gs              # Dictionary of all temporal slices
pxt.m               # List of available time step indices
gstslice.lgi        # Voxel labels for slice extraction
gstslice.mprop      # Grain properties (volume, diameter)
```

### Slice Extraction Strategy (Preview)

Part D will:
1. Loop through `pxt.m` (time slices)
2. For each time slice, extract 2D slices along X, Y, Z axes
3. Compare against EBSD MDF reference (from Parts A–B)
4. Score representativeness per axis
5. Recommend best time slice for twin introduction

---

## Implementation Checklist

- [ ] §17: Configuration cell (copy as-is)
- [ ] §18: MCGS instantiation and dashboard loading
- [ ] §19: Run simulation (verbose=False)
- [ ] §20: Select time slices for analysis
- [ ] §21: Characterize grain morphology
- [ ] §22: Summary statistics

---

## Notes

- Part C is **sequential and deterministic** (once simulation runs, output is fixed)
- All 100 time steps are stored in `pxt.gs` automatically
- Part D will **selectively analyze** time slices based on representativeness scoring
- No parallel or background processing needed in Part C
- Typical runtime: 15–30 seconds for full simulation

---

**Status:** Ready for implementation  
**Next step:** Implement Part C into repgen3mcgs1.ipynb

