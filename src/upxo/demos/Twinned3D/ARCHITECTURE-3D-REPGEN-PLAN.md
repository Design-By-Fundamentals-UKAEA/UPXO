# 3D Representative Twinned Grain Structure Generation
## Architecture & Integration Plan

**Project:** Extending 2D repgen2mcgs to 3D (CCZr copper twinning)  
**Location:** `src/upxo/demos/Twinned3D/`  
**Status:** Planning phase (no implementation yet)  
**Date:** June 2026

---

## 1. Current State Analysis

### 1.1 Existing 3D MC Grain Structure Generation

**File:** `demos/gsgen/mcgs3d01.ipynb`  
**Module:** `upxo.ggrowth.mcgs`

```python
from upxo.ggrowth.mcgs import mcgs

pxt = mcgs(study='independent', input_dashboard='twinned3d-1.xls')
pxt.simulate(verbose=False)
pxt.gs[0], pxt.gs[1], ..., pxt.gs[N]  # Temporal slices
```

**Status:** ✓ Fully functional, generates 3D grain structures at multiple MC time steps

### 1.2 Existing 3D Twin Morphology Generation

**File:** `demos/twins/mcgs3d02.ipynb`  
**Module:** `upxo.geoEntities.plane` + `gstslice.instantiate_twins()`

**Current Approach:**
```python
twspec = {
    'n': [5, 10, 3],                    # Twin counts per dimension
    'tv': np.array([5, -3.5, 5]),       # Twin vectors
    'dlk': np.array([1.0, -1.0, 1.0]),  # Delta length/kinetics
    'dnw': np.array([0.5, 0.5, 0.5]),   # Delta normal width
    'dno': np.array([0.5, 0.5, 0.5]),   # Delta normal orientation
    'tdis': 'normal',                    # Thickness distribution
    'tpar': {'loc': 1.12, 'scale': 0.1, 'val': 1},
    'vf': [0.05, 1.00],                 # Volume fraction range
    'sep_bzcz': False
}

twgenspec = {
    'seedsel': 'random_gb',              # Seed selection strategy
    'K': 20,                             # Number of grains
    'bidir_tp': False,                   # Bidirectional twin plane
    'checks': [True, True]               # Validity checks
}

gstslice.instantiate_twins(
    ninstances=2,
    twspec=twspec,
    twgenspec=twgenspec
)
```

**Status:** ✓ Morphologically working, but **CRYSTALLOGRAPHICALLY LIMITED**

### 1.3 2D Representative Grain Structure (Reference Implementation)

**File:** `demos/repGen/repgen2mcgs4.ipynb`

**Complete workflow:**
- Part A: EBSD pre-processing (MDF, CSL segregation, parent/twin identification)
- Part B: Grain role analysis (parent vs. primary/secondary twins)
- Part C: MC grain structure generation
- Part D: Representativeness ranking (5 metrics: Wasserstein, KS, AD, energy, ratio)
- Part E: Twin host allocation
- Part F: EBSD orientation sampling
- Part G: Twin lamella introduction + MDF verification
- Part H: Conformal meshing

**Status:** ✓ Complete pipeline with crystallographic validation via MDF

---

## 2. Crystallographic Limitations in Current 3D Twin Generation

### 2.1 What's Missing

| Capability | Current Status | Required For |
|---|---|---|
| Habit plane calculation (quaternion → 3D plane) | ✗ Not implemented | Crystallographic accuracy |
| Σ3 twin orientation assignment | ✗ Not implemented | CSL relationship validation |
| MDF computation for twins | ✗ Not implemented | Representativeness verification |
| Conflict-free adjacent grain orientations | ✗ Not implemented | Preventing grain merger |
| EBSD orientation pool (2D→3D) | ✗ Not implemented | Realistic texture distribution |
| CSL misorientation verification | ✗ Not implemented | Quality assurance |

### 2.2 Current Approach (Morphological Only)

```
Plane-based slicing (geometric)
    ↓
Voxel selection based on distance
    ↓
Twin lamella creation (NO orientation assignment)
    ↓
Result: Correct morphology, MISSING crystallography
```

### 2.3 Required Approach (Morphological + Crystallographic)

```
Plane-based slicing (geometric) ✓
    ↓
Voxel selection based on distance ✓
    ↓
Habit plane calculation from parent quaternion ← NEW
    ↓
Twin orientation assignment via Σ3 rotation ← NEW
    ↓
Conflict-free adjacent grain checking ← NEW
    ↓
MDF computation and validation ← NEW
    ↓
Result: Correct morphology + crystallography
```

---

## 3. Input Dashboard: twinned3d-1.xls

**Location:** `src/upxo/demos/Twinned3D/twinned3d-1.xls`  
**File size:** 63 KB  
**Created:** 2023, last modified June 24, 2026

**Structure:**
- Sheet `mcgs`: 141 rows × 9 columns — MC grain structure parameters
- Sheet `materials`: Material specifications
- Sheet `s_skip`: Skipped configurations
- Sheet `sori`: Orientation parameters

**Usage in pipeline:**
```python
pxt = mcgs(study='independent', 
          input_dashboard=r'C:\...\twinned3d-1.xls')
pxt.simulate(verbose=False)

# Access temporal slices
for tslice in pxt.m:
    gs_3d = pxt.gs[tslice]  # 3D grain structure at time step tslice
```

---

## 4. Proposed 3D Repgen Architecture

### 4.1 Overall Pipeline Flow

```
┌─────────────────────────────────────────────────────────────┐
│ EXPERIMENTAL INPUT: 2D EBSD Data (Copper-Chromium-Zirconium)│
│ • Grain structure with twins                                 │
│ • Misorientation distribution (MDF)                         │
│ • Twin area fractions (TVF) per role                        │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ PART A: EBSD PRE-PROCESSING (from repgen2mcgs4)             │
│ • Load CTF file, clean & characterize                       │
│ • Compute MDF, identify CSL pairs                           │
│ • Segregate parents, primary/secondary twins                │
│ • Compute TVF by generation                                 │
│ • Build merged EBSD LFI                                     │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ PART B: EBSD GRAIN ROLE ANALYSIS                            │
│ • Extract morphological properties per role                 │
│ • Compute statistics (mean, std, Q1/Q2/Q3)                  │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ PART C: 3D MC GRAIN STRUCTURE GENERATION                    │
│ • Load twinned3d-1.xls input dashboard                      │
│ • Generate 3D MC grain structures at N time steps           │
│ • Output: pxt.gs[0], pxt.gs[1], ..., pxt.gs[N]             │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ PART D: 3D REPRESENTATIVENESS ASSESSMENT (NEW LOGIC)        │
│                                                              │
│ For each MC temporal slice:                                 │
│   FOR axis in [X, Y, Z]:  (user-configurable)              │
│     FOR each slice position:                                │
│       Extract 2D slice perpendicular to axis                │
│       Compute 2D MDF, morphology (area, etc.)               │
│       Compare to EBSD reference (5 metrics)                 │
│       Record: pass/fail                                     │
│   Aggregate: % pass per axis                                │
│   Accept if: % pass ≥ P (default 60%) per axis              │
│                                                              │
│ Output: Ranked list of best-representative MC slices        │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ PART E: TWIN HOST ALLOCATION (from repgen2mcgs4)            │
│ • Select best-representative 3D MC slice                    │
│ • Target twin hosting fraction from EBSD TVF               │
│ • Designate largest grains as hosts                         │
│ • Achieve target hosting fraction                           │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ PART F: ORIENTATION ASSIGNMENT (HYBRID)                     │
│                                                              │
│ HOST GRAINS (with twins):                                   │
│   • Sample parent quaternions from EBSD pool                │
│   • Conflict-free assignment (adjacent hosts differ)        │
│   • Apply Σ3 rotation → twin orientation (stored)           │
│                                                              │
│ NON-HOST GRAINS (background):                               │
│   • Build 3D orientation pool from 2D EBSD                  │
│   • Replicate + small random jitter (±2°)                   │
│   • Conflict-free assignment vs neighbors                   │
│   • Result: Realistic 3D texture                            │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ PART G: TWIN LAMELLA INTRODUCTION (CRYSTALLOGRAPHIC)        │
│                                                              │
│ For each host grain:                                        │
│   1. Calculate habit planes from parent quaternion          │
│   2. Select one {111} plane (randomly or stress-based)      │
│   3. Compute Σ3 twin orientation (60° about ⟨111⟩)         │
│   4. Generate lamellae geometry (parallel slabs)            │
│   5. Assign twin orientation (stored per sub-block)         │
│   6. Repeat for secondary twins if TVF > threshold          │
│                                                              │
│ Output: Full 3D structure with orientations                 │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ PART H: MDF VERIFICATION (2D SLICES & 3D)                   │
│                                                              │
│ 2D slice MDF:                                               │
│   • Extract slices along X, Y, Z                            │
│   • Compute MDF for each slice                              │
│   • Overlay on EBSD reference (should match)                │
│                                                              │
│ 3D MDF:                                                     │
│   • Compute all grain-boundary pairs in 3D                  │
│   • Build complete MDF with host-host, host-twin,           │
│     twin-twin boundaries                                    │
│   • Verify Σ3 peak (~60°) present                          │
│   • Match texture to EBSD                                   │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ PART I: VISUALIZATION (2D + 3D)                             │
│                                                              │
│ 2D Slices:                                                  │
│   • User-specified axis + position                          │
│   • IPF coloring + grain boundaries                         │
│   • matplotlib output                                       │
│                                                              │
│ 3D Volume:                                                  │
│   • PyVista rendering with nipy_spectral cmap               │
│   • Per-role transparency (hosts opaque, twins, nonhosts)   │
│   • Interactive rotation/zoom                              │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ FINAL OUTPUT: 3D REPRESENTATIVE TWINNED GRAIN STRUCTURE     │
│                                                              │
│ Files:                                                      │
│   • 3D LGI (voxel labels): numpy array                      │
│   • Orientations (quaternions): per grain dict              │
│   • Twin metadata: parent/twin relationships                │
│   • MDF data: validation against EBSD                       │
│   • Statistics: volume fractions, texture                   │
│   • Visualizations: 2D slices + 3D volume                   │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 Key Integration Points

| Component | Source | Adaptation |
|---|---|---|
| EBSD loading & MDF | repgen2mcgs4 Part A | Reuse with CCZr data |
| Property selection | repgen2mcgs4 Part B | Reuse same properties |
| MC generation | mcgs3d01 | Reuse MCGS class |
| Representativeness metrics | repgen2mcgs4 Part D | Adapt to 2D slices from 3D |
| Twin host allocation | repgen2mcgs4 Part E | Reuse logic for 3D grains |
| Orientation sampling | repgen2mcgs4 Part F | Build 3D pool from 2D EBSD |
| Habit plane calculation | NEW SECTION | Quaternion → 3D {111} planes |
| Twin orientation assignment | mcgs3d02 + NEW LOGIC | Add crystallographic Σ3 |
| MDF computation | NEW SECTION | All 3D boundary pairs |

---

## 5. Implementation Roadmap

### Phase 1: Framework Setup (Weeks 1–2)
- [ ] Set up notebook structure (Parts A–I)
- [ ] Load twinned3d-1.xls and generate 3D MC structures
- [ ] Implement 2D slice extraction (X, Y, Z orthogonal)
- [ ] Load EBSD reference data (copper CCZr)

### Phase 2: Representativeness Assessment (Weeks 2–3)
- [ ] Implement slice representativeness metrics (Wasserstein, KS, AD, energy, ratio)
- [ ] Per-axis pass/fail determination
- [ ] Overall 3D acceptance logic (P-percentage per axis)
- [ ] Rank best-representative MC slices

### Phase 3: Crystallographic Framework (Weeks 3–4)
- [ ] Implement habit plane calculation (quaternion → 3D)
- [ ] Implement Σ3 orientation assignment
- [ ] Implement conflict-free adjacent grain checking
- [ ] Build 3D orientation pool from 2D EBSD

### Phase 4: Twin Introduction & Validation (Weeks 4–5)
- [ ] Integrate morphological twin generation (existing `instantiate_twins`)
- [ ] Add crystallographic orientation assignment to twins
- [ ] Implement MDF computation (2D slices + full 3D)
- [ ] Add MDF verification against EBSD

### Phase 5: Visualization & Output (Weeks 5–6)
- [ ] Implement 2D slice visualization (matplotlib)
- [ ] Implement 3D PyVista visualization (nipy_spectral, per-role opacity)
- [ ] Export to numpy format (`.npy` files)
- [ ] Generate comprehensive reports

---

## 6. Critical Design Decisions

### 6.1 Multi-Directional 2D Slice Validation (User Innovation)

Rather than attempting 3D EBSD validation (limited data), validate:
- Extract 2D slices perpendicular to X, Y, Z axes
- Compare each 2D slice against experimental 2D EBSD
- Aggregate: all three axes must pass for 3D acceptance

**Advantage:** Leverages abundant 2D EBSD data, detects anisotropy

### 6.2 Morphology Reuse with Orientation Variants

- Generate unique 3D morphology (voxel LGI)
- Reuse same LGI, regenerate orientations for variants
- **Savings:** ~66% memory for orientation-only variants

### 6.3 2D→3D EBSD Pool Construction

- No 3D EBSD data available
- Build 3D pool by replicating 2D EBSD
- Apply small jitter (±2°) per layer for realism
- **Result:** Realistic 3D texture from limited 2D data

---

## 7. Configuration Defaults

```python
# 3D representativeness assessment
CONFIG_3D = {
    'n_slices_x': 10,
    'n_slices_y': 10,
    'n_slices_z': 10,
    'test_along_x': True,
    'test_along_y': True,
    'test_along_z': True,
    'p_percentage': 60.0,  # Default acceptance threshold
}

# Twin specifications (from existing twspec structure)
TWSPEC_3D = {
    'vf': [0.05, 1.00],           # Volume fraction range
    'tdis': 'normal',             # Thickness distribution
    'tpar': {'loc': 1.12, 'scale': 0.1, 'val': 1},
}

# Crystallographic constraints
HABIT_PLANE_PLANES = 4             # Four {111} planes per grain
SIGMA3_ANGLE_DEG = 60              # Σ3 twin rotation angle
MIN_ADJACENT_MISORI_DEG = 5.0      # Minimum separation between adjacent grains
```

---

## 8. Expected Outputs

### 8.1 Per MC Temporal Slice

```
Representative MC Slice Analysis
  Time step: 90
  Domain size: 60 × 60 × 60 voxels (60 × 60 × 60 μm)
  Number of grains: 2294
  
  X-normal slices (10 slices):
    Pass: 8/10 (80%) → ✓ ACCEPTED (≥60%)
  
  Y-normal slices (10 slices):
    Pass: 6/10 (60%) → ✓ ACCEPTED (≥60%)
  
  Z-normal slices (10 slices):
    Pass: 5/10 (50%) → ✗ REJECTED (<60%)
  
  Overall: ✗ NOT ELIGIBLE (Z-axis failed)
```

### 8.2 Final Output Package

```
Twinned3D_Representative_Structure/
├── 3d_grain_structure.npy          # 3D LGI (60×60×60)
├── host_grain_quaternions.json     # Parent orientations
├── twin_orientations.json          # Twin orientations (Σ3)
├── grain_roles.json                # Parent/primary/secondary labels
├── mdf_ebsd_reference.npy          # Reference MDF
├── mdf_3d_computed.npy             # Computed MDF (all boundaries)
├── representativeness_report.txt   # Full validation report
├── 2d_slices/
│   ├── x_normal_slice_30.png       # 2D visualization
│   ├── y_normal_slice_30.png
│   └── z_normal_slice_30.png
└── visualization_3d.html           # PyVista interactive viewer
```

---

## 9. Next Steps (Before Implementation)

1. **Clarification needed:**
   - Confirm twinned3d-1.xls MC domain size and voxel count
   - Confirm copper CCZr EBSD file location and format
   - Confirm expected twin volume fractions for copper

2. **Scope discussion:**
   - Should we support multiple MC time slices or just best match?
   - Should twin host allocation use EBSD-driven targeting or user-specified count?
   - Should we export to conformal mesh format in this pipeline or keep separate?

3. **Code search needed:**
   - Where is CSL relationship data stored in UPXO?
   - Where is the quaternion-to-Euler conversion utility?
   - Where is the plane class used for habit plane calculations?

---

**Ready for discussion and refinement before implementation begins.**
