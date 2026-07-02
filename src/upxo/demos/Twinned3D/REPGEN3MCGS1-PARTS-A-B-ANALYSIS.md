# repgen3mcgs1.ipynb — Parts A & B Analysis

**Status:** Ready for copy with NO CHANGES  
**Date:** June 2026

---

## Summary

**Part A (EBSD Pre-processing):** Copy verbatim ✓  
**Part B (Grain Role Analysis):** Copy verbatim ✓  

Both parts are identical between 2D and 3D pipelines because:
- We use the same 2D EBSD reference data
- Properties analysis applies equally to 2D grains and 3D grains
- No changes required to the physics or methodology

---

## Part A: EBSD Pre-processing (§1–14)

### Sections (unchanged)

| Section | Content | 3D Status | Changes |
|---|---|---|---|
| §1 | Configuration (file paths, tolerances) | Use CCZr copper file | Only file path + `INPUT_DASHBOARD` → `twinned3d-1.xls` |
| §2 | Import statements | All imports needed | **NONE** |
| §3 | EBSD loading, subsampling, cropping | 2D EBSD processing | **NONE** |
| §4 | Initialize repgen2d object | Creates rg object | **NONE** |
| §5 | Compute MDF | Misorientation distribution | **NONE** |
| §6 | Select MDF peaks | Interactive selection | **NONE** |
| §7 | Plot MDF with peaks | Visualization | **NONE** |
| §8 | Segregate CSL pairs | Parent/twin identification | **NONE** |
| §9 | Compute CSL volume fractions | TVF statistics | **NONE** |
| §10 | Identify parent grains | Parent/primary/secondary classification | **NONE** |
| §11 | Compute EBSD TVF | Twin area fractions | **NONE** |
| §12 | Plot EBSD TVF | Visualization | **NONE** |
| §13 | Build merged EBSD LFI | Merge grain structure | **NONE** |
| §14 | Merge provenance | Statistics | **NONE** |

### Code Changes Required

**Configuration cell only:**

```python
# BEFORE (2D repgen2mcgs4):
CTF_FILE        = r'C:\Development\EBSD datasets\UKAEA__OFHCCu\OFHC_Cu_dataset\EBSD_pre\warp_out_s2.ctf'
INPUT_DASHBOARD = r'C:\Development\UPXO\upxo_library\src\upxo\demos\repGen\tgs2d.xls'

# AFTER (3D repgen3mcgs1):
CTF_FILE        = r'C:\Development\EBSD datasets\UKAEA__OFHCCu\OFHC_Cu_dataset\EBSD_pre\warp_out_s2.ctf'  # SAME
INPUT_DASHBOARD = r'C:\Development\UPXO\upxo_library\src\upxo\demos\Twinned3D\twinned3d-1.xls'  # CHANGED
```

**Everything else: VERBATIM COPY**

---

## Part B: Grain Role Property Analysis (§15–16)

### Sections (unchanged)

| Section | Content | 3D Status | Changes |
|---|---|---|---|
| §15 | Select GS properties for twin analysis | Interactive widget | **NONE** |
| §16 | Read properties and view distributions | Statistics by grain role | **NONE** |

### Physical Reasoning: Why No Changes Needed

The 2D repgen2mcgs4 notebook analyzes **grain morphological properties** (area, perimeter, aspect ratio, eccentricity, solidity, etc.) segregated by **grain role** (pure parent, primary twin, secondary twin, intermediate twin, non-role).

For 3D repgen3mcgs1:
- **3D grains** (from 3D MC structure) have equivalent morphological properties
- **Same grain roles** can be assigned to 3D grains
- **Same properties are computed** (volume, surface area, aspect ratio, etc.)
- The **logic for property analysis is identical**

The role-based segregation is performed on the EBSD data (Part A), then applied to the MC grains (Parts C–F). The properties themselves are dimensionality-agnostic.

### Code Changes Required

**NONE.** Copy §15–16 verbatim.

The user will:
1. Run §15 interactive widget to select properties (same choices available)
2. Run §16 to compute statistics on EBSD grains segregated by role

Example output for 3D (identical structure to 2D):
```
Grain role morphological property statistics (EBSD):

Pure parents (n=549):
  volnv     : mean=1827.3 ± 2156.2  min=6.4   max=24364.8
  ecc       : mean=0.842 ± 0.142    min=0.174 max=1.000

Primary twins (n=88):
  volnv     : mean=892.7 ± 1125.4   min=10.0  max=8950.3
  ecc       : mean=0.825 ± 0.165    min=0.148 max=0.998

Secondary twins (n=35):
  volnv     : mean=645.2 ± 897.1    min=8.5   max=5123.4
  ecc       : mean=0.818 ± 0.171    min=0.131 max=0.997

... (more roles)
```

---

## Implementation Instruction for repgen3mcgs1.ipynb

### Step 1: Create new notebook
```
C:\Development\UPXO\upxo_library\src\upxo\demos\Twinned3D\repgen3mcgs1.ipynb
```

### Step 2: Copy structure from repgen2mcgs4.ipynb

**Copy EXACTLY:**
- Title cell: "# repgen3d pipeline v1 — full 3D twin-aware workflow" (updated title)
- Part A header: "## Part A — EBSD pre-processing"
- §1 through §14: All cells (configuration, imports, EBSD processing, MDF, CSL, TVF, merged LFI)
- Part B header: "## Part B — Grain role property analysis (twin GS)"
- §15 through §16: Property selection and statistics

### Step 3: ONLY configuration change

In configuration cell (§1):
- Keep `CTF_FILE` pointing to copper EBSD
- Change `INPUT_DASHBOARD` to `twinned3d-1.xls`
- Keep `MIN_GRAIN_SIZE`, `MISORI_TOL`, `CROP_REGION` as-is

### Step 4: After Part B, NEW sections begin

After §16 output, we will create:
- **Part C:** 3D MC grain structure generation (from twinned3d-1.xls)
- **Part D:** 3D representativeness assessment (2D slices, multi-axis validation)
- **Part E–onwards:** Twin allocation, orientation assignment, etc. (adapted from repgen2mcgs4)

---

## Verification Checklist (Before Implementation)

- [ ] Confirm copper CCZr EBSD file location correct
- [ ] Confirm twinned3d-1.xls dashboard will generate sufficient temporal slices
- [ ] Confirm repgen2d class methods (`compute_mdf_ebsd`, `segregate_csl_pairs`, etc.) work with the copper data
- [ ] Confirm no data preprocessing differences (different EBSD conventions?)

---

## Expected Part A Output (Example)

```
Loaded EBSD data (dimensions: 513 x 363 pixels, step size: 0.8 um)
[crop] region=[1, 1, 99, 99]%  →  cropped 355×503

[rechar_lfi] Map 355×503  |  unknown pixels: 3619 (2.03%)
clean_and_rechar_from_rdr() completed in 2.7 s
Non-positive pixels remaining : 111
Grains after cleaning         : 1058

Property statistics:
  area      : mean=107.9 ± 212.3 µm²
  perimeter : mean=40.4 ± 45.5 µm
  eq_diameter : mean=9.2 ± 7.2 µm

MDF peaks identified: 12
CSL pairs segregated: Σ3, Σ7, Σ9, Σ27...

Merged EBSD:
  Grains total    : 673
  Grains with twins : 212
  Twin-hosting fraction : 0.3150
```

---

## Expected Part B Output (Example)

```
Selected properties: [area, perimeter, aspect_ratio, eccentricity, solidity]

Grain role statistics:

Pure parents (n=549):
  area      : Q1=23.4  Q2=78.5  Q3=156.2 µm²
  eccentricity : Q1=0.76 Q2=0.85 Q3=0.91

Primary twins (n=88):
  area      : Q1=12.1  Q2=45.3  Q3=98.7 µm²
  eccentricity : Q1=0.71 Q2=0.83 Q3=0.89

... (all roles)

Property distributions visualized (KDE curves).
```

---

## Summary: Ready to Copy

✓ **Part A:** Exact copy (1 config line change)  
✓ **Part B:** Exact copy (0 changes)  
✓ **Both:** No physics or methodology changes  
✓ **Only:** Dashboard file path updated to 3D MC input

**Proceed with implementation upon user approval.**

---

**End of Analysis**
