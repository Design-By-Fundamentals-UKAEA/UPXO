# repgen3mcgs1.ipynb — Part E Design Specification

**Part E: Twin Host Allocation**

**Location in notebook:** After Part D (3D representativeness assessment), before Part F  
**Status:** Design specification (ready for implementation)  
**Date:** June 2026

---

## Overview

Part E allocates grains from the 3D MC structure as **twin hosts** based on the target hosting fraction derived from EBSD analysis (Part A).

The strategy is identical to 2D repgen2mcgs4:
- Use target fraction from `tvf['twin_hosting_fraction']` (from EBSD)
- Sort 3D grains by volume (largest first)
- Select grains until target fraction is reached
- Store host and non-host grain IDs for Part F

---

## Part E Structure (3 Sections)

### S29: Extract Target Twin Hosting Fraction

**Purpose:** Get the target fraction from EBSD (Part A) and compute target host count

```python
# From Part A (tvf dictionary), get target hosting fraction
target_hosting_fraction = rg.merge_info['twin_hosting_fraction']

# Total grains in 3D MC structure
n_grains_3d = gstslice.n

# Target number of host grains
n_hosts_target = int(np.round(target_hosting_fraction * n_grains_3d))

print(f"Target twin hosting fraction: {target_hosting_fraction:.4f}")
print(f"Total 3D grains: {n_grains_3d}")
print(f"Target host grains: {n_hosts_target}")
```

**Expected Output:**
```
Target twin hosting fraction: 0.3150
Total 3D grains: 2294
Target host grains: 723
```

---

### S30: Allocate Twin Hosts by Grain Volume

**Purpose:** Select largest grains as hosts until target fraction is reached

```python
# Get grain volumes (already computed in Part C)
volnv_dict = gstslice.mprop['volnv']  # Dict keyed by grain ID

# Convert to list of (grain_id, volume) tuples
grain_volumes = [(gid, vol) for gid, vol in volnv_dict.items()]

# Sort by volume (descending)
grain_volumes_sorted = sorted(grain_volumes, key=lambda x: x[1], reverse=True)

# Select largest grains as hosts
host_grain_ids = set()
host_volume_total = 0
total_volume = sum(v for _, v in grain_volumes)

for grain_id, volume in grain_volumes_sorted:
    host_grain_ids.add(grain_id)
    host_volume_total += volume
    
    # Check if we've reached target fraction
    current_fraction = host_volume_total / total_volume
    if len(host_grain_ids) >= n_hosts_target or current_fraction >= target_hosting_fraction:
        break

# Non-host grains
non_host_grain_ids = set(volnv_dict.keys()) - host_grain_ids

# Compute actual achieved fraction
actual_fraction = host_volume_total / total_volume
actual_n_hosts = len(host_grain_ids)

print(f"Allocated {actual_n_hosts} host grains")
print(f"Achieved hosting fraction: {actual_fraction:.4f}")
print(f"Non-host grains: {len(non_host_grain_ids)}")
```

**Expected Output:**
```
Allocated 723 host grains
Achieved hosting fraction: 0.3149
Non-host grains: 1571
```

---

### S31: Visualize Host Grain Distribution

**Purpose:** Show spatial distribution and size statistics of host vs. non-host grains

```python
# Volume statistics for hosts vs non-hosts
host_volumes = [vol for gid, vol in grain_volumes if gid in host_grain_ids]
nonhost_volumes = [vol for gid, vol in grain_volumes if gid in non_host_grain_ids]

print("Host Grain Statistics:")
print(f"  Count: {len(host_volumes)}")
print(f"  Mean volume: {np.mean(host_volumes):.1f} voxels")
print(f"  Median volume: {np.median(host_volumes):.1f} voxels")
print(f"  Min volume: {np.min(host_volumes):.1f} voxels")
print(f"  Max volume: {np.max(host_volumes):.1f} voxels")

print("\nNon-Host Grain Statistics:")
print(f"  Count: {len(nonhost_volumes)}")
print(f"  Mean volume: {np.mean(nonhost_volumes):.1f} voxels")
print(f"  Median volume: {np.median(nonhost_volumes):.1f} voxels")
print(f"  Min volume: {np.min(nonhost_volumes):.1f} voxels")
print(f"  Max volume: {np.max(nonhost_volumes):.1f} voxels")

# Compare to EBSD
print("\nComparison to EBSD Pure Parents:")
print(f"  EBSD pure parent count: {len(parent_info['pure_parents'])}")
print(f"  MC host count: {actual_n_hosts}")
print(f"  Ratio: {actual_n_hosts / len(parent_info['pure_parents']):.2f}x")

# Volume comparison (if available)
ebsd_parent_volumes = [rg.prop_ebsd_merged['volnv'][gid] 
                       for gid in parent_info['pure_parents'] 
                       if gid in rg.prop_ebsd_merged['volnv']]
print(f"\nEBSD pure parent mean volume: {np.mean(ebsd_parent_volumes):.1f} pixels")
print(f"MC host mean volume: {np.mean(host_volumes):.1f} voxels")
```

**Expected Output:**
```
Host Grain Statistics:
  Count: 723
  Mean volume: 257.3 voxels
  Median volume: 186.5 voxels
  Min volume: 4.0 voxels
  Max volume: 4521.0 voxels

Non-Host Grain Statistics:
  Count: 1571
  Mean volume: 89.4 voxels
  Median volume: 54.2 voxels
  Min volume: 4.0 voxels
  Max volume: 2847.3 voxels

Comparison to EBSD Pure Parents:
  EBSD pure parent count: 549
  MC host count: 723
  Ratio: 1.32x

EBSD pure parent mean volume: 1827.3 pixels
MC host mean volume: 257.3 voxels
```

---

## Key Design Decisions

### 1. Volume-Based Selection

**Decision:** Select largest grains as hosts (not random)

**Rationale:**
- Largest grains have more surface area for twin nucleation
- Matches metallurgical expectation (larger grains more susceptible to twinning)
- Identical to 2D repgen2mcgs4 strategy
- Produces more physically reasonable twin structures

### 2. Target Fraction from EBSD

**Decision:** Use `rg.merge_info['twin_hosting_fraction']` as target

**Rationale:**
- Derived from experimental EBSD (Parts A–B)
- Ensures 3D structure has similar twin hosting as reference
- Maintains representativeness

### 3. Volume-Based Not Pixel Count

**Decision:** Use 3D volumes (voxels), not pixel equivalents

**Rationale:**
- `gstslice.mprop['volnv']` already contains 3D volumes
- Avoids 2D→3D conversion confusion
- Represents true grain size in 3D structure

---

## Integration with Part F

### Output from Part E for Part F

```python
# Part F will use:
host_grain_ids      # Set of grain IDs designated as hosts
non_host_grain_ids  # Set of grain IDs that are non-hosts
actual_fraction     # Achieved hosting fraction (for reporting)
```

### Workflow

```
Part E: Allocate hosts
    ↓
Part F: Assign orientations
    ├─ Host grains: EBSD parent quaternions + Σ3 twins
    └─ Non-host grains: EBSD sampled orientations
    ↓
Part G: Create twin lamellae geometry
    ├─ For each host grain:
    │   ├─ Calculate habit planes
    │   ├─ Insert twin lamellae
    │   └─ Assign twin orientations
    └─ For non-host grains: unchanged
    ↓
Part H: MDF verification
```

---

## Expected File Sizes & Counts

| Item | Value |
|---|---|
| Host grain count | ~700–800 (depends on grain size distribution) |
| Non-host grain count | ~1400–1600 |
| Storage (host/non-host lists) | <100 KB |
| Computation time | <1 second |

---

## Implementation Checklist

- [ ] S29: Extract target hosting fraction from EBSD
- [ ] S30: Allocate hosts by volume, achieve target fraction
- [ ] S31: Visualize and compare host/non-host statistics

---

## Notes

- Part E is **deterministic** (no randomness once grain volumes are fixed)
- Selection happens once per 3D structure (not resampled)
- Host allocation independent of orientation (orientation assigned in Part F)
- All grain IDs stored as Python sets for efficient lookup

---

**Status:** Ready for implementation  
**Next step:** Implement Part E into repgen3mcgs1.ipynb

