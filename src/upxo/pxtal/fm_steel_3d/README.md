# `upxo.pxtal.fm_steel_3d`

3D **ferritic–martensitic steel** hierarchical microstructure generation from a labelled feature image (**LFI**).

Canonical public docs (user-facing):  
[Use Cases: FM Steels](https://github.com/Design-By-Fundamentals-UKAEA/UPXO/wiki/Use-Cases-FM-Steels) ·  
[3D Grain Structure Generation](https://github.com/Design-By-Fundamentals-UKAEA/UPXO/wiki/3D-Grain-Structure-Generation) ·  
[GUI Applications](https://github.com/Design-By-Fundamentals-UKAEA/UPXO/wiki/GUI-Applications)

There is **no** class named `FMSteelGenerator`. Start from **`FMSteel3DBase`**.

---

## Pipeline (chainable)

Each stage returns a **new** object (upstream instance is not mutated):

```text
LFI  →  FMSteel3DBase
     →  FMSteel3DWithPAGs          (generate_pag_clusters)
     →  FMSteel3DWithBlocks        (assign_pag_orientations, then generate_blocks)
     →  FMSteel3DWithOrientations  (assign_orientations / KS)
     →  FMSteel3DWithSubBlocks     (optional generate_subblocks)
```

Phases: martensite (transformed PAGs) vs retained austenite (whole-PAG, no hierarchy) — see `phases_3d.py`.

### Minimal example

```python
import numpy as np
from upxo.pxtal.fm_steel_3d import FMSteel3DBase, MeshExporter3D

lfi = ...  # 3D int array, grain IDs >= 1
fm = FMSteel3DBase.from_lfi(
    lfi,
    physical_dimensions=(100.0, 100.0, 100.0),
    voxel_size=1.0,
    connectivity=6,
    random_seed=42,
)
fm_pag = fm.generate_pag_clusters(
    pag_size_distribution={'sizes': [3, 4, 6], 'probs': [0.25, 0.5, 0.25]},
    pag_grain_fraction=0.8,
    random_seed=42,
)
fm_pag.assign_pag_orientations(pag_ori_mode='random', random_seed=42)
fm_blk = fm_pag.generate_blocks(block_thickness_range=(2.0, 5.0), random_seed=42)
fm_ori = fm_blk.assign_orientations(
    ks_variant_selection='random_per_block', random_seed=42
)
# optional: fm_sub = fm_ori.generate_subblocks(...)
```

### Launch GUI

- **Windows:** double-click `guiLaunchers/Launch_FM_Steel_GUI.bat` (repo root)  
- **Python:** `from upxo.pxtal.fm_steel_3d.gui_launcher import launch_gui; launch_gui()`

---

## Layout (this package)

| Path | Role |
|---|---|
| `base_3d.py` | `FMSteel3DBase`, `PhysicalDimensions` |
| `with_pags_3d.py` / `with_blocks_3d.py` / `with_orientations_3d.py` / `with_subblocks_3d.py` | Pipeline stages |
| `pag_clustering_3d.py`, `pag_technique_selector_3d.py` | PAG techniques / levers |
| `block_generator_3d.py`, `subblock_generator_3d.py` | Slab slicing workers |
| `orientation_assigner_3d.py` | KS variants + sub-block scatter |
| `mesh_exporter_3d.py` | Abaqus C3D8 / C3D4 / C3D20 / C3D10 export (**implemented**) |
| `phases_3d.py`, `feature_props_3d.py`, `geom_metrics_3d.py` | Phases / metrics |
| `viz/` | PyVista / matplotlib workers |
| `gui/`, `gui_launcher.py` | Tkinter wizard |
| `assets/mesh_limits.json` | Voxel-count soft/hard limits |

**Not canonical:** copies under `src/upxo/scripts/HierarchyGS/FMSteel3D/*_impl.py` (legacy / scripts; often gitignored). Prefer this package.

---

## Related demos

- `src/upxo/demos/FMSteel3D/block_level_01.ipynb` (when present in clone)  
- Tests: `tests/fm_steel_3d/`

Install: `pip install upxo` (Python ≥ 3.13). Mesh export interaction/step INP includes may be stubs — edit before submitting FE jobs.
