# `upxo.pxtal.twinned_simple_3d`

3D **twinned FCC** representative microstructures (e.g. **OFHC-Cu**, **CuCrZr**): host grains → primary / secondary **Σ3** twin lamellae, cleaning, representativeness checks, Abaqus export.

Canonical public docs:  
[Use Cases: CuCrZr & OFHC-Cu](https://github.com/Design-By-Fundamentals-UKAEA/UPXO/wiki/Use-Cases-CuCrZr-OFHC-Cu) ·  
[3D Grain Structure Generation](https://github.com/Design-By-Fundamentals-UKAEA/UPXO/wiki/3D-Grain-Structure-Generation)

---

## Public API

| Class | Role |
|---|---|
| `TwinnedSimple3DBase` | Wrap 3D LFI; morphology; twin-host allocation |
| `TwinGenerator3D` | Primary / secondary twin introduction |
| `OrientationAssigner3D` | Host / non-host orientation assignment |
| `StructureCleaner3D` | Spike removal, lobe splitting |
| `RepresentativenessValidator3D` | Morphological / crystallographic slice validation |

Also: `abaqus_exporter_3d.AbaqusExporter3D`, `viz_3d` helpers.

### Minimal shape

```python
from upxo.pxtal.twinned_simple_3d import TwinnedSimple3DBase, TwinGenerator3D

# lfi_3d: from MCGS3D / Voronoi / etc. Prefer name LFI; attribute may still be gs.lgi (deprecated).
twin_base = TwinnedSimple3DBase(lgi=lfi_3d, voxel_size=1.0, rng_seed=42)
# or: TwinnedSimple3DBase.from_mcgs(pxt, tslice_key=pxt.m[-1], ...)

twin_base.allocate_twin_hosts_spatial(target_hosting_fraction=0.3, seed=42)
# or allocate_twin_hosts(...)

twin_gen = TwinGenerator3D(
    base=twin_base,
    n_lamellae_per_host=2,
    twin_nucleation_site='random_gb',
    meshing_route='conformal',
    rng_seed=42,
)
# introduce_primary_twins(...) needs EBSD-derived twin thickness / TVF targets
# — see demos under demos/Twinned3D/; do not invent placeholders for production studies.
```

## Layout (this package)

| Path | Role |
|---|---|
| `base_3d.py` | Host structure + allocation |
| `twin_generator_3d.py` | Twin embedding |
| `orientation_3d.py` | Orientation assignment |
| `cleaning_3d.py` | Structure cleaner |
| `repr_validator_3d.py` | Representativeness |
| `abaqus_exporter_3d.py` | Partitioned Abaqus `.inp` |
| `viz_3d.py` | IPF slices / role colours |

---

## Related demos

- `src/upxo/demos/Twinned3D/` — e.g. `repOFHCCu3d_ebsdVf.1.0.ipynb`, `repgen3mcgs*`  
- EBSD import: `upxo.interfaces.defdap.ebsd_reader.EBSDReader` (`.ctf` / `.crc`; `pip install upxo[ebsd]`)

Install: `pip install upxo` (Python ≥ 3.13). Export interaction/step INP files are often stubs — edit before FE submission.
