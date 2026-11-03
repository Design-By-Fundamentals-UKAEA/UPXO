# mtex validation suite

Tests that compare UPXO's orientation maths with mtex 1.0.0 for cubic (m-3m) crystals, and that
check the geometry stated on the wiki theory pages. They use synthetic inputs only: no EBSD file
and no measured orientation appears in this folder.

## Layers

| Layer | Needs mtex | What it holds |
|---|---|---|
| Frozen reference | no | Values produced once with mtex and stored in `reference/` with the versions and date of the run |
| Invariants | no | Properties that hold whatever mtex gives: symmetry, bounds, sample-frame invariance, convention round-trips |
| Independent oracle | no | `_oracle.py`: numpy/scipy reimplementations that share no code with UPXO |
| Live parity | yes | `test_mtex_live.py`: UPXO against mtex on the same inputs; skipped without mtex |

## Files

| File | Covers |
|---|---|
| `test_ks_variant_set.py` | The 24 KS variants: independent construction, equality with mtex's variants, the 276 pair angles (10 distinct values and counts), 4 packets of 6, 3 Bain groups of 8, angles inside and between packets, independence from the parent |
| `test_twin_geometry.py` | Twin operator, Σ3, shared plane, Σ9 between variants, Σ1/Σ9 after two twins and Σ3/Σ27a/Σ27b after three, `compute_s3_habit_plane_3d`, `compute_s3_lamella_angle_2d`, Brandon and Palumbo-Aust tolerances, overlap of the Σ5/Σ7/Σ9 windows |
| `test_invariants.py` | Disorientation against the independent reduction, symmetry, triangle inequality, 62.7994° bound, sample-frame and crystal-relabelling invariance, g-matrix route, stored-quaternion and active-quaternion conventions |
| `test_plane_trace_formulas.py` | Trace direction, apparent width `t0 / sqrt(1 - nz^2)` by direct geometry, mean width of isotropic lamellae, identifying the plane from a noisy trace |
| `test_mtex_live.py` | Matrix convention, random-pair disorientation, KS variants of random parents, twin and Σ27 angles, {111} normals in the sample frame |
| `tolerances.py` | Every tolerance, with its reason |

Older tests that compare with frozen mtex values stay where they are: `tests/xtalphy/test_cubic_disorientation_vs_mtex.py`,
`test_sigma3_twin_vs_mtex.py`, `test_ipf_vs_mtex.py`, `test_texops_cubic_conventions.py`,
`test_cubic_conventions_texture_model.py` and `tests/viz/test_pole_figure_vs_mtex.py`.

## Run

```
pytest tests/validation_mtex -v          # all; the live module is skipped without mtex
pytest tests/validation_mtex -k "not live"
```

## Regenerate the frozen values

```
python validation/generate_reference.py   # needs mtex; rewrites reference/ks_mtex.json
```

The file records the mtex, numpy and Python versions and the date. A change in a frozen value after
a regeneration is a change in mtex.

## Checks that the tests catch the old defects

Three defects of the earlier code were reintroduced in memory and each made the suite fail: the
disorientation reduced on the sample side, the habit plane computed from g instead of its
transpose, and the KS operators applied on the martensite side.

## Not covered

- Non-cubic symmetry.
- Grain detection, grain boundary smoothing and other EBSD-map operations (checked on a measured
  map outside this folder, with results that cannot be reproduced from it).
- ODF and texture index.
- Whole pipelines under both backends; the mtex backend does not exist yet.
