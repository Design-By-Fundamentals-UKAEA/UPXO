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
| Synthetic ground truth | no (mtex optional) | `synthetic.py`: grain structures with a known answer, so the twin/parent harness is checked before it is trusted on real, noisy maps |
| Live parity | yes | `test_mtex_live.py` and the live-mtex tests in `test_synthetic_structures.py`: UPXO against mtex on the same inputs; skipped without mtex |

## Files

| File | Covers |
|---|---|
| `test_ks_variant_set.py` | The 24 KS variants: independent construction, equality with mtex's variants, the 276 pair angles (10 distinct values and counts), 4 packets of 6, 3 Bain groups of 8, angles inside and between packets, independence from the parent |
| `test_twin_geometry.py` | Twin operator, Σ3, shared plane, Σ9 between variants, Σ1/Σ9 after two twins and Σ3/Σ27a/Σ27b after three, `compute_s3_habit_plane_3d`, `compute_s3_lamella_angle_2d`, Brandon and Palumbo-Aust tolerances, overlap of the Σ5/Σ7/Σ9 windows |
| `test_invariants.py` | Disorientation against the independent reduction, symmetry, triangle inequality, 62.7994° bound, sample-frame and crystal-relabelling invariance, g-matrix route, stored-quaternion and active-quaternion conventions |
| `test_plane_trace_formulas.py` | Trace direction, apparent width `t0 / sqrt(1 - nz^2)` by direct geometry, mean width of isotropic lamellae, identifying the plane from a noisy trace |
| `synthetic.py` | Voronoi grain structures and Sigma3 twin lamellae with known ground truth (grain id, area, parent/twin identity); not a test file itself |
| `_twin_harness.py` | Runs UPXO's own `segregate_csl_pairs` / `identify_parent_grains` / `csl_volume_fractions` on a label field + per-grain Euler angles; shared with the local real-data runs |
| `test_synthetic_structures.py` | The generator's own consistency; `compute_mdf_from_quats` against the oracle on a labelled structure; the twin/parent harness recovers the known truth exactly; the twin-only vs Sigma3-linked area-fraction distinction; mtex run live on the same structure (grain partition, Sigma3 pairs) |
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
transpose, and the KS operators applied on the martensite side. A fourth mutation, the stored
quaternion built as the quaternion of R instead of g, was reintroduced into `synthetic.py` the
same way and made 7 of the 13 tests in `test_synthetic_structures.py` fail.

## A pitfall found while building this (kept here so it isn't rediscovered)

mtex always reserves `phaseId == 0` for "not indexed", whatever `CSList[0]` is. A real, indexed
phase needs `phaseId >= 1`, mapped to `CSList[phaseId - 1]`. Building an in-memory `EBSD` object
with a placeholder "not indexed" entry at `CSList[0]` and pointing real pixels at `phaseId = 1`
silently puts them on that placeholder's symmetry instead, so every disorientation computed from
the grain table comes out unreduced -- the symptom was every boundary `misorientation.angle()`
landing near 180 degrees. For one phase, skip the placeholder: `CSList = [cs]`,
`phaseId = ones(...)`.

## Not covered

- Non-cubic symmetry.
- Grain-boundary smoothing and other EBSD-map operations beyond grain detection and the Sigma3
  pipeline (checked on a measured map outside this folder, with results that cannot be
  reproduced from it).
- ODF and texture index.
- Whole pipelines under both backends; the mtex backend does not exist yet.
