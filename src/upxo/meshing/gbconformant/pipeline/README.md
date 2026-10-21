# Conformal meshing pipeline

Runs the grain-conformal tetrahedral meshing stages in order, from a labelled grain structure to
an Abaqus deck, and saves each stage's results so a later run only redoes what changed.

## From Python

```python
from upxo.meshing.gbconformant.pipeline import run

result = run(RAW_FOLDER="path/to/raw_export", HIERARCHY_LEVEL="blk",
             GRAIN_ID_SEED=42, through="export_abaqus")
result.raise_for_failure()
print(result.export_folder())
```

Settings are passed by name (`run({...})` or keywords). Unknown names and unusable values raise
`SettingsError` before any work starts. `result.ran`, `result.restored` and `result.skipped` list the
stages by what happened to them; a failing stage is reported in `result.failed_stage` and
`result.error` instead of raising.

`on_event` receives one dict per step (`start`, `done`, `progress`, `restored`, `stale`, `skipped`,
`failed`), each naming its `stage`.

## From the command line

```
python -m upxo.meshing.gbconformant.pipeline stages
python -m upxo.meshing.gbconformant.pipeline settings [--stage STAGE] [--json]
python -m upxo.meshing.gbconformant.pipeline run (--raw-folder FOLDER [--level blk] | --twinned-folder FOLDER) \
    [--preset NAME] [--set KEY=VALUE ...] [--settings FILE.json] [--subset-box index X0 Y0 Z0 X1 Y1 Z1] \
    [--through STAGE] [--run-name NAME] [--data-dir DIR] [--no-resume] [-v | -q] [--json]
python -m upxo.meshing.gbconformant.pipeline presets [NAME] [--json]
python -m upxo.meshing.gbconformant.pipeline estimate (--voxels N | --shape NX NY NZ) [--preset NAME] [--set ...]
python -m upxo.meshing.gbconformant.pipeline status RUN_FOLDER
python -m upxo.meshing.gbconformant.pipeline report RUN_FOLDER [--sections ...] [--no-figures]
```

Exit codes: 0 finished, 1 a stage failed, 2 the command line or a setting was unusable.

## Inputs

* `DATASET_TYPE=fm_steel`: a raw export folder (`<prefix>_lgi.npy`, `<prefix>_orientations.pkl`,
  `raw_export.pkl`). `HIERARCHY_LEVEL` picks `pag`, `pck`, `blk` or `sblk`; empty means the finest found.
* `DATASET_TYPE=twinned_fcc`: a twinned FCC structure folder `TWINNED_FOLDER` holding
  `lgi_twinned_clean.npy` (axes z, y, x), `orientations_all.pkl` (unit quaternion [w, x, y, z] per
  grain), `twin_role.pkl` and `twin_parent_of.pkl`. `TWINNED_VOXEL_SIZE` and `TWINNED_UNITS` give the
  voxel edge length. Every grain is one feature, named as the voxel deck of the twinned structure names
  it (`es_cell_host_12`, `es_cell_ptwin_340`, `es_cell_stwin_a_...`, `es_cell_nonpart_...`), with the
  element sets `es_role_*` and `es_family_<host>` that join grains by role and by host family. The
  orientations are converted to Bunge-Euler angles in degrees, wrapped to [0, 360).
* `DATASET_TYPE=labels`: one 3-D integer `.npy` image in `SAMPLE_PATH`, with `SPACING`.

`STAGE_SUBSET` crops the input to a box first (by voxel index, percent or physical length). A subset
run keeps everything in its own numbered folder `<RUN_NAME>_subset_001`, `_002`, ... and numbers its
exports inside it.

## Settings

All settings, with defaults, types, limits and the stages they decide, are listed by the `settings`
command and held in `settings.py`. Settings saved by an older version are migrated on load
(`load_versioned`) and every value that had to be reset is reported.

## Presets

`presets` lists the built-in presets; `--preset NAME` starts a run from one and `--set` changes win. A
preset is a group of values for the settings it controls, taken from the development notebooks
or from runs on the development crop of the FM-steel data; each preset's notes say which. Applying a
preset sets those settings, and every other setting that some preset controls goes back to its default,
so presets can be switched between. The seed, the input and the run name are never part of a preset.

From Python: `presets.apply_preset(config, "quick_look")`, `presets.changes(config, name)` for what
would change, `presets.which_preset(config)`, `presets.recommend(voxels)`.

## Forecast

`estimate` (and `estimate.forecast(voxels, config, history)`) gives the tetrahedra, nodes, time and
peak memory to expect for a crop, each with a range. The constants were fitted to the FM-steel
development data: a crop of 8,800 voxels under eight settings and a crop of 27,000 voxels. What the
runs showed:

* The surface size factor `GMSH_SIZE_FACTOR` sets the tetrahedron count: 0.5 gave 2.43 times the
  tetrahedra of the default 0.75, 1.0 gave 0.79 times, 1.5 gave 0.74 times.
* `TET_MESH_SIZE_FACTOR` changed the count only when its size was smaller than the surface size
  (0.6 gave 1.53 times); 1.0, 1.5 and 2.5 gave the same count.
* Tetrahedra grew as the voxel count to the power 0.91; time as the voxel count to the power 0.80.
  Time did not follow the tetrahedron count on the crops measured, and the same settings took between
  171 and 345 s on one computer, so the time range is wide.
* Peak memory of the main process was about 330 MB plus 0.00096 MB per tetrahedron. Worker processes
  add their own memory.

The finished runs in the data directory refine the forecast: each run's `run_manifest.json` holds a
`results` section (voxels, grains, tetrahedra, nodes, seconds, peak memory, the size factors), and the
forecast is scaled by how far those runs differ from the fitted law. Runs made before this section existed
are read from their stage timings and their tetrahedron report.

## After a failure

`result.suggestions` (and the command line, after `stopped at ...`) lists settings that bear on the
failure the stage reported, from the rules in `advice.py`: each has a proposed value and the reason.
Nothing changes until it is applied: add the `--set` lines to the same command (the earlier stages are
restored), or call `advice.apply_suggestion(config, suggestion)`. A proposal equal to the current value is
not offered.

## Run folder

`<data directory>/runs/<RUN_NAME>/` (the data directory is `UPXO_CONFORMAL_DATA`, or `--data-dir`):

* `checkpoints/<stage>/` - the results of each finished stage, as JSON and `.npz` files with a format
  and version tag. No pickles.
* `run_manifest.json` - settings, changes from the defaults, seeds, input file names and sizes, the
  saved stages with their signatures, stage timings, and package versions.
* the stage outputs: `*.vtp`, `*.vtu`, `*.json` audits and the `abaqus/` decks.

## Resuming

`run` restores a stage only while its own settings, the input files and every earlier stage are
unchanged (a signature chained through the stages decides). Anything else is run again, and the
stages after it. `--no-resume` ignores the saved results.

## One stage in a separate process

```python
from upxo.meshing.gbconformant.pipeline.background import StageProcess, NotReady

job = StageProcess(state, "tet_mesh")      # NotReady lists why it cannot be run this way
job.start()
while not job.done:
    for event in job.poll():                # progress messages from the stage
        print(event["message"])
    time.sleep(0.1)
if job.outcome == "finished":
    job.apply()                             # the stage's results are now in `state`
```

The stage runs in a fresh Python process that restores the saved results of the stages before it,
runs the stage and saves its results. `state` changes only when `apply()` is called, so a cancelled
(`job.cancel()` stops the process and everything it started at once) or failed stage leaves it as
it was. It needs saved results (`CHECKPOINT_RESULTS`) and every earlier stage either done and saved
or switched off. `run_stage_separately(state, stage, on_event, should_cancel)` does the waiting.
A failed job keeps its folder `<run folder>/.worker/<stage>_<time>/` with the process output.

## The deck

`export_abaqus` writes the partitioned C3D4 deck. Besides the element sets, materials, sections, face
node sets and static step it holds:

* `04b_nsets_boundaries.inp` (`EXPORT_BOUNDARY_SETS`): `ns_gb_all` (nodes shared by two or more
  features), `ns_junction_all` (shared by three or more) and `ns_gb_<element set>` for the boundary nodes
  of each feature. Two meshed grains merged into one feature have no boundary between them.
* `fm_mesh.msh` (`EXPORT_MSH`) and `fm_mesh.vtu` (`EXPORT_VTU`): the deck's nodes and tetrahedra in the
  deck's numbering and units. The `.msh` is Gmsh 2.2 ASCII with one physical group per element set and
  element data `euler_deg`, `feature_label`, `global_id` and `ipf_x`, `ipf_y`, `ipf_z`. The `.vtu` holds the
  same as cell data, the IPF colours as 8-bit RGB. The IPF colour is that of
  `upxo.xtalphy.crystal_orientation.ipf_color`.
* `07_interactions.inp`: a template, comments only. It reports, for each axis, how many nodes of the low
  face have a partner on the opposite face at the same position, and shows the lines of a periodic
  constraint, a tie and the face set names. No constraint is active. Opposite faces of a conformal mesh need
  not carry matching nodes (30 of about 1,000 matched on a twinned crop).
* `03e_elsets_groups.inp`: for datasets that name groups of features (twinned FCC: roles and host families).
* `03f_elsets_layers.inp` (`EXPORT_BOUNDARY_LAYER`): `es_gb_layer`, the elements with a node on a grain
  boundary, and `es_interior`, the others. The share of elements in the layer is in `deck_info.json`; on a
  structure of small grains it can be half of the mesh or more.
* `boundary_classes.csv`, `04c_nsets_boundary_classes.inp` and `fm_interfaces.vtp` (`EXPORT_BOUNDARY_CLASSES`): every
  triangle that separates two features is an interface; each pair of neighbouring features has one row in the table
  with its two element sets, the kind of boundary, the area, the misorientation angle and whether it is low-angle.
  FM steel, from the finest level: a sub-block boundary separates two sub-blocks of one block, a block boundary two
  blocks of one packet, a packet boundary two packets of one PAG, and a PAG boundary two PAGs; the kinds a structure
  can have depend on the level it was meshed at (block level has no sub-block boundaries). Twinned FCC: a twin boundary
  separates two grains of one host family, any other boundary between grains is a grain boundary. The misorientation
  is the cubic one (the symmetry setting `TEXTURE_CRYSTAL_SYMMETRY`; other symmetries give no angle and no low-angle
  sets); a boundary is low-angle below `BOUNDARY_LOW_ANGLE_DEGREES` (default 15). The node sets are `ns_gb_<kind>`
  (`ns_gb_block`, `ns_gb_packet`, `ns_gb_pag`; `ns_gb_twin`, `ns_gb_grain`), `ns_gb_lowangle` and `ns_gb_highangle`; a
  node on the edge between two kinds is in both sets. `deck_info.json`, the run summary and the export report give, per
  kind, the number of boundaries, the share of the boundary area, the area-weighted mean misorientation and the
  low-angle share. The surface file holds the interface triangles with the kind, misorientation, area and the two
  features as cell data. The areas sum to the boundary area `fidelity` measures for the same mesh.
* `features.csv` (`EXPORT_FEATURE_TABLE`): one row per feature that holds elements: element set, material, UMAT
  grain number, Bunge-Euler angles, `pag_id`, `packet_id`, `block_id`, `subblock_id` and the names of their element
  sets, raw voxels, tetrahedra, volume and volume fraction. `pag_id` is the PAG number of the element set name; the
  other ids number the packets, blocks and sub-blocks present from 1 in ascending key order, and are -1 (blank in the
  table) where a level does not apply. Twinned FCC gives `role`, `family_id` and `family_elset` instead. The same
  ids are cell data in `fm_mesh.vtu` and element data in `fm_mesh.msh`, so a result can be summed by hierarchy
  level (twinned FCC: `family_id` and `role_code`, with 0 non-host, 1 host, 2 primary twin, 3 and 4 the two kinds of
  secondary twin).
* `deck_check.json` (`EXPORT_DECK_CHECK`): the deck files read back and checked: numbering of nodes and elements,
  connectivity, unused nodes, positive tetrahedron volumes, every element in exactly one section set, sections with
  existing element sets and materials, no empty sets, sets within range, step boundary conditions on existing node
  sets, and the files the master file includes. The result is in the export report and the run summary. With
  `REQUIRE_DECK_CHECK` a failure stops the stage; otherwise it is reported and the deck is kept. A deck that passes is
  consistent as a file; it has not been run in Abaqus.

## Load cases and the size of the output

The analysis step of the FM-steel deck is built from a load case (`fm_steel_abaqus/loading.py`).

* `FM_LOAD_MODE=displacement` (default): one step that moves the loaded face by `FM_APPLIED_DISPLACEMENT` in
  `FM_STEP_TIME`, as before.
* `FM_LOAD_MODE=strain`: the history is given as strains and a strain rate (`FM_STRAIN_RATE`, per time unit); the step
  time follows from them and the displacement is the strain times the domain length along the load axis, in output
  units, multiplied by the length scale when written.
  * `FM_LOAD_CASE=monotonic`: from zero to `FM_TARGET_STRAIN`, as `FM_LOAD_SENSE` tension or compression.
  * `FM_LOAD_CASE=cyclic`: from zero to the maximum strain, then `FM_CYCLES` times down to the minimum and up to the
    maximum, as a tabular `*Amplitude` on the loaded face (the loaded face's displacement is the peak strain times the
    length; the table holds signed fractions of it). The cycle is given by `FM_CYCLE_DEFINITION`: `min_max`
    (`FM_STRAIN_MIN`, `FM_STRAIN_MAX`, the default) or `amplitude_r` (`FM_STRAIN_AMPLITUDE` and `FM_R_RATIO`, the
    minimum over the maximum strain: maximum = 2 amplitude / (1 - R), minimum = R maximum; R = -1 is fully reversed,
    0.1 is tension-tension, above 1 is compression-compression). The longest increment is cut to a tenth of the shortest
    segment so that the peaks are reached.
* `FM_HOLD_TIME` (both modes): a last step `Hold` that states the final displacement again on the loaded face and keeps
  it for that long. Field and history output requests are repeated in each step.
* In strain mode the increments and the output interval are cut down to what fits the step time (the deck's notes and
  `deck_info.json` say what was cut). The settings of the other mode are ignored.

The load case is in `deck_info.json` (`load_case`), the export report and the run summary. The syntax of the amplitude
table, the propagation of the displacement into the hold step and the repeated output requests have been checked in
the written files only (`deck_check.json` confirms every amplitude a boundary condition uses is defined); they have not
been run in Abaqus, so run one deck of each load case through an Abaqus data check first.

`FM_OUTPUT_ELSET` limits the element field output to an element set of the deck (`*Element Output, elset=...`, for
example `es_gb_layer`); an unknown name stops the export and lists some of the names.

`deck_info.json` (`output_estimate`), the export report, `estimate` and the forecast give the degrees of freedom (three per
node), the number of field output frames (the first frame, then one per output interval of each step) and a rough
size of the field output: every requested variable is stored per frame for every node or element it covers, as
4-byte values (the columns of a variable: U and RF 3, S, LE and NE 6, `SDV` as many as `FM_N_DEPVAR`) plus a 4-byte label
per entry, with no compression, and history output not counted. The figure is a first estimate; the constant
`ODB_SCALE` in `fm_steel_abaqus/outputs.py` is where a measured ODB size is used to tune it.

## Job scripts, the data check and reproducing a deck

`JOB_SCRIPTS` writes these beside the deck (`fm_steel_abaqus/jobs.py`):

* `submit.bat` and `submit.sh`: `abaqus job=<JOB_NAME> input=model_master.inp [user=<UMAT_FILE>] cpus=<JOB_CPUS>
  [memory="<JOB_MEMORY_GB> gb"] interactive` on this machine (the `.bat` uses `call`, with DOS line ends);
* `datacheck.bat` and `datacheck.sh`: the same with `datacheck` on one CPU, under the job name `<JOB_NAME>_datacheck`;
* `submit.slurm`: `#SBATCH` lines for the job name, CPUs, `--mem` (when set), `--time` (`JOB_TIME_HOURS`) and
  `--partition` (`SLURM_PARTITION`, when set), and the same command with `cpus=${SLURM_CPUS_PER_TASK}`. Submit with
  `sbatch submit.slurm`.

`ABAQUS_COMMAND` is the command or full path that starts Abaqus. A path pasted with its quotes (`UMAT_FILE`) is taken
as it is. The job name may hold letters, digits, underscore and hyphen; the command and the UMAT path must not hold
quotes, line ends or any of `` ` $ % ; & | < > `` (an export stops with a message when one does). With no `UMAT_FILE` the
scripts carry a note that `user=` is missing.

`datacheck DECK_FOLDER [--abaqus COMMAND] [--timeout SECONDS]` runs the data check where Abaqus is installed, with the
job settings the deck was written with, and reads the `***ERROR` and `***WARNING` lines from the job's `.dat`, `.msg`
and `.log` files. It writes `abaqus_datacheck.json` (`ok` means Abaqus exited with 0 and no error line was found) and
exits 0 when the check passed, 1 when it failed or could not be run, 2 for a folder that is no deck. Abaqus has not
been run against by UPXO's tests: the command is tested against a stand-in.

`EXPORT_PROVENANCE` writes `provenance.json` into the deck folder: the command that rebuilds the deck, the settings that
differ from the defaults, the seeds, the input files with sizes and times, the crop and the package versions.
`reproduce DECK_OR_RUN_FOLDER` prints the command (from `provenance.json`, or rebuilt from the settings in a run
folder's `run_manifest.json`). The command is the `run` command with the input as a flag (`--raw-folder`,
`--twinned-folder` or `--labels`), `--level`, `--subset-box` for the crop, `--run-name` and one `--set KEY=VALUE` for
every setting that differs from its default; the spacings the stages fill in are left out (a labelled image's spacing is
an input and is kept). Run in the same data directory it restores what is still valid and rebuilds the rest. From Python:
`provenance.reproduce_command(config)`. The command is written for the system it was made on (paths and quoting).

## Results and reports

```python
from upxo.meshing.gbconformant.pipeline import results
from upxo.meshing.gbconformant.pipeline.run import open_run

state = open_run("path/to/run folder")          # the saved results, with the settings of run_manifest.json
print(results.summary_text(state))               # input, cleaning, surface, mesh, export, time per stage
for block in results.quality_blocks(state): ...  # worst elements against the limits, per stage
figures = results.quality_figures(state) + results.fidelity_figures(state)
session = results.report_session(state)          # <run folder>/report
results.populate_report(session, state)          # summary, settings, quality, fidelity, deliverables
```

`python -m upxo.meshing.gbconformant.pipeline report RUN_FOLDER [--sections ...] [--no-figures]` does the
last three lines and writes `report/report.html` with its images. A section already in the report is
rebuilt instead of repeated, and entries added by hand are kept.

The fidelity section compares the voxel structure with the mesh: junction line length per line and per
grain, sphere-equivalent grain diameter, and pole figures, before and after (needs an FM-steel raw folder
as the input and the before/after comparison stage).

### Per-grain quality, per-feature fidelity and the merge log

* `results.quality_blocks(state)` also lists the share of tetrahedra beyond the usual limits (smallest
  dihedral below 15, 20 and the `TET_TARGET_MIN_DIHEDRAL` value, largest above 150 and 165, minSICN, aspect
  ratio), and the ten grains with the smallest dihedral angle with their element-set names.
  `results.quality_grain_table(state)` is every grain's row (`quality_per_grain.csv` in a report).
* `results.grain_data(state)` measures every feature before (voxels) and after (mesh): volume, boundary
  area, neighbours lost and gained, and aspect ratio. A feature that lost more than `FIDELITY_SHRINK_LIMIT`
  of its volume is flagged `shrunk`, one with no tetrahedra `lost`. The voxel boundary area is the area of
  the stepped faces, so it exceeds the area of a smooth surface; the ratio shows that, not a defect.
* The deck export writes `merge_log.json` and `merge_log.csv` beside the deck when grains were merged: for
  each absorbed feature, the feature whose orientation it now carries and the cubic misorientation between
  them. It reports only.

## Runs, stale results and sizes

* `runs.list_runs()` lists the run folders of the data directory, newest first: the input, the subset box
  of a subset run, how far the run got, and how much it saved. `describe_run(folder)` does one folder.
* `PipelineState.invalidate_from(stage, reason)` keeps the reason for every stage that held results, with
  the time (`state.stale_reasons`, `state.stale_reason_text(stage)`); a restore that finds other settings
  records its reason too. `state.invalidation_preview(stage)` says which finished stages an edit would
  discard and how long they took.
* After an invalidation, `saved_results.rewind` reads back from the run folder the copy of each shared result
  (labels, surface, tetrahedra) that the stage before the dropped one saved, so the dropped stage can run
  again from where the earlier stages left off. It needs the earlier copies to be kept
  (`CHECKPOINT_KEEP_HISTORY`).
* `units.physical_text(state, setting, value)` gives the physical size of a setting measured in voxels
  (`Setting.unit`), for instance `0.75 x voxel edge = 0.75 um`.
* `results.stage_summary_block(state)` is one table of every stage that ran: status, seconds, key numbers.

## Disk use of runs

`describe_run(folder)` and `list_runs()` report, for each run, `disk_bytes` (everything in the folder), `history_bytes`
(the older copies of results in `checkpoints`) and `older_export_bytes` (see below). Three actions free space:

* `drop_run_history(folder)` keeps each saved result only in the last stage that wrote it (the labels, for instance,
  are saved by every cleaning stage). The run still loads as a whole with the same results; what is lost is running a
  stage again from the results of the stage before it. It returns the bytes freed.
* `delete_older_exports(folder)` keeps the newest export in `abaqus`, `surfaces` and `volume` (a deck and its mesh-only
  deck count as one export) and deletes the others. Every export writes a new folder, and on a crop of 111,628
  tetrahedra each deck folder with its `.msh` was about 40 MB, so repeated exports are usually most of a run's disk:
  one run held nine of them, 363 MB of 389 MB, against 18 MB of saved stage results once the history was dropped.
* `delete_run(folder, root=None, keep=None)` deletes a run folder. Only a folder directly inside the runs directory that is
  a run (it holds a manifest or saved stage results) is deleted; the runs directory itself, folders inside a run,
  any other folder and the run `keep` (the one that is loaded) raise `RunDeletionError`. Read-only files do not stop it.

## The crop and the hierarchy

The subset stage reports how many PAGs, packets and blocks (sub-blocks at sub-block level; grains and host families for
twinned FCC) the crop holds, next to the number in the whole structure, and how many features the box cuts
(`counts` and `cut_features` in the report and in `subset.json`). `RVE_MIN_PAGS`, `RVE_MIN_PACKETS` and
`RVE_MIN_BLOCKS` are empty by default; for a level with a minimum set, a crop below it gets a warning in the report
(`warnings`) and the summary. The minimums change no stage's results.

## Quality by region

The stage `region_quality` runs after the volume stages and before the comparison and the exports. It reports the
quality of the tetrahedra and of the surface triangles at three places of the finished mesh, and how that quality
changes with the distance from them. It reads `verified_tets` and changes nothing; `STAGE_REGION_QUALITY` switches it
off.

**Features.** `gb_surface` is the set of faces that separate two grains. `junction_lines` is the set of mesh edges shared
by three or more regions (grains, and the box exterior when `REGION_INCLUDE_RVE_FACES` is on). `junction_points` is the
set of nodes where junction lines end, branch or change the set of regions they separate; where four grains meet, a
generic structure has one. A line end on a box face is not a point unless the box exterior counts as a region.
`REGION_GB_SURFACE`, `REGION_JUNCTION_LINES` and `REGION_JUNCTION_POINTS` choose the features reported.

**Layers.** An element is in layer 1 of a feature when it touches the feature. It is in layer 2 when it shares a node with
a layer-1 element and is not in layer 1, in layer 3 when it shares a node with a layer-2 element and is not in layer 1 or
2, and so on up to `REGION_ORDER` (1 to 50). `REGION_TOUCH` decides what touching is: `face or edge` means sharing a face
with the grain boundary surface or an edge with a junction line, and `node` means sharing any node. Junction points are
touched at a node in both cases. Triangles are grown only along the surface; every triangle of the grain boundary surface
lies on it, so that feature has one layer for triangles. An element can be in the layers of several features.

**What is shown.** The tables list every layer 1 to N of each feature, the elements in layers 1 to N together, the whole
mesh and the elements beyond the layers shown. `REGION_ORDER_MODE` chooses what the plots and the selected arrays use:
`layers 1 to N` (every element within N layers) or `only layer N` (the ring N layers out). For tetrahedra the table gives
the smallest and largest dihedral angle, minSICN and the aspect ratio; for triangles the smallest and largest angle and the
shape quality; both count the elements below and above the limits of the quality stages
(`MIN_TET_DIHEDRAL_DEGREES`, `TET_TARGET_MIN_DIHEDRAL`, `TET_MAX_DIHEDRAL`, `SURFACE_MIN_ANGLE_DEGREES`).

**Leaving elements out.** `REGION_EXCLUDE_LINES` and `REGION_EXCLUDE_POINTS` remove from the grain boundary surface the
elements within `REGION_EXCLUDE_ORDER` layers of the junction lines, or of the junction points. The layers of the
other features are unchanged.

**Files.** With `REGION_WRITE_FILES` the stage writes `region_quality/region_quality.csv` (every statistic of every row),
`region_tets.vtu` (each tetrahedron with `grain`, the measures, `layer_<feature>` and `selected_<feature>`) and
`region_triangles.vtp` (each triangle with its two grains, the measures, `layer_<feature>` and `selected_<feature>`). A
layer of 0 means beyond the layers asked for. The result kept with the run holds the statistics, a histogram of every
measure for every row, and the layer of every element; `results.region_blocks`, `results.region_result_figures` and
`results.region_flat_block` build the tables, the figures and the spreadsheet from it, and the quality section of a report
includes them.

On the crop of 111,628 tetrahedra the stage finds 121 junction lines (581 mesh edges) and 53 junction points.

## Before and after comparison

The comparison measures the grain structure at five points of the pipeline and compares any two of them. Each is a
`Snapshot` (`fm_steel_abaqus/snapshots.py`) built from the artefacts of the run when asked for and kept until an
artefact or a setting changes:

| Representation | Taken from | Grain volume and aspect ratio |
| --- | --- | --- |
| Voxel structure | the labels of the chosen hierarchy level | yes |
| Surface before smoothing | `interfaces_smoothed.original_points` | no (interior boundaries only) |
| Surface after smoothing | `interfaces_smoothed.points` | no (interior boundaries only) |
| Remeshed surface before tet meshing | `repaired_rve` with the points kept in `pre_tet_surface_points` | yes |
| Tetrahedral mesh | `verified_tets` | yes |

`COMPARE_BEFORE` and `COMPARE_AFTER` choose the pair (default: voxel structure and tetrahedral mesh); the first must come
earlier in the list above than the second. Every plot and table is labelled with the two names, never with "before" and
"after" alone, and a count in a legend says what it counts: `Voxel structure: 271 grains`, `Tetrahedral mesh: 268 grains`.
Grains the voxel structure has and the mesh lacks were removed by the cleaning stages before the surface existed; a
footnote on the plot says so.

The tet mesher moves surface nodes during smoothing, so the surface before tet meshing is copied (as
`pre_tet_surface_points`, the node coordinates only) when `tet_mesh` starts. A run made before this existed does not have
it, and the remeshed surface is then reported as not kept. The grain volumes of that surface come from the surface
triangles, with the tetrahedra saying which side of each triangle is inside which grain.

**Parameters.** Grain volume, sphere-equivalent diameter, aspect ratio, number of neighbours (coordination number: grains
that share a boundary), boundary area per grain, junction line length per line, junction line length per grain, grain
boundary misorientation, grain mean boundary misorientation and pole figures. A parameter that needs a grain volume cannot be read from the two open surfaces.
Boundary areas count the interior grain boundaries only (the faces of the box are not counted), the same for every
representation; the area of the stepped voxel faces exceeds that of a smooth surface enclosing the same voxels.

**Misorientation** needs cubic symmetry and an orientation for every grain. The boundary misorientation is the angle
across each boundary between two grains, weighted by the area of the boundary, and is shown (`COMPARE_MISORIENTATION_NORM`)
as the fraction of the total boundary area per degree, as the boundary area per unit volume of the structure per degree,
or as the area per degree. The grain mean boundary misorientation is, for each grain, the area-weighted mean of the
misorientations across all its boundaries; the grains are counted once each or by their share of the volume
(`COMPARE_GRAIN_WEIGHT`, a side without grain volumes counts grains).

**Units.** `COMPARE_PLOT_UNITS` is `auto`, `microns`, `mm` or `m` and applies to every plot and table; areas and volumes follow
as the square and cube. `auto` picks, for each plot or table, the unit that gives the numbers about two digits before the
decimal point. The numbers are computed in the output units (`FM_OUTPUT_UNITS`) and converted for display, so changing the
display units does not rerun anything.

**Plots.** `COMPARE_PLOT_KIND` is `histogram`, `kde`, `histogram + kde`, `cdf`, `box` or `violin`. A histogram uses
`COMPARE_BINS` bins, or bins of `COMPARE_BIN_WIDTH` (in the unit of the quantity on the axis) when that gives from 2 to
400 bins; whole-number quantities such as the number of neighbours get one bin per number. A KDE is a Gaussian kernel density with the
bandwidth `COMPARE_KDE_BANDWIDTH` (the standard deviation of the Gaussian, in the unit of the axis) or Silverman's rule; the
value used is printed under the plot. A quantity weighted by area or volume has no box or violin plot and is drawn as
histogram plus KDE. `COMPARE_SHOW_MEAN`, `COMPARE_SHOW_STD` and `COMPARE_SHOW_MEDIAN` draw a vertical line at the mean (and a
shorter dotted one at the median) and a horizontal bar from one standard deviation below to one above the mean for each
side, and put the numbers in the legend. `COMPARE_COLOR_BEFORE` and `COMPARE_COLOR_AFTER` are colour names or `#rrggbb`
codes.

`comparison_view.comparison_figure(state, parameter, view)` draws one parameter as a `distribution`, as the `matched`
change grain by grain (or junction set by junction set), or as `pole` figures; `comparison_tables(state)` and
`comparison_text(state)` give the tables of the chosen pair. `results.fidelity_figure` keeps its earlier keys.
