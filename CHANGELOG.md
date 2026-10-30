# CHANGELOG

All notable changes to UPXO are documented in this file.

## [Unreleased]

### Added

- **`OrientationAssigner3D.assign_pag_orientations_with_hagb`** now sets `self.last_hagb_fallback_count` to the number of PAGs that could not meet the HAGB threshold within `max_attempts`, previously only visible by parsing its `RuntimeWarning`. `FMSteel3DWithPAGs.assign_pag_orientations` surfaces it as `self.hagb_fallback_count` for `pag_ori_mode='hagb_constrained'` and `'textured'` with `hagb_constraint=True`.
- **Orientation conventions page** (`docs/orientation_conventions.rst`) and the `euler_bunge_to_matrix` docstring: which rotation each orientation format holds (Euler angles, R, g, stored and active quaternions), the side on which symmetry operators and twin or Kurdjumov-Sachs relationships are composed, and how directions and plane normals are rotated between frames.
- **`compute_grain_role_property_distributions`** (`viz.ebsdviz`) accepts a `'full'` group: every indexed grain, whatever its role. The default groups are unchanged.
- **CSL and parent/twin grain maps** (`plot_csl_grain_map`, `plot_parent_twin_map`, `plot_combined_parent_twin_map`): the legend is drawn outside the map, `legend_fontsize` sets its size, and `step_size` (micrometres per pixel) adds a length scale bar. The `repgen2d` map methods pass the EBSD step size.
- **`PoleFigure.plot_density`**: `colorbar_label` sets the colour bar title and `colorbar_orientation` is `'vertical'` or `'horizontal'`.
- **`detect_texture_component_peaks`**: `verbose=True` prints the major steps and the progress of the density pass.
- **numba backend for the MC small-grain and spike removal** (`twinned_simple_3d.fast_cleaning`): `TwinnedSimple3DBase.clean_temporal_slices` and `clean_temporal_slices_recursive` take `backend='auto'` (default), `'numba'` or `'numpy'`. The numba kernels give the same labels and counts as the numpy code, much faster. `'auto'` uses numba when it is installed and the numpy code otherwise; `'numba'` raises `ImportError` without numba.
- **`backend` option for the twinned FCC export index** (`AbaqusExporter3D`, `AbaqusExporter3D.build_index`): `'auto'` (default), `'numba'` or `'numpy'`. With numba the grain-element index is built from one sort instead of one scan per grain. The written files are identical for both backends. `'auto'` is numba when it is installed and numpy otherwise; `'numba'` raises `ImportError` without numba.
- **`backend` option for `StructureCleaner3D`** (`clean`, `clean_recursive`): `'auto'` (default), `'numba'` or `'numpy'`. With numba the spike pass is a numba kernel and the lobe split takes component sizes from one pass and relabels every split lobe with one lookup instead of scanning the whole array twice per lobe, which is much faster on structures with many lobes. The cleaned structure, orientations, roles and split events are identical for both backends. `'auto'` is numba when it is installed and numpy otherwise; `'numba'` raises `ImportError` without numba.
- **Grain-boundary and junction overlays** (`viz.ebsd_overlays`): `overlay_grain_boundaries` draws smoothed boundary lines, or pixel boundaries when there is no smoothed structure, and `overlay_junctions` marks triple and quadruple junctions with a chosen marker, size and colours. `junction_points` returns them.
- **`apply_boundary_overlay`** (`viz.ebsd_overlays`) draws the boundary and junction overlay on a map figure and replaces its own earlier overlay, in axes of any pixel size (`data_per_pixel`).
- **`render_voxel_sets`** (`twinned_simple_3d.viz_3d`) can colour voxels by id with a colour map: `color_by='grain'` uses the grain id in the global grain structure, `color_by='feature'` gives every connected cluster inside one grain its own id (`build_feature_ids`).
- **`smooth_structure`** (`viz.ebsd_smoothing`): the smoothed grain structure of a label map by technique A or B, with an optional `subsample` that smooths every n-th pixel and returns the result in full-resolution coordinates.
- **Texture component map**: `assign_grains_to_texture_components` gives each grain its nearest component by cubic disorientation, and `texture_component_map_figure` draws the map with one colour per component, a legend outside the map and a length scale.
- **`euler_map_figure`**: `title`, `show_axes`, `show_scale_bar`, `font_sizes`, `xlim`, `ylim` and `dpi` options.
- **Weighted pole figures** (`PoleFigure(weights=...)`, `texture.pole_figure`, `COMPARE_GRAIN_WEIGHT`): `PoleFigure` takes one non-negative weight per orientation, and the MUD density becomes the weighted mean of the kernels (the numba and numpy paths agree). The before/after comparison weights every grain by its volume when the grains are counted by volume, so its pole figures and their difference show what the merge and the mesh did to the volume of each orientation, not only which grains were lost.
- **Dataset fields in the before/after comparison** (`fm_steel_abaqus.parameters`, `fm_steel_abaqus.snapshots.dataset_columns`): the parameters of the comparison also hold the Bunge-Euler angles (`euler_phi1`, `euler_phi`, `euler_phi2`) and the sphere-equivalent diameter of the PAG, packet, block (FM steel) or family (twinned FCC) each grain belongs to, before and after, with a matched view for the group diameters. A level the structure does not have, or is meshed at, is reported as unavailable with the reason.
- **Voxel deck of the conformal meshing pipeline** (`gbconformant` stage `export_abaqus_voxel`, `fm_steel_abaqus.voxel_deck`): an FM-steel or twinned FCC Abaqus deck with one C3D8 element per voxel, written beside the tetrahedral deck in the run's `abaqus` folder under the same number (`NNN_voxel` beside `NNN_fm`). The voxels are the cleaned labels at their own resolution, optionally resampled by a scale factor (`VOXEL_RESCALE`, `VOXEL_SCALE_FACTOR`; coarsening takes the label most of the voxels carry), or the tetrahedral mesh voxelized (`VOXEL_SOURCE='voxelized tet mesh'`, `VOXEL_TET_VOXEL_SIZE`). It has the files, element and node sets, materials, sections and step of the tetrahedral deck, under the same names. The stage is off by default (`STAGE_EXPORT_VOXEL`).
- **Hexahedra in the FM-steel deck writer** (`fm_steel_abaqus.write_deck`, `fm_steel_abaqus.elements`): the writer, the deck check, the mesh files, the boundary classes and the junction and region code accept `(M, 8)` C3D8 hexahedra as well as `(M, 4)` C3D4 tetrahedra. Decks of tetrahedra are unchanged; `deck_info.json` gains `elements`.
- **Further element sets from a provider module** (`SET_PROVIDER`, `fm_steel_abaqus.extra_sets`): a module named by the setting can add element sets, node sets and surfaces to the FM-steel decks; the writer checks their names and records the module in `deck_info.json` and `provenance.json`. Empty by default: the decks are unchanged. `AbaqusExporter3D(extra_elsets=...)` adds further element-set files to the twinned FCC voxel deck in the same way.

### Changed

- **Less memory for large FM-steel decks** (`fm_steel_abaqus.write_deck`, `deckcheck.check_deck`, `interfaces.interface_faces`, `boundary_sets.boundary_nodes`, `elements.element_faces`): the writer finds the nodes in use by counting instead of sorting every element corner, the boundary classes build only the faces next to a boundary, the boundary nodes and the volume check work in blocks, and the deck check reads the files back in blocks instead of as lists of text lines. The decks written are byte-identical.
- **Memory estimate for the voxel deck** (`fm_steel_abaqus.voxel_deck.estimate_voxel_deck`, `voxel_grid_shape`): the stage reports the peak memory and the disk space the deck needs, and logs a warning when the peak would use most of the free memory.

### Fixed

- **Twinned FCC decks and reports said `PAG`** (`fm_steel_abaqus.hierarchy.Hierarchy.level_text`): a twinned structure has only grains, but the deck heading, the header of `03a_elsets_pag.inp`, `deck_info.json`, the export summary and the before/after comparison gave its level as `pag`. They say `grain`. File and element-set names are unchanged.
- **Twinned FCC voxel deck with voids** (`AbaqusExporter3D`): a voxel with label 0 or less got an element but no section, so Abaqus rejected the deck. A void voxel now gets no element (the other elements keep their numbers), a node that only void voxels touch is not written, and the node sets of the box faces leave those nodes out; a box face that is entirely void raises an error that names it. Decks without voids are byte-identical to before.
- **Output element set of the FM-steel deck** (`FM_OUTPUT_ELSET`, `DeckSettings.output_elset`): the step named the set without its instance, but the set is defined in the Part, so Abaqus could not find it. It is written as `FM_STEEL_RVE-1.<set>`.
- **Set names of the FM-steel deck**: names were written as typed, so an empty name, a name with a space, comma or period, a name longer than 80 characters, or two sets with one name (Abaqus ignores case) gave a deck Abaqus rejects or misreads. The export now stops before writing anything and names each set and the rule it breaks. The deck check reports names defined twice, names that break the rules, surfaces that name missing elements or faces, and element output sets that do not exist.
- **`repqual/` network assessment** (`KREPR`, 2D and 3D): the generalized measures `calculate_rkf_wd_on_generalized`, `calculate_rkf_ksp_on_generalized`, `calculate_rkf_ed_on_generalized` and `calculate_rkf_ed_nlsd_generalized` computed their matrix and threw it away; they now return it. The 2D `KREPR` called a method that does not exist when the target and sample sets were both 3D; that call is gone.
- **`mc2repr` divergence tests** (`repqual.mcgs2d_representativeness_assesser`): the `kldiv` and `jsdiv` tests, which were in the default test list but did nothing, now give the Kullback-Leibler divergence (nats) and the Jensen-Shannon distance (base 2, 0 to 1) of the target and sample property distributions on shared histogram bins (`kl_divergence`, `js_distance`, `shared_bin_probabilities`). Their results are in `performance[sample][parameter]['kldiv' | 'jsdiv']`.

Cubic orientation maths now agrees with mtex. `euler_bunge_to_matrix` returns Rz Rx Rz, the transpose of the Bunge g matrix, and the 24 cubic symmetry operators act on its crystal side; several functions applied them on the sample side, or compared orientations in the sample frame. Orientation pools, texture component names, boundary disorientation distributions and misorientation statistics produced by earlier versions used the wrong values. The reference tests carry frozen mtex 1.0.0 values, so they need neither mtex nor an EBSD file.

- **`cubic_misorientation` and `cubic_misorientation_scalar`** (`xtalphy.crystal_orientation`) returned wrong disorientations for large angles, by up to 56 deg. `grain_boundary_misorientation_distribution` and `gbmd_from_lfi` use them. The returned axis is now in crystal coordinates, and a 3x3 argument is read in the convention of `euler_bunge_to_matrix`.
- **`grain_avg_quats`** averaged pixels that are different symmetric variants of one orientation as plain quaternions, which can land tens of degrees from every pixel of the grain. It is now the cubic symmetry-aware mean (`symmetry=True`; `symmetry=False` keeps the plain mean). `EBSDReader.rechar_lfi` and `EBSDReader.grain_average_euler_deg` use it, and `rechar_lfi` derives the Euler angles of the pixels it fills from the mean quaternion.
- **`fcc_symmetrise_ori` and `detect_texture_component_peaks`**: the symmetric equivalents were up to 63 deg from their input, and the peaks reported the Euler angles of the inverse orientation, so copper, brass, Goss and S were never recognised by name.
- **`tops.synth_fcc`, `tops.synth_fcc_quats` and `tops.cubic_misorientation_V3`** (`xtalphy.texops`): the symmetric equivalents were wrong, so a copper-only texture had 8% of its orientations within 15 deg of copper; it now has at least 95%. `synth_fcc_quats` returned the quaternion of Rz Rx Rz, the inverse of the stored convention that EBSD grain means and the Abaqus Euler export use. `cubic_misorientation_V3` summed the wrong matrix entries.
- **`FCCTexture` and `texOps.fcc.fcc_symmetrise_ori`**: crystal and sample symmetry operators were on the wrong sides. `texOps.fcc.matrix_to_euler_bunge` was not the inverse of `euler_bunge_to_matrix` (it swapped phi1 and phi2).
- **FM steel disorientation** (`fm_steel_3d`): `OrientationAssigner3D.disorientation_deg`, the HAGB constraint of `assign_pag_orientations_with_hagb` and `assign_isolated_grain_orientations`, and `compute_misorientation_distribution` formed the relative rotation of the Bunge g matrices in the sample frame, so grains could be assigned orientations closer than the HAGB threshold (7.8 deg measured against 15 deg requested). `cubic_misorientation_old1` applied no symmetry at all. All now share `disorientation_from_g`, and the 24 variants of one Kurdjumov-Sachs parent are separated by exactly the tabulated set of angles.
- **Kurdjumov-Sachs variants** (`OrientationAssigner3D.get_ks_rotations`) were built from the wrong one of the two in-plane BCC directions and symmetrised on the wrong side; block orientations assigned from them were not members of the KS family.
- **FM steel block orientations** (`OrientationAssigner3D.assign_orientations_to_all_blocks`) composed each block's KS variant on the wrong side of its parent PAG orientation (`pag_g @ variant` instead of `variant @ pag_g`, confirmed against mtex's `variants(p2c, parentOri)`), so within-PAG block disorientations did not land on the tabulated KS angles (one measured case: 45.4 deg where the nearest tabulated value is 47.11 deg).
- **FM steel slice disorientation histograms** (`disorientation_stats.slice_disorientation_histograms`, the `blk_within`, `blk_across`, `sb_within` and `sb_across` types): used a plain rotation angle with no cubic symmetry, so none of the 276 mutual pairs of one parent's 24 Kurdjumov-Sachs variants landed on the tabulated set and some reached 180 deg; the `pag_adj` histogram was already correct. They now use `disorientation_from_g_batch`, a vectorised sibling of `disorientation_from_g` added for this.
- **Inverse-pole-figure colours and the colour key** (`xtalphy.ipf`, `ebsdviz.build_ipf_rgb`, `crystal_orientation.ipf_color`, `PoleFigure` IPF colouring, `twinned_simple_3d.viz_3d.plot_ipf_triangle_key`, the FM-steel IPF map colours): the colour was the normalised absolute crystal direction, which was not brought into the fundamental sector, so orientations related by a cubic symmetry operation had different colours, and the key showed one corner of the colour space. `build_ipf_rgb` also read DefDAP-stored quaternions as crystal-to-sample ones, colouring the inverse orientation. The colour is now mtex's default key for cubic crystals, reproduced to machine precision and pinned by tests with frozen mtex colours; the key is drawn on the equal-area projection of the sector [001]-[101]-[111]. `build_ipf_rgb` takes `quat_convention` (default `'defdap'`, the stored form every caller uses). `PoleFigure` IPF colours follow the mtex key for cubic symmetry only.
- **Sigma3 twin habit plane and lamella trace** (`compute_s3_habit_plane_3d`, `compute_s3_lamella_angle_2d`, used by the twinned FCC pipeline in 2D and 3D): the {111} plane normals were rotated into the sample frame with the quaternion read as crystal-to-sample, but the pipeline's quaternions are DefDAP-stored (sample-to-crystal). The lamella planes were up to 87 deg from the {111} planes of their grains, and a twin did not share the plane of its parent. They now match mtex's `orientation * Miller(1,1,1)`. The twin orientation itself (`TwinGenerator3D._apply_sigma3_variant`, the 60 deg <111> rotation composed in the crystal frame) was already correct and is unchanged; the four variants of one parent are mutually 38.94 deg (Sigma9) apart.
- **`EBSDReader.rechar_lfi`** (the fill of unindexed pixels, run by `characterise`): the work for each unindexed region included passes over the whole label field, so maps with thousands of regions were very slow: in assigning regions to grains, in counting the filled pixels, and in the grain averaging (three passes over every pixel, building 24 symmetry variants per pixel). The regions are now assigned with array operations (largest neighbouring grain, lowest id on a tie), only the grains that receive a region are averaged, and `_reduce_to_reference` rates the 24 variants without building them. Labels, quaternions and Euler angles are unchanged. A map with no indexed pixel raised `ValueError`; it is now left unchanged.

### Deprecated

- Unused methods that carry the same defects now raise a `FutureWarning` naming the defect and the replacement: `tops.cubic_misorientation_V1` and `_V2`, `tops.fcc_symmetrise_ori_V1`, `tops.fcc_variants_in_FZ`, `tops.fcc_variants_in_FZ_old`, `tops.tc_variants_in_FZ`, and the orientation, texture and block-orientation methods of `mcgs3_grain_structure` that were not already deprecated.

## [1.4.0] — 2026-10-21

### Fixed

- **Units lost in `clean_small_grains` and `remove_spikes`** (`fm_steel_3d`): both build a new `FMSteel3DBase` from the cleaned label grid without passing `units`, so the class default (`'microns'`) replaced the unit the structure was generated in. PAGs, blocks, orientations and sub-blocks take their unit from that object, so every later stage inherited the wrong one. Both pass `units=self.units`.
- **`MeshExporter3D` applied displacement and `length_scale`**: node coordinates are written multiplied by `length_scale`, but the applied displacement was not, so for `length_scale != 1` the load and the geometry were in different unit systems. The displacement is multiplied by `length_scale` too, and the heading comment of `model_master.inp` states when `length_scale` was applied.
- **`MeshExporter3D` element-set names and sections**: the section definitions follow the element-set prefix, so a subclass that renames element sets no longer leaves sections that point at the default names. With default settings the files written are byte-identical to before for all four element types at the PAG, block and sub-block levels.
- **Ensemble seed-config files**: `steps_ensemble.write_seed_config` wrote files without `injection_points`, which the loader requires, so they could not be read back. It uses the same writer as `write_ensemble_seed_config`.
- **`d3v2p1` surface angle repair**: the no-sharpening check measured the opening before a relocation with the node already at its new position, so it could not reject a relocation at the moved node. It measures the opening at the old position, as `d3v2p0` does.
- **2D conformal mesher in a worker thread**: Gmsh's default interrupt handler calls `signal.signal()`, which Python permits only on the main thread. `conformal_mesher2d` initialises Gmsh with `interruptible=False`, so the caller owns cancellation.
- **`write_subsampled_ctf`** raises when the destination resolves to the source file, which it would have truncated.
- **API reference build** (`docs/`): NumPy was mocked, so modules that evaluate NumPy expressions or annotations such as `np.ndarray | None` on import failed to import, and autodoc left out them and the modules that import them. The documentation workflow and `docs/requirements.txt` install NumPy, and it is no longer mocked. The malformed reStructuredText in three docstrings (`fm_steel_3d.deck_export`, `fm_steel_3d.grain_structure_summary.voxel_spacing`, `gbconformant.pipeline.background`) is corrected.
- **matplotlib 3.9 and later**: `matplotlib.cm.get_cmap` was removed, so plotting through `gsviz`, `meshviz`, `mcgs2Cont`, `mcgs2_temporal_slice`, `nonConformalMesher`, `fm_steel_3d` grain-structure plots and the representativeness assessment raised `AttributeError: module 'matplotlib.cm' has no attribute 'get_cmap'`. They use `plt.get_cmap`.
- **NumPy 2**: `np.NaN` (removed) in the representativeness assessment is `np.nan`; `np.in1d` (deprecated) in `fdbops`, `point2d` and `point3d` is `np.isin`.
- **gmsh on Linux**: a missing system library (`libGLU.so.1`) raised a bare `OSError` from `import gmsh`. `upxo._sup.optional_imports.import_gmsh` raises an `ImportError` that names the packages to install, and every gmsh import in the library uses it.

### Changed

- **`mcgsV1_1` is superseded by `mcgs_v2`.** `mcgsV1_1` re-exports `mcgs_v2` and `MCGSConfig` under the old names and emits a `DeprecationWarning`, so existing imports keep working. The `fm_steel_3d` and `twinned_simple_3d` step scripts import `mcgs_v2`.
- **`MeshExporter3D.configure_analysis`** (`fm_steel_3d.mesh_exporter_3d`): the `applied_strain` argument is replaced by `applied_displacement`, written to the `*Boundary` line as given (times `length_scale`). The strain form multiplied by the voxel count and voxel size along the load axis, so a caller had to know the physical length of the exported domain. `_write_steps` no longer takes `NX`, `NY`, `NZ` and `dx`.
- **Twinned FCC pipeline**: `calculate_lfi` and `from_mcgs` skip the PyVista grid and the mechanical properties of each temporal slice, which nothing in the pipeline reads. `rank_temporal_slices_by_n` accepts an explicit list of slice keys (`tslice_keys`), which overrides `start` and `step`.
- **Quieter grain detection**: `find_grains` has `print_msg` and `char_morphology_of_grains` has `verbose`, both defaulting to the previous output, for callers that print their own summary.
- **`expand_grain_quats_to_voxels`** (`xtalphy.crystal_orientation`) indexes the quaternions by a lookup table, so its cost no longer grows with the number of grains. The output is the same; negative labels still map to zeros.
- **Reports** (`reporting`): tables accept the same input shapes wherever a report takes one, and `normalize_table_data` is public for callers that build a table from them. Caption line breaks are kept (a caption is escaped first and its newlines then become `<br>`), so multi-paragraph captions no longer collapse into one line. `render_html(live_refresh_s=...)` adds a refresh tag, so a report that is still growing can be watched in a browser.
- **Python 3.14** is listed in the classifiers; the install-levels workflow already tested it.

### Added

- **`mcgs_v2`** (`ggrowth.mcgs_v2`): Monte Carlo grain growth in 2D and 3D from a plain Python configuration (`MCGSConfig`), with no Excel input dashboard. `dim=2` runs algorithms `'200'`, `'201'` and `'202'` on a `(ny, nx)` state array followed by `detect_grains()`; `dim=3` (the default) runs `'300a'` and `'300b'`. `xmax`, `ymax` and `zmax` are inclusive.
- **Conformal meshing pipeline** (`meshing/gbconformant/pipeline`): the grain-conformal tetrahedral meshing stages of `d3v2p0` and `d3v2p1` in one ordered run, from a labelled grain structure to an Abaqus deck, in Python (`run`) and from the command line (`python -m upxo.meshing.gbconformant.pipeline`).
  - Stages: input (load, subset, grain-id mapping), cleaning (small grains, spikes, pinches, voxel topology, thin grains, upscale), surface (interface smoothing, geometry guard and relaxation, point-contact separation, Gmsh remesh, RVE closing, joint remesh), validation and repair, tetrahedral meshing with dihedral smoothing, swaps and node insertion, region quality and before/after comparison (reports only), and export of surfaces, volume mesh and Abaqus decks. Each stage can be switched off by a setting.
  - Inputs: an FM steel raw export folder at the PAG, packet, block or sub-block level, a twinned FCC structure folder, or one 3D integer `.npy` label image. `STAGE_SUBSET` crops the input to a box given by voxel index, percent or physical length.
  - Settings are named, typed and limited; unknown names and unusable values raise `SettingsError` before any work starts, and settings saved by an older version are migrated on load. Presets, run-size forecast (tetrahedra, nodes, time and memory with ranges), and suggested setting changes after a failed stage.
  - Each finished stage saves its results (JSON and `.npz` files with a format and version tag, no pickles) in the run folder with a `run_manifest.json`. A later run restores a stage only while its own settings, the input files and every earlier stage are unchanged. The `status` command lists the saved stage results of a run folder and `report` writes its HTML report; `reproduce` prints the command that rebuilds a deck from its `provenance.json`, and `datacheck` runs the Abaqus data check on a deck folder (it needs Abaqus).
  - `background.StageProcess` and `run_stage_separately` run one stage in a separate process, report its progress, cancel it at any moment (the process and everything it started) and take its results into the caller's state only when it finishes.
  - Tests: `tests/meshing/gbconformant/pipeline/`.
- **FM steel Abaqus decks for conformal tetrahedral meshes** (`meshing/gbconformant/fm_steel_abaqus`): resolves which raw feature each meshed grain stands for after cleaning and merging, and writes a partitioned C3D4 deck with element sets by hierarchy level, materials with the grain orientations, sections, boundary node sets, junction lines, a constraint template (no constraint is active), interface triangles, a load history from a displacement or from a strain and strain rate, an output-size estimate, job scripts and `.msh` / `.vtu` copies of the mesh. Also: a feature ownership audit (`audit`), a merge log of the absorbed features with the misorientation between each and the feature whose orientation it now has (`mergelog`), per-feature fidelity of the mesh to the raw voxels (`fidelity`), a before/after comparison of the raw features against the meshed grains (`comparison`, `snapshots`, `parameters`, `plots`), pole figures of the raw and the meshed features and their difference (`texture`), checks of a written deck read back from its files (`deckcheck`), and twinned FCC structure folders as input (`twinned`). Tests: `tests/meshing/gbconformant/fm_steel_abaqus/`.
- **Progress reporting** (`upxo._sup.progress`): computation code reports how far it has got (`report`, `track`, `stage`, `plan`) and whoever starts the computation chooses what to do with it (`progress_scope` takes any callable; `ListSink`, `StdoutSink` and `TqdmSink` are provided). The listener follows the call stack through a `contextvars.ContextVar`; with none installed every call is a near-free no-op, and `check_cancel` and `ProgressCancelled` let a listener stop the computation. Reported by: the 3D MCGS algorithms, the twinned FCC pipeline (cleaning passes, slice validation, labelling, slice ranking, Abaqus export steps, twin introduction), EBSD map reading, and the `d3v2p0` geometry guard, guard relaxation, interface smoothing, pinch cleaning, unpinch, voxel topology and surface quality stages (each accepts a progress callback).
- **Configuration module** (`upxo.configuration`): portable, versioned configuration of the locations UPXO reads data from and writes exports to.
- **`write_rows`** (`upxo._sup.text_rows`): writes a 2D integer or float array to a text handle with the text `np.savetxt` produces, formatting each block of rows with one `%` operation. Input it cannot reproduce exactly (binary handles, empty input, unusual formats, object or bool arrays) goes to `np.savetxt` unchanged. `discrete_topology` writes its `.msh` text with it.
- **Plot-quality presets** (`viz.plot_quality`): `"quick"`, `"high"` and `"publication"` set fonts, line widths, palette and DPI together; `apply_preset` sets the process-wide default, `quality_context` scopes a preset to a block and `savefig_dpi` gives the export DPI. Settings passed to a plotting call take precedence.
- **FM steel (`fm_steel_3d`) modules**:
  - `deck_export`: `ConfigurableMeshExporter3D` extends `MeshExporter3D` with element-set prefixes and switches per hierarchy level, optional grain-boundary, junction-line and junction-point element sets, renamed or switched-off face node sets (the faces the analysis step loads and supports are always written), origin and domain-edge node sets, a requested hierarchy level including `'packet'`, and optional flat output. `export_deck` runs a whole export from plain arguments. Output at default settings equals that of `MeshExporter3D`.
  - `MeshExporter3D` hooks for subclasses: `_elset_prefix`, `_elset_enabled`, `_extra_node_sets` and `_part_include_files`.
  - `mesh_rescaling`: `mesh_state_for_scale` returns a read-through state that rebuilds the label grid and every voxel-location mapping at another resolution without modifying the source; `scaled_grid_shape` validates the factor.
  - `voxel_interface_masks`: `grain_boundary_voxels`, `junction_line_voxels` and `junction_point_voxels` for a 3D label grid.
  - `geom_metrics_3d.feature_distribution`: volume, surface area, equivalent-sphere diameter, sphericity, neighbour count, triple-line length, junction-point count or bounding-box aspect ratio per feature, in physical units. `slice_metrics_2d.slice_metric_values` and `pooled_slice_metric` give a metric per label of one slice and pooled over evenly spaced slices normal to chosen axes.
  - `slice_region_stats`: `pag_slice_distributions` collects area, aspect ratio, perimeter, solidity and neighbour count (wrapping at the slice edges) of PAG and grain regions on slices in the xy, yz or xz plane.
  - `disorientation_stats`: `hierarchical_disorientations` (adjacent and random PAG pairs, block pairs within and across PAGs, sub-block pairs within and across blocks) and `slice_disorientation_histograms`.
  - `grain_structure_summary`: `voxel_spacing` and `grain_structure_summary` (grain count, voxels and volume per grain, mean equivalent-sphere diameter, sphericity and neighbour count).
  - `visualization_maps` and `hierarchy_colouring`: label maps of the PAG, block and sub-block hierarchy and colour composites by hierarchy level.
  - `hierarchy_queries`: lookups of voxels, orientations and ancestors (PAG, packet, block, sub-block) by level.
  - `intercept_advice`: intercept grain sizes along the directions blocks are sliced in, and the interquartile range of the intercept lengths pooled over x, y and z.
  - `orientation_files`: `read_euler_table` reads Bunge angles per block from an Excel, tab-separated or comma-separated file.
  - `ensemble_seed_config`: `write_ensemble_seed_config` (no overwriting), `load_ensemble_seed_config` (validates, migrates older schema versions) and `list_ensemble_seed_configs`.
  - `distribution_export`: `export_distributions` writes property distributions to csv, Excel, text and pickle files.
  - `viz.distribution_plots`: `plot_kde`, `plot_kde_overlay` and `histogram_bin_count`.
  - Tests: `tests/fm_steel_3d/`.
- **Twinned FCC**: `island_twin_3d.introduce_island_twin_3d`, bounded oblate-ellipsoid Sigma3 twin inclusions that touch no boundary of the grain or region containing them, which the plane-slab carving of `TwinGenerator3D` cannot produce. `TwinGenerator3D(min_lamella_thickness_um=...)` is a floor on the lamella half-width in physical units, applied on top of the voxel-based floor. `feature_props_3d`: parent, host, twin and complete grain-size extraction (circle-equivalent diameter and intercept grain size) matching the EBSD side, the host-grain counterpart used before twins exist, per-axis slice properties, and the lines of the final summary.
- **2D orientation (EBSD) maps**: `xtalphy.gb_disorientation_map.compute_gb_disorientation` assigns the disorientation of each grain pair to the smoothed boundary lines between them; `viz.ebsd_maps` draws Euler, IPF, grain-boundary and grain-boundary-disorientation figures; `viz.ebsd_smoothing` builds grain polygons by segment-based (`smooth_technique_a`) or Voronoi-based (`smooth_technique_b`) smoothing, and `disorientation_from_polygons` builds the boundary disorientation result from either; `viz.ebsdviz.build_grain_role_sets` and `compute_four_category_grain_sets` give the parent, host, twin and complete grain-id sets; `viz.vizDistr.plot_slice_kde_curve_set` draws one reference density curve and a set of synthetic ones.
- **2D grain structures**: `GrainManifold2D.grain_boundary_ridges` returns the raw Voronoi ridge data between cells of different grain labels, captured before same-label cells are unioned; `gridops.planar_intercept_grain_size` gives the mean linear-intercept grain size along one axis of a 2D label array.
- **`d3v2p0`**: `spike_cleaning` (iterative removal of spike voxels from a label array) and `surface_export` (PyVista views of surface complexes and tetrahedral meshes).
- **`d3v2p1`**: faster versions of the surface stages (`geometry_guard`, `guard_relaxation`, crossing repair, `surface_quality`, `interfaces`), with the `d3v2p0` names and arguments plus `backend` and `n_workers`; their output is the same as `d3v2p0`'s.
  - `mesh_tables`: integer-key edge and face tables (`unique_rows`, `EdgeIncidence`, `node_element_index`), used by `small_facet_angles`, `is_disk_chart`, `smooth_interfaces`, `surface_intersections` and the face conformity check in `gmsh_tets`.
  - Tet verification, tet smoothing set-up and tet swaps are faster; their output is the same as before.
  - `improve_surface_angles(backend='numba')` (or `UPXO_BACKEND=numba`) runs a compiled candidate search in `d3v2p0`'s order of work; `'auto'` does not select it. If numba is missing, its kernels do not compile or its self-test fails, the parallel or numpy tier runs and the report says why.
  - `remesh_interfaces_gmsh` and `remesh_closed_rve_gmsh` take `retry_failed_only=True`: a chart that came out of a pass without a Gmsh warning keeps its mesh in later passes. The result is equivalent to a full re-mesh, not identical; the default is unchanged.
  - Worker processes start through a light module (`_pool_init`) that imports nothing heavy, and shared data goes in shared memory. `backend.check_backend` and `backend.requested_backend` (the backend after `UPXO_BACKEND` is applied).
- **CI**: `.github/workflows/install-levels.yml` builds the wheel and installs it at every installation level (base, viz, mesh, ebsd, all) on Linux and Windows with Python 3.13 and 3.14, and at the minimum declared dependency versions; `.github/scripts/install_check.py` checks an installed `upxo` at one level.
- `tests/test_removed_third_party_apis.py` (no tracked module uses a removed matplotlib, NumPy or SciPy API) and `tests/test_optional_imports.py`.
- README and wiki: the Linux system libraries that `gmsh` needs.

## [1.3.1] — 2026-10-06

### Fixed

- **Modules that failed to import**: `FMSteels` (an empty `__slots__` entry), `ImageConfig` (a mutable dataclass default), and several modules that imported helpers from locations that do not exist in the installed package (`heirGs/pdomain_2.py`, `pxtal/polyxtal.py`, `geoEntities/edge2d.py`, `_sup/dataTypeHandlers.py`, `parswep/mcgs2d_parameter_sweeping.py`, `initialize.py`, and the two `external/ellipsoid_fit` scripts). Each now imports from the real package path.
- **Modules that ran code on import**: `_sup/progress_bar.py`, `interfaces/package_check.py` and `mechanics/twins_calculations.py` executed demo code or printed output when imported. They now do so only when run as a script.
- **`fm_steel_3d` transformation steps** (`apply_transform`, `sweep_clean`) imported their helpers from a package that is not shipped. The helpers are now in `pxtal/fm_steel_3d/transform_shared.py`.
- **Dependency declarations**: `netlsd`, used by the `nlsd` graph-similarity measure of the representativeness assessment (`repqual`, `repgen`), is now a base dependency in `pyproject.toml`, `setup.py` and `requirements.txt`. `setup.py` also lacked `statsmodels`, which the other two listed.
- **Functions with no working replacement now say so**: `gops.PROFILE_up2d_INST`, `dataTypeHandlers.coords_to_UpxoPointList`, `pdomain_2.make_PDI`, the `mulpoint2d` and edge branches of `pops.make_polygon`, the `sortjp` values other than `'ignore'` in `find_gb_jp`, and `grid.plotgs` raise `NotImplementedError` with a message, in place of failing on an import of a module that no longer exists. `initialize.py` raises it for its Voronoi paths.
- `pdomain_2` prints tables with pandas, so `tabulate` is not needed.

### Changed

- **Default output folder**: the FM steel and twinned FCC pipeline reports and exports, the FM steel mesh exporter and the twinned Abaqus exporter wrote to a `data` folder two levels above the installed package, which is inside the Python environment for a pip install and fails on a read-only system installation. They now write to `<checkout>/data` in a source checkout and to `./data` under the working directory otherwise (`upxo._sup.data_dir.default_data_dir`). Tests: `tests/test_default_data_dir.py`.
- **README and wiki**: tested Python versions (3.13, 3.14), platforms (no `gmsh` wheel on Linux aarch64, so no `[mesh]` or `[all]` there), dependency versions from the declared minimum to the newest, and install notes (`netlsd` built from source, Windows long paths).

### Removed

- **`meshing/confMesh3d`** (the SurfaceNets pipeline: `complex_builder`, `surface_nets`, `surface_remesh`, `tetgen_mesh`, `volume_mesh`, `validation`, `export`, `config`) and **`meshing/pxtalmesh_01.py`** (the `confMesh2d` class built on `pygmsh`). 3D conformal meshing is in `meshing/gbconformant` (`d3v2p0`, `d3v2p1`), which uses neither of them.
- No code in the library uses `trimesh`, `pygmsh` or `orix`. The 3D branch of `set_pxtal` that imported `orix` raises `NotImplementedError`.
- References to the removed modules in docstrings, `docs/conf.py`, `docs/workflows.rst` (Workflow 15 describes the `d3v2p0` / `d3v2p1` pipeline) and the `confMesh2d_*` demonstration notebooks.

### Added

- `tests/test_import_integrity.py`: every tracked module imports in a fresh interpreter and prints nothing; every import statement in the package and in the tracked notebooks resolves (module, imported name, declared dependency); `pyproject.toml`, `setup.py` and `requirements.txt` agree; entry points and Sphinx listings resolve.
- `tests/test_repaired_legacy_paths.py`: the repaired import paths and the `NotImplementedError` messages above.

## [1.3.0] — 2026-10-06

### Added

- **Twinned FCC 3D Abaqus export with the reference UMAT materials, mm units and a load step**: `twinned_simple_3d.abaqus_exporter_3d.AbaqusExporter3D` output now follows the reference model used to validate the target UMAT (`T25C.inp`).
  - **C3D4 export**: `element_type='C3D4'` splits every voxel into 6 tetrahedra (Kuhn decomposition, same table as `fm_steel_3d.mesh_exporter_3d`); neighbouring voxels share the same body diagonal so the mesh is conforming. `element_type` is now validated against `SUPPORTED_ELEMENT_TYPES = ('C3D8', 'C3D4')` — previously any string was accepted and written above 8-node brick connectivity, producing an input file Abaqus rejects. New `n_elements` property.
  - **Materials**: new default `material_format='reference_umat'` — `*Depvar 1` then `*User Material` with 6 constants (Bunge angles in degrees wrapped into `[0, 360)`, a sequential grain number, then two constants copied from the reference whose meaning is set by the UMAT). `n_depvar` defaults to `1`. `'bunge_euler'`/`'orientation'` remain available.
  - **Units and loading**: new `length_scale` (default `1e-3`) multiplies every node coordinate to write mm. New uniaxial static step (`write_uniaxial_static_step`): the max face of `load_axis` (default `z`) displaced by `applied_strain` (default `0.2`) × the domain length, the min face of every axis held in its own direction; `step_time` (default `80`) configurable. Interactions are no longer written (matching the reference) and no longer warn about it.
  - `export_abaqus_mesh` (the automation step) passes `length_scale`/`load_axis`/`applied_strain`/`step_time` through, with defaults matching the exporter.
  - Fixed: the export step reported `nx*ny*nz` elements regardless of element type — six times too few for C3D4. Now reports `AbaqusExporter3D.n_elements`. Also fixed `write()` leaving a disabled element-set file's placeholder open.
  - Tests: `tests/twinned_simple_3d/test_abaqus_exporter_3d_model.py`, element-type and export-step tests in the same suite.
- **EBSD → smoothed mesh → Abaqus (2D)**: end-to-end pipeline from a real EBSD `.ctf` map to a conformal 2D mesh with per-grain, EBSD-measured Bunge-Euler orientations in the exported Abaqus `.inp`.
  - `interfaces/defdap/ebsd_reader.py`: `EBSDReader.split_disconnected_grains(connectivity=4)` — a grain id spread across spatially disconnected pixel regions (most commonly from `crop()` slicing an irregular grain in two) is relabelled so every id is one connected region; `euler_ebsd`/`quat_ebsd` untouched. `EBSDReader.grain_average_euler_deg()` — per-grain Bunge-Euler angles (degrees) from a mean of `quat_ebsd` over each grain's pixels (renormalised, positive-hemisphere), exposed as a standalone public method (the same averaging approach `rechar_lfi` already used internally).
  - `meshing/writer_ABQ.py`: `export_confmesh2d_inp` gains `material_format='bunge_euler'` — writes one `*Material` + `*User Material, constants=3` (the grain's Bunge-Euler angles) + `*Depvar` per grain, from a `grain_euler_deg` dict, mirroring the convention already used by the 3D exporter (`twinned_simple_3d.abaqus_exporter_3d.AbaqusExporter3D`). Default behaviour (`material_format='isotropic'`) unchanged; also gains an `elastic_constants` override for that path.
  - Demo: `src/upxo/demos/ebsdOps/ebsd_to_abaqus_2d.ipynb` — EBSD read → crop → characterise → split disconnected grains → grain-averaged orientation → Technique A geometrification/smoothing → conformal mesh → element-quality statistics → Abaqus export → interactive PyVista mesh-vs-grain-boundary view.
  - Wiki: new [EBSD to Abaqus (2D)](https://github.com/Design-By-Fundamentals-UKAEA/UPXO/wiki/EBSD-to-Abaqus-2D) page, cross-linked from Data I/O, Meshing, and Use Cases.
  - Tests: `tests/interfaces/defdap/test_ebsd_reader.py`, `tests/meshing/test_writer_abq.py` — synthetic-array fixtures, no DefDAP / `.ctf` file / Gmsh dependency.
- **Editable 2D grain polygons**: grain boundaries built from UPXO geometry objects, convertible to and from Shapely, with per-grain scalar properties.
  - `geoEntities/polygon2d.py`: `Polygon2d` (one grain, built on `ring2d`) and `NestedPolygon2d` (a host with hole polygons, nestable). `edit_segment` replaces the interior nodes of one boundary segment in place, keeping its two end nodes; `subdivide_segment(seg_index, n=None, at_fractions=None)` inserts interior points by arc-length fraction. `polygons_to_prop_dataframe` / `apply_prop_dataframe` move per-grain scalars to and from a DataFrame.
  - `geoEntities/polygon2d_from_shapely.py`: `polygon_collection_from_shapely(cells, gid_key=None, tol=1e-6)` converts `{gid: Polygon | MultiPolygon | GeometryCollection}` (for example `GrainManifold2D.cells`, or a Voronoi tessellation) into `{gid: Polygon2d | NestedPolygon2d}`. Neighbouring grains reference the same `MSline2d` object for a shared wall, so an edit made through one grain is seen by the other, the boundary stays gap-free and total area is conserved. For a `MultiPolygon` cell the largest part is the value and the rest are in `props['extra_parts']`.
  - `pxtal/geometrification.py`: `polygonised_grain_structure.construct_geometric_xtals_from_gbcoords(..., dtype='upxo')` returns `Polygon2d` / `NestedPolygon2d` built directly from `self.GB`, without copying.
  - `geoEntities/mulsline2d.py`: `ring2d.translate(dx, dy, seen=None)`.
  - Demos: `src/upxo/demos/geom/poly2d01.ipynb` to `poly2d06.ipynb` — `Polygon2d` basics, the Shapely converter, `subdivide_segment` with a hand-rolled boundary-roughening preview, and conversion of Voronoi, Monte Carlo (Technique B) and Technique A grain structures.
  - Tests: `tests/geoEntities/test_polygon2d*.py`, `tests/pxtal/test_geometrification_polygon2d.py`, `tests/pxtal/test_polygon2d_from_shapely_grainmanifold.py`.
- **3D conformal tet meshing: element-quality controls** (`meshing/gbconformant/d3v2p0`). Every angle limit is a user input; tet limits are reported, not enforced.
  - Thin grains: `thin_freeze.detect_thin_grains` (inscribed-thickness criterion) selects grains to freeze; `local_thickness.local_thickness` / `node_local_thickness` give local grain thickness from voxel labels.
  - Interface smoothing: `interfaces.smooth_interfaces` gains `thickness_cap_fraction`, `minimum_wedge_angle`, `minimum_corner_angle` and `iterations_by_pair` (per grain pair). `guard_relaxation.relax_guarded_interfaces` re-smooths regions rolled back by the geometry guard.
  - Facet openings: `facet_angles.edge_openings` and `facet_angles.sharpens` implement the no-sharpening rule (an opening below the limit may not get sharper; one at or above it may not fall below it). Gmsh remeshes (`gmsh_interfaces`, `gmsh_closed`) take `minimum_remesh_opening` and restore source charts where a remesh creates a sharper opening.
  - Surface triangles: `surface_angles.improve_surface_angles` raises the smallest triangle angles by diagonal flips, node relocation on the surface or curve, and short-edge collapses, with `wedge_limit` for the no-sharpening rule. `surface_metrics.voxel_like_fraction` reports the share of grain-boundary area that is still voxel-like.
  - Tetrahedra: `tet_angles.dihedral_angles` / `dihedral_summary`; `gmsh_tets.mesh_repaired_rve_gmsh` gains `netgen_all_volumes`, `optimize_threshold` and `minimum_dihedral` (reported); `gmsh_tets.save_grain_tetrahedra`. `tet_smoothing.smooth_grain_tetrahedra` moves interior and grain-surface nodes to raise the smallest dihedral angle (floors anchored to the starting angles, neighbour floor, maximum angle, surface triangle floors). `tet_swaps.swap_grain_tetrahedra` (2-3 and 3-2 swaps inside grains) and `tet_swaps.insert_grain_tetrahedra` (node insertion).
  - Views: `grain_subset_view` (grains above a size percentile), `tet_quality_view` (clipped tets coloured by dihedral angle), `quality_plots.triangle_quality_distribution` / `tet_quality_distribution` (one histogram per measure, counting the elements within a fraction of the way from the worst value to the best possible) and `quality_plots.show_surface` (grain-boundary surface with or without edges).
  - Tests: `tests/meshing/gbconformant/d3v2p0/`.
- **3D conformal tet meshing: faster stages** (`meshing/gbconformant/d3v2p1`). Same function names and arguments as `d3v2p0`, plus `backend` and `n_workers`; stages without a faster version are imported from `d3v2p0`.
  - Faster stages: grain-interface and joint Gmsh remesh, pre-tet surface validation, triangle intersection search, surface angle repair, tet meshing (one Gmsh model per grain, grains in parallel), tet dihedral smoothing (8-colour block phases in parallel) and tet swaps.
  - Results: the remeshes, validation, intersection search and swaps give the same output as `d3v2p0`; angle repair and tet smoothing follow the same rules in a different order, so their meshes are equivalent, not identical.
  - `backend.py`: CPU detection (physical and usable cores), automatic worker count (physical cores), tiers `'numpy'` (serial, pure numpy), `'parallel'` (worker processes) and `'numba'` (compiled kernels), chosen by argument, then `UPXO_BACKEND` / `UPXO_N_WORKERS`, then automatically. Without numba or worker processes every stage runs serially in numpy; if worker processes fail, the stage reruns serially. Each stage report records the tier and worker count used.
  - numba kernels (`numba_intersections`, `numba_smoothing`, `numba_swaps`) for the intersection search, smoothing trial evaluation and surface checks, and swap acceptance.
  - Tests: `tests/meshing/gbconformant/d3v2p1/`.
- **Twinned FCC 3D: pole-figure sample symmetry and EBSD-against-synthetic comparison** (`twinned_simple_3d/steps/steps_visualization_export.py`).
  - `sample_frame_quats` converts DefDAP passive quaternions to the sample frame and replicates them through the sample-symmetry group (180 deg rotations about the selected RD/TD/ND axes; default all three).
  - `plot_pole_figure` gains `apply_sample_symmetry` / `use_rd` / `use_td` / `use_nd`, `projection` (`'stereographic'` or `'equal_area'`), `hemisphere` and `plot_mode` (`'density'`, `'scatter'`, `'hybrid'`).
  - `plot_pole_figure_comparison` (EBSD, synthetic and overlay or density difference, on one shared colour scale) and `pretwin_pole_figure_populations` (EBSD parent grains against assigned twin hosts).
  - Tests: `tests/twinned_simple_3d/test_pole_figure_steps.py`.
- **Twinned FCC 3D Abaqus export: step increment and field-output controls**: `AbaqusExporter3D` and `export_abaqus_mesh` take `initial_inc`, `min_inc`, `max_inc`, `max_increments`, `output_interval` and `element_outputs` / `node_outputs` (defaults `('LE', 'NE', 'S', 'SDV')` / `('U',)`), validated by `validate_step_controls`. The output database gets `step_time / output_interval + 1` frames.
- **Demonstration notebooks** (`src/upxo/demonstrations/`): 3D conformal meshing (`CM02`, `CM03`, `CM04`, `CM04_dev`) and 2D conformal meshing (`confMesh2d_*`).

### Changed

- **`pxtal/geometrification.py`, Technique A with island grains**: `self.GB` and `self.GBCoords` now include island grains, keyed by original grain id (in 1.2.1 an island id had no entry). A new `self.GB_holes` maps a host grain id to the `ring2d` of each island it directly encloses; an island nested inside another island is a hole of its immediate parent. An island's `ring2d` is the same object in its own `self.GB` entry and in its host's `self.GB_holes` list. Tests: `tests/pxtal/test_geometrification_islands.py`.
- `pxtal/_gb_topology.py` (new): `junction_points_from_polygons` and `assemble_ring_from_wall_segments`, extracted from `polygonised_grain_structure`; `get_junction_points_from_grain_intersections` and `flip_segments_to_reorder_GBS` call them. Behaviour is unchanged.

### Fixed

- **`GrainManifold2D.smooth_interfaces`**: cells could extend past the label-image domain after smoothing. `_laplacian_step` holds only vertices lying exactly on the domain edge, and the negative-`mu` Taubin pass can move other boundary-adjacent vertices outward. `smooth_interfaces` now ends with `trim_to_rve(bounds=(0, 0, width, height))`. `trim_to_rve(bounds)` remains available for cropping to a sub-window. Tests: `tests/pxtal/test_grainmanifold2d_domain_clip.py`.

- **Twinned FCC 3D texture comparison** (`twinned_simple_3d/steps/steps_distribution_viewer.py`): `compute_pole_figure_delta_mud_iqr` used the DefDAP quaternions without converting them to the sample frame, and neither it nor `plot_texture_residual` applied sample symmetry. Both now use `sample_frame_quats`, like every other pole figure, with `apply_sample_symmetry` / `use_rd` / `use_td` / `use_nd`.

### Removed

- **`pyvoro` dropped from the `mesh` extra.** It had no real dependents in the library (only two demo scripts called it) and was a recurring install failure on Windows. `mesh` is now `tetgen` + `gmsh` only. Also fixes drift found while doing this: `setup.py`'s `mesh` extra was missing `gmsh` (unlike `pyproject.toml`'s); `requirements.txt` had `tetgen` active/uncommented unlike every other extras entry in that file.
- **`src/upxo/scripts/` removed from the repository entirely.** Development-only scratch scripts, never meant to ship; moved (with history preserved via `git subtree split`) to a standalone, purely local repo (`upxo_scripts`) that is never pushed anywhere.
- **`src/upxo/demos/` is no longer tracked by this repository.** Demo notebooks and their supporting files now live at [UPXO-demos](https://github.com/Design-By-Fundamentals-UKAEA/UPXO-demos) (MIT licensed), extracted with full history preserved. The folder itself still exists on disk in a working checkout of this repo for active local development (some of its contents currently support in-progress conformal-meshing work), but nothing under it is committed or pushed from here going forward, and it is not part of the installed package — it never shipped in the wheel/sdist in the first place.
- **Unused modules no longer tracked**: `algorithms/alg310_bu.py`, `interfaces/abaqus/upxo_aba.py` and `pxtalops/morphochar.py`. Nothing imported them and none of them compiled (incomplete function body, Python 2 `print` statement, invalid syntax), so they are no longer part of the package.
- **References to the discontinued cleaving mesh generator** (`meshing/gbconformant/cleaving3dV1P0`, never part of a release) removed from the `meshing` package docstring and from a skipped package-layout test.

## [1.2.1] — 2026-09-22

### Fixed

- **`pxtal/geometrification.py`**: Technique A (`polygonised_grain_structure`) now handles island grains (a grain fully enclosed by another) correctly. `pix_to_geom` splits the structure into a hole-filled version plus one geometrification pass per island cluster (recursing for nested islands), then cuts each island out of its host with a Shapely `difference`. Previously any structure containing an island grain raised `IndexError` or `AttributeError`, or produced a host with no interior ring.
- Junction-point extraction (`get_junction_points_from_grain_intersections`) recorded only the first end of a `LineString` intersection; a grain with a four-grain corner could lose a junction point and raise `A linearring requires at least 4 coordinates`. Both ends are now recorded.
- `set_grain_loc_ids` classified domain corners with an `elif` chain, so a grain spanning two corners (e.g. wrapping a one-pixel grain on the domain edge) lost the second corner's boundary segment; its ring then failed to close and the polygon collapsed to zero area. Corner tests are now independent; the boundary-segment consolidation step removes the resulting duplicate segments and adds any domain-edge segment still missing.
- Island-containing structures: `self.GB` / `self.GBCoords` were keyed by the filled sub-structure's local relabelled ids rather than original grain ids, so indexing by `gid` could silently return a different grain's boundary ring. Now remapped to original ids; an island grain has no entry (its boundary is a hole in its host, not a same-level ring) rather than returning wrong data.
- `set_grain_centroids_raw`'s vectorised fast path could raise `IndexError`, or silently return a wrong centroid, for a caller-supplied `gid` absent from the label image; it now returns `NaN` for that case, matching the original per-grain loop.
- `ring2d.clone()` copied `coords`/`conn0`/`conn1` by reference despite documenting an independent copy; now copied by value.
- **`GrainManifold2D`** (Technique B): single-pixel and other small grains, including islands, could collapse to a sliver under repeated Taubin smoothing passes — small-grain vertices are now frozen by default (`thin_grain_px`). Mirrored ghost seeds in `_generate_clipped_polygons` could carve spurious voids near the domain edge into the raw (untrimmed) manifold — guard-seed placement fixed. `generate_constrained_hybrid_seeds` could miss a single-pixel or one-pixel-wide grain's boundary entirely (a central-difference gradient blind spot) — a 4-neighbour-difference pass, plus a final per-label seed guarantee independent of the boundary-sampling stride, close both gaps.
- **`viz/gsviz.plot_multipolygon_geometric`**: interior rings (holes) were drawn as opaque white polygons, hiding an island grain drawn earlier in iteration order. Each polygon is now one compound path so holes are true cut-outs.
- **`repqual/`**: `determine_distr_type` used target values for every sample; `mc2repr` threshold setters could store invalid values and loop forever; NLSD results overwrote `rkf['ed']` instead of their own key.
- **`pxtal/`**: zero-twin case in `remove_overlaps_in_twins` handled; `repgen3d` missing `__slots__` prevented construction; undefined `np` reference in `geoEntities`' abstract coords body removed.

### Performance

- Technique A (`polygonised_grain_structure.pix_to_geom` + `smooth_gbsegs`): roughly 15x faster (930 grains: 54 s → 3.7 s, then further to ~1.9 s in a second pass; scaling improved from roughly quadratic to roughly linear in grain count). All-pairs geometry tests replaced with Shapely `STRtree` candidate queries; array scans replaced with hash lookups; one method removed as dead work (its result was overwritten before being read); per-segment properties cached instead of recomputed for every neighbour pair; `deepcopy` in `smooth_gbsegs` replaced with new `MSline2d.clone()` / `ring2d.clone()`.
- Technique B (`GrainManifold2D.smooth_interfaces` + `_generate_clipped_polygons`): the vertex-adjacency graph is now built once per smoothing run instead of once per Taubin pass; Voronoi cells that do not cross the RVE boundary skip the clipping step. Roughly 20-30% faster depending on grain-structure regularity. Neither technique is reliably faster than the other across all grain-structure types — see `confMesh2d_compare_techniques.ipynb` for current, measured numbers on a given case.

### Removed

- `rasterio` is no longer a UPXO dependency. `pxtal/geometrification.py`'s `polygonize()` and `pxtal_ori_map_2d.py`'s `polygonize_voronoi_grid` now use the new pure NumPy/Shapely `_sup/raster_polygonize.py`, matching `rasterio.features.shapes`'s output structure. The `[io]` extra is removed from `pyproject.toml` / `setup.py` / `requirements.txt`; `docs/conf.py`'s mock-imports list and the README / docs install instructions updated to match.

### Added

- **`_sup/raster_polygonize.py`**: pure NumPy/Shapely label-to-polygon tracer (4-connected, hole-aware).
- **`_sup/raster_islands.py`**: detects grains fully enclosed by another grain (islands) ahead of polygonisation.
- **Demo**: `confMesh2d_compare_techniques.ipynb` — Technique A vs Technique B on the same label image, with geometry, mesh-quality and timing comparisons.
- Wiki: new 2D Grain Structure Geometrification page documenting both techniques, island handling, and how to choose between them.

### Infrastructure

- CI: Sphinx docs now also deploy on pushes to `dev` (previously `main` only); tests now also run on pushes to `main` (previously `dev` only).

---

## [1.2.0] — 2026-09-11

### Added

#### 2D Voronoi Tessellation Engine
- **`pxtal/voronoi_tessellation_2d/`**: Standalone high-fidelity 2D Voronoi geometry engine — periodic boundary conditions, Laguerre/power diagrams (weighted Voronoi), centroidal Voronoi tessellation (CVT) via Lloyd relaxation, and grain-boundary interface perturbation
- **`pxtal/vortess2d.gtess2d`**: Wired onto the new engine — `periodic`, `weights`, `cvt_iterations`, `perturb_factor` exposed as constructor keywords; new `from_mpoint2d` and working `from_shapely_mulpolygon` constructors; `bounds`/`info`/`seeds` properties; `__len__`
- **`pxtal/geotess.geotess2d`**: Completed previously-stubbed container/topology methods — `__getitem__`/`__setitem__`/`__repr__`, `make_seeds_random`/`make_seeds_pdisc` (Bridson sampling), first/second nearest-neighbour topology, boundary/internal grain filtering; `perturb_grain_boundaries` now delegates to the new engine
- **Tests**: `tests/pxtal/test_vortess2d.py` covering the engine, `gtess2d`, and `geotess2d` integration

#### Twinned FCC 3D Demo Automation
- **`pxtal/twinned_simple_3d/steps/`**: Automation-ready wrapper package for the Twinned FCC pipeline, mirroring `fm_steel_3d/steps/`
- **Demos**: `twinned_fcc_bas0.ipynb`/`bas1.ipynb` (minimal, annotated), `twinned_fcc_int0.ipynb`/`int1.ipynb` (full pipeline with validation/diagnostics), `twinned_fcc_adv0.ipynb`/`adv1.ipynb` (deeper tuning, mapping optimization, slice sweep) under `src/upxo/demos/Twinned3D/`

#### Conformal Meshing Extensions
- **`meshing/confMesh3d/`**: Optional surface remeshing stage (`surface_remesh.py`) and a TetGen tet-meshing backend (`tetgen_mesh.py`)

#### 2D conformal meshing (raw Gmsh)
- **`confMesh2dGMSH`**: Shared grain-boundary line tags, island/void interiors, Gmsh physical-group ELSETs, `from_geometric_pxtal`, safer `try`/`finally` Gmsh sessions, vectorised mesh extract
- **`viz.meshviz.plot_conformal_2d_by_grain`**: Grain-ELSET fill with optional GB overlay and face NSETs (used by `confMesh2dGMSH.plot_by_grain` and `gsmesh2d.visualize_gs_mesh`)
- **`writer_ABQ.export_confmesh2d_inp`**: Compact 1..N nodes; CPS3/CPS6/CPS4/CPS8 or CPE3/CPE6/CPE4/CPE8 from connectivity width; `GRAIN_*` ELSETs, `NS_LEFT`/`RIGHT`/`TOP`/`BOTTOM`/`GB` NSETs, optional dummy sections (`summarize_inp` helper)
- **2D mesh fidelity**: `prepare_grain_polygons` (snap / orient); Threshold `dist_min`/`dist_max` actually applied; optional Laplace2D optimize; clockwise elements reversed after extract; `fidelity_report` (mesh vs Shapely area/GB length) and `quality_report` (aspect ratio, min angle)
- **Demos** (force-tracked under `src/upxo/demos/confMesh/`): `confMesh2d_gmsh.ipynb` (canonical), `confMesh2d_export.ipynb` (plot + Abaqus INP), `confMesh2d_mcgs.ipynb` (MCGS teaching path), `confMesh2d_geometrify.ipynb` (polygonise then `mesh_gs`), `confMesh2d_technique_b.ipynb` (Technique B then `mesh_gs`); capability notebooks `confMesh2d_tri_vs_quad`, `confMesh2d_islands_voids`, `confMesh2d_voronoi`, `confMesh2d_size_field`, `confMesh2d_quadratic`, `confMesh2d_quality`, `confMesh2d_nsets`
- **Tests**: `tests/meshing/test_confmesh2d_gmsh.py`

#### Centralized Reporting
- **`reporting/`**: Session/entries/HTML-rendering core for building structured pipeline reports
- **`pxtal/fm_steel_3d/raw_export.py`**: Raw pipeline-state export across PAG/Packet/Block/Sub-block hierarchy levels
- Reporting integration wired into `fm_steel_3d` and `twinned_simple_3d`

#### Self-Representativeness Assessment (twinned_simple_3d)
- `selfrepr_morphology.py`: Per-grain morphological parameters for EBSD self-representativeness studies
- `representativeness_metrics.py`: Registry of two-sample distribution-comparison similarity metrics
- `selfrepr_qualification.py`: Threshold-based qualification of representativeness results
- `subsetting_2d.py`: 2D rectangular sub-domain extraction (sibling of the existing 3D `subsetting.py`)
- `stride_study.py`: Moving-window stride/tiling study support
- `mc_qualification.py`: Monte-Carlo candidate qualification
- `base_3d.percentile_trim`: Percentile-of-range outlier trimming, alongside the existing IQR-based trim
- `crystal_orientation.compute_grain_csl_participation`: Per-grain parent/twin role tally across all CSL types

#### FM Steel Packet-Level Tooling
- `orientation_mean_3d.py`: Crystallographically-correct packet mean orientation
- `slice_metrics_2d.py`: Fast numpy-only 2D slice geometric metrics

#### FM Steel 3D Demo Automation
- **`pxtal/fm_steel_3d/steps/`**: New automation-ready wrapper package, one thin module per pipeline stage (start, base grain structure, cleaning, transformations, PAG clustering, block generation, sub-block generation, visualization, raw export, mesh export, ensemble seed config); lives inside the installed package (unlike Twinned3D's own `steps/`, which sits under `demos/`) so it is directly importable by automation scripts, not just notebooks
- `geom_metrics_3d.feature_aspect_ratio_bbox`: Free-function per-feature 3D bounding-box aspect ratio, generalizing the existing 2D/instance-bound versions
- `viz/grain_structure_viz_3d.GrainStructureViz3D.build_distinguishable_cmap`: Shuffled large-N colormap so per-feature ID coloring stays distinguishable past `tab20`'s 20-color limit
- `viz/grain_structure_viz_3d.GrainStructureViz3D.lfi_to_polydata`: New `jupyter_backend` param to embed an interactive PyVista widget (e.g. `'trame'`) in a notebook cell instead of a blocking native window
- **Demos**: `fm_steel_3d_bas0.ipynb` (concise, full pipeline through mesh export/ensemble config), `fm_steel_3d_bas1.ipynb` (annotated, + Quick-Demo Mode toggle) under `src/upxo/demos/FMSteel3D/`

#### Visualization
- Side-by-side voxel-grid comparison render
- IPF triangle key plot and side-by-side LFI comparison render
- Pole-figure size-colored scatter overlay with its own colorbar
- Histogram-with-slice-band plotting (`vizDistr.plot_hist_with_slice_band`)
- Per-axes property-distribution plotting (`vizDistr.plot_property_distribution_on_axes`)
- Grain-role-aware EBSD visualization for CSL parent/twin analysis

#### Other Core Additions
- `charops.boundary_grain_fraction`: Fraction of grains touching the domain boundary, for 2D labelled images
- `gsdataops/grid_ops`: Majority-vote-safe downsampling and anisotropic stretch
- `pxtal/gridops`: Raw per-crossing lengths exposed from `axis_intercept_grain_size`

#### Testing
- Reporting integration tests (fm_steel_3d, twinned_simple_3d)
- FM steel raw-export and packet-orientation-mean tests
- GB-conformant meshing (`meshing/gbconformant/d3v2p0`) test suite

### Changed

- **Twinned FCC orientation assignment**: `max_retries` exposed; MDF bin width/angle range now derived from `mdf_ref` instead of a hardcoded 65°; `neighbour_frac` added to Pool B host allocation with do/do-not fallback; MRF Gibbs/MAP initialization reuses `self.fallback_quats` instead of drawing a fresh unseeded sample
- **`xtalphy/texops.py`**: Fixed the same Bunge ZXZ Euler-decomposition bug present in `crystal_orientation.matrix_to_euler_bunge`; added texture-component detection; ported previously interactive-only quaternion helpers into core
- **Packaging**: `requirements.txt` enables `tetgen`; optional extra `[mesh]` now also installs `gmsh>=4.13`
- **`confMesh2d` (pygmsh)** is deprecated; new 2D conformal work must use `confMesh2dGMSH` / `gsmesh2d.mesh_gs`
- Removed redundant tracked demos `confMesh2d9.ipynb` and `mesh2d01.ipynb`
- The voxel-lattice cleaving mesh generator (`meshing/cleaving/`) is untracked pending relocation to `meshing/gbconformant/cleaving3dV1P0/` (WIP, not part of this release)

### Not Included (Deferred)

- MC Metropolis-acceptance refinement (permanently out of scope per architecture decision)
- GB-energy/crystallography coupling for block selection (deferred to future versions)
- Multi-CSL twin registry beyond Sigma-3 Path A (CSL Path B deferred post-Path A validation)
- Cleaving mesh generator migration to `meshing/gbconformant/cleaving3dV1P0/` (WIP, untracked)

---

## [1.1.0] — 2026-08-04

**First official TestPyPI release of UPXO.**

### Added

#### Core Pipeline Modules
- **`pxtal/fm_steel_3d/`**: Hierarchical microstructure generation for FM steels
  - `base_3d.py`: Core FMSteelGenerator with PAG generation and clustering
  - `orientation_3d.py`: Texture-guided crystallographic orientation assignment
  - `cleaning_3d.py`: Twin and artifact removal
  - `feature_props_3d.py`: Role-stratified property queries (martensite/austenite)
  - `repr_validator_3d.py`: Post-generation validation metrics
  - `viz_3d.py`: Role-colored visualization and refinement helpers

- **`pxtal/twinned_simple_3d/`**: Twinned FCC microstructure generation
  - `base_3d.py`: Core TwinnedSimple3DBase with spatial-dispersal host allocation
  - `orientation_3d.py`: Texture-guided orientation with Rodrigues calculations
  - `twin_generator_3d.py`: Multi-path Sigma-3 twin embedding
  - `feature_props_3d.py`: Twin and matrix property queries
  - `cleaning_3d.py`: Twin-specific artifact removal
  - `repr_validator_3d.py`: Post-twin validation metrics
  - `viz_3d.py`: Twin-colored visualization

- **Conformal Meshing (`meshing/confMesh3d/`)**: Five-stage pipeline for grain-boundary-aligned tetrahedral meshing
  - Surface extraction via marching cubes
  - Complex builder for grain boundary geometry
  - Mesh validation and quality checks
  - Volume mesh generation via gmsh
  - Abaqus and VTK export

#### Material Module Refactoring
- **`material/`**: Type-safe material registry and properties
  - `identity.py`: MaterialIdentity with name, alloy, composition fields
  - `processing.py`: ProcessingRoute and ProcessingStep with deformation type classification
  - `provenance.py`: Provenance tracking (source, method, parameters, timestamp)
  - `registry.py`: MaterialRegistry with typed instance storage and soft validation
  - `texture.py`: TextureComponentProfile for crystal-family-generic texture modeling

#### Operations Helpers (New)
- **`gbops/gbpoint_ops.py`**: Grain-boundary point extraction and analysis
- **`netops/neighops.py`**: Neighbor graph connectivity utilities
- **`propOps/morphops.py`**: Morphological property helpers (volumes, binning)
- **`imageOps/labelops.py`**: Label reindexing and manipulation
- **`fdbOps/fdbops.py`**: Feature database entry helpers

#### EBSD Integration
- Support for EBSD data import (CIF, HDF5 via DefDAP)
- Texture ODF (Orientation Density Function) profiling from EBSD scans
- EBSD-guided synthetic microstructure generation
- Crystal orientation validation against experimental data

#### Documentation & Examples
- Comprehensive wiki (18+ pages) covering all capabilities
- Detailed workflow examples (20 workflows covering 2D/3D generation, meshing, visualization)
- Updated README with all new capabilities highlighted
- Demo notebooks for FM Steel and Twinned FCC pipelines

#### Testing
- 12 new test modules with 1,280 lines of test code
- Material registry tests (initialization, ingestion, provenance, validation)
- Operations helper tests (gb_ops, net_ops, prop_ops, image_ops, fdb_ops)
- Visualization smoke tests (ebsdviz, vizDistr)
- Grain structure and Voronoi constructor tests
- 82/87 tests passing (93.1%)

### Changed

- **`pxtalops/twin3d.py`**: Added optional `host_coords` parameter to `introduce_twin_lamella_3d()` for coordinate caching optimization
- **README.md**: Expanded Core Capabilities and Microstructures Supported sections; fixed typos
- **Workflows documentation**: Expanded from 8 to 20 workflows covering all major capabilities

### Fixed

- Typos in README: "pertaining multi-scale" → "pertaining to multi-scale", "teknology" → "technology", "powerpoint" → "power plant", "visibility" → "viability"
- Test suite: Resolved 26 initial test failures through systematic debugging and API alignment

### Infrastructure

- Added `.gitignore` entries for legacy local-only folders and generated output directories
- Organized data, sessions, and generated output structures
- Sphinx build workflow configured for automated API documentation generation

### Not Included (Deferred)

- MC Metropolis-acceptance refinement (permanently out of scope per architecture decision)
- GB-energy/crystallography coupling for block selection (deferred to future versions)
- Multi-CSL twin registry beyond Sigma-3 Path A (CSL Path B deferred post-Path A validation)

---

## [1.0.0] — Development Only (Never Published)

Initial development version. Features and APIs evolved significantly before first TestPyPI release.
