"""Temporal-Slice Sweep -- advanced-tier extension, run at the end of the
pipeline.

Part G's ranking picks a single "best" temporal slice cheaply, from 2D
cross-sectional grain-count/property comparisons alone -- it never runs
the actual downstream pipeline, so it cannot know a slice's ACHIEVED
total twin volume fraction (that number only exists once host allocation,
orientation assignment, twin generation, and cleaning have all actually
run for that slice). This module re-runs that full downstream chain for
every requested saved temporal slice and reports which ones reach a given
percentage of the EBSD-partitioned twin-volume-fraction target -- the
only way to answer that question directly, at the cost of repeating the
most expensive part of the pipeline once per slice swept.

Every slice's downstream chain is independent of every other slice's
(TwinnedSimple3DBase.from_mcgs copies its input label grid rather than
aliasing shared state), so the per-slice work below is written as one
module-level function, `_run_one_slice`, callable identically from a
plain sequential loop or from a worker process pool.
"""


def _run_one_slice(tslice_key, pxt, rg, parent_info, vf_targets, twin_thickness,
                    host_alloc_kwargs, orientation_kwargs, twin_gen_kwargs,
                    clean_kwargs, target_total_vf, vf_threshold_pct,
                    output_dir, sweep_master_folder, export_passing):
    """Runs host allocation, orientation assignment, twin generation, and
    cleaning for one temporal slice, and reports its achieved total twin
    volume fraction against `target_total_vf`. Module-level (rather than
    nested inside sweep_temporal_slices) so it can be sent by reference
    to a worker process on platforms -- Windows included -- whose
    multiprocessing start method cannot pickle a closure.

    Returns
    -------
    dict : on success, 'tslice_key', 'achieved_total_vf', 'target_total_vf',
    'pct_of_target', 'passes', 'achieved_hosting_fraction',
    'n_host_grains', 'n_conflicts', 'n_grains_clean', 'export' (the dict
    returned by export_raw(), or None if this slice did not qualify for
    export). On failure (an exception raised during that slice's
    pipeline): 'tslice_key', 'error' only.
    """
    import numpy as np
    from .steps_host_allocation import allocate_hosts
    from .steps_orientation_assignment import assign_orientations
    from .steps_twin_generation import generate_twins, clean_structure
    from .steps_visualization_export import export_raw

    try:
        base = allocate_hosts(pxt, tslice_key, **host_alloc_kwargs)
        assigner = assign_orientations(base, rg, parent_info, **orientation_kwargs)
        tg, tvf = generate_twins(base, rg, parent_info, assigner, twin_thickness,
                                  vf_targets=vf_targets, **twin_gen_kwargs)
        cleaner, n_passes_run = clean_structure(tg, **clean_kwargs)
    except Exception as e:
        return {'tslice_key': tslice_key, 'error': str(e)}

    twin_gids = {g for g, role in cleaner.twin_role_clean.items()
                 if role in ('primary_twin', 'secondary_twin')}
    twin_vox = int(np.isin(cleaner.lgi_clean, list(twin_gids)).sum())
    total_vox = int(np.sum(cleaner.lgi_clean > 0))
    achieved_total_vf = twin_vox / total_vox if total_vox > 0 else 0.0
    pct_of_target = 100 * achieved_total_vf / target_total_vf if target_total_vf > 0 else 0.0
    passes = pct_of_target >= vf_threshold_pct

    export_result = None
    if export_passing and passes:
        export_result = export_raw(
            cleaner, output_dir=output_dir,
            master_folder=f"{sweep_master_folder}/tslice_{tslice_key}")

    return {
        'tslice_key': tslice_key,
        'achieved_total_vf': achieved_total_vf,
        'target_total_vf': target_total_vf,
        'pct_of_target': pct_of_target,
        'passes': passes,
        'achieved_hosting_fraction': base.actual_hosting_fraction,
        'n_host_grains': len(base.host_grain_ids),
        'n_conflicts': assigner.n_conflicts,
        'n_grains_clean': len(cleaner.twin_role_clean),
        'export': export_result,
    }


def _print_row(tslice_key, row):
    if 'error' in row:
        print(f"  tslice_key={tslice_key}: FAILED -- {row['error']}")
        return
    status = 'PASS' if row['passes'] else 'below threshold'
    print(f"  tslice_key={tslice_key}: achieved_total_vf={row['achieved_total_vf']:.4f}  "
          f"pct_of_target={row['pct_of_target']:.1f}%  "
          f"hosting_fraction={row['achieved_hosting_fraction']:.4f}  "
          f"n_conflicts={row['n_conflicts']}  "
          f"n_grains={row['n_grains_clean']}  {status}")


# Populated once per worker process by _init_worker, then read by
# _run_one_slice_worker -- keeps pxt/rg/parent_info and the kwargs dicts
# from being re-pickled once per slice (they are pickled once, at pool
# start-up, per worker process instead).
_worker_state = {}


def _init_worker(pxt, rg, parent_info, vf_targets, twin_thickness,
                  host_alloc_kwargs, orientation_kwargs, twin_gen_kwargs,
                  clean_kwargs, target_total_vf, vf_threshold_pct,
                  output_dir, sweep_master_folder, export_passing):
    _worker_state.update(
        pxt=pxt, rg=rg, parent_info=parent_info, vf_targets=vf_targets,
        twin_thickness=twin_thickness, host_alloc_kwargs=host_alloc_kwargs,
        orientation_kwargs=orientation_kwargs, twin_gen_kwargs=twin_gen_kwargs,
        clean_kwargs=clean_kwargs, target_total_vf=target_total_vf,
        vf_threshold_pct=vf_threshold_pct, output_dir=output_dir,
        sweep_master_folder=sweep_master_folder, export_passing=export_passing)


def _run_one_slice_worker(tslice_key):
    s = _worker_state
    return _run_one_slice(
        tslice_key, s['pxt'], s['rg'], s['parent_info'], s['vf_targets'],
        s['twin_thickness'], s['host_alloc_kwargs'], s['orientation_kwargs'],
        s['twin_gen_kwargs'], s['clean_kwargs'], s['target_total_vf'],
        s['vf_threshold_pct'], s['output_dir'], s['sweep_master_folder'],
        s['export_passing'])


def sweep_temporal_slices(pxt, rg, parent_info, vf_targets, twin_thickness,
                           tslice_keys=None, vf_threshold_pct=90.0,
                           host_alloc_kwargs=None, orientation_kwargs=None,
                           twin_gen_kwargs=None, clean_kwargs=None,
                           export_passing=True, output_dir=None,
                           base_filename="temporal_slice_sweep", verbose=True,
                           n_workers=1):
    """Re-runs host allocation, orientation assignment, twin generation,
    and cleaning for every requested temporal slice, and reports whether
    each slice's achieved total twin volume fraction reaches
    `vf_threshold_pct` of the EBSD-partitioned target (Stage-1 +
    Secondary-2a + Secondary-2b from `vf_targets`).

    `tslice_keys`: None or an empty list sweeps every slice `pxt` saved
    except index 0 (the Monte Carlo simulation's un-annealed seed state,
    excluded for the same reason Part G's own ranking excludes it -- see
    Part G's markdown). Given an explicit list, only the keys actually
    present in `pxt.m` are used; any others are dropped with a printed
    note rather than raising an error.

    `host_alloc_kwargs`/`orientation_kwargs`/`twin_gen_kwargs`/
    `clean_kwargs`: passed through to
    steps_host_allocation.allocate_hosts() /
    steps_orientation_assignment.assign_orientations() /
    steps_twin_generation.generate_twins() /
    steps_twin_generation.clean_structure() respectively -- pass the same
    dicts of named parameters already configured earlier in the notebook
    so every swept slice is evaluated under identical settings.

    `n_workers`: 1 (default) sweeps every slice sequentially, in this
    process, exactly as before. An integer greater than 1 instead
    distributes the requested slices across that many worker processes
    (`concurrent.futures.ProcessPoolExecutor`) -- process-based rather
    than thread-based, because each slice's work is ordinary
    CPU/NumPy-bound Python that the interpreter's global lock would
    prevent threads from actually overlapping. `pxt`, `rg`, `parent_info`,
    and the four kwargs dicts are pickled once per worker process (not
    once per slice) via the pool's initializer. `n_workers` is capped at
    `min(n_workers, os.cpu_count(), number of slices actually swept)`.
    With more than one worker, per-slice progress prints as each slice
    finishes rather than as it starts, since several are in flight
    concurrently; the returned list is still ordered to match the order
    slices were swept in, identically to the sequential case.

    Data export protocol: every slice whose achieved fraction reaches
    `vf_threshold_pct` (when `export_passing=True`, the default) shares
    ONE master folder for the whole sweep,
    `<output_dir>/TwinnedFCC/Grain Structures/<base_filename><N>/`
    (auto-numbered via steps_visualization_export.next_master_folder(),
    picked once at the start of this sweep -- not once per slice, so
    every passing slice in this run lands under the SAME master folder,
    while a later, separate sweep gets a fresh one and never collides
    with this one). Inside it, each passing slice gets its own
    `tslice_<key>/` subfolder (via
    steps_visualization_export.export_raw()'s `master_folder` argument),
    so results from different slices in the same sweep never collide or
    overwrite one another either. Abaqus mesh export is deliberately NOT
    run here, as it is comparatively expensive per slice; re-run Parts
    I-P for one specific `tslice_key` afterward if a mesh is needed for a
    particular passing slice.

    Returns
    -------
    list of dict : one row per swept slice, in the order swept -- see
    _run_one_slice()'s docstring for each row's shape.
    """
    import os
    from .steps_visualization_export import next_master_folder, DEFAULT_OUTPUT_DIR

    host_alloc_kwargs = dict(host_alloc_kwargs or {})
    orientation_kwargs = dict(orientation_kwargs or {})
    twin_gen_kwargs = dict(twin_gen_kwargs or {})
    clean_kwargs = dict(clean_kwargs or {})
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR
    sweep_master_folder = next_master_folder(output_dir, base_filename) if export_passing else None

    available = [k for k in pxt.m if k != 0]
    if not tslice_keys:
        keys_to_run = available
    else:
        keys_to_run = [k for k in tslice_keys if k in available]
        dropped = [k for k in tslice_keys if k not in available]
        if dropped:
            print(f"Sweep: ignoring requested slice keys not present in this run: {dropped}")

    target_total_vf = (vf_targets['tvf_stage1'] + vf_targets['tvf_secondary_2a']
                        + vf_targets['tvf_secondary_2b'])

    if n_workers <= 1 or len(keys_to_run) <= 1:
        rows = []
        for tslice_key in keys_to_run:
            if verbose:
                print(f"--- Sweeping tslice_key={tslice_key} ---")
            row = _run_one_slice(
                tslice_key, pxt, rg, parent_info, vf_targets, twin_thickness,
                host_alloc_kwargs, orientation_kwargs, twin_gen_kwargs,
                clean_kwargs, target_total_vf, vf_threshold_pct, output_dir,
                sweep_master_folder, export_passing)
            rows.append(row)
            if verbose:
                _print_row(tslice_key, row)
        return rows

    from concurrent.futures import ProcessPoolExecutor, as_completed
    n_workers = min(n_workers, os.cpu_count() or 1, len(keys_to_run))
    if verbose:
        print(f"Sweeping {len(keys_to_run)} slices across {n_workers} worker processes ...")
    rows_by_key = {}
    with ProcessPoolExecutor(
            max_workers=n_workers, initializer=_init_worker,
            initargs=(pxt, rg, parent_info, vf_targets, twin_thickness,
                      host_alloc_kwargs, orientation_kwargs, twin_gen_kwargs,
                      clean_kwargs, target_total_vf, vf_threshold_pct,
                      output_dir, sweep_master_folder, export_passing)) as pool:
        futures = {pool.submit(_run_one_slice_worker, k): k for k in keys_to_run}
        for future in as_completed(futures):
            tslice_key = futures[future]
            row = future.result()
            rows_by_key[tslice_key] = row
            if verbose:
                _print_row(tslice_key, row)
    return [rows_by_key[k] for k in keys_to_run]
