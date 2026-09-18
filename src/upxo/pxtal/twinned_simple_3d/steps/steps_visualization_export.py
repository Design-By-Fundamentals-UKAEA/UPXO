"""Visualization & Export -- Part O of the Twinned FCC walkthrough.

Covers a 3D PyVista render, a raw npy/pickle dump, and an Abaqus mesh +
elsets/nodesets export. Elset/nodeset configuration is pure data (no
compute step of its own) -- it is all consumed by export_abaqus_mesh
below via its role_enabled/role_prefix/nset_config arguments. Finishes
by rendering the accumulated upxo.reporting.ReportSession to report.html.
"""
from pathlib import Path

from .steps_start import DEFAULT_OUTPUT_DIR

# Per-role-level elset defaults (every level enabled, prefix "es_<level>_").
DEFAULT_ROLE_ENABLED = {
    'non_host': True, 'host': True, 'primary_twin': True,
    'stwin_a': True, 'stwin_b': True,
}
DEFAULT_ROLE_PREFIX = {
    'non_host': 'es_nonhost_', 'host': 'es_host_', 'primary_twin': 'es_primary_',
    'stwin_a': 'es_seca_', 'stwin_b': 'es_secb_',
}
# Per-face nodeset defaults -- origins/rve_edges nodesets are not
# implemented and are omitted here.
DEFAULT_NSET_CONFIG = {
    'XMIN': {'enabled': True, 'prefix': 'ns_x-'}, 'XMAX': {'enabled': True, 'prefix': 'ns_x+'},
    'YMIN': {'enabled': True, 'prefix': 'ns_y-'}, 'YMAX': {'enabled': True, 'prefix': 'ns_y+'},
    'ZMIN': {'enabled': True, 'prefix': 'ns_z-'}, 'ZMAX': {'enabled': True, 'prefix': 'ns_z+'},
}


def render_3d_structure(cleaner, role_opacity=None, role_color=None, nonhost_cmap=None):
    """Interactive PyVista 3D render (renders inline if called from a
    Jupyter notebook with PyVista's Jupyter backend enabled; opens a
    separate window otherwise). No return value -- this is a plotting
    side effect.

    role_opacity/role_color default to None, which lets render_3d apply
    its own tuned defaults -- notably primary_twin at partial opacity
    (0.6, not fully solid), so secondary twins nucleated at or inside the
    primary-twin/host interface stay visible instead of being hidden
    behind an opaque primary-twin surface. Pass your own dicts only if
    you deliberately want different opacities/colours.
    """
    import numpy as np
    from upxo.pxtal.twinned_simple_3d.viz_3d import render_3d

    lgi_xyz = np.transpose(cleaner.lgi_clean, (2, 1, 0))
    render_3d(lgi_xyz, cleaner.twin_role_clean, role_opacity=role_opacity,
              role_color=role_color, nonhost_cmap=nonhost_cmap)


def render_ipf_slice(cleaner, axis=2, slice_idx=None, sample_direction=(0., 0., 1.)):
    """Plots an IPF-coloured 2D slice through the cleaned structure --
    plain matplotlib (unlike render_3d_structure above, this has no
    interactive-window step and runs fine in a headless/automated
    execution).

    `axis`: 0=X, 1=Y, 2=Z. `slice_idx`: None (default) uses the domain's
    mid-slice.

    Returns
    -------
    (fig, ax)
    """
    from upxo.pxtal.twinned_simple_3d.viz_3d import plot_ipf_slice
    return plot_ipf_slice(cleaner.lgi_clean, cleaner.all_quats_clean, axis=axis,
                           slice_idx=slice_idx, sample_direction=sample_direction)


def sgs_pole_figure_stages(cleaner):
    """Per-grain orientations for the five SGS (synthetic structure)
    pole-figure populations, matching
    steps_ebsd_analysis_2.ebsd_pole_figure_stages()'s shape exactly for
    direct pairing in steps_distribution_viewer.plot_texture_residual():
    'full' (every grain), 'host' (every non-twin grain -- both allocated
    hosts and untouched non-hosts, the synthetic analogue of EBSD's
    'parents'), 'primary_twins', 'secondary_twins', and 'all_twins'.

    Returns
    -------
    dict : {'full', 'host', 'primary_twins', 'secondary_twins',
    'all_twins'} -- each a {'gids': ndarray, 'quats': ndarray}.
    """
    import numpy as np

    role_filters = {
        'full': None,
        'host': ('host', 'non_host'),
        'primary_twins': ('primary_twin',),
        'secondary_twins': ('secondary_twin',),
        'all_twins': ('primary_twin', 'secondary_twin'),
    }
    stages = {}
    for stage_key, wanted_roles in role_filters.items():
        if wanted_roles is None:
            gids = list(cleaner.all_quats_clean.keys())
        else:
            gids = [g for g in cleaner.all_quats_clean.keys()
                    if cleaner.twin_role_clean.get(g) in wanted_roles]
        quats = np.array([cleaner.all_quats_clean[g] for g in gids], dtype=np.float64)
        stages[stage_key] = {'gids': np.array(gids), 'quats': quats}
    return stages


def plot_pole_figure(stage_data, pole_family='100', plot_mode='density',
                      half_width_deg=7.5, title=None, ax=None):
    """Plots one population's pole figure -- a {'gids', 'quats'} dict
    from ebsd_pole_figure_stages()/sgs_pole_figure_stages().

    plot_mode: 'density' (default -- MUD contour) or 'scatter'.
    half_width_deg: the density kernel's angular half-width (degrees) --
    ignored for plot_mode='scatter'.

    Returns
    -------
    (fig, ax)
    """
    from upxo.viz.xphy.pole_figure import PoleFigure

    quats = stage_data['quats'].copy()
    quats[:, 1:] *= -1   # crystal->sample conjugation, same as render_ipf_slice's source data
    pf = PoleFigure(quats, convention='quaternion', gids=stage_data['gids'])
    if plot_mode == 'scatter':
        return pf.plot_scatter(pole_family=pole_family, ax=ax, title=title)
    return pf.plot_density(pole_family=pole_family, ax=ax, half_width_deg=half_width_deg, title=title)


def plot_pole_figure_overlay(stage_a, label_a, stage_b, label_b, pole_family='100',
                              marker_a='o', marker_b='^', color_a='steelblue',
                              color_b='darkorange', title=None):
    """Overlays two populations' scatter pole figures on one axes,
    distinguished by marker shape (not just colour) -- e.g. parents vs.
    twins on the same plot. Always scatter -- density contours from two
    different-sized populations can't be meaningfully overlaid the same
    way; use plot_pole_figure() per population for density instead.

    Returns
    -------
    (fig, ax)
    """
    import matplotlib.pyplot as plt
    from upxo.viz.xphy.pole_figure import PoleFigure

    fig, ax = plt.subplots(figsize=(6, 6))
    for stage_data, label, marker, color in (
            (stage_a, label_a, marker_a, color_a), (stage_b, label_b, marker_b, color_b)):
        quats = stage_data['quats'].copy()
        quats[:, 1:] *= -1
        pf = PoleFigure(quats, convention='quaternion', gids=stage_data['gids'])
        pf.plot_scatter(pole_family=pole_family, ax=ax, color_by='fixed',
                         color=color, marker=marker, label=f'{label} (n={len(stage_data["gids"])})')
    ax.legend(loc='upper right')
    default_title = f"{{{pole_family}}} Pole Figure -- {label_a} vs. {label_b}"
    ax.set_title(title or default_title, fontsize=12, fontweight='bold')
    return fig, ax


def next_master_folder(output_dir=DEFAULT_OUTPUT_DIR, base_filename="grain_structure"):
    """Picks the next unused "<base_filename><N>" folder name under
    <output_dir>/TwinnedFCC/Grain Structures -- the collision-avoidance
    convention export_raw() uses by default, and the one
    steps_temporal_slice_sweep.sweep_temporal_slices() uses once per
    sweep (not once per slice) to give every slice in that sweep a
    shared master folder. Creates the "Grain Structures" directory if it
    does not already exist (needed to check what's already there), but
    NOT the returned folder itself -- that happens when something is
    actually written into it.

    Returns
    -------
    str : the chosen folder name (not a full path), e.g. "grain_structure3".
    """
    out_base = Path(output_dir) / "TwinnedFCC" / "Grain Structures"
    out_base.mkdir(parents=True, exist_ok=True)
    idx = 1
    while (out_base / f"{base_filename}{idx}").exists():
        idx += 1
    return f"{base_filename}{idx}"


def export_raw(cleaner, output_dir=DEFAULT_OUTPUT_DIR, base_filename="grain_structure", master_folder=None):
    """Dumps the cleaned structure's raw arrays (label field,
    orientations, twin role/parent maps) as .npy/.pkl -- the lightest-
    weight, format-agnostic save, useful for reloading straight back into
    Python later without re-running the pipeline.

    Files are written directly under
    <output_dir>/TwinnedFCC/Grain Structures/<master_folder>/. When
    `master_folder` is None (the default), a fresh, auto-numbered
    "<base_filename><N>" folder is picked via next_master_folder(), so
    repeated exports never overwrite each other. Pass an explicit
    `master_folder` (e.g. "<sweep_folder>/tslice_<key>", as
    sweep_temporal_slices() does) to nest this export inside a folder
    shared across a whole sweep instead.

    Returns
    -------
    dict : {'out_dir', 'master_folder', 'files' (name -> byte size),
    'n_grains', 'timestamp'}
    """
    import numpy as np
    import pickle
    import datetime

    if master_folder is None:
        master_folder = next_master_folder(output_dir, base_filename)
    out_dir = Path(output_dir) / "TwinnedFCC" / "Grain Structures" / master_folder
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "lgi_twinned_clean.npy", cleaner.lgi_clean)
    with open(out_dir / "orientations_all.pkl", "wb") as f:
        pickle.dump(cleaner.all_quats_clean, f)
    with open(out_dir / "twin_role.pkl", "wb") as f:
        pickle.dump(cleaner.twin_role_clean, f)
    with open(out_dir / "twin_parent_of.pkl", "wb") as f:
        pickle.dump(cleaner.twin_parent_of_clean, f)

    file_names = ["lgi_twinned_clean.npy", "orientations_all.pkl", "twin_role.pkl", "twin_parent_of.pkl"]
    return {
        "out_dir": str(out_dir),
        "master_folder": master_folder,
        "files": {name: (out_dir / name).stat().st_size for name in file_names},
        "n_grains": len(cleaner.twin_role_clean),
        "timestamp": datetime.datetime.now().isoformat(timespec='seconds'),
    }


def build_abaqus_index(cleaner, verbose=True):
    """Builds the grain-element index the Abaqus export needs (which
    elements belong to which grain/role/family/variant). Cache this
    yourself if exporting the same cleaned structure more than once --
    it's the expensive part.
    """
    from upxo.pxtal.twinned_simple_3d.abaqus_exporter_3d import AbaqusExporter3D
    return AbaqusExporter3D.build_index(
        cleaner.lgi_clean, cleaner.twin_role_clean, cleaner.twin_parent_of_clean, verbose=verbose)


def export_abaqus_mesh(cleaner, tg, index, output_dir=DEFAULT_OUTPUT_DIR,
                        base_filename="twinned_fcc_mesh", voxel_size_um=1.0,
                        element_type="C3D8", material_format="bunge_euler",
                        n_depvar=100, write_role_elsets=True, write_family_elsets=True,
                        write_variant_elsets=True, role_enabled=None, role_prefix=None,
                        family_prefix="es_family_", variant_prefix="es_variant_",
                        nset_config=None, material_level="feature"):
    """Writes the full Abaqus .inp mesh (voxel-conformal
    C3D8 elements by default) plus elsets (by role/family/variant) and
    nodesets (domain faces), into a fresh, auto-numbered subfolder so
    repeated exports never overwrite each other.

    `index`: from build_abaqus_index() above. `tg`: the TwinGenerator3D
    from steps_twin_generation.generate_twins (needed for family/variant
    elset provenance).

    Returns
    -------
    dict : {'output_dir', 'n_elements', 'n_elsets', 'n_nsets', 'files'
    (name -> byte size)}
    """
    from upxo.pxtal.twinned_simple_3d.abaqus_exporter_3d import AbaqusExporter3D

    out_base = Path(output_dir) / "TwinnedFCC" / "ABQInputFiles"
    out_base.mkdir(parents=True, exist_ok=True)
    idx = 1
    while (out_base / f"{base_filename}{idx}").exists():
        idx += 1
    out_dir = out_base / f"{base_filename}{idx}"

    exporter = AbaqusExporter3D(
        twin_role=cleaner.twin_role_clean, twin_parent_of=cleaner.twin_parent_of_clean,
        all_quats=cleaner.all_quats_clean, twinmake=tg, voxel_size_um=voxel_size_um,
        element_type=element_type, material_format=material_format, n_depvar=n_depvar,
        write_role_elsets=write_role_elsets, write_family_elsets=write_family_elsets,
        write_variant_elsets=write_variant_elsets,
        role_enabled=role_enabled or DEFAULT_ROLE_ENABLED,
        role_prefix=role_prefix or DEFAULT_ROLE_PREFIX,
        family_prefix=family_prefix, variant_prefix=variant_prefix,
        nset_config=nset_config or DEFAULT_NSET_CONFIG,
        material_level=material_level, index=index,
    )
    exporter.write(str(out_dir))

    file_sizes = {f.name: f.stat().st_size for f in sorted(Path(out_dir).glob("*.inp"))}
    return {
        "output_dir": str(out_dir),
        "n_elements": exporter.nx * exporter.ny * exporter.nz,
        "n_elsets": len(exporter._grain_elems) + exporter.n_role_elsets_written,
        "n_nsets": exporter.n_nsets_written,
        "files": file_sizes,
    }


def finish_report(report):
    """Renders the accumulated report (every add_image/add_table call
    made throughout this notebook) to report.html.

    Returns
    -------
    pathlib.Path : path to the written report.html.
    """
    from upxo.reporting.render_html import render_html
    return render_html(report)
