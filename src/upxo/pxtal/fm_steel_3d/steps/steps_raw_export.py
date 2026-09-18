"""Raw Data Exports -- Part I of the FM Steel 3D walkthrough.

Thin wrapper around `upxo.pxtal.fm_steel_3d.raw_export` (built and tested
against the GUI's own Raw Data Exports page) -- no reimplementation here.
`export_raw_data` (and the functions it calls) only ever access `app`
duck-typed via `getattr(app, "_fm_base", None)` etc., so a notebook can
hand it any small object exposing the same five attributes instead of the
real Tkinter App -- that's exactly what `PipelineStages` is for.
"""


class PipelineStages:
    """Duck-typed stand-in for the GUI App's five pipeline-stage attributes,
    so `upxo.pxtal.fm_steel_3d.raw_export.export_raw_data` (and
    `level_availability`/`collect_level_data`) work unmodified from a
    notebook. Pass whichever stage objects exist so far; the rest default
    to None and are reported as unavailable/skipped."""

    def __init__(self, fm_base=None, fm_with_pags=None, fm_with_blocks=None,
                 fm_with_orientations=None, fm_with_subblocks=None):
        self._fm_base = fm_base
        self._fm_with_pags = fm_with_pags
        self._fm_with_blocks = fm_with_blocks
        self._fm_with_orientations = fm_with_orientations
        self._fm_with_subblocks = fm_with_subblocks


def export_raw(stages, out_dir, formats=('npy', 'pkl', 'npz'), levels=None, prefixes=None):
    """Exports `stages`' data to `out_dir`. See
    `upxo.pxtal.fm_steel_3d.raw_export.export_raw_data` for the full
    contract (per-level "<prefix>_lgi" arrays, per-level
    "<prefix>_orientations.pkl", a shared bundle for everything else).

    Returns
    -------
    dict : {"out_dir", "files", "n_entries", "levels_exported", "levels_skipped"}
    """
    from upxo.pxtal.fm_steel_3d.raw_export import export_raw_data
    return export_raw_data(stages, out_dir, list(formats), levels=levels, prefixes=prefixes)
