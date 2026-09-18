"""Mesh Export -- Part J of the FM Steel 3D walkthrough.

Thin wrapper around `mesh_exporter_3d.MeshExporter3D`. `fm_state` accepts
any of FMSteel3DWithPAGs/WithBlocks/WithOrientations/WithSubBlocks --
the exporter auto-detects the lowest hierarchy level present for
element-set/material/section assignment. Default element-set naming
prefixes from the base class are used as-is (per-elset prefix
customization is a GUI-only convenience, wired up by a GUI-internal
subclass in runner.py -- not needed here).
"""
from pathlib import Path


def export_mesh(fm_state, out_dir, folder_name='fm_steel_mesh', element_type='C3D8',
                 output_unit='microns', verbosity=0, custom_message=""):
    """Exports `fm_state` as an Abaqus voxel-conformal mesh into
    `<out_dir>/<folder_name>/` (auto-created if missing --
    MeshExporter3D.set_output_base_path requires the path to already exist).

    `element_type`: one of 'C3D8' (linear hex), 'C3D4' (linear tet),
    'C3D20' (quadratic hex), 'C3D10' (quadratic tet).

    Returns
    -------
    pathlib.Path : the folder actually written to.
    """
    from upxo.pxtal.fm_steel_3d.mesh_exporter_3d import MeshExporter3D

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exporter = MeshExporter3D(verbosity=verbosity)
    exporter.set_output_units(output_unit)
    exporter.set_output_base_path(str(out_dir))

    export_fn = {
        'C3D8': exporter.export_c3d8, 'C3D4': exporter.export_c3d4,
        'C3D20': exporter.export_c3d20, 'C3D10': exporter.export_c3d10,
    }[element_type]
    return export_fn(fm_state, folder_name, custom_message=custom_message)
