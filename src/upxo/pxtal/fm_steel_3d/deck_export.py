"""Abaqus deck export with chosen set names, optional sets and a chosen level.

``ConfigurableMeshExporter3D`` extends ``MeshExporter3D`` with:

* element-set name prefixes and an on/off switch for each hierarchy level;
* optional element sets of grain-boundary, junction-line and junction-point
  voxels;
* renamed or switched-off face node sets (the faces the analysis step loads
  and supports are always written);
* optional node sets at the origin and along the edges of the domain;
* a requested hierarchy level for materials and sections, including the
  ``'packet'`` level;
* optional writing of the files straight into the output directory.

``export_deck`` runs a whole export from plain arguments: rescaling, exporter
set-up and the call for the chosen element type.

Options (all optional; ``options`` is a plain dict)
---------------------------------------------------
* ``export_level``: ``'auto'`` (default), ``'subblock'``, ``'block'``,
  ``'packet'`` or ``'pag'``.
* ``elset_<kind>``: bool, kinds pag, pck, blk, subblk and iso (default True).
* ``prefix_<kind>``: str, default ``'es_<kind>_'``.
* ``elset_gb``, ``elset_jl``, ``elset_jp``: bool, default False; the set is
  named ``<prefix>all``.
* ``prefix_gb``, ``prefix_jl``, ``prefix_jp``: str, default ``'es_gb_'``,
  ``'es_jl_'`` and ``'es_jp_'``.
* ``nset_<face>``: bool, faces x_min, x_max, y_min, y_max, z_min and z_max
  (default True).
* ``prefix_<face>``: str, default the library face-set names.
* ``nset_origins``: bool, default False; the set is named ``prefix_origins``
  (``'ns_origin'``).
* ``nset_rve_edges``: bool, default False; the set is named
  ``prefix_rve_edges`` (``'ns_rve_edge'``).
"""
import os
import re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from .mesh_exporter_3d import (
    MeshExporter3D, DEFAULT_ELSET_PREFIXES, FACE_NSET_NAMES)
from .mesh_rescaling import mesh_state_for_scale
from .voxel_interface_masks import (
    grain_boundary_voxels, junction_line_voxels, junction_point_voxels)

__all__ = ['EXPORT_LEVELS', 'ELEMENT_TYPES', 'ConfigurableMeshExporter3D',
           'export_deck', 'find_isolated_grain_orientations',
           'next_instance_directory', 'as_bool', 'parse_integer_list',
           'parse_positive_int']

EXPORT_LEVELS = ('auto', 'subblock', 'block', 'packet', 'pag')
ELEMENT_TYPES = ('C3D8', 'C3D4', 'C3D20', 'C3D10')

_FACE_OPTION_KEYS = {'XMIN': ('nset_x_min', 'prefix_x_min'),
                     'XMAX': ('nset_x_max', 'prefix_x_max'),
                     'YMIN': ('nset_y_min', 'prefix_y_min'),
                     'YMAX': ('nset_y_max', 'prefix_y_max'),
                     'ZMIN': ('nset_z_min', 'prefix_z_min'),
                     'ZMAX': ('nset_z_max', 'prefix_z_max')}

_FEATURE_SETS = (('gb', 'Grain boundary', 'es_gb_', grain_boundary_voxels),
                 ('jl', 'Junction line', 'es_jl_', junction_line_voxels),
                 ('jp', 'Junction point', 'es_jp_', junction_point_voxels))


def as_bool(value, default=False):
    """Read a boolean that may have been stored as text ('true', 'off', ...)."""
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {'true', '1', 'yes', 'on'}:
            return True
        if text in {'false', '0', 'no', 'off', ''}:
            return False
        return bool(default)
    return bool(value)


def parse_integer_list(text) -> List[int]:
    """Integers from comma-separated text; blank entries are skipped.

    Raises ValueError if an entry is not an integer.
    """
    return [int(x.strip()) for x in str(text).split(',') if x.strip()]


def parse_positive_int(value) -> Optional[int]:
    """A positive integer from a number or text, or None.

    None, blank text, zero and negative values give None. Text that is not an
    integer raises ValueError.
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    number = int(text)
    return number if number > 0 else None


def next_instance_directory(base_dir) -> str:
    """Create and return ``base_dir/instanceN``, N one above the highest existing."""
    base_dir = str(base_dir)
    max_n = 0
    if os.path.isdir(base_dir):
        for name in os.listdir(base_dir):
            m = re.match(r'^instance(\d+)$', name, re.IGNORECASE)
            if m and os.path.isdir(os.path.join(base_dir, name)):
                max_n = max(max_n, int(m.group(1)))
    inst_path = os.path.join(base_dir, f"instance{max_n + 1}")
    os.makedirs(inst_path, exist_ok=True)
    return inst_path


def find_isolated_grain_orientations(fm_state) -> dict:
    """``isolated_grain_orientations`` of the state, found by following its
    ``_parent`` chain; an empty dict if none is found."""
    obj = fm_state
    while obj is not None:
        oris = getattr(obj, 'isolated_grain_orientations', None)
        if oris is not None:
            return oris
        obj = getattr(obj, '_parent', None)
    return {}


class ConfigurableMeshExporter3D(MeshExporter3D):
    """``MeshExporter3D`` with configurable set names, optional sets and level.

    Parameters
    ----------
    options : dict, optional
        See the module docstring.
    flat_output : bool
        If True the files are written straight into the output base
        directory and ``folder_name`` is not used to make a subfolder.
    **kwargs
        Passed to ``MeshExporter3D`` (``verbosity``, ``phase_id``,
        ``mat_data_flag``).
    """

    __slots__ = ('options', 'flat_output')

    def __init__(self, options: Optional[dict] = None, *, flat_output: bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        self.options: dict = dict(options or {})
        self.flat_output: bool = bool(flat_output)
        level = self.options.get('export_level', 'auto')
        if level not in EXPORT_LEVELS:
            raise ValueError(f"export_level must be one of {EXPORT_LEVELS}, got {level!r}.")

    # ── option access ─────────────────────────────────────────────────────────

    def _flag(self, key: str, default: bool) -> bool:
        return as_bool(self.options.get(key, default), default)

    def _text(self, key: str, default: str) -> str:
        return str(self.options.get(key, default) or default)

    def _elset_prefix(self, kind: str) -> str:
        return self._text(f'prefix_{kind}', DEFAULT_ELSET_PREFIXES[kind])

    def _elset_enabled(self, kind: str) -> bool:
        return self._flag(f'elset_{kind}', True)

    # ── output directory, level, boundary sets, master file ──────────────────

    def _resolve_output_dir(self, folder_name: str):
        if not self.flat_output:
            return super()._resolve_output_dir(folder_name)
        self._output_base.mkdir(parents=True, exist_ok=True)
        return self._output_base

    def _boundary_nset_config(self):
        required = ({'XMIN', 'YMIN', 'ZMIN', f'{self.load_axis.upper()}MAX'}
                    if self.write_step else set())
        return {face: {
            'enabled': self._flag(enabled_key, True) or face in required,
            'prefix': self._text(prefix_key, FACE_NSET_NAMES[face]),
        } for face, (enabled_key, prefix_key) in _FACE_OPTION_KEYS.items()}

    def _extra_node_sets(self, NX: int, NY: int, NZ: int,
                         node_mask: np.ndarray) -> List[Tuple[str, List[int]]]:
        stride_i = (NY + 1) * (NZ + 1)
        sets: List[Tuple[str, List[int]]] = []

        if self._flag('nset_origins', False):
            sets.append((self._text('prefix_origins', 'ns_origin'), [1]))

        if self._flag('nset_rve_edges', False):
            # A node lies on an edge of the domain when it is on two or more of
            # the three pairs of opposite faces.
            on_i = np.zeros(NX + 1, dtype=int)
            on_i[[0, NX]] = 1
            on_j = np.zeros(NY + 1, dtype=int)
            on_j[[0, NY]] = 1
            on_k = np.zeros(NZ + 1, dtype=int)
            on_k[[0, NZ]] = 1
            faces_touched = on_i[:, None, None] + on_j[None, :, None] + on_k[None, None, :]
            i, j, k = np.nonzero((faces_touched >= 2) & node_mask)
            ids = (i * stride_i + j * (NZ + 1) + k + 1).tolist()
            sets.append((self._text('prefix_rve_edges', 'ns_rve_edge'), ids))

        return sets

    def _part_include_files(self, has_subblk: bool) -> List[str]:
        names = ['01_nodes.inp', '02_elements.inp']
        if self._elset_enabled('pag'):
            names.append('03a_elsets_pag.inp')
        if self._elset_enabled('pck'):
            names.append('03b_elsets_pck.inp')
        names.append('03c_elsets_blk.inp')
        if has_subblk and self._elset_enabled('subblk'):
            names.append('03d_elsets_subblk.inp')
        names.append('06_sections.inp')
        return names

    def _detect_lowest_level(self, fm_state) -> str:
        """The requested ``export_level`` where the stored state supports it.

        The state can be older than the request, e.g. generated at block level
        and exported with a sub-block request. 'subblock' and 'block' are then
        honoured only when the state holds that level; otherwise the level is
        detected from the state instead of failing inside the elset writer.
        """
        requested = self.options.get('export_level', 'auto')
        if requested == 'subblock' and getattr(fm_state, 'all_subblocks', None):
            return 'subblock'
        if requested == 'block' and getattr(fm_state, 'all_blocks', None):
            return 'block'
        if requested in ('packet', 'pag'):
            return requested
        detected = super()._detect_lowest_level(fm_state)
        if requested in ('subblock', 'block'):
            print(f"Note: export level {requested!r} is not available on the stored "
                  f"structure (generate that level first) -- exporting at "
                  f"'{detected}' level instead.")
        return detected

    # ── isolated grains and voxel-feature element sets ───────────────────────

    def _write_elsets_isolated(self, f, fm_state, NY: int, NZ: int,
                               active_mask: np.ndarray, em: int) -> set:
        """Isolated-grain elsets (every isolated grain, not only those without a
        retained-austenite PAG), then the optional grain-boundary,
        junction-line and junction-point elsets."""
        written: set = set()
        if self._elset_enabled('iso'):
            isolated = getattr(fm_state, 'isolated_grains', set())
            prefix = self._elset_prefix('iso')
            if isolated:
                f.write("** Isolated grain elsets\n")
                for gid in sorted(isolated):
                    vox = fm_state.grain_locs.get(gid)
                    if vox is not None and len(vox):
                        eids = self._vox_to_hex_eids(vox, NY, NZ, active_mask)
                        if eids.size == 0:
                            continue
                        self._write_elset_ids(f, f"{prefix}{gid}", eids, em)
                        written.add(gid)

        for kind, title, default_prefix, mask_of in _FEATURE_SETS:
            if not self._flag(f'elset_{kind}', False):
                continue
            f.write(f"** {title} elsets\n")
            prefix = self._text(f'prefix_{kind}', default_prefix)
            ii, jj, kk = np.where(mask_of(fm_state.lgi) & active_mask)
            if ii.size > 0:
                hex_eids = (ii * NY * NZ + jj * NZ + kk + 1).astype(np.int64)
                self._write_elset_ids(f, f"{prefix}all", hex_eids, em)

        return written

    # ── packet level ──────────────────────────────────────────────────────────

    @staticmethod
    def _packets_of(fm_state) -> list:
        g2pag = fm_state.grain_to_pag_id
        g2pkt = fm_state.grain_to_local_pkt_idx
        unique_packets = set()
        for gid in fm_state.grain_locs.keys():
            if gid in g2pag and gid in g2pkt:
                unique_packets.add((g2pag[gid], g2pkt[gid]))
        return sorted(unique_packets)

    def _write_materials(self, f, fm_state, level: str,
                         block_id_map: dict, sb_id_map: dict,
                         global_id_map: dict, phase_id: int, mat_data_flag: int,
                         written_pags: set, written_blocks: set,
                         written_subblocks: set, written_iso: set) -> None:
        """The 'packet' level is not known to the base class, so it is written
        here; every other level is delegated to the base class."""
        if level != "packet":
            super()._write_materials(f, fm_state, level, block_id_map, sb_id_map,
                                     global_id_map, phase_id, mat_data_flag,
                                     written_pags, written_blocks, written_subblocks, written_iso)
            return

        f.write(
            "** *User Material stubs with Bunge-Euler angles (phi1, Phi, phi2, degrees).\n"
            "** constants=6: phi1, Phi, phi2 [Bunge-Euler deg],"
            " grain_id [global elset ID], phase_id, mat_data_flag.\n"
            "** Replace with your CPFEM constitutive definition as needed.\n**\n"
        )

        def _mat(mat_name: str, phi1: float, PHI: float, phi2: float, grain_id: int, extra_comment: str = "") -> None:
            f.write(f"*Material, name={mat_name}\n")
            if extra_comment:
                f.write(f"** {extra_comment}\n")
            f.write(
                f"** Bunge-Euler (deg): phi1={phi1:.4f}, Phi={PHI:.4f}, phi2={phi2:.4f}\n"
                f"*User Material, constants=6\n"
                f"{phi1:.6f}, {PHI:.6f}, {phi2:.6f},"
                f" {grain_id}, {phase_id}, {mat_data_flag}\n"
                f"*Depvar\n100\n**\n"
            )

        for pag_id, pkt_idx in self._packets_of(fm_state):
            ori = fm_state.pag_orientations.get(pag_id, (0.0, 0.0, 0.0))
            gid_val = global_id_map.get(("pck", pag_id, pkt_idx), -1)
            _mat(f"MAT_PCK_{pag_id}_{pkt_idx}", *ori, grain_id=gid_val)

        iso_oris = find_isolated_grain_orientations(fm_state)
        for gid in sorted(getattr(fm_state, "isolated_grains", set())):
            if gid not in written_iso:
                continue
            vox = fm_state.grain_locs.get(gid)
            if vox is None or len(vox) == 0:
                continue
            gid_val = global_id_map.get(("iso", gid), -1)
            ori_iso = iso_oris.get(gid, (0.0, 0.0, 0.0))
            comment = (f"Isolated grain {gid}: FCC orientation assigned." if gid in iso_oris
                       else f"Isolated grain {gid}: no orientation assigned — placeholder (0,0,0).")
            _mat(f"MAT_ISO_{gid}", *ori_iso, grain_id=gid_val, extra_comment=comment)

    def _write_sections(self, f, fm_state, level: str,
                        block_id_map: dict, sb_id_map: dict,
                        written_pags: set, written_blocks: set,
                        written_subblocks: set, written_iso: set) -> None:
        if level != "packet":
            super()._write_sections(f, fm_state, level, block_id_map, sb_id_map,
                                    written_pags, written_blocks, written_subblocks, written_iso)
            return

        f.write("** *Solid Section — lowest hierarchy level only.\n**\n")
        p_pck = self._elset_prefix('pck')
        p_iso = self._elset_prefix('iso')

        for pag_id, pkt_idx in self._packets_of(fm_state):
            eset = f"{p_pck}{pag_id}_{pkt_idx}"
            f.write(f"*Solid Section, elset={eset}, material=MAT_PCK_{pag_id}_{pkt_idx}\n,\n")

        for gid in sorted(getattr(fm_state, "isolated_grains", set())):
            if gid not in written_iso:
                continue
            f.write(f"*Solid Section, elset={p_iso}{gid}, material=MAT_ISO_{gid}\n,\n")


def export_deck(fm_state, element_type: str, output_directory, *,
                folder_name: str = 'model', output_unit: Optional[str] = None,
                scale_factor: float = 1.0, max_voxels: Optional[int] = None,
                force: bool = False, custom_message: str = '',
                ignore_ids=(), analysis: Optional[dict] = None,
                options: Optional[dict] = None, verbosity: int = 1,
                flat_output: bool = True) -> Path:
    """Export ``fm_state`` as an Abaqus deck and return the directory written.

    Parameters
    ----------
    fm_state :
        FM steel pipeline state. It is not modified; with ``scale_factor``
        other than 1 the deck is made from a rescaled copy of the grid.
    element_type : {'C3D8', 'C3D4', 'C3D20', 'C3D10'}
    output_directory :
        Existing directory the files are written to (with ``flat_output``) or
        in which ``<folder_name>N`` is created.
    output_unit : {'microns', 'mm', 'm'}, optional
        Unit of the node coordinates; the unit of the state if not given.
    scale_factor : float
        Voxel-count scale factor of the exported grid (> 0).
    max_voxels : int, optional
        Overrides the hard voxel limit of the element type.
    force : bool
        Skip the interactive confirmation above the hard limit.
    ignore_ids : iterable of int
        Grain ids exported as void.
    analysis : dict, optional
        Keyword arguments of ``MeshExporter3D.configure_analysis``
        (``write_step`` may be given as text).
    options : dict, optional
        See the module docstring.
    """
    if element_type not in ELEMENT_TYPES:
        raise ValueError(f"element_type must be one of {ELEMENT_TYPES}, got {element_type!r}.")

    export_state = mesh_state_for_scale(fm_state, scale_factor)

    exporter = ConfigurableMeshExporter3D(options, flat_output=flat_output,
                                          verbosity=verbosity)
    exporter.set_output_base_path(str(output_directory))
    if output_unit is not None:
        exporter.set_output_units(output_unit)
    if analysis:
        settings = dict(analysis)
        if 'write_step' in settings:
            settings['write_step'] = as_bool(settings['write_step'], True)
        exporter.configure_analysis(**settings)
    ids = [int(i) for i in ignore_ids]
    if ids:
        exporter.ignore_lfi_ids(ids)

    export = {'C3D8': exporter.export_c3d8, 'C3D4': exporter.export_c3d4,
              'C3D20': exporter.export_c3d20, 'C3D10': exporter.export_c3d10}[element_type]
    return export(export_state, folder_name, output_unit=output_unit,
                  max_voxels=max_voxels, force=force, custom_message=custom_message)
