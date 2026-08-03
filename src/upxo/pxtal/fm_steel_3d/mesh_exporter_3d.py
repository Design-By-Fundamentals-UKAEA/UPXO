"""
MeshExporter3D — Abaqus FEM mesh export for FM steel 3-D microstructures.

Supported element types
-----------------------
C3D8   Linear hexahedral (1 element / voxel)
C3D4   Linear tetrahedral, Kuhn decomposition (6 elements / voxel)
C3D20  Quadratic hexahedral (1 element / voxel, 20 nodes: 8 corners + 12 mid-edge)
C3D10  Quadratic tetrahedral (6 elements / voxel, 10 nodes each: 4 corners + 6 mid-edge)

Output structure
----------------
<output_base>/<folder_name>1/
    model_master.inp        — heading + *INCLUDE pointers
    01_nodes.inp
    02_elements.inp
    03a_elsets_pag.inp
    03b_elsets_pck.inp
    03c_elsets_blk.inp
    03d_elsets_subblk.inp   — omitted when no sub-blocks present
    04_nsets_bc.inp         — ns.x-, ns.x+, ns.y-, ns.y+, ns.z-, ns.z+
    05_materials.inp        — *User Material stubs with Bunge-Euler angles
    06_sections.inp         — *Solid Section (lowest hierarchy level only)
    07_interactions.inp     — stub
    08_steps_output.inp     — stub

Elset naming
------------
es_pag_{pag_id}
es_pck_{pag_id}_{local_pkt_idx}
es_blk_{pag_id}_{local_pkt_idx}_{blk_local}
es_subblk_{pag_id}_{local_pkt_idx}_{blk_local}_{sb_local}
es_iso_{grain_id}

Global elset ID (A in *User Material)
--------------------------------------
Every elset is assigned a unique 0-based integer in write order:
  PAG elsets → packet elsets → block elsets → [sub-block elsets]
  → [isolated grain elsets (sorted by grain ID, if any exist)]
At block level  A ∈ {n+m … n+m+p−1} ∪ {n+m+p … n+m+p+k−1}
At sub-block level  A ∈ {n+m+p … n+m+p+q−1} ∪ {n+m+p+q … n+m+p+q+k−1}
(n=PAGs, m=packets, p=blocks, q=sub-blocks, k=isolated grains)

Node numbering (all element types share the same corner nodes)
--------------
node_id(i,j,k) = i*(NY+1)*(NZ+1) + j*(NZ+1) + k + 1    (1-based)

Element numbering
-----------------
C3D8 / C3D20:  elem_id(i,j,k) = i*NY*NZ + j*NZ + k + 1  (1-based, one per voxel)
C3D4 / C3D10:  elem_id = 6*(i*NY*NZ + j*NZ + k) + tet_local + 1  (tet_local in 0..5)
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple

from .phases_3d import PHASE_MARTENSITE, PHASE_RETAINED_AUSTENITE, PHASE_NAMES


# ── Module-level constants ────────────────────────────────────────────────────

DEFAULT_LIMITS: Dict[str, Dict[str, int]] = {
    'C3D8':  {'max_voxels': 5_000_000, 'warn_voxels': 2_000_000},
    'C3D4':  {'max_voxels': 2_000_000, 'warn_voxels': 1_000_000},
    'C3D20': {'max_voxels': 1_500_000, 'warn_voxels':   750_000},
    'C3D10': {'max_voxels': 1_000_000, 'warn_voxels':   500_000},
}

# Try to overlay mutable limits from JSON assets; fall back silently to hardcoded values.
_LIMITS_JSON = Path(__file__).parent / 'assets' / 'mesh_limits.json'
try:
    import json as _json
    with open(_LIMITS_JSON) as _f:
        _loaded = _json.load(_f).get('DEFAULT_LIMITS', {})
    for _etype, _vals in _loaded.items():
        if _etype in DEFAULT_LIMITS and isinstance(_vals, dict):
            DEFAULT_LIMITS[_etype].update(
                {k: int(v) for k, v in _vals.items() if isinstance(v, (int, float))}
            )
    del _json, _f, _loaded, _etype, _vals
except Exception:
    pass
finally:
    del _LIMITS_JSON

_CONFIRM_PHRASES: Set[str] = {
    'i accept large mesh', 'i understand the file will be large',
    'just do it', "i don't care", 'go for it',
    'full martensite ahead', 'steel yourself', 'forge ahead',
    'quench my thirst', 'temper tantrum',
    'kuru',    # "do" — Sanskrit imperative
}

_CANCEL_PHRASES: Set[str] = {
    'abort',
    'tyaja',   # "abandon/renounce" — Sanskrit
}

_BYTES_PER_ELEM: Dict[str, int] = {
    'C3D8': 90, 'C3D4': 60, 'C3D20': 280, 'C3D10': 170,
}
_BYTES_PER_NODE = 50
_TETS_PER_VOX = 6

# Package root: mesh_exporter_3d.py is at src/upxo/pxtal/fm_steel_3d/
# Four parents up → src/  Five parents up → project root
_PACKAGE_ROOT = Path(__file__).parents[4]
_DEFAULT_OUTPUT_BASE = _PACKAGE_ROOT / 'data' / 'ABQInputFiles'

_UPXO_HEADER = (
    "** Made with UPXO: UKAEA Poly-XTAL Operations.\n"
    "** Visit: https://github.com/Design-By-Fundamentals-UKAEA/UPXO\n"
    "** or: pip install upxo\n"
    "**\n"
)

# Kuhn decomposition — 6 tets sharing the body diagonal (i,j,k)→(i+1,j+1,k+1).
# Each row: (n1,n2,n3,n4) as (di,dj,dk) offsets from voxel min-corner.
# Node ordering satisfies positive Abaqus C3D4 Jacobian:
#   face 1-2-3 outward normal points AWAY from node 4 (apex).
_KUHN_TETS: Tuple = (
    ((0, 0, 0), (1, 1, 0), (1, 0, 0), (1, 1, 1)),  # base at z=k
    ((0, 0, 0), (1, 0, 0), (1, 0, 1), (1, 1, 1)),  # base at y=j
    ((0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 1, 1)),  # base at z=k, y-side
    ((0, 0, 0), (0, 1, 1), (0, 1, 0), (1, 1, 1)),  # base at x=i
    ((0, 0, 0), (1, 0, 1), (0, 0, 1), (1, 1, 1)),  # base at y=j, z-side
    ((0, 0, 0), (0, 0, 1), (0, 1, 1), (1, 1, 1)),  # base at x=i, yz-side
)


class MeshExporter3D:
    """
    Worker for generating Abaqus FEM meshes from FM steel 3-D microstructures.

    Usage example
    -------------
    exporter = MeshExporter3D(verbosity=1)
    exporter.set_output_units('mm')
    exporter.set_output_base_path(r'C:/MyProject/FEMInputs')
    exporter.ignore_lfi_ids([0])
    out_dir = exporter.export_c3d8(fm_state, 'run_001',
                                   custom_message='Made by Dr. A. Smith, Univ. X')
    """

    __slots__ = ('_verbosity', '_limits', '_output_unit', '_output_base',
                 '_ignore_ids', '_phase_id', '_mat_data_flag')

    def __init__(self, verbosity: int = 0,
                 phase_id: int = PHASE_MARTENSITE, mat_data_flag: int = 0):
        self._verbosity: int = int(verbosity)
        self._limits: Dict = {k: dict(v) for k, v in DEFAULT_LIMITS.items()}
        self._output_unit: Optional[str] = None
        self._output_base: Path = _DEFAULT_OUTPUT_BASE
        self._ignore_ids: Set[int] = set()
        if int(phase_id) != PHASE_MARTENSITE:
            raise ValueError(
                f"phase_id={phase_id} is not supported. "
                f"Only phase_id={PHASE_MARTENSITE} (martensite/ferrite, the "
                "transformed matrix) is a configurable value here; retained "
                f"austenite is auto-tagged phase_id={PHASE_RETAINED_AUSTENITE} "
                "wherever a retained-austenite PAG is present (see "
                "retained_austenite_pag_ids on FMSteel3DWithPAGs) and is not "
                "user-overridable. Other phases (e.g. delta-ferrite, bainite) "
                "are not yet supported."
            )
        self._phase_id: int = int(phase_id)
        if int(mat_data_flag) not in (0, 1):
            raise ValueError(
                f"mat_data_flag={mat_data_flag} is invalid. Must be 0 or 1."
            )
        self._mat_data_flag: int = int(mat_data_flag)

    # ── Configuration ─────────────────────────────────────────────────────────

    def set_output_units(self, unit: str) -> None:
        """Set physical unit for node coordinates written to .inp files."""
        if unit not in {'microns', 'mm', 'm'}:
            raise ValueError(
                f"Invalid unit '{unit}'. Valid: {{'microns', 'mm', 'm'}}. "
                "State unchanged."
            )
        self._output_unit = unit
        if self._verbosity > 0:
            print(f"Output unit set to: '{unit}'")

    def set_voxel_limit(self, elem_type: str, new_max: int,
                        new_warn: Optional[int] = None) -> None:
        """Override hard voxel limit (and optionally warn threshold) for one element type."""
        if elem_type not in self._limits:
            raise ValueError(
                f"Unknown element type '{elem_type}'. Valid: {list(self._limits)}"
            )
        old = self._limits[elem_type]['max_voxels']
        self._limits[elem_type]['max_voxels'] = int(new_max)
        if new_warn is not None:
            self._limits[elem_type]['warn_voxels'] = int(new_warn)
        if self._verbosity > 0:
            print(f"{elem_type} voxel limit: {old:,} -> {new_max:,}")

    def set_output_base_path(self, path: str) -> None:
        """Override default output base directory. The path must already exist."""
        p = Path(path)
        if not p.exists():
            raise ValueError(f"Output base path does not exist: {p}")
        self._output_base = p
        if self._verbosity > 0:
            print(f"Output base path: {p}")

    def ignore_lfi_ids(self, ids: List[int]) -> None:
        """
        Register feature IDs to treat as voids before meshing.
        IDs <= 0 are always void regardless of this list.
        """
        if not all(isinstance(i, int) for i in ids):
            raise TypeError("All IDs must be plain Python int.")
        negatives = [i for i in ids if i < 0]
        if negatives:
            print(f"Note: {negatives} are already void by default. Registered anyway.")
        self._ignore_ids.update(ids)
        if self._verbosity > 0:
            print(f"Void IDs (cumulative): {sorted(self._ignore_ids)}")

    # ── Pre-mesh utilities ────────────────────────────────────────────────────

    def _build_active_mask(self, lfi: np.ndarray) -> np.ndarray:
        """(NX,NY,NZ) bool — True for voxels that become elements."""
        mask = lfi > 0
        if self._ignore_ids:
            mask &= ~np.isin(lfi, list(self._ignore_ids))
        return mask

    def _build_active_node_mask(self, active_mask: np.ndarray,
                                NX: int, NY: int, NZ: int) -> np.ndarray:
        """
        (NX+1,NY+1,NZ+1) bool — True for every corner node that borders
        at least one active voxel. Derived from active_mask so no orphan
        nodes can appear in the written output.
        """
        node_mask = np.zeros((NX + 1, NY + 1, NZ + 1), dtype=bool)
        ii, jj, kk = np.where(active_mask)
        for di in (0, 1):
            for dj in (0, 1):
                for dk in (0, 1):
                    node_mask[ii + di, jj + dj, kk + dk] = True
        return node_mask

    @staticmethod
    def _unit_scale(src: str, dst: str) -> float:
        _scales = {
            ('microns', 'microns'): 1.0,  ('microns', 'mm'): 1e-3, ('microns', 'm'): 1e-6,
            ('mm', 'microns'): 1e3,        ('mm', 'mm'): 1.0,       ('mm', 'm'): 1e-3,
            ('m', 'microns'): 1e6,         ('m', 'mm'): 1e3,        ('m', 'm'): 1.0,
        }
        return _scales[(src, dst)]

    def _estimate_file_mb(self, n_vox: int, elem_type: str) -> float:
        """Rough .inp file-size estimate in MB (assumes cubic domain)."""
        cbrt = round(n_vox ** (1 / 3))
        NX = NY = NZ = cbrt
        n_nodes = (NX + 1) * (NY + 1) * (NZ + 1)
        if elem_type in ('C3D20', 'C3D10'):
            n_nodes += (NX * (NY+1) * (NZ+1) + (NX+1) * NY * (NZ+1)
                        + (NX+1) * (NY+1) * NZ)
        if elem_type == 'C3D10':
            n_nodes += NX*NY*(NZ+1) + (NX+1)*NY*NZ + NX*(NY+1)*NZ + NX*NY*NZ
        n_elem = n_vox * (_TETS_PER_VOX if elem_type in ('C3D4', 'C3D10') else 1)
        return (n_nodes * _BYTES_PER_NODE + n_elem * _BYTES_PER_ELEM[elem_type]) / 1e6

    def _resolve_output_dir(self, folder_name: str) -> Path:
        """
        Create <output_base>/<folder_name>N/ where N is the lowest available suffix.
        First attempt uses N=1; increments if the folder already exists.
        """
        base = self._output_base
        base.mkdir(parents=True, exist_ok=True)
        idx = 1
        candidate = base / f"{folder_name}{idx}"
        while candidate.exists():
            idx += 1
            candidate = base / f"{folder_name}{idx}"
        candidate.mkdir(parents=True)
        print(f"Output directory created: {candidate}")
        return candidate

    def check_voxel_threshold(self, n_vox: int, elem_type: str,
                              max_voxels_override: Optional[int] = None) -> Dict:
        """Pure (no I/O, no prompting) size/threshold check -- the computation
        `_preflight_check` uses internally, exposed as its own public method so
        a caller that cannot answer an interactive `input()` prompt (this is
        exactly the problem the GUI hit: `_preflight_check`'s hard-limit branch
        blocks on `input()` with no attached stdin when run from a background
        thread, permanently wedging the app-wide busy-lock) can decide up front
        whether to ask its own confirmation dialog, then call export with
        force=True to bypass `_preflight_check`'s interactive prompt entirely.

        Returns
        -------
        dict with keys: exceeds_hard, exceeds_warn, hard, warn, est_mb
        """
        limits = self._limits[elem_type]
        hard = int(max_voxels_override) if max_voxels_override else limits['max_voxels']
        warn = limits['warn_voxels']
        return {
            'exceeds_hard': n_vox >= hard,
            'exceeds_warn': n_vox >= warn,
            'hard': hard,
            'warn': warn,
            'est_mb': self._estimate_file_mb(n_vox, elem_type),
        }

    def _preflight_check(self, n_vox: int, elem_type: str,
                         max_voxels_override: Optional[int],
                         force: bool) -> None:
        """
        Three-tier consent system:
          < warn_voxels   → silent proceed
          < hard limit    → print warning, proceed
          >= hard limit   → require typed consent (3 attempts)
        """
        limits = self._limits[elem_type]
        hard = int(max_voxels_override) if max_voxels_override else limits['max_voxels']
        warn = limits['warn_voxels']
        est_mb = self._estimate_file_mb(n_vox, elem_type)

        if n_vox < warn:
            return

        if n_vox < hard:
            print(f"WARNING: {elem_type} on {n_vox:,} voxels (~{est_mb:.0f} MB estimated).")
            return

        # Hard limit reached — require explicit consent
        print(f"\n  Element type        : {elem_type}")
        print(f"  Voxel count         : {n_vox:,}  (hard limit: {hard:,})")
        print(f"  Estimated file size : ~{est_mb:.0f} MB\n")
        print("  Type one of the accepted phrases to proceed:")
        print('    "i accept large mesh"  |  "just do it"  |  "i don\'t care"')
        print('    "go for it"  |  "full martensite ahead"  |  "steel yourself"')
        print('    "forge ahead"  |  "quench my thirst"  |  "temper tantrum"')
        print('    "kuru"  (do — Sanskrit)')
        print('  Type "tyaja" (renounce) or "abort" to cancel.\n')

        if force:
            print("  force=True — bypassing consent.")
            return

        for attempt in range(1, 4):
            try:
                response = input(f"  [{attempt}/3] > ").strip().lower()
            except KeyboardInterrupt:
                print()
                raise RuntimeError("Mesh export cancelled (Ctrl+C).")
            if response in _CONFIRM_PHRASES:
                return
            if response in _CANCEL_PHRASES:
                raise RuntimeError("Mesh export cancelled (tyaja/abort).")
            remaining = 3 - attempt
            if remaining:
                print(f'  "{response}" not recognised — {remaining} attempt(s) left.')
        raise RuntimeError(
            "Mesh export aborted: confirmation not received after 3 attempts."
        )

    # ── ID / map helpers ──────────────────────────────────────────────────────

    def _build_block_id_map(self, fm_state) -> Dict:
        """Map block_id → (pag_id, local_pkt_idx, blk_local) for canonical names."""
        result: Dict = {}
        if not hasattr(fm_state, 'grain_to_blocks_map'):
            return result
        g2pag = fm_state.grain_to_pag_id
        g2pkt = fm_state.grain_to_local_pkt_idx
        for grain_id, blk_list in fm_state.grain_to_blocks_map.items():
            pag_id = g2pag.get(grain_id)
            pkt_idx = g2pkt.get(grain_id)
            if pag_id is None:
                continue
            for blk_local, block_id in enumerate(blk_list, start=1):
                result[block_id] = (pag_id, pkt_idx, blk_local)
        return result

    def _build_subblock_id_map(self, fm_state, block_id_map: Dict) -> Dict:
        """Map subblock_id → (pag_id, pkt_idx, blk_local, sb_local)."""
        result: Dict = {}
        if not hasattr(fm_state, 'block_to_subblocks_map'):
            return result
        for block_id, (pag_id, pkt_idx, blk_local) in block_id_map.items():
            for sb_local, sb_id in enumerate(
                    fm_state.block_to_subblocks_map.get(block_id, []), start=1):
                result[sb_id] = (pag_id, pkt_idx, blk_local, sb_local)
        return result

    def _detect_lowest_level(self, fm_state) -> str:
        """
        Determine which hierarchy level has orientations to drive materials/sections.
        Sub-block and block levels require their orientation dicts to be present.
        Falls back to 'pag' when orientations have not yet been assigned.
        """
        if (hasattr(fm_state, 'all_subblocks') and fm_state.all_subblocks
                and hasattr(fm_state, 'subblock_orientations')):
            return 'subblock'
        if (hasattr(fm_state, 'all_blocks') and fm_state.all_blocks
                and hasattr(fm_state, 'block_orientations')):
            return 'block'
        return 'pag'

    @staticmethod
    def _leftover_isolated_grains(fm_state) -> set:
        """isolated_grains not covered by a tracked retained_austenite_pag_ids
        entry.

        A grain covered by a tracked retained-austenite PAG gets a proper
        PAG-level *Elset/*Material/*Solid Section (es_pag_*/MAT_PAG_*, with
        the PAG's own correctly-assigned FCC orientation -- see
        _write_materials); the legacy per-grain es_iso_*/MAT_ISO_* path is
        only needed for grains that have no such PAG identity to fall back
        on (FMSteel3DBase.generate_pag_clusters's pre-filter path, which
        excludes grains before clustering ever forms a PAG around them).
        """
        isolated = getattr(fm_state, 'isolated_grains', set())
        if not isolated:
            return set()
        retained_pag_ids = getattr(fm_state, 'retained_austenite_pag_ids', set())
        if not retained_pag_ids:
            return set(isolated)
        clusters_dict = getattr(fm_state, 'clusters_dict', {})
        covered: set = set()
        for pid in retained_pag_ids:
            covered.update(clusters_dict.get(pid, []))
        return isolated - covered

    def _count_phase_active_elements(self, fm_state, active_mask: np.ndarray,
                                     NY: int, NZ: int, em: int, n_active: int) -> Dict[int, int]:
        """Active element count per phase (see phases_3d.py), for the
        *Heading* summary. Retained austenite = active elements belonging to
        a tracked retained_austenite_pag_ids PAG, plus any leftover
        (legacy, untracked) isolated grains; martensite = everything else.
        """
        retained_ids = getattr(fm_state, 'retained_austenite_pag_ids', set())
        retained_grains: set = set()
        clusters_dict = getattr(fm_state, 'clusters_dict', {})
        for pid in retained_ids:
            retained_grains.update(clusters_dict.get(pid, []))
        retained_grains |= self._leftover_isolated_grains(fm_state)

        retained_active = 0
        grain_locs = getattr(fm_state, 'grain_locs', {})
        for gid in retained_grains:
            vox = grain_locs.get(gid)
            if vox is not None and len(vox):
                retained_active += int(self._vox_to_hex_eids(vox, NY, NZ, active_mask).size)
        retained_active *= em

        return {
            PHASE_RETAINED_AUSTENITE: retained_active,
            PHASE_MARTENSITE: max(0, n_active - retained_active),
        }

    def _build_global_id_map(self, fm_state, blk_map: Dict, sb_map: Dict) -> Dict:
        """
        Assign a unique 0-based global integer ID to every elset, in write order:
          PAG elsets → packet elsets → block elsets → [sub-block elsets]
          → [isolated grain elsets, sorted by grain ID]

        Returns
        -------
        dict  keys: ('pag', pag_id) | ('pck', pag_id, pkt_idx) |
                    ('blk', block_id) | ('sb', sb_id) | ('iso', gid)
              values: int global ID
        """
        gmap: Dict = {}
        counter = 0

        for pag_id in fm_state.clusters_dict:
            gmap[('pag', pag_id)] = counter
            counter += 1

        g2pag = getattr(fm_state, 'grain_to_pag_id', {})
        g2pkt = getattr(fm_state, 'grain_to_local_pkt_idx', {})
        seen_pck: set = set()
        for gid in fm_state.grain_locs:
            if gid not in g2pag:
                continue
            key = (g2pag[gid], g2pkt[gid])
            if key not in seen_pck:
                seen_pck.add(key)
                gmap[('pck', key[0], key[1])] = counter
                counter += 1

        for block_id in blk_map:
            gmap[('blk', block_id)] = counter
            counter += 1

        for sb_id in sb_map:
            gmap[('sb', sb_id)] = counter
            counter += 1

        isolated = self._leftover_isolated_grains(fm_state)
        for gid in sorted(isolated):
            vox = fm_state.grain_locs.get(gid)
            if vox is not None and len(vox):
                gmap[('iso', gid)] = counter
                counter += 1

        return gmap

    def _vox_to_hex_eids(self, vox_coords: np.ndarray,
                         NY: int, NZ: int,
                         active_mask: np.ndarray) -> np.ndarray:
        """Compute 1-based hex element IDs from (n,3) voxel coordinate array."""
        ii, jj, kk = vox_coords[:, 0], vox_coords[:, 1], vox_coords[:, 2]
        valid = active_mask[ii, jj, kk]
        ii, jj, kk = ii[valid], jj[valid], kk[valid]
        return (ii * NY * NZ + jj * NZ + kk + 1).astype(np.int64)

    # ── Node writer ───────────────────────────────────────────────────────────

    def _write_nodes(self, f, NX: int, NY: int, NZ: int,
                     dx: float, node_mask: np.ndarray) -> None:
        """Write *Node section, streaming one i-slab at a time."""
        f.write("*Node\n")
        stride_i = (NY + 1) * (NZ + 1)
        buf: List[str] = []
        for i in range(NX + 1):
            js, ks = np.where(node_mask[i])
            if js.size == 0:
                continue
            for j_v, k_v in zip(js.tolist(), ks.tolist()):
                nid = i * stride_i + j_v * (NZ + 1) + k_v + 1
                buf.append(
                    f"{nid:>10d}, {i*dx:>14.6e}, {j_v*dx:>14.6e}, {k_v*dx:>14.6e}\n"
                )
            if len(buf) >= 8192:
                f.writelines(buf)
                buf.clear()
        if buf:
            f.writelines(buf)

    # ── Element writers ───────────────────────────────────────────────────────

    def _write_elements_c3d8(self, f, NX: int, NY: int, NZ: int,
                              active_mask: np.ndarray) -> None:
        """Write *Element, type=C3D8 section streaming by i-slab."""
        f.write("*Element, type=C3D8\n")
        stride_i = (NY + 1) * (NZ + 1)
        buf: List[str] = []

        for i in range(NX):
            js, ks = np.where(active_mask[i])
            if js.size == 0:
                continue
            for j_v, k_v in zip(js.tolist(), ks.tolist()):
                eid = i * NY * NZ + j_v * NZ + k_v + 1
                n1 = i     * stride_i + j_v     * (NZ+1) + k_v     + 1
                n2 = (i+1) * stride_i + j_v     * (NZ+1) + k_v     + 1
                n3 = (i+1) * stride_i + (j_v+1) * (NZ+1) + k_v     + 1
                n4 = i     * stride_i + (j_v+1) * (NZ+1) + k_v     + 1
                n5 = i     * stride_i + j_v     * (NZ+1) + (k_v+1) + 1
                n6 = (i+1) * stride_i + j_v     * (NZ+1) + (k_v+1) + 1
                n7 = (i+1) * stride_i + (j_v+1) * (NZ+1) + (k_v+1) + 1
                n8 = i     * stride_i + (j_v+1) * (NZ+1) + (k_v+1) + 1
                buf.append(
                    f"{eid:>10d}, {n1:>10d}, {n2:>10d}, {n3:>10d}, {n4:>10d},"
                    f" {n5:>10d}, {n6:>10d}, {n7:>10d}, {n8:>10d}\n"
                )
            if len(buf) >= 8192:
                f.writelines(buf)
                buf.clear()
        if buf:
            f.writelines(buf)

    def _write_elements_c3d4(self, f, NX: int, NY: int, NZ: int,
                              active_mask: np.ndarray) -> None:
        """Write *Element, type=C3D4 — 6 Kuhn tets per active voxel, streaming."""
        f.write("*Element, type=C3D4\n")
        stride_i = (NY + 1) * (NZ + 1)
        buf: List[str] = []

        for i in range(NX):
            js, ks = np.where(active_mask[i])
            if js.size == 0:
                continue
            for j_v, k_v in zip(js.tolist(), ks.tolist()):
                tet_base = 6 * (i * NY * NZ + j_v * NZ + k_v)
                for t, offsets in enumerate(_KUHN_TETS):
                    eid = tet_base + t + 1
                    nodes = [
                        (i + di) * stride_i + (j_v + dj) * (NZ+1) + (k_v + dk) + 1
                        for di, dj, dk in offsets
                    ]
                    buf.append(
                        f"{eid:>10d}, {nodes[0]:>10d}, {nodes[1]:>10d},"
                        f" {nodes[2]:>10d}, {nodes[3]:>10d}\n"
                    )
            if len(buf) >= 8192:
                f.writelines(buf)
                buf.clear()
        if buf:
            f.writelines(buf)

    # ── Midside-node active-mask builders ────────────────────────────────────

    @staticmethod
    def _build_active_xmid_mask(active_mask: np.ndarray,
                                NX: int, NY: int, NZ: int) -> np.ndarray:
        """(NX, NY+1, NZ+1) — True for x-mid nodes bordering ≥1 active voxel."""
        m = np.zeros((NX, NY + 1, NZ + 1), dtype=bool)
        m[:, :NY,  :NZ] |= active_mask
        m[:, :NY,  1:]  |= active_mask
        m[:, 1:,   :NZ] |= active_mask
        m[:, 1:,   1:]  |= active_mask
        return m

    @staticmethod
    def _build_active_ymid_mask(active_mask: np.ndarray,
                                NX: int, NY: int, NZ: int) -> np.ndarray:
        """(NX+1, NY, NZ+1) — True for y-mid nodes bordering ≥1 active voxel."""
        m = np.zeros((NX + 1, NY, NZ + 1), dtype=bool)
        m[:NX, :, :NZ]  |= active_mask
        m[:NX, :, 1:]   |= active_mask
        m[1:,  :, :NZ]  |= active_mask
        m[1:,  :, 1:]   |= active_mask
        return m

    @staticmethod
    def _build_active_zmid_mask(active_mask: np.ndarray,
                                NX: int, NY: int, NZ: int) -> np.ndarray:
        """(NX+1, NY+1, NZ) — True for z-mid nodes bordering ≥1 active voxel."""
        m = np.zeros((NX + 1, NY + 1, NZ), dtype=bool)
        m[:NX, :NY, :] |= active_mask
        m[:NX, 1:,  :] |= active_mask
        m[1:,  :NY, :] |= active_mask
        m[1:,  1:,  :] |= active_mask
        return m

    @staticmethod
    def _build_active_xydm_mask(active_mask: np.ndarray,
                                NX: int, NY: int, NZ: int) -> np.ndarray:
        """(NX, NY, NZ+1) — xy-face-diag mids (z=k face); active if adjacent voxel active."""
        m = np.zeros((NX, NY, NZ + 1), dtype=bool)
        m[:, :, :NZ] |= active_mask
        m[:, :, 1:]  |= active_mask
        return m

    @staticmethod
    def _build_active_xzdm_mask(active_mask: np.ndarray,
                                NX: int, NY: int, NZ: int) -> np.ndarray:
        """(NX, NY+1, NZ) — xz-face-diag mids (y=j face); active if adjacent voxel active."""
        m = np.zeros((NX, NY + 1, NZ), dtype=bool)
        m[:, :NY, :] |= active_mask
        m[:, 1:,  :] |= active_mask
        return m

    @staticmethod
    def _build_active_yzdm_mask(active_mask: np.ndarray,
                                NX: int, NY: int, NZ: int) -> np.ndarray:
        """(NX+1, NY, NZ) — yz-face-diag mids (x=i face); active if adjacent voxel active."""
        m = np.zeros((NX + 1, NY, NZ), dtype=bool)
        m[:NX, :, :] |= active_mask
        m[1:,  :, :] |= active_mask
        return m

    # ── Node writer for C3D20 ─────────────────────────────────────────────────

    def _write_nodes_c3d20(self, f, NX: int, NY: int, NZ: int, dx: float,
                           node_mask: np.ndarray,
                           xmid_mask: np.ndarray,
                           ymid_mask: np.ndarray,
                           zmid_mask: np.ndarray) -> None:
        """Write *Node for C3D20: corner nodes then x/y/z midside nodes.

        Global ID layout (Nc = (NX+1)(NY+1)(NZ+1) corner nodes):
          xmid(i,j,k)  = Nc + i*(NY+1)*(NZ+1) + j*(NZ+1) + k + 1
          ymid(i,j,k)  = Nc + NX*(NY+1)*(NZ+1) + i*NY*(NZ+1) + j*(NZ+1) + k + 1
          zmid(i,j,k)  = Nc + NX*(NY+1)*(NZ+1) + (NX+1)*NY*(NZ+1) + i*(NY+1)*NZ + j*NZ + k + 1
        """
        f.write("*Node\n")
        buf: List[str] = []

        stride_c = (NY + 1) * (NZ + 1)
        for i in range(NX + 1):
            js, ks = np.where(node_mask[i])
            for j_v, k_v in zip(js.tolist(), ks.tolist()):
                nid = i * stride_c + j_v * (NZ + 1) + k_v + 1
                buf.append(
                    f"{nid:>10d}, {i*dx:>14.6e}, {j_v*dx:>14.6e}, {k_v*dx:>14.6e}\n"
                )
            if len(buf) >= 8192:
                f.writelines(buf); buf.clear()

        Nc     = (NX + 1) * (NY + 1) * (NZ + 1)
        xm_off = Nc
        xm_si  = (NY + 1) * (NZ + 1)
        for i in range(NX):
            js, ks = np.where(xmid_mask[i])
            for j_v, k_v in zip(js.tolist(), ks.tolist()):
                nid = xm_off + i * xm_si + j_v * (NZ + 1) + k_v + 1
                buf.append(
                    f"{nid:>10d}, {(i+0.5)*dx:>14.6e},"
                    f" {j_v*dx:>14.6e}, {k_v*dx:>14.6e}\n"
                )
            if len(buf) >= 8192:
                f.writelines(buf); buf.clear()

        ym_off = Nc + NX * (NY + 1) * (NZ + 1)
        ym_si  = NY * (NZ + 1)
        for i in range(NX + 1):
            js, ks = np.where(ymid_mask[i])
            for j_v, k_v in zip(js.tolist(), ks.tolist()):
                nid = ym_off + i * ym_si + j_v * (NZ + 1) + k_v + 1
                buf.append(
                    f"{nid:>10d}, {i*dx:>14.6e},"
                    f" {(j_v+0.5)*dx:>14.6e}, {k_v*dx:>14.6e}\n"
                )
            if len(buf) >= 8192:
                f.writelines(buf); buf.clear()

        zm_off = Nc + NX * (NY + 1) * (NZ + 1) + (NX + 1) * NY * (NZ + 1)
        zm_si  = (NY + 1) * NZ
        for i in range(NX + 1):
            js, ks = np.where(zmid_mask[i])
            for j_v, k_v in zip(js.tolist(), ks.tolist()):
                nid = zm_off + i * zm_si + j_v * NZ + k_v + 1
                buf.append(
                    f"{nid:>10d}, {i*dx:>14.6e},"
                    f" {j_v*dx:>14.6e}, {(k_v+0.5)*dx:>14.6e}\n"
                )
            if len(buf) >= 8192:
                f.writelines(buf); buf.clear()

        if buf:
            f.writelines(buf)

    # ── Element writer for C3D20 ──────────────────────────────────────────────

    def _write_elements_c3d20(self, f, NX: int, NY: int, NZ: int,
                               active_mask: np.ndarray, Nc: int) -> None:
        """Write *Element, type=C3D20 — 1 element per active voxel.

        Abaqus C3D20 node ordering (same base as C3D8 + 12 mid-edge nodes):
          9=mid(1-2)  10=mid(2-3)  11=mid(3-4)  12=mid(4-1)
          13=mid(5-6) 14=mid(6-7)  15=mid(7-8)  16=mid(8-5)
          17=mid(1-5) 18=mid(2-6)  19=mid(3-7)  20=mid(4-8)
        """
        f.write("*Element, type=C3D20\n")
        cn_si  = (NY + 1) * (NZ + 1)

        xm_off = Nc
        xm_si  = (NY + 1) * (NZ + 1)

        ym_off = Nc + NX * (NY + 1) * (NZ + 1)
        ym_si  = NY * (NZ + 1)

        zm_off = Nc + NX * (NY + 1) * (NZ + 1) + (NX + 1) * NY * (NZ + 1)
        zm_si  = (NY + 1) * NZ

        buf: List[str] = []
        for i in range(NX):
            js, ks = np.where(active_mask[i])
            if js.size == 0:
                continue
            for j_v, k_v in zip(js.tolist(), ks.tolist()):
                eid = i * NY * NZ + j_v * NZ + k_v + 1

                n1  = i     * cn_si + j_v     * (NZ+1) + k_v     + 1
                n2  = (i+1) * cn_si + j_v     * (NZ+1) + k_v     + 1
                n3  = (i+1) * cn_si + (j_v+1) * (NZ+1) + k_v     + 1
                n4  = i     * cn_si + (j_v+1) * (NZ+1) + k_v     + 1
                n5  = i     * cn_si + j_v     * (NZ+1) + (k_v+1) + 1
                n6  = (i+1) * cn_si + j_v     * (NZ+1) + (k_v+1) + 1
                n7  = (i+1) * cn_si + (j_v+1) * (NZ+1) + (k_v+1) + 1
                n8  = i     * cn_si + (j_v+1) * (NZ+1) + (k_v+1) + 1

                n9  = xm_off + i     * xm_si + j_v     * (NZ+1) + k_v     + 1
                n10 = ym_off + (i+1) * ym_si + j_v     * (NZ+1) + k_v     + 1
                n11 = xm_off + i     * xm_si + (j_v+1) * (NZ+1) + k_v     + 1
                n12 = ym_off + i     * ym_si + j_v     * (NZ+1) + k_v     + 1
                n13 = xm_off + i     * xm_si + j_v     * (NZ+1) + (k_v+1) + 1
                n14 = ym_off + (i+1) * ym_si + j_v     * (NZ+1) + (k_v+1) + 1
                n15 = xm_off + i     * xm_si + (j_v+1) * (NZ+1) + (k_v+1) + 1
                n16 = ym_off + i     * ym_si + j_v     * (NZ+1) + (k_v+1) + 1
                n17 = zm_off + i     * zm_si + j_v     * NZ     + k_v     + 1
                n18 = zm_off + (i+1) * zm_si + j_v     * NZ     + k_v     + 1
                n19 = zm_off + (i+1) * zm_si + (j_v+1) * NZ     + k_v     + 1
                n20 = zm_off + i     * zm_si + (j_v+1) * NZ     + k_v     + 1

                buf.append(
                    f"{eid:>10d}, {n1:>10d}, {n2:>10d}, {n3:>10d}, {n4:>10d},"
                    f" {n5:>10d}, {n6:>10d}, {n7:>10d}, {n8:>10d},\n"
                    f"           {n9:>10d}, {n10:>10d}, {n11:>10d}, {n12:>10d},"
                    f" {n13:>10d}, {n14:>10d}, {n15:>10d}, {n16:>10d},\n"
                    f"           {n17:>10d}, {n18:>10d}, {n19:>10d}, {n20:>10d}\n"
                )
            if len(buf) >= 2048:
                f.writelines(buf); buf.clear()
        if buf:
            f.writelines(buf)

    # ── Node writer for C3D10 ─────────────────────────────────────────────────

    def _write_nodes_c3d10(self, f, NX: int, NY: int, NZ: int, dx: float,
                           node_mask: np.ndarray,
                           xmid_mask: np.ndarray, ymid_mask: np.ndarray,
                           zmid_mask: np.ndarray,
                           xydm_mask: np.ndarray, xzdm_mask: np.ndarray,
                           yzdm_mask: np.ndarray,
                           bdm_mask: np.ndarray) -> None:
        """Write *Node for C3D10: corner + axis-mid + face-diag-mid + body-diag-mid.

        ID ranges (Nc = (NX+1)(NY+1)(NZ+1)):
          xmid(i,j,k)  = Nc + i*(NY+1)*(NZ+1) + j*(NZ+1) + k + 1
          ymid(i,j,k)  = Nc+NX*(NY+1)*(NZ+1) + i*NY*(NZ+1) + j*(NZ+1) + k + 1
          zmid(i,j,k)  = above + (NX+1)*NY*(NZ+1) + i*(NY+1)*NZ + j*NZ + k + 1
          xydm(i,j,k)  = N_zm_end + i*NY*(NZ+1) + j*(NZ+1) + k + 1
          xzdm(i,j,k)  = N_xydm_end + i*(NY+1)*NZ + j*NZ + k + 1
          yzdm(i,j,k)  = N_xzdm_end + i*NY*NZ + j*NZ + k + 1
          bdm(i,j,k)   = N_yzdm_end + i*NY*NZ + j*NZ + k + 1
        """
        f.write("*Node\n")
        buf: List[str] = []

        stride_c = (NY + 1) * (NZ + 1)
        for i in range(NX + 1):
            js, ks = np.where(node_mask[i])
            for j_v, k_v in zip(js.tolist(), ks.tolist()):
                nid = i * stride_c + j_v * (NZ + 1) + k_v + 1
                buf.append(
                    f"{nid:>10d}, {i*dx:>14.6e}, {j_v*dx:>14.6e}, {k_v*dx:>14.6e}\n"
                )
            if len(buf) >= 8192:
                f.writelines(buf); buf.clear()

        Nc = (NX + 1) * (NY + 1) * (NZ + 1)

        xm_off = Nc;  xm_si = (NY + 1) * (NZ + 1)
        for i in range(NX):
            js, ks = np.where(xmid_mask[i])
            for j_v, k_v in zip(js.tolist(), ks.tolist()):
                nid = xm_off + i * xm_si + j_v * (NZ + 1) + k_v + 1
                buf.append(
                    f"{nid:>10d}, {(i+0.5)*dx:>14.6e},"
                    f" {j_v*dx:>14.6e}, {k_v*dx:>14.6e}\n"
                )
            if len(buf) >= 8192:
                f.writelines(buf); buf.clear()

        ym_off = Nc + NX * (NY + 1) * (NZ + 1);  ym_si = NY * (NZ + 1)
        for i in range(NX + 1):
            js, ks = np.where(ymid_mask[i])
            for j_v, k_v in zip(js.tolist(), ks.tolist()):
                nid = ym_off + i * ym_si + j_v * (NZ + 1) + k_v + 1
                buf.append(
                    f"{nid:>10d}, {i*dx:>14.6e},"
                    f" {(j_v+0.5)*dx:>14.6e}, {k_v*dx:>14.6e}\n"
                )
            if len(buf) >= 8192:
                f.writelines(buf); buf.clear()

        zm_off = ym_off + (NX + 1) * NY * (NZ + 1);  zm_si = (NY + 1) * NZ
        for i in range(NX + 1):
            js, ks = np.where(zmid_mask[i])
            for j_v, k_v in zip(js.tolist(), ks.tolist()):
                nid = zm_off + i * zm_si + j_v * NZ + k_v + 1
                buf.append(
                    f"{nid:>10d}, {i*dx:>14.6e},"
                    f" {j_v*dx:>14.6e}, {(k_v+0.5)*dx:>14.6e}\n"
                )
            if len(buf) >= 8192:
                f.writelines(buf); buf.clear()

        N_zm_end  = zm_off + (NX + 1) * (NY + 1) * NZ
        xydm_off  = N_zm_end;  xydm_si = NY * (NZ + 1)
        for i in range(NX):
            js, ks = np.where(xydm_mask[i])
            for j_v, k_v in zip(js.tolist(), ks.tolist()):
                nid = xydm_off + i * xydm_si + j_v * (NZ + 1) + k_v + 1
                buf.append(
                    f"{nid:>10d}, {(i+0.5)*dx:>14.6e},"
                    f" {(j_v+0.5)*dx:>14.6e}, {k_v*dx:>14.6e}\n"
                )
            if len(buf) >= 8192:
                f.writelines(buf); buf.clear()

        xzdm_off  = xydm_off + NX * NY * (NZ + 1);  xzdm_si = (NY + 1) * NZ
        for i in range(NX):
            js, ks = np.where(xzdm_mask[i])
            for j_v, k_v in zip(js.tolist(), ks.tolist()):
                nid = xzdm_off + i * xzdm_si + j_v * NZ + k_v + 1
                buf.append(
                    f"{nid:>10d}, {(i+0.5)*dx:>14.6e},"
                    f" {j_v*dx:>14.6e}, {(k_v+0.5)*dx:>14.6e}\n"
                )
            if len(buf) >= 8192:
                f.writelines(buf); buf.clear()

        yzdm_off  = xzdm_off + NX * (NY + 1) * NZ;  yzdm_si = NY * NZ
        for i in range(NX + 1):
            js, ks = np.where(yzdm_mask[i])
            for j_v, k_v in zip(js.tolist(), ks.tolist()):
                nid = yzdm_off + i * yzdm_si + j_v * NZ + k_v + 1
                buf.append(
                    f"{nid:>10d}, {i*dx:>14.6e},"
                    f" {(j_v+0.5)*dx:>14.6e}, {(k_v+0.5)*dx:>14.6e}\n"
                )
            if len(buf) >= 8192:
                f.writelines(buf); buf.clear()

        bdm_off   = yzdm_off + (NX + 1) * NY * NZ;  bdm_si = NY * NZ
        for i in range(NX):
            js, ks = np.where(bdm_mask[i])
            for j_v, k_v in zip(js.tolist(), ks.tolist()):
                nid = bdm_off + i * bdm_si + j_v * NZ + k_v + 1
                buf.append(
                    f"{nid:>10d}, {(i+0.5)*dx:>14.6e},"
                    f" {(j_v+0.5)*dx:>14.6e}, {(k_v+0.5)*dx:>14.6e}\n"
                )
            if len(buf) >= 8192:
                f.writelines(buf); buf.clear()

        if buf:
            f.writelines(buf)

    # ── Element writer for C3D10 ──────────────────────────────────────────────

    def _write_elements_c3d10(self, f, NX: int, NY: int, NZ: int,
                               active_mask: np.ndarray, Nc: int) -> None:
        """Write *Element, type=C3D10 — 6 Kuhn tets × 10 nodes per active voxel.

        Abaqus C3D10: N5=mid(1-2) N6=mid(2-3) N7=mid(1-3)
                       N8=mid(1-4) N9=mid(2-4) N10=mid(3-4).
        """
        f.write("*Element, type=C3D10\n")
        cn_si    = (NY + 1) * (NZ + 1)

        xm_off   = Nc;  xm_si  = (NY + 1) * (NZ + 1)
        ym_off   = Nc + NX * (NY + 1) * (NZ + 1);  ym_si = NY * (NZ + 1)
        zm_off   = ym_off + (NX + 1) * NY * (NZ + 1);  zm_si = (NY + 1) * NZ
        N_zm_end = zm_off + (NX + 1) * (NY + 1) * NZ

        xydm_off = N_zm_end;  xydm_si = NY * (NZ + 1)
        xzdm_off = xydm_off + NX * NY * (NZ + 1);  xzdm_si = (NY + 1) * NZ
        yzdm_off = xzdm_off + NX * (NY + 1) * NZ;  yzdm_si = NY * NZ
        bdm_off  = yzdm_off + (NX + 1) * NY * NZ;  bdm_si  = NY * NZ

        buf: List[str] = []
        for i in range(NX):
            js, ks = np.where(active_mask[i])
            if js.size == 0:
                continue
            for j_v, k_v in zip(js.tolist(), ks.tolist()):
                tet_base = 6 * (i * NY * NZ + j_v * NZ + k_v)

                c000  = i     * cn_si + j_v     * (NZ+1) + k_v     + 1
                c100  = (i+1) * cn_si + j_v     * (NZ+1) + k_v     + 1
                c110  = (i+1) * cn_si + (j_v+1) * (NZ+1) + k_v     + 1
                c010  = i     * cn_si + (j_v+1) * (NZ+1) + k_v     + 1
                c001  = i     * cn_si + j_v     * (NZ+1) + (k_v+1) + 1
                c101  = (i+1) * cn_si + j_v     * (NZ+1) + (k_v+1) + 1
                c111  = (i+1) * cn_si + (j_v+1) * (NZ+1) + (k_v+1) + 1
                c011  = i     * cn_si + (j_v+1) * (NZ+1) + (k_v+1) + 1

                xm_ijk   = xm_off + i * xm_si + j_v     * (NZ+1) + k_v     + 1
                xm_ij1k  = xm_off + i * xm_si + (j_v+1) * (NZ+1) + k_v     + 1
                xm_ijk1  = xm_off + i * xm_si + j_v     * (NZ+1) + (k_v+1) + 1
                xm_ij1k1 = xm_off + i * xm_si + (j_v+1) * (NZ+1) + (k_v+1) + 1

                ym_ijk   = ym_off + i     * ym_si + j_v * (NZ+1) + k_v     + 1
                ym_i1jk  = ym_off + (i+1) * ym_si + j_v * (NZ+1) + k_v     + 1
                ym_ijk1  = ym_off + i     * ym_si + j_v * (NZ+1) + (k_v+1) + 1
                ym_i1jk1 = ym_off + (i+1) * ym_si + j_v * (NZ+1) + (k_v+1) + 1

                zm_ijk   = zm_off + i     * zm_si + j_v     * NZ + k_v + 1
                zm_i1jk  = zm_off + (i+1) * zm_si + j_v     * NZ + k_v + 1
                zm_ij1k  = zm_off + i     * zm_si + (j_v+1) * NZ + k_v + 1
                zm_i1j1k = zm_off + (i+1) * zm_si + (j_v+1) * NZ + k_v + 1

                xydm_ijk  = xydm_off + i * xydm_si + j_v * (NZ+1) + k_v     + 1
                xydm_ijk1 = xydm_off + i * xydm_si + j_v * (NZ+1) + (k_v+1) + 1

                xzdm_ijk  = xzdm_off + i * xzdm_si + j_v     * NZ + k_v + 1
                xzdm_ij1k = xzdm_off + i * xzdm_si + (j_v+1) * NZ + k_v + 1

                yzdm_ijk  = yzdm_off + i     * yzdm_si + j_v * NZ + k_v + 1
                yzdm_i1jk = yzdm_off + (i+1) * yzdm_si + j_v * NZ + k_v + 1

                bdm_ijk   = bdm_off + i * bdm_si + j_v * NZ + k_v + 1

                # Tet 0: n1=c000 n2=c110 n3=c100 n4=c111
                # N5=mid(1,2)=xydm(i,j,k)  N6=mid(2,3)=ymid(i+1,j,k)
                # N7=mid(1,3)=xmid(i,j,k)  N8=mid(1,4)=bdm
                # N9=mid(2,4)=zmid(i+1,j+1,k)  N10=mid(3,4)=yzdm(i+1,j,k)
                buf.append(
                    f"{tet_base+1:>10d},"
                    f" {c000:>10d}, {c110:>10d}, {c100:>10d}, {c111:>10d},\n"
                    f"           {xydm_ijk:>10d}, {ym_i1jk:>10d},"
                    f" {xm_ijk:>10d}, {bdm_ijk:>10d},"
                    f" {zm_i1j1k:>10d}, {yzdm_i1jk:>10d}\n"
                )
                # Tet 1: n1=c000 n2=c100 n3=c101 n4=c111
                # N5=xmid(i,j,k)  N6=zmid(i+1,j,k)  N7=xzdm(i,j,k)
                # N8=bdm  N9=yzdm(i+1,j,k)  N10=ymid(i+1,j,k+1)
                buf.append(
                    f"{tet_base+2:>10d},"
                    f" {c000:>10d}, {c100:>10d}, {c101:>10d}, {c111:>10d},\n"
                    f"           {xm_ijk:>10d}, {zm_i1jk:>10d},"
                    f" {xzdm_ijk:>10d}, {bdm_ijk:>10d},"
                    f" {yzdm_i1jk:>10d}, {ym_i1jk1:>10d}\n"
                )
                # Tet 2: n1=c000 n2=c010 n3=c110 n4=c111
                # N5=ymid(i,j,k)  N6=xmid(i,j+1,k)  N7=xydm(i,j,k)
                # N8=bdm  N9=xzdm(i,j+1,k)  N10=zmid(i+1,j+1,k)
                buf.append(
                    f"{tet_base+3:>10d},"
                    f" {c000:>10d}, {c010:>10d}, {c110:>10d}, {c111:>10d},\n"
                    f"           {ym_ijk:>10d}, {xm_ij1k:>10d},"
                    f" {xydm_ijk:>10d}, {bdm_ijk:>10d},"
                    f" {xzdm_ij1k:>10d}, {zm_i1j1k:>10d}\n"
                )
                # Tet 3: n1=c000 n2=c011 n3=c010 n4=c111
                # N5=yzdm(i,j,k)  N6=zmid(i,j+1,k)  N7=ymid(i,j,k)
                # N8=bdm  N9=xmid(i,j+1,k+1)  N10=xzdm(i,j+1,k)
                buf.append(
                    f"{tet_base+4:>10d},"
                    f" {c000:>10d}, {c011:>10d}, {c010:>10d}, {c111:>10d},\n"
                    f"           {yzdm_ijk:>10d}, {zm_ij1k:>10d},"
                    f" {ym_ijk:>10d}, {bdm_ijk:>10d},"
                    f" {xm_ij1k1:>10d}, {xzdm_ij1k:>10d}\n"
                )
                # Tet 4: n1=c000 n2=c101 n3=c001 n4=c111
                # N5=xzdm(i,j,k)  N6=xmid(i,j,k+1)  N7=zmid(i,j,k)
                # N8=bdm  N9=ymid(i+1,j,k+1)  N10=xydm(i,j,k+1)
                buf.append(
                    f"{tet_base+5:>10d},"
                    f" {c000:>10d}, {c101:>10d}, {c001:>10d}, {c111:>10d},\n"
                    f"           {xzdm_ijk:>10d}, {xm_ijk1:>10d},"
                    f" {zm_ijk:>10d}, {bdm_ijk:>10d},"
                    f" {ym_i1jk1:>10d}, {xydm_ijk1:>10d}\n"
                )
                # Tet 5: n1=c000 n2=c001 n3=c011 n4=c111
                # N5=zmid(i,j,k)  N6=ymid(i,j,k+1)  N7=yzdm(i,j,k)
                # N8=bdm  N9=xydm(i,j,k+1)  N10=xmid(i,j+1,k+1)
                buf.append(
                    f"{tet_base+6:>10d},"
                    f" {c000:>10d}, {c001:>10d}, {c011:>10d}, {c111:>10d},\n"
                    f"           {zm_ijk:>10d}, {ym_ijk1:>10d},"
                    f" {yzdm_ijk:>10d}, {bdm_ijk:>10d},"
                    f" {xydm_ijk1:>10d}, {xm_ij1k1:>10d}\n"
                )

            if len(buf) >= 2048:
                f.writelines(buf); buf.clear()
        if buf:
            f.writelines(buf)

    # ── Elset writers ─────────────────────────────────────────────────────────

    @staticmethod
    def _write_elset_ids(f, name: str, hex_eids: np.ndarray,
                         elem_multiplier: int = 1) -> None:
        """
        Write one *Elset block.
        elem_multiplier=6: each hex eid is expanded to 6 consecutive tet eids.
        """
        if hex_eids.size == 0:
            return
        if elem_multiplier == 1:
            ids = hex_eids
        else:
            base = (hex_eids.astype(np.int64) - 1) * elem_multiplier
            ids = np.concatenate([base + t + 1 for t in range(elem_multiplier)])
            ids = np.sort(ids)
        flat = ids.tolist()
        f.write(f"*Elset, elset={name}\n")
        for start in range(0, len(flat), 10):
            f.write(', '.join(str(x) for x in flat[start:start + 10]) + '\n')

    def _write_elsets_pag(self, f, fm_state, NY: int, NZ: int,
                          active_mask: np.ndarray, em: int) -> set:
        """Write PAG elsets; return the set of pag_ids that actually got a
        non-empty *Elset* (a PAG can end up with zero active voxels once
        ignore_lfi_ids() has voided out all its member grains, or if it is a
        singleton PAG whose sole grain is voided) -- callers must not emit a
        *Material*/*Solid Section* for any pag_id missing from this set, or
        Abaqus will reject the deck for referencing an undefined elset."""
        f.write("** PAG elsets\n")
        written: set = set()
        for pag_id, grain_ids in fm_state.clusters_dict.items():
            parts = []
            for gid in grain_ids:
                vox = fm_state.grain_locs.get(gid)
                if vox is not None and len(vox):
                    parts.append(self._vox_to_hex_eids(vox, NY, NZ, active_mask))
            if parts:
                merged = np.concatenate(parts)
                if merged.size:
                    self._write_elset_ids(f, f"es_pag_{pag_id}", merged, em)
                    written.add(pag_id)
        return written

    def _write_elsets_pck(self, f, fm_state, NY: int, NZ: int,
                          active_mask: np.ndarray, em: int) -> None:
        f.write("** Packet elsets\n")
        g2pag = fm_state.grain_to_pag_id
        g2pkt = fm_state.grain_to_local_pkt_idx
        for gid, vox in fm_state.grain_locs.items():
            if gid not in g2pag or vox is None or len(vox) == 0:
                continue
            pag_id = g2pag[gid]
            pkt_idx = g2pkt[gid]
            eids = self._vox_to_hex_eids(vox, NY, NZ, active_mask)
            self._write_elset_ids(f, f"es_pck_{pag_id}_{pkt_idx}", eids, em)

    def _write_elsets_blk(self, f, fm_state, NY: int, NZ: int,
                          active_mask: np.ndarray, em: int,
                          block_id_map: Dict) -> set:
        """Write block elsets; return the set of block_ids that actually got
        a non-empty *Elset* (see _write_elsets_pag for why this matters)."""
        f.write("** Block elsets\n")
        written: set = set()
        for block_id, vox in fm_state.all_blocks.items():
            if block_id not in block_id_map or vox is None or len(vox) == 0:
                continue
            pag_id, pkt_idx, blk_local = block_id_map[block_id]
            eids = self._vox_to_hex_eids(vox, NY, NZ, active_mask)
            if eids.size == 0:
                continue
            self._write_elset_ids(
                f, f"es_blk_{pag_id}_{pkt_idx}_{blk_local}", eids, em
            )
            written.add(block_id)
        return written

    def _write_elsets_subblk(self, f, fm_state, NY: int, NZ: int,
                              active_mask: np.ndarray, em: int,
                              subblock_id_map: Dict) -> set:
        """Write sub-block elsets; return the set of sb_ids that actually got
        a non-empty *Elset* (see _write_elsets_pag for why this matters)."""
        f.write("** Sub-block elsets\n")
        written: set = set()
        for sb_id, vox in fm_state.all_subblocks.items():
            if sb_id not in subblock_id_map or vox is None or len(vox) == 0:
                continue
            pag_id, pkt_idx, blk_local, sb_local = subblock_id_map[sb_id]
            eids = self._vox_to_hex_eids(vox, NY, NZ, active_mask)
            if eids.size == 0:
                continue
            self._write_elset_ids(
                f, f"es_subblk_{pag_id}_{pkt_idx}_{blk_local}_{sb_local}", eids, em
            )
            written.add(sb_id)
        return written

    def _write_elsets_isolated(self, f, fm_state, NY: int, NZ: int,
                                active_mask: np.ndarray, em: int) -> set:
        """Write isolated-grain elsets for the legacy flat-isolated-grain
        case only (see _leftover_isolated_grains); return the set of grain
        ids that actually got a non-empty *Elset* (see _write_elsets_pag for
        why this matters). Grains covered by a tracked retained-austenite
        PAG get a proper PAG-level elset from _write_elsets_pag instead."""
        isolated = self._leftover_isolated_grains(fm_state)
        written: set = set()
        if not isolated:
            return written
        f.write("** Isolated grain elsets (legacy, no tracked retained-austenite PAG)\n")
        for gid in sorted(isolated):
            vox = fm_state.grain_locs.get(gid)
            if vox is not None and len(vox):
                eids = self._vox_to_hex_eids(vox, NY, NZ, active_mask)
                if eids.size == 0:
                    continue
                self._write_elset_ids(f, f"es_iso_{gid}", eids, em)
                written.add(gid)
        return written

    # ── Boundary-condition node set writer ───────────────────────────────────

    def _write_nsets_bc(self, f, NX: int, NY: int, NZ: int,
                        node_mask: np.ndarray) -> None:
        """Write ns.x-, ns.x+, ns.y-, ns.y+, ns.z-, ns.z+ node sets."""
        stride_i = (NY + 1) * (NZ + 1)

        def _ids_x(i_fixed: int) -> List[int]:
            js, ks = np.where(node_mask[i_fixed])
            return [i_fixed * stride_i + j * (NZ+1) + k + 1
                    for j, k in zip(js.tolist(), ks.tolist())]

        def _ids_y(j_fixed: int) -> List[int]:
            is_, ks = np.where(node_mask[:, j_fixed, :])
            return [iv * stride_i + j_fixed * (NZ+1) + k + 1
                    for iv, k in zip(is_.tolist(), ks.tolist())]

        def _ids_z(k_fixed: int) -> List[int]:
            is_, js = np.where(node_mask[:, :, k_fixed])
            return [iv * stride_i + j * (NZ+1) + k_fixed + 1
                    for iv, j in zip(is_.tolist(), js.tolist())]

        for name, ids in [
            ('ns_x-', _ids_x(0)),  ('ns_x+', _ids_x(NX)),
            ('ns_y-', _ids_y(0)),  ('ns_y+', _ids_y(NY)),
            ('ns_z-', _ids_z(0)),  ('ns_z+', _ids_z(NZ)),
        ]:
            f.write(f"*Nset, nset={name}\n")
            for start in range(0, len(ids), 10):
                f.write(', '.join(str(x) for x in ids[start:start + 10]) + '\n')

    # ── Material and section writers ──────────────────────────────────────────

    def _write_materials(self, f, fm_state, level: str,
                         block_id_map: Dict, sb_id_map: Dict,
                         global_id_map: Dict,
                         phase_id: int, mat_data_flag: int,
                         written_pags: set, written_blocks: set,
                         written_subblocks: set, written_iso: set) -> None:
        """Write *Material stubs.

        written_pags/written_blocks/written_subblocks/written_iso are the id
        sets that _write_elsets_pag/blk/subblk/isolated actually emitted a
        non-empty *Elset* for (a feature can end up with zero active voxels
        after ignore_lfi_ids(), or be a singleton PAG whose sole grain was
        voided). Any id missing from the relevant set is skipped here too --
        otherwise this would emit a *Material* for an elset that was never
        written, and (worse) _write_sections would reference it in a
        *Solid Section*, which Abaqus rejects outright.

        phase_id is the id for the transformed matrix (martensite,
        PHASE_MARTENSITE); retained-austenite PAGs always get
        PHASE_RETAINED_AUSTENITE regardless, since they are a different
        phase by construction, not a configurable choice.
        """
        f.write(
            "** *User Material stubs with Bunge-Euler angles (phi1, Phi, phi2, degrees).\n"
            "** constants=6: phi1, Phi, phi2 [Bunge-Euler deg],"
            " grain_id [global elset ID], phase_id, mat_data_flag.\n"
            f"** phase_id: {PHASE_MARTENSITE}={PHASE_NAMES[PHASE_MARTENSITE]}, "
            f"{PHASE_RETAINED_AUSTENITE}={PHASE_NAMES[PHASE_RETAINED_AUSTENITE]}.\n"
            "** Replace with your CPFEM constitutive definition as needed.\n**\n"
        )

        retained_ids = getattr(fm_state, 'retained_austenite_pag_ids', set())

        def _mat(mat_name: str, phi1: float, PHI: float, phi2: float,
                 grain_id: int, extra_comment: str = '',
                 phase_id_override: Optional[int] = None) -> None:
            eff_phase = phase_id if phase_id_override is None else phase_id_override
            f.write(f"*Material, name={mat_name}\n")
            if extra_comment:
                f.write(f"** {extra_comment}\n")
            f.write(
                f"** Bunge-Euler (deg): phi1={phi1:.4f}, Phi={PHI:.4f}, phi2={phi2:.4f}\n"
                f"*User Material, constants=6\n"
                f"{phi1:.6f}, {PHI:.6f}, {phi2:.6f},"
                f" {grain_id}, {eff_phase}, {mat_data_flag}\n"
                f"*Depvar\n100\n**\n"
            )

        if level == 'subblock':
            for sb_id, (pag_id, pkt_idx, blk_local, sb_local) in sb_id_map.items():
                if sb_id not in written_subblocks:
                    continue
                ori = fm_state.subblock_orientations.get(sb_id, (0.0, 0.0, 0.0))
                gid_val = global_id_map.get(('sb', sb_id), -1)
                _mat(f"MAT_SUBBLK_{pag_id}_{pkt_idx}_{blk_local}_{sb_local}", *ori,
                     grain_id=gid_val)
        elif level == 'block':
            for block_id, (pag_id, pkt_idx, blk_local) in block_id_map.items():
                if block_id not in written_blocks:
                    continue
                ori = fm_state.block_orientations.get(block_id, (0.0, 0.0, 0.0))
                gid_val = global_id_map.get(('blk', block_id), -1)
                _mat(f"MAT_BLK_{pag_id}_{pkt_idx}_{blk_local}", *ori,
                     grain_id=gid_val)
        elif level == 'pag':
            for pag_id, ori in fm_state.pag_orientations.items():
                if pag_id not in written_pags:
                    continue
                gid_val = global_id_map.get(('pag', pag_id), -1)
                is_retained = pag_id in retained_ids
                _mat(f"MAT_PAG_{pag_id}", *ori, grain_id=gid_val,
                     extra_comment=f"Retained austenite PAG {pag_id} (untransformed)."
                                   if is_retained else '',
                     phase_id_override=PHASE_RETAINED_AUSTENITE if is_retained else None)

        # Retained-austenite PAGs never produced blocks/sub-blocks (see
        # generate_blocks()), so when the rest of the structure's lowest
        # level is 'block' or 'subblock' they still need their own PAG-level
        # *Material here -- the level=='pag' branch above only covers them
        # when there are no blocks anywhere in the structure at all.
        if level != 'pag':
            for pag_id in sorted(retained_ids):
                if pag_id not in written_pags:
                    continue
                ori = fm_state.pag_orientations.get(pag_id, (0.0, 0.0, 0.0))
                gid_val = global_id_map.get(('pag', pag_id), -1)
                _mat(f"MAT_PAG_{pag_id}", *ori, grain_id=gid_val,
                     extra_comment=f"Retained austenite PAG {pag_id} (untransformed).",
                     phase_id_override=PHASE_RETAINED_AUSTENITE)

        get_iso_ori = getattr(fm_state, 'get_isolated_grain_orientation', None)
        for gid in sorted(self._leftover_isolated_grains(fm_state)):
            if gid not in written_iso:
                continue
            gid_val = global_id_map.get(('iso', gid), -1)
            ori = get_iso_ori(gid) if get_iso_ori is not None else None
            if ori is not None:
                _mat(f"MAT_ISO_{gid}", *ori, grain_id=gid_val,
                     extra_comment=f"Isolated grain {gid} (retained austenite).",
                     phase_id_override=PHASE_RETAINED_AUSTENITE)
            else:
                _mat(f"MAT_ISO_{gid}", 0.0, 0.0, 0.0, grain_id=gid_val,
                     extra_comment=f"Isolated grain {gid}: no KS orientation — edit as needed.",
                     phase_id_override=PHASE_RETAINED_AUSTENITE)

    def _write_sections(self, f, fm_state, level: str,
                        block_id_map: Dict, sb_id_map: Dict,
                        written_pags: set, written_blocks: set,
                        written_subblocks: set, written_iso: set) -> None:
        """Write *Solid Section cards -- see _write_materials for why every
        loop here is filtered against the written_* sets: a *Solid Section*
        referencing an elset that _write_elsets_* skipped (zero active
        voxels) makes Abaqus reject the whole input deck."""
        f.write("** *Solid Section — lowest hierarchy level only.\n**\n")

        retained_ids = getattr(fm_state, 'retained_austenite_pag_ids', set())

        if level == 'subblock':
            for sb_id, (pag_id, pkt_idx, blk_local, sb_local) in sb_id_map.items():
                if sb_id not in written_subblocks:
                    continue
                eset = f"es_subblk_{pag_id}_{pkt_idx}_{blk_local}_{sb_local}"
                mat  = f"MAT_SUBBLK_{pag_id}_{pkt_idx}_{blk_local}_{sb_local}"
                f.write(f"*Solid Section, elset={eset}, material={mat}\n,\n")
        elif level == 'block':
            for block_id, (pag_id, pkt_idx, blk_local) in block_id_map.items():
                if block_id not in written_blocks:
                    continue
                eset = f"es_blk_{pag_id}_{pkt_idx}_{blk_local}"
                mat  = f"MAT_BLK_{pag_id}_{pkt_idx}_{blk_local}"
                f.write(f"*Solid Section, elset={eset}, material={mat}\n,\n")
        elif level == 'pag':
            for pag_id in fm_state.clusters_dict:
                if pag_id not in written_pags:
                    continue
                f.write(
                    f"*Solid Section, elset=es_pag_{pag_id},"
                    f" material=MAT_PAG_{pag_id}\n,\n"
                )

        # See _write_materials: retained-austenite PAGs never get a block/
        # sub-block section, so they need their own PAG-level section here
        # whenever the rest of the structure's lowest level went further.
        if level != 'pag':
            for pag_id in sorted(retained_ids):
                if pag_id not in written_pags:
                    continue
                f.write(
                    f"*Solid Section, elset=es_pag_{pag_id},"
                    f" material=MAT_PAG_{pag_id}\n,\n"
                )

        for gid in sorted(self._leftover_isolated_grains(fm_state)):
            if gid not in written_iso:
                continue
            f.write(f"*Solid Section, elset=es_iso_{gid}, material=MAT_ISO_{gid}\n,\n")

    # ── Stub writers ──────────────────────────────────────────────────────────

    @staticmethod
    def _write_interactions_stub(f) -> None:
        f.write(
            "** Interactions / constraints\n"
            "** TODO: Add periodic BCs, tie constraints, or contact definitions.\n"
        )

    @staticmethod
    def _write_steps_stub(f) -> None:
        f.write(
            "** Step definition\n"
            "** TODO: Define *Step, boundary conditions, and *Output requests.\n"
        )

    # ── Master .inp writer ────────────────────────────────────────────────────

    def _write_master(self, out_dir: Path, elem_type: str, has_subblk: bool,
                      custom_message: str, src_unit: str, dst_unit: str,
                      dx: float, NX: int, NY: int, NZ: int, n_active: int,
                      phase_element_counts: Optional[Dict[int, int]] = None) -> None:
        with open(out_dir / 'model_master.inp', 'w') as f:
            f.write(_UPXO_HEADER)
            if custom_message:
                f.write(f"** {custom_message}\n**\n")
            phase_line = ""
            if phase_element_counts and n_active:
                parts = [
                    f"{PHASE_NAMES.get(pid, f'phase{pid}')}={cnt:,} "
                    f"({100.0 * cnt / n_active:.1f}%)"
                    for pid, cnt in sorted(phase_element_counts.items())
                    if cnt > 0
                ]
                if parts:
                    phase_line = f"** Phase breakdown: {', '.join(parts)}\n"
            f.write(
                f"*Heading\n"
                f"** FM steel microstructure — element type: {elem_type}\n"
                f"** Grid: {NX} x {NY} x {NZ} voxels  "
                f"|  Active elements: {n_active:,}\n"
                f"{phase_line}"
                f"** Voxel size: {dx:.6g} {dst_unit}  "
                f"(structure defined in {src_unit})\n"
                f"** Domain: {NX*dx:.4g} x {NY*dx:.4g} x {NZ*dx:.4g} {dst_unit}\n**\n"
                f"*PREPRINT, ECHO=NO, MODEL=NO, HISTORY=NO, CONTACT=NO\n**\n"
                f"*INCLUDE, INPUT=01_nodes.inp\n"
                f"*INCLUDE, INPUT=02_elements.inp\n"
                f"*INCLUDE, INPUT=03a_elsets_pag.inp\n"
                f"*INCLUDE, INPUT=03b_elsets_pck.inp\n"
                f"*INCLUDE, INPUT=03c_elsets_blk.inp\n"
            )
            if has_subblk:
                f.write("*INCLUDE, INPUT=03d_elsets_subblk.inp\n")
            f.write(
                "*INCLUDE, INPUT=04_nsets_bc.inp\n"
                "*INCLUDE, INPUT=05_materials.inp\n"
                "*INCLUDE, INPUT=06_sections.inp\n"
                "*INCLUDE, INPUT=07_interactions.inp\n"
                "*INCLUDE, INPUT=08_steps_output.inp\n"
            )

    # ── Shared export pipeline ────────────────────────────────────────────────

    def _export_core(self, fm_state, folder_name: str, elem_type: str,
                     output_unit: Optional[str], max_voxels: Optional[int],
                     force: bool, custom_message: str,
                     phase_id: Optional[int] = None,
                     mat_data_flag: Optional[int] = None) -> Path:
        """Shared pipeline for all element types."""
        eff_phase_id  = phase_id      if phase_id      is not None else self._phase_id
        eff_mat_flag  = mat_data_flag if mat_data_flag is not None else self._mat_data_flag
        if int(eff_phase_id) != PHASE_MARTENSITE:
            raise ValueError(
                f"phase_id={eff_phase_id} is not supported. "
                f"Only phase_id={PHASE_MARTENSITE} (martensite, the transformed "
                "matrix) is configurable; retained austenite is auto-tagged "
                f"phase_id={PHASE_RETAINED_AUSTENITE} automatically."
            )
        if int(eff_mat_flag) not in (0, 1):
            raise ValueError(
                f"mat_data_flag={eff_mat_flag} is invalid. Must be 0 or 1."
            )

        if hasattr(fm_state, 'ensure_isolated_grain_orientations'):
            fm_state.ensure_isolated_grain_orientations()

        NX, NY, NZ = fm_state.lgi.shape
        n_vox = NX * NY * NZ
        self._preflight_check(n_vox, elem_type, max_voxels, force)

        active_mask = self._build_active_mask(fm_state.lgi)
        node_mask   = self._build_active_node_mask(active_mask, NX, NY, NZ)
        is_tet      = elem_type in ('C3D4', 'C3D10')
        n_active    = int(active_mask.sum()) * (_TETS_PER_VOX if is_tet else 1)

        src_unit = getattr(fm_state, 'units', 'microns')
        dst_unit = output_unit or self._output_unit or src_unit
        dx = fm_state.voxel_size * self._unit_scale(src_unit, dst_unit)

        level         = self._detect_lowest_level(fm_state)
        blk_map       = self._build_block_id_map(fm_state)
        sb_map        = self._build_subblock_id_map(fm_state, blk_map)
        global_id_map = self._build_global_id_map(fm_state, blk_map, sb_map)
        has_subblk    = level == 'subblock'
        em            = _TETS_PER_VOX if is_tet else 1

        out_dir = self._resolve_output_dir(folder_name)

        Nc = (NX + 1) * (NY + 1) * (NZ + 1)

        _v = self._verbosity > 0
        _total = 10 + (1 if has_subblk else 0)
        _step  = [0]

        def _done(fname):
            _step[0] += 1
            if _v:
                print(f"  [{_step[0]:2d}/{_total}] Written: {fname}")

        def _starting(label, total=None):
            # Companion to _done(): steps before this only ever printed on
            # completion, so a slow/stuck step was indistinguishable from the
            # previous one still finishing -- this pinpoints which step it is.
            if _v:
                t = total if total is not None else _total
                print(f"  [{_step[0] + 1:2d}/{t}] Starting: {label}...")

        # 01 nodes
        with open(out_dir / '01_nodes.inp', 'w') as f:
            if elem_type == 'C3D20':
                xm_m = self._build_active_xmid_mask(active_mask, NX, NY, NZ)
                ym_m = self._build_active_ymid_mask(active_mask, NX, NY, NZ)
                zm_m = self._build_active_zmid_mask(active_mask, NX, NY, NZ)
                self._write_nodes_c3d20(f, NX, NY, NZ, dx, node_mask,
                                        xm_m, ym_m, zm_m)
            elif elem_type == 'C3D10':
                xm_m   = self._build_active_xmid_mask(active_mask, NX, NY, NZ)
                ym_m   = self._build_active_ymid_mask(active_mask, NX, NY, NZ)
                zm_m   = self._build_active_zmid_mask(active_mask, NX, NY, NZ)
                xydm_m = self._build_active_xydm_mask(active_mask, NX, NY, NZ)
                xzdm_m = self._build_active_xzdm_mask(active_mask, NX, NY, NZ)
                yzdm_m = self._build_active_yzdm_mask(active_mask, NX, NY, NZ)
                bdm_m  = active_mask.copy()
                self._write_nodes_c3d10(f, NX, NY, NZ, dx, node_mask,
                                        xm_m, ym_m, zm_m,
                                        xydm_m, xzdm_m, yzdm_m, bdm_m)
            else:
                self._write_nodes(f, NX, NY, NZ, dx, node_mask)
        _done('01_nodes.inp')

        # 02 elements
        with open(out_dir / '02_elements.inp', 'w') as f:
            if elem_type == 'C3D8':
                self._write_elements_c3d8(f, NX, NY, NZ, active_mask)
            elif elem_type == 'C3D4':
                self._write_elements_c3d4(f, NX, NY, NZ, active_mask)
            elif elem_type == 'C3D20':
                self._write_elements_c3d20(f, NX, NY, NZ, active_mask, Nc)
            elif elem_type == 'C3D10':
                self._write_elements_c3d10(f, NX, NY, NZ, active_mask, Nc)
        _done('02_elements.inp')

        # 03a PAG elsets
        with open(out_dir / '03a_elsets_pag.inp', 'w') as f:
            written_pags = self._write_elsets_pag(f, fm_state, NY, NZ, active_mask, em)
        _done('03a_elsets_pag.inp')

        # 03b packet elsets
        with open(out_dir / '03b_elsets_pck.inp', 'w') as f:
            self._write_elsets_pck(f, fm_state, NY, NZ, active_mask, em)
        _done('03b_elsets_pck.inp')

        # 03c block + isolated elsets
        written_blocks: set = set()
        with open(out_dir / '03c_elsets_blk.inp', 'w') as f:
            if hasattr(fm_state, 'all_blocks') and fm_state.all_blocks:
                written_blocks = self._write_elsets_blk(f, fm_state, NY, NZ, active_mask, em, blk_map)
            written_iso = self._write_elsets_isolated(f, fm_state, NY, NZ, active_mask, em)
        _done('03c_elsets_blk.inp')

        # 03d sub-block elsets
        written_subblocks: set = set()
        if has_subblk:
            with open(out_dir / '03d_elsets_subblk.inp', 'w') as f:
                written_subblocks = self._write_elsets_subblk(f, fm_state, NY, NZ, active_mask, em, sb_map)
            _done('03d_elsets_subblk.inp')

        # 04 BC node sets
        _starting('04_nsets_bc.inp')
        with open(out_dir / '04_nsets_bc.inp', 'w') as f:
            self._write_nsets_bc(f, NX, NY, NZ, node_mask)
        _done('04_nsets_bc.inp')

        # 05 materials
        _starting('05_materials.inp')
        with open(out_dir / '05_materials.inp', 'w') as f:
            self._write_materials(f, fm_state, level, blk_map, sb_map,
                                  global_id_map, eff_phase_id, eff_mat_flag,
                                  written_pags, written_blocks,
                                  written_subblocks, written_iso)
        _done('05_materials.inp')

        # 06 sections
        _starting('06_sections.inp')
        with open(out_dir / '06_sections.inp', 'w') as f:
            self._write_sections(f, fm_state, level, blk_map, sb_map,
                                 written_pags, written_blocks,
                                 written_subblocks, written_iso)
        _done('06_sections.inp')

        # 07 interactions
        with open(out_dir / '07_interactions.inp', 'w') as f:
            self._write_interactions_stub(f)
        _done('07_interactions.inp')

        # 08 steps
        with open(out_dir / '08_steps_output.inp', 'w') as f:
            self._write_steps_stub(f)
        _done('08_steps_output.inp')

        # master
        _starting('model_master.inp', total=_total + 1)
        phase_counts = self._count_phase_active_elements(fm_state, active_mask, NY, NZ, em, n_active)
        self._write_master(out_dir, elem_type, has_subblk, custom_message,
                           src_unit, dst_unit, dx, NX, NY, NZ, n_active,
                           phase_element_counts=phase_counts)
        if _v:
            print(f"  [{_total + 1}/{_total + 1}] Written: model_master.inp")

        return out_dir

    # ── Public export API ─────────────────────────────────────────────────────

    def export_c3d8(self, fm_state, folder_name: str,
                    output_unit: Optional[str] = None,
                    max_voxels: Optional[int] = None,
                    force: bool = False,
                    custom_message: str = "",
                    phase_id: Optional[int] = None,
                    mat_data_flag: Optional[int] = None) -> Path:
        """
        Export as C3D8 (linear hexahedral) Abaqus mesh.

        Parameters
        ----------
        fm_state :
            FM steel pipeline state: FMSteel3DWithPAGs, ...WithBlocks,
            ...WithOrientations, or ...WithSubBlocks.
        folder_name : str
            Base folder name; suffix '1' appended, auto-incremented if taken.
        output_unit : str, optional
            Override output unit for this call ('microns', 'mm', 'm').
        max_voxels : int, optional
            Override C3D8 hard voxel limit for this call.
        force : bool
            Skip the consent prompt even when hard limit is exceeded.
        custom_message : str
            Written to *Heading (e.g. 'Made by Dr. A. Smith, Univ. X').
        phase_id : int, optional
            Override instance phase_id for this call (must be 2).
        mat_data_flag : int, optional
            Override instance mat_data_flag for this call (0 or 1).

        Returns
        -------
        Path
            Path to created output directory.
        """
        out_dir = self._export_core(fm_state, folder_name, 'C3D8',
                                    output_unit, max_voxels, force, custom_message,
                                    phase_id, mat_data_flag)
        print(f"C3D8 mesh written: {out_dir}")
        return out_dir

    def export_c3d4(self, fm_state, folder_name: str,
                    output_unit: Optional[str] = None,
                    max_voxels: Optional[int] = None,
                    force: bool = False,
                    custom_message: str = "",
                    phase_id: Optional[int] = None,
                    mat_data_flag: Optional[int] = None) -> Path:
        """
        Export as C3D4 (linear tet) Abaqus mesh — 6 Kuhn tets per voxel.

        Parameters
        ----------
        (same as export_c3d8)

        Returns
        -------
        Path
            Path to created output directory.
        """
        out_dir = self._export_core(fm_state, folder_name, 'C3D4',
                                    output_unit, max_voxels, force, custom_message,
                                    phase_id, mat_data_flag)
        print(f"C3D4 mesh written: {out_dir}")
        return out_dir

    def export_c3d20(self, fm_state, folder_name: str,
                     output_unit: Optional[str] = None,
                     max_voxels: Optional[int] = None,
                     force: bool = False,
                     custom_message: str = "",
                     phase_id: Optional[int] = None,
                     mat_data_flag: Optional[int] = None) -> Path:
        """Export as C3D20 (quadratic hexahedral) Abaqus mesh.

        Each voxel maps to one 20-node element (8 corners + 12 mid-edge nodes).
        Hard limit 1.5 M voxels; warn at 750 k.
        """
        out_dir = self._export_core(fm_state, folder_name, 'C3D20',
                                    output_unit, max_voxels, force, custom_message,
                                    phase_id, mat_data_flag)
        print(f"C3D20 mesh written: {out_dir}")
        return out_dir

    def export_c3d10(self, fm_state, folder_name: str,
                     output_unit: Optional[str] = None,
                     max_voxels: Optional[int] = None,
                     force: bool = False,
                     custom_message: str = "",
                     phase_id: Optional[int] = None,
                     mat_data_flag: Optional[int] = None) -> Path:
        """Export as C3D10 (quadratic tetrahedral) Abaqus mesh.

        Each voxel is decomposed into 6 Kuhn tets, each with 10 nodes
        (4 corners + 6 mid-edge nodes).  Hard limit 1 M voxels; warn at 500 k.
        """
        out_dir = self._export_core(fm_state, folder_name, 'C3D10',
                                    output_unit, max_voxels, force, custom_message,
                                    phase_id, mat_data_flag)
        print(f"C3D10 mesh written: {out_dir}")
        return out_dir

    # ── Deprecated stubs ──────────────────────────────────────────────────────

    def mesh_fm_steel_c3d8(self, *args, **kwargs):
        raise AttributeError("Renamed to export_c3d8(). Update your call site.")

    def mesh_fm_steel_c3d20(self, *args, **kwargs):
        raise AttributeError("Renamed to export_c3d20(). Update your call site.")

    def write_fm_steel_abaqus_inp(self, *args, **kwargs):
        raise AttributeError("Removed. Use export_c3d8() or export_c3d4().")


__all__ = ['MeshExporter3D', 'DEFAULT_LIMITS']
