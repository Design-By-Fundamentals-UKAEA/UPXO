"""
FMSteel3DWithSubBlocks: Final state class with full FM steel hierarchy including sub-blocks.

Extends FMSteel3DWithOrientations by adding sub-block (lath) level subdivision
within each martensitic block, with per-sub-block orientation spread.

Classes:
    FMSteel3DWithSubBlocks: Complete FM steel state class with sub-block hierarchy.
"""

import numpy as np
from typing import Optional, Dict, List, Tuple

from .phases_3d import (
    PHASE_MARTENSITE, PHASE_RETAINED_AUSTENITE, PHASE_NAMES,
    retained_austenite_voxels_and_orientations,
)


class FMSteel3DWithSubBlocks:
    """
    Complete 3D FM steel microstructure with sub-block (lath) hierarchy.

    Hierarchy: grains -> PAGs -> packets -> blocks -> sub-blocks

    Attributes
    ----------
    _parent : FMSteel3DWithOrientations
        Parent orientation state (full block hierarchy with orientations).
    all_subblocks : dict[str, np.ndarray]
        {subblock_id: (n_voxels, 3) voxel coordinate array}.
        Sub-block naming: 'SB_{block_id}_{local}' where block_id follows
        the 'B_{pag_id}_{packet_id}_{local}' convention.
    block_to_subblocks_map : dict[str, list[str]]
        {block_id: [subblock_id, ...]} — maps every block to its sub-blocks.
    subblock_orientations : dict[str, tuple]
        {subblock_id: (phi1, Phi, phi2)} in degrees — parent block orientation
        plus a small intra-block spread perturbation.
    subblock_slicing_normals : dict[str, np.ndarray]
        {subblock_id: unit_normal (3,)} — inherited from parent block.
    _random_seed : int or None
    """

    __slots__ = (
        '_parent',
        'all_subblocks',
        'block_to_subblocks_map',
        'subblock_orientations',
        'subblock_slicing_normals',
        '_random_seed',
        '_verbosity',
        '_log_sink',
    )

    def __init__(self,
                 parent,
                 all_subblocks: Dict[str, np.ndarray],
                 block_to_subblocks_map: Dict[str, List[str]],
                 subblock_orientations: Dict[str, Tuple[float, float, float]],
                 subblock_slicing_normals: Optional[Dict[str, np.ndarray]] = None,
                 random_seed: Optional[int] = None,
                 verbosity: Optional[int] = None,
                 log_sink=None):
        self._parent = parent
        self.all_subblocks = all_subblocks
        self.block_to_subblocks_map = block_to_subblocks_map
        self.subblock_orientations = subblock_orientations
        self.subblock_slicing_normals = subblock_slicing_normals or {}
        self._random_seed = random_seed
        self._verbosity = int(getattr(parent, '_verbosity', 0) if verbosity is None else verbosity)
        self._log_sink = getattr(parent, '_log_sink', None) if log_sink is None else log_sink

    # ------------------------------------------------------------------
    # Delegation — read-only access to full parent hierarchy
    # ------------------------------------------------------------------

    @property
    def lgi(self) -> np.ndarray:
        return self._parent.lgi

    @property
    def grain_locs(self) -> Dict[int, np.ndarray]:
        return self._parent.grain_locs

    @property
    def clusters_dict(self) -> Dict[int, List[int]]:
        return self._parent.clusters_dict

    @property
    def all_blocks(self) -> Dict[str, np.ndarray]:
        return self._parent.all_blocks

    @property
    def block_orientations(self) -> Dict[str, Tuple[float, float, float]]:
        return self._parent.block_orientations

    @property
    def pag_orientations(self) -> Dict[int, Tuple[float, float, float]]:
        return self._parent.pag_orientations

    @property
    def grain_to_blocks_map(self) -> Dict[int, List[str]]:
        """grain_id -> [block_id, ...] (keys are grain_ids = packet ids)."""
        return self._parent.grain_to_blocks_map

    @property
    def grain_to_pag_id(self) -> Dict[int, int]:
        """Reverse lookup grain_id -> pag_id."""
        return self._parent.grain_to_pag_id

    @property
    def grain_to_local_pkt_idx(self) -> Dict[int, int]:
        """1-based local packet ordinal within each PAG: grain_id -> local_idx."""
        return self._parent.grain_to_local_pkt_idx

    @property
    def grain_to_plane_idx(self) -> Dict[int, int]:
        """grain_id -> {111}FCC habit-plane index (0-3) from parent block level."""
        return self._parent.grain_to_plane_idx

    @property
    def grain_orientations(self) -> Dict[int, Tuple[float, float, float]]:
        """Grain-level BCC orientations from parent orientations level."""
        return self._parent.grain_orientations

    @property
    def block_slicing_normals(self) -> Dict[str, np.ndarray]:
        """block_id -> unit normal of the {111}FCC habit plane used to slice
        it, from parent block level (see FMSteel3DWithBlocks docstring)."""
        return self._parent.block_slicing_normals

    @property
    def block_to_variant_idx(self) -> Dict[str, int]:
        """block_id -> KS variant index (0-5) within its packet, from parent
        orientations level (see FMSteel3DWithOrientations docstring)."""
        return self._parent.block_to_variant_idx

    @property
    def physical_dimensions(self):
        return self._parent.physical_dimensions

    @property
    def voxel_size(self) -> float:
        return self._parent.voxel_size

    @property
    def units(self) -> str:
        """Physical unit string ('microns', 'mm', 'm')."""
        return self._parent.units

    @property
    def isolated_grains(self):
        """Isolated grains from parent PAG level."""
        return self._parent.isolated_grains

    @property
    def retained_austenite_pag_ids(self):
        """Retained-austenite PAG IDs from parent PAG level (see
        FMSteel3DWithPAGs docstring)."""
        return self._parent.retained_austenite_pag_ids

    def ensure_isolated_grain_orientations(self, random_seed: Optional[int] = None) -> None:
        """See FMSteel3DWithPAGs.ensure_isolated_grain_orientations."""
        self._parent.ensure_isolated_grain_orientations(random_seed=random_seed)

    def get_isolated_grain_orientation(self, gid: int) -> Optional[Tuple[float, float, float]]:
        """See FMSteel3DWithPAGs.get_isolated_grain_orientation."""
        return self._parent.get_isolated_grain_orientation(gid)

    def available_phases(self) -> List[int]:
        """See FMSteel3DWithOrientations.available_phases."""
        phases = []
        if self.all_subblocks:
            phases.append(PHASE_MARTENSITE)
        if self.isolated_grains:
            phases.append(PHASE_RETAINED_AUSTENITE)
        return phases

    def get_phase_voxels_and_orientations(
        self, phase_id: int
    ) -> Tuple[Dict, Dict[int, Tuple[float, float, float]]]:
        """See FMSteel3DWithOrientations.get_phase_voxels_and_orientations.

        PHASE_MARTENSITE here is sub-block granularity (all_subblocks,
        subblock_orientations) -- the finer level this class adds; retained
        austenite has no sub-block subdivision, so it stays grain granularity
        exactly as at the block level.
        """
        if phase_id == PHASE_MARTENSITE:
            return dict(self.all_subblocks), dict(self.subblock_orientations)
        if phase_id == PHASE_RETAINED_AUSTENITE:
            return retained_austenite_voxels_and_orientations(self)
        raise ValueError(
            f"Unknown phase_id={phase_id}. Available phases for this "
            f"structure: {self.available_phases()} "
            f"({[PHASE_NAMES.get(p) for p in self.available_phases()]})."
        )

    @property
    def n_grains(self) -> int:
        return self._parent.n_grains

    @property
    def n_pags(self) -> int:
        return self._parent.n_pags

    @property
    def n_blocks(self) -> int:
        return self._parent.n_blocks

    @property
    def n_subblocks(self) -> int:
        return len(self.all_subblocks)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def reassign_subblock_orientations(
        self,
        intrablock_ori_spread_deg: float = 2.0,
        random_seed: Optional[int] = None,
    ) -> None:
        """Re-assign sub-block orientations in-place with a new spread value."""
        from .orientation_assigner_3d import OrientationAssigner3D
        ori_assigner = OrientationAssigner3D()
        self.subblock_orientations = ori_assigner.assign_subblock_orientations(
            all_subblocks=self.all_subblocks,
            block_orientations=self.block_orientations,
            intrablock_ori_spread_deg=intrablock_ori_spread_deg,
            random_seed=random_seed,
        )

    def get_subblock_statistics(self) -> Dict:
        """Return basic statistics on the sub-block structure."""
        sb_sizes = [len(v) for v in self.all_subblocks.values()]
        if not sb_sizes:
            return {'n_subblocks': 0, 'min_voxels_per_subblock': 0,
                    'max_voxels_per_subblock': 0, 'mean_voxels_per_subblock': 0.0,
                    'n_blocks_with_subblocks': 0,
                    'mean_subblocks_per_block': 0.0}

        sbb = [len(v) for v in self.block_to_subblocks_map.values() if v]
        return {
            'n_subblocks': len(self.all_subblocks),
            'min_voxels_per_subblock': int(min(sb_sizes)),
            'max_voxels_per_subblock': int(max(sb_sizes)),
            'mean_voxels_per_subblock': float(np.mean(sb_sizes)),
            'n_blocks_with_subblocks': len(sbb),
            'mean_subblocks_per_block': float(np.mean(sbb)) if sbb else 0.0,
        }


__all__ = ['FMSteel3DWithSubBlocks']
