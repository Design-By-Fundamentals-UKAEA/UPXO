"""
FMSteel3DWithBlocks: State class representing PAGs partitioned into martensitic blocks.

This module contains the class for FM steel structures after block generation.
It adds block-level hierarchical structure (blocks within packets within PAGs).

Classes:
    FMSteel3DWithBlocks: PAGs + blocks state class.
"""

import numpy as np
from typing import Optional, Dict, List, Tuple


class FMSteel3DWithBlocks:
    """
    FM steel grain structure with block hierarchy.
    
    Holds the PAG structure plus computed martensitic blocks within each PAG.
    Blocks represent the smallest scale of microstructural organization in this
    model (below blocks are grains, above blocks are PAGs).
    
    This is an intermediate state in the pipeline. It is created by calling
    generate_blocks() on a FMSteel3DWithPAGs instance, and transitions to
    FMSteel3DWithOrientations by calling assign_orientations().
    
    Attributes
    ----------
    _parent : FMSteel3DWithPAGs
        Reference to parent PAG structure (read-only intermediate state).
    
    all_blocks : dict[str, np.ndarray]
        Maps block_id (str) → (n_voxels, 3) array of voxel coordinates.
        Block naming convention: 'B_{pag_id}_{grain_id}_{local_id}'.

    grain_to_blocks_map : dict[int, list[str]]
        Maps grain_id (= packet identifier, global integer) → list of block_ids
        produced from that packet.  Keys are grain_ids, NOT local packet ordinals.
        Use parent.grain_to_local_pkt_idx to convert grain_id to a local ordinal.

    slicing_planes : dict
        Stores the Plane objects used for slicing (for visualization/debugging).
    
    _random_seed : int
        Random seed used for block generation.
    """
    
    __slots__ = (
        '_parent',
        'all_blocks',
        'grain_to_blocks_map',
        'grain_to_plane_idx',
        'block_slicing_normals',
        'slicing_planes',
        '_random_seed',
        '_verbosity',
        '_log_sink',
    )

    def __init__(self,
                 parent,
                 all_blocks: Dict[str, np.ndarray],
                 grain_to_blocks_map: Dict[int, List[str]],
                 grain_to_plane_idx: Optional[Dict[int, int]] = None,
                 block_slicing_normals: Optional[Dict[str, np.ndarray]] = None,
                 slicing_planes: Optional[Dict] = None,
                 random_seed: Optional[int] = None,
                 verbosity: Optional[int] = None,
                 log_sink=None):
        """
        Initialize FMSteel3DWithBlocks.

        Typically called internally by FMSteel3DWithPAGs.generate_blocks().
        Direct instantiation allowed but not recommended.

        Parameters
        ----------
        parent : FMSteel3DWithPAGs
            Parent PAG structure instance.
        all_blocks : dict
            Block voxel data: {block_id: voxel_coords_array}.
        grain_to_blocks_map : dict
            {grain_id: [block_id, ...]} — maps each packet (identified by its
            grain_id) to the block_ids produced from it.
        grain_to_plane_idx : dict, optional
            Maps grain_id to the {111} plane index used for that packet.
        block_slicing_normals : dict, optional
            {block_id: unit_normal_array} — the slicing plane normal used to
            create each block. Required by SubBlockGenerator3D.
        slicing_planes : dict, optional
            Plane objects used for slicing (for debugging).
        random_seed : int, optional
            Random seed used for this block generation.
        """
        self._parent = parent
        self.all_blocks = all_blocks
        self.grain_to_blocks_map = grain_to_blocks_map
        self.grain_to_plane_idx = grain_to_plane_idx or {}
        self.block_slicing_normals = block_slicing_normals or {}
        self.slicing_planes = slicing_planes or {}
        self._random_seed = random_seed
        self._verbosity = int(getattr(parent, '_verbosity', 0) if verbosity is None else verbosity)
        self._log_sink = getattr(parent, '_log_sink', None) if log_sink is None else log_sink

    def _emit(self, level: int, msg: str, component: str = 'ORI') -> None:
        if self._verbosity < int(level):
            return
        text = f"[{component}][L{int(level)}] {msg}"
        if self._log_sink is not None:
            self._log_sink(text)
        else:
            print(text)
    
    # ========== Delegation properties (read-only access to parents) ==========
    
    @property
    def lgi(self) -> np.ndarray:
        """Labeled grain image from base parent."""
        return self._parent.lgi
    
    @property
    def grain_locs(self) -> Dict[int, np.ndarray]:
        """Grain voxel coordinates from base parent."""
        return self._parent.grain_locs
    
    @property
    def clusters_dict(self) -> Dict[int, List[int]]:
        """PAG clustering from parent."""
        return self._parent.clusters_dict
    
    @property
    def neigh_clid(self) -> Dict[int, List[int]]:
        """PAG adjacency from parent."""
        return self._parent.neigh_clid
    
    @property
    def pag_orientations(self) -> Dict[int, Tuple[float, float, float]]:
        """PAG orientations from parent."""
        return self._parent.pag_orientations

    @property
    def grain_to_pag_id(self) -> Dict[int, int]:
        """Reverse lookup grain_id -> pag_id, derived from parent clusters_dict."""
        return self._parent.grain_to_pag_id

    @property
    def grain_to_local_pkt_idx(self) -> Dict[int, int]:
        """1-based local packet ordinal within each PAG: grain_id -> local_idx."""
        return self._parent.grain_to_local_pkt_idx
    
    @property
    def n_grains(self) -> int:
        """Total grains in base structure."""
        return self._parent.n_grains
    
    @property
    def physical_dimensions(self):
        """Physical domain size."""
        return self._parent.physical_dimensions
    
    @property
    def n_pags(self) -> int:
        """Number of PAGs."""
        return self._parent.n_pags

    @property
    def n_blocks(self) -> int:
        """Number of blocks."""
        return len(self.all_blocks)

    @property
    def voxel_size(self) -> float:
        """Voxel size."""
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
    def isolated_grain_orientations(self):
        """Isolated-grain orientations from parent PAG level (see
        FMSteel3DWithPAGs docstring)."""
        return self._parent.isolated_grain_orientations

    @property
    def retained_austenite_pag_ids(self):
        """Retained-austenite PAG IDs from parent PAG level (see
        FMSteel3DWithPAGs docstring). These PAG IDs were skipped by
        generate_blocks(), so they never appear in all_blocks/
        grain_to_blocks_map."""
        return self._parent.retained_austenite_pag_ids

    @property
    def neigh_gid(self) -> Dict[int, List[int]]:
        """Grain-level neighbour graph from base level."""
        return self._parent.neigh_gid

    @property
    def n_retained_austenite_pags(self) -> int:
        """Number of retained-austenite PAGs from parent PAG level."""
        return self._parent.n_retained_austenite_pags

    @property
    def n_transformed_pags(self) -> int:
        """Number of transformed (martensite-forming) PAGs from parent PAG level."""
        return self._parent.n_transformed_pags

    def ensure_isolated_grain_orientations(self, random_seed: Optional[int] = None) -> None:
        """See FMSteel3DWithPAGs.ensure_isolated_grain_orientations."""
        self._parent.ensure_isolated_grain_orientations(random_seed=random_seed)

    def get_isolated_grain_orientation(self, gid: int) -> Optional[Tuple[float, float, float]]:
        """See FMSteel3DWithPAGs.get_isolated_grain_orientation."""
        return self._parent.get_isolated_grain_orientation(gid)

    # ========== Pipeline continuation ==========
    
    def assign_orientations(self,
                            ks_variant_selection: str = 'random_per_block',
                            random_seed: Optional[int] = None) -> 'FMSteel3DWithOrientations':
        """
        Assign BCC crystal orientations to blocks via the Kurdjumov-Sachs relationship.

        PAG orientations must already be set (via ``assign_pag_orientations`` on the
        parent ``FMSteel3DWithPAGs``). Each block receives one of the 24 KS variants
        from the physically appropriate packet for its habit plane.

        Parameters
        ----------
        ks_variant_selection : str, optional
            ``'random_per_block'`` — adjacency-aware greedy graph-colouring:
            adjacent blocks within a packet receive different KS variants where
            possible, with a random choice among the free variants.
            ``'deterministic'`` — always assigns KS variant index 0 from the
            packet; reproducible for geometry verification.
            Default ``'random_per_block'``.
        random_seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        FMSteel3DWithOrientations
            New instance with computed orientations; does not modify self.
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        self._emit(1, f"Assigning block orientations for {len(self.all_blocks)} blocks")
        
        from .orientation_assigner_3d import OrientationAssigner3D
        from .with_orientations_3d import FMSteel3DWithOrientations
        
        ori_assigner = OrientationAssigner3D(verbosity=self._verbosity, log_sink=self._log_sink)
        block_orientations, block_to_variant_idx = ori_assigner.assign_orientations_to_all_blocks(
            all_blocks=self.all_blocks, clusters_dict=self.clusters_dict,
            pag_orientations=self.pag_orientations,
            grain_to_plane_idx=self.grain_to_plane_idx,
            ks_variant_selection=ks_variant_selection,
            random_seed=random_seed
        )

        self._emit(
            1,
            f"Assigned orientations to {len(block_orientations)} blocks across {len(self.clusters_dict)} PAGs",
        )
        
        return FMSteel3DWithOrientations(
            parent=self, grain_orientations={},
            block_orientations=block_orientations,
            block_to_variant_idx=block_to_variant_idx,
            pag_orientations=self.pag_orientations, random_seed=random_seed,
            verbosity=self._verbosity, log_sink=self._log_sink
        )
    
    def assign_custom_block_orientations(
        self,
        euler_dict: Dict[str, Tuple[float, float, float]],
        random_seed: Optional[int] = None,
    ) -> 'FMSteel3DWithOrientations':
        """
        Bypass KS variant assignment with a user-supplied block orientation dict.

        Use this as injection point C: apply any arbitrary orientation distribution
        to the existing block morphology without re-running the KS pipeline.

        Parameters
        ----------
        euler_dict : dict
            {block_id: (phi1, Phi, phi2)} — Bunge ZXZ Euler angles in degrees.
            Extra keys not in ``all_blocks`` are ignored (warning issued).
            Missing blocks will have no orientation in the returned object.
        random_seed : int, optional
            Stored in the returned ``FMSteel3DWithOrientations`` for reproducibility.

        Returns
        -------
        FMSteel3DWithOrientations
        """
        import warnings
        from .with_orientations_3d import FMSteel3DWithOrientations

        known = set(self.all_blocks.keys())
        provided = set(euler_dict.keys())

        extra = provided - known
        if extra:
            warnings.warn(
                f"assign_custom_block_orientations: {len(extra)} key(s) in euler_dict "
                f"not in all_blocks, ignored (first 5: {sorted(extra)[:5]}).",
                UserWarning, stacklevel=2,
            )

        missing = known - provided
        if missing:
            warnings.warn(
                f"assign_custom_block_orientations: {len(missing)} block(s) have no "
                f"entry in euler_dict — they carry no orientation "
                f"(first 5: {sorted(missing)[:5]}).",
                UserWarning, stacklevel=2,
            )

        valid_dict = {k: v for k, v in euler_dict.items() if k in known}

        self._emit(
            1,
            f"Custom block orientations: {len(valid_dict)} applied, "
            f"{len(missing)} missing, {len(extra)} extra ignored.",
        )

        return FMSteel3DWithOrientations(
            parent=self,
            grain_orientations={},
            block_orientations=valid_dict,
            pag_orientations=self.pag_orientations,
            random_seed=random_seed,
            verbosity=self._verbosity,
            log_sink=self._log_sink,
        )

    def get_block_statistics(self) -> Dict:
        """Compute statistics on block structure."""
        block_sizes = [len(v) for v in self.all_blocks.values()]
        if block_sizes:
            bpp = [len(b) for b in self.grain_to_blocks_map.values()]
            return {
                'n_blocks': len(self.all_blocks),
                'min_voxels_per_block': min(block_sizes),
                'max_voxels_per_block': max(block_sizes),
                'mean_voxels_per_block': float(np.mean(block_sizes)),
                'n_packets': len(self.grain_to_blocks_map),
                'blocks_per_packet_mean': float(np.mean(bpp)) if bpp else 0.0
            }
        return {'n_blocks': 0, 'min_voxels_per_block': 0, 'max_voxels_per_block': 0,
                'mean_voxels_per_block': 0.0, 'n_packets': 0, 'blocks_per_packet_mean': 0.0}


__all__ = ['FMSteel3DWithBlocks']
