"""
FMSteel3DWithOrientations: Final state class with full FM steel hierarchy and orientations.

This module contains the complete FM steel microstructure class with all
hierarchical levels (grains → PAGs → packets → blocks) and crystal orientations
assigned to blocks via Kurdjumov-Sachs relationship.

Classes:
    FMSteel3DWithOrientations: Final complete FM steel state class.
"""

import numpy as np
from typing import Optional, Dict, List, Tuple
from itertools import permutations, product as _iproduct

from .phases_3d import (
    PHASE_MARTENSITE, PHASE_RETAINED_AUSTENITE, PHASE_NAMES,
    retained_austenite_voxels_and_orientations,
)


class FMSteel3DWithOrientations:
    """
    Complete 3D FM steel microstructure with full hierarchy and orientations.
    
    This is the final state in the FM steel generation pipeline. It contains:
    - Base grain structure
    - PAG partitioning
    - Block hierarchy within each PAG
    - BCC crystal orientations assigned to blocks
    
    This state is ready for mesh generation, visualization, or export.
    
    Attributes
    ----------
    _parent : FMSteel3DWithBlocks
        Reference to parent block structure (read-only complete structure).
    
    grain_orientations : dict[int, tuple[float, float, float]]
        Maps grain ID → BCC Euler angles.
        Format: {grain_id: (phi1_deg, Phi_deg, phi2_deg)}.
    
    block_orientations : dict[str, tuple[float, float, float]]
        Maps block_id → BCC Euler angles.
        Format: {block_id: (phi1_deg, Phi_deg, phi2_deg)}.
    
    pag_orientations : dict[int, tuple[float, float, float]]
        Maps PAG ID → FCC (parent austenite) Euler angles.
        Format: {pag_id: (phi1_deg, Phi_deg, phi2_deg)}.
    
    _random_seed : int
        Random seed used for orientation assignment.
    """
    
    __slots__ = (
        '_parent',
        'grain_orientations',
        'block_orientations',
        'block_to_variant_idx',
        'pag_orientations',
        '_random_seed',
        '_verbosity',
        '_log_sink',
    )

    def __init__(self,
                 parent,
                 grain_orientations: Dict[int, Tuple[float, float, float]],
                 block_orientations: Dict[str, Tuple[float, float, float]],
                 pag_orientations: Optional[Dict[int, Tuple[float, float, float]]] = None,
                 block_to_variant_idx: Optional[Dict[str, int]] = None,
                 random_seed: Optional[int] = None,
                 verbosity: Optional[int] = None,
                 log_sink=None):
        """
        Initialize FMSteel3DWithOrientations.

        Typically called internally by FMSteel3DWithBlocks.assign_orientations().
        Direct instantiation allowed but not recommended.

        Parameters
        ----------
        parent : FMSteel3DWithBlocks
            Parent block structure instance.
        grain_orientations : dict
            Grain BCC orientations: {grain_id: (phi1, Phi, phi2)}.
        block_orientations : dict
            Block BCC orientations: {block_id: (phi1, Phi, phi2)}.
        pag_orientations : dict, optional
            PAG FCC orientations (if not already in parent).
        block_to_variant_idx : dict, optional
            {block_id: KS variant index (0-5)} within its packet -- produced
            by OrientationAssigner3D.assign_orientations_to_all_blocks().
            Empty for orientations assigned via assign_custom_block_orientations
            (no KS variant selection took place). Default empty dict.
        random_seed : int, optional
            Random seed used for orientation assignment.
        """
        self._parent = parent
        self.grain_orientations = grain_orientations
        self.block_orientations = block_orientations
        self.block_to_variant_idx = block_to_variant_idx or {}
        self.pag_orientations = pag_orientations or parent.pag_orientations
        self._random_seed = random_seed
        self._verbosity = int(getattr(parent, '_verbosity', 0) if verbosity is None else verbosity)
        self._log_sink = getattr(parent, '_log_sink', None) if log_sink is None else log_sink

    def _emit(self, level: int, msg: str, component: str = 'SUBBLOCK') -> None:
        if self._verbosity < int(level):
            return
        text = f"[{component}][L{int(level)}] {msg}"
        if self._log_sink is not None:
            self._log_sink(text)
        else:
            print(text)
    
    # ========== Full delegation (read-only access to entire hierarchy) ==========
    
    @property
    def lgi(self) -> np.ndarray:
        """Labeled grain image from base."""
        return self._parent.lgi
    
    @property
    def grain_locs(self) -> Dict[int, np.ndarray]:
        """Grain voxel coordinates from base."""
        return self._parent.grain_locs
    
    @property
    def clusters_dict(self) -> Dict[int, List[int]]:
        """PAG clustering."""
        return self._parent.clusters_dict
    
    @property
    def all_blocks(self) -> Dict[str, np.ndarray]:
        """Block voxel data."""
        return self._parent.all_blocks
    
    @property
    def grain_to_blocks_map(self) -> Dict[int, List[str]]:
        """grain_id -> [block_id, ...] mapping (keys are grain_ids = packet ids)."""
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
    def block_slicing_normals(self) -> Dict[str, np.ndarray]:
        """block_id -> unit normal of the {111}FCC habit plane used to slice
        it, from parent block level (see FMSteel3DWithBlocks docstring)."""
        return self._parent.block_slicing_normals

    @property
    def n_grains(self) -> int:
        """Total grains."""
        return self._parent.n_grains
    
    @property
    def n_blocks(self) -> int:
        """Total blocks."""
        return len(self.all_blocks)
    
    @property
    def n_pags(self) -> int:
        """Total PAGs."""
        return len(self.clusters_dict)
    
    @property
    def physical_dimensions(self):
        """Physical domain size."""
        return self._parent.physical_dimensions

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

    # ========== Analysis & statistics ==========
    
    def get_full_hierarchy_statistics(self) -> Dict[str, any]:
        """
        Compute comprehensive statistics across all hierarchy levels.
        
        Returns
        -------
        dict
            Keys include:
            - n_grains, grain_voxel_stats
            - n_pags, pags_grains_stats
            - n_blocks, block_voxel_stats
            - n_isolated_grains
            - total_voxels_in_fm_structure
        """
        stats = {
            'n_grains': self.n_grains,
            'n_pags': self.n_pags,
            'n_blocks': self.n_blocks,
            'n_block_orientations_assigned': len(self.block_orientations),
            'n_pag_orientations_assigned': len(self.pag_orientations)
        }
        if self.n_pags > 0:
            grains_per_pag = [len(g) for g in self.clusters_dict.values()]
            stats['mean_grains_per_pag'] = float(np.mean(grains_per_pag))
            blocks_per_pag = []
            for pid in self.clusters_dict.keys():
                n_b = sum(1 for bid in self.all_blocks.keys()
                         if int(bid.split('_')[1]) == pid)
                blocks_per_pag.append(n_b)
            stats['mean_blocks_per_pag'] = float(np.mean(blocks_per_pag)) if blocks_per_pag else 0.0
        return stats
    
    def compute_misorientation_distribution(self,
                                             n_sample_pairs: int = 5000,
                                             random_seed: Optional[int] = None
                                             ) -> Dict[str, np.ndarray]:
        """
        Compute cubic-symmetry-reduced misorientation angles between block pairs.

        Samples n_sample_pairs pairs in each category (within-PAG and
        across-PAG) and returns the disorientation angle (minimum over
        all 24 cubic symmetry operators).

        Within-PAG distribution should show peaks at the KS-predicted angles
        (~10.5°, ~14.9°, ~20.6°, ~47.1°, ~60°).  Across-PAG distribution
        should be roughly uniform (no preferred angle).

        Parameters
        ----------
        n_sample_pairs : int, optional
            Target number of sampled pairs per category.  Default 5000.
        random_seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        dict
            'within_pag' : ndarray of disorientation angles in degrees
            'across_pag' : ndarray of disorientation angles in degrees
        """
        from .orientation_assigner_3d import OrientationAssigner3D

        if random_seed is not None:
            np.random.seed(random_seed)

        # 24 proper cubic symmetry operators (det = +1, permutation × sign matrices)
        ops = []
        for p in permutations(range(3)):
            P = np.eye(3)[list(p)]
            for signs in _iproduct([-1, 1], repeat=3):
                S = P * np.array(signs)
                if abs(np.linalg.det(S) - 1) < 1e-6:
                    ops.append(S)
        sym_ops = np.array(ops)   # (24, 3, 3)

        assigner = OrientationAssigner3D()
        block_ids = list(self.block_orientations.keys())
        n_blocks  = len(block_ids)

        # Pre-compute rotation matrices for all blocks  (n_blocks, 3, 3)
        R_arr = np.empty((n_blocks, 3, 3))
        for k, bid in enumerate(block_ids):
            ea  = self.block_orientations[bid]
            R   = assigner.cubic_euler_bunge_to_matrix_v1(
                      np.array([ea[0]]), np.array([ea[1]]), np.array([ea[2]]),
                      degrees=True)
            R_arr[k] = R[0] if R.ndim == 3 else R

        # PAG membership from block name  B_{pag_id}_{grain_id}_{local}
        pag_arr = np.array([int(bid.split('_')[1]) for bid in block_ids])

        # Index blocks by PAG for within-PAG sampling
        pag_to_idx: Dict[int, List[int]] = {}
        for k, pid in enumerate(pag_arr):
            pag_to_idx.setdefault(int(pid), []).append(k)
        multi_pags = [(pid, np.array(idxs))
                      for pid, idxs in pag_to_idx.items() if len(idxs) >= 2]

        sym_ops_T = sym_ops.transpose(0, 2, 1)   # (24, 3, 3) — precomputed transposes

        def _disori(i: int, j: int) -> float:
            dR     = R_arr[i].T @ R_arr[j]
            # Full cubic-cubic bicrystal symmetry:
            #   H1 @ dR @ H2.T  for all (H1, H2) in G × G  →  576 combinations
            # Step 1: apply all left operators  →  (24, 3, 3)
            M1     = sym_ops @ dR
            # Step 2: apply all right operators  →  (24, 24, 3, 3)
            M_all  = M1[:, None] @ sym_ops_T[None, :]
            traces = M_all[:, :, 0, 0] + M_all[:, :, 1, 1] + M_all[:, :, 2, 2]
            # Maximum trace  ↔  minimum rotation angle (arccos is decreasing)
            return float(np.degrees(
                np.arccos(np.clip((traces.max() - 1) / 2, -1.0, 1.0))))

        # Within-PAG pairs (sample with replacement across PAGs)
        within_angles: List[float] = []
        if multi_pags:
            for _ in range(n_sample_pairs):
                pid, idxs = multi_pags[np.random.randint(len(multi_pags))]
                i, j = np.random.choice(idxs, 2, replace=False)
                within_angles.append(_disori(int(i), int(j)))

        # Across-PAG pairs (reject same-PAG draws)
        across_angles: List[float] = []
        budget = n_sample_pairs * 4   # ~4× oversampling covers rejection overhead
        for _ in range(budget):
            if len(across_angles) >= n_sample_pairs:
                break
            i, j = np.random.choice(n_blocks, 2, replace=False)
            if pag_arr[i] != pag_arr[j]:
                across_angles.append(_disori(int(i), int(j)))

        return {
            'within_pag': np.array(within_angles),
            'across_pag': np.array(across_angles),
        }

    def get_ks_variant_statistics(self) -> Dict:
        """How evenly the 6 KS variants within each packet were actually used.

        For every (pag_id, plane_idx) packet with 2+ blocks, tallies how many
        blocks received each variant index that was actually assigned, then
        summarises that per-packet distribution with the same
        size_balance_metrics (cv, min_max_ratio, gini) used elsewhere for
        packet_size_balance -- perfectly even usage across whichever variants
        were used gives cv=0, min_max_ratio=1, gini=0; a packet where every
        block ended up sharing one variant (the worst case the adjacency-
        aware graph-colouring in assign_orientations_to_all_blocks tries to
        avoid) sits at the opposite extreme.

        Requires block_to_variant_idx, which is only populated when
        orientations were assigned via assign_orientations_to_all_blocks
        (i.e. FMSteel3DWithBlocks.assign_orientations()) -- empty if custom
        block orientations were injected instead.
        """
        empty = {'n_packets_with_variants': 0,
                 'cv': {'min': 0.0, 'max': 0.0, 'mean': 0.0, 'median': 0.0},
                 'min_max_ratio': {'min': 0.0, 'max': 0.0, 'mean': 0.0, 'median': 0.0},
                 'gini': {'min': 0.0, 'max': 0.0, 'mean': 0.0, 'median': 0.0}}
        if not self.block_to_variant_idx:
            return empty

        from upxo.pxtalops.grain_splitting_3d import size_balance_metrics

        grain_to_plane_idx = self.grain_to_plane_idx
        packet_variant_counts: Dict[Tuple[int, int], Dict[int, int]] = {}
        for block_id, v_idx in self.block_to_variant_idx.items():
            parts = block_id.split('_')
            pag_id = int(parts[1])
            gid = int(parts[2])
            plane_idx = grain_to_plane_idx.get(gid)
            if plane_idx is None:
                continue
            counts = packet_variant_counts.setdefault((pag_id, plane_idx), {})
            counts[v_idx] = counts.get(v_idx, 0) + 1

        cvs, ratios, ginis = [], [], []
        for counts in packet_variant_counts.values():
            sizes = list(counts.values())
            if len(sizes) < 2:
                continue
            m = size_balance_metrics(sizes)
            cvs.append(m['cv'])
            ratios.append(m['min_max_ratio'])
            ginis.append(m['gini'])

        def _agg(vals):
            if not vals:
                return {'min': 0.0, 'max': 0.0, 'mean': 0.0, 'median': 0.0}
            return {'min': float(np.min(vals)), 'max': float(np.max(vals)),
                   'mean': float(np.mean(vals)), 'median': float(np.median(vals))}

        return {
            'n_packets_with_variants': len(packet_variant_counts),
            'cv': _agg(cvs),
            'min_max_ratio': _agg(ratios),
            'gini': _agg(ginis),
        }

    def available_phases(self) -> List[int]:
        """Phase ids actually present in this structure, for populating a
        phase-selector dropdown (e.g. the Block-Level IPF Map panel).

        PHASE_MARTENSITE is present whenever any blocks exist; PHASE_
        RETAINED_AUSTENITE is present whenever isolated_grains is non-empty
        (the flattened view covering both PAG-covered and leftover-isolated
        retained-austenite grains -- see FMSteel3DWithPAGs docstring).
        """
        phases = []
        if self.all_blocks:
            phases.append(PHASE_MARTENSITE)
        if self.isolated_grains:
            phases.append(PHASE_RETAINED_AUSTENITE)
        return phases

    def get_phase_voxels_and_orientations(
        self, phase_id: int
    ) -> Tuple[Dict, Dict[int, Tuple[float, float, float]]]:
        """Feature voxels + orientations for one phase, for phase-filtered
        IPF maps (see viz/orientation_viz_3d.py and gui/pages_viz.py).

        PHASE_MARTENSITE -> block granularity: (all_blocks, block_orientations),
        exactly what the existing Block-Level IPF Map already renders (blocks
        only ever exist for transformed PAGs, so no filtering is needed).

        PHASE_RETAINED_AUSTENITE -> grain granularity (retained austenite is
        never split into packets/blocks): every retained-austenite grain
        keyed by grain_id, orientation from either its retained PAG or its
        own isolated_grain_orientations entry.

        Returns
        -------
        (features, orientations) : (dict, dict)
            features:     {feature_id: (n_voxels, 3) voxel coordinate array}
            orientations: {feature_id: (phi1, Phi, phi2) Bunge-Euler degrees}
        """
        if phase_id == PHASE_MARTENSITE:
            return dict(self.all_blocks), dict(self.block_orientations)
        if phase_id == PHASE_RETAINED_AUSTENITE:
            return retained_austenite_voxels_and_orientations(self)
        raise ValueError(
            f"Unknown phase_id={phase_id}. Available phases for this "
            f"structure: {self.available_phases()} "
            f"({[PHASE_NAMES.get(p) for p in self.available_phases()]})."
        )

    def get_misorientation_statistics(self) -> Dict[str, any]:
        """
        Compute grain-grain and block-block misorientation statistics.
        
        Uses cubic_misorientation_old1 (from parent class) to compute
        misorientation angles between neighboring grains/blocks.
        
        Returns
        -------
        dict
            Keys: 'grain_grain_misori', 'block_block_misori',
            (each is dict of statistics: min, max, mean, etc.)
        """
        from .orientation_assigner_3d import OrientationAssigner3D
        ori_assigner = OrientationAssigner3D()
        
        if len(self.block_orientations) >= 2:
            block_ids = list(self.block_orientations.keys())
            n_samples = min(10, len(block_ids) * (len(block_ids) - 1) // 2)
            block_misoris = []
            for _ in range(n_samples):
                bid1, bid2 = np.random.choice(block_ids, 2, replace=False)
                angle, _, _ = ori_assigner.cubic_misorientation_old1(
                    self.block_orientations[bid1], self.block_orientations[bid2], degrees=True)
                block_misoris.append(angle)
            return {
                'n_block_pairs_sampled': n_samples,
                'mean_block_misorientation_deg': float(np.mean(block_misoris)),
                'std_block_misorientation_deg': float(np.std(block_misoris)),
                'block_misorientations': block_misoris
            }
        return {}
    
    def build_euler_angle_3d_maps(
        self,
        level: str = 'block',
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build dense 3D Euler angle maps (phi1, Phi, phi2) over the full RVE.

        Each voxel receives the Bunge ZXZ Euler angles (degrees) of its parent
        block.  Voxels not covered by any block (PAG boundaries, isolated grains)
        retain the zero initialisation.

        Parameters
        ----------
        level : str, optional
            Hierarchy level to draw orientations from.  Currently only ``'block'``
            is supported.  Default ``'block'``.

        Returns
        -------
        tuple of (phi1_3d, Phi_3d, phi2_3d)
            Three float32 arrays, each shaped like ``self.lgi``.
            Angles are in degrees (Bunge ZXZ convention).
        """
        if level != 'block':
            raise ValueError(
                f"level={level!r} is not supported; only 'block' is available in "
                "FMSteel3DWithOrientations.  Use FMSteel3DWithSubBlocks for 'subblock'."
            )
        shape = self.lgi.shape
        phi1_3d = np.zeros(shape, dtype=np.float32)
        Phi_3d  = np.zeros(shape, dtype=np.float32)
        phi2_3d = np.zeros(shape, dtype=np.float32)

        for block_id, vox in self.all_blocks.items():
            ea = self.block_orientations.get(block_id)
            if ea is None or len(vox) == 0:
                continue
            # vox columns: (z, y, x) — from np.argwhere convention in base_3d
            phi1_3d[vox[:, 0], vox[:, 1], vox[:, 2]] = float(ea[0])
            Phi_3d[vox[:, 0], vox[:, 1], vox[:, 2]]  = float(ea[1])
            phi2_3d[vox[:, 0], vox[:, 1], vox[:, 2]] = float(ea[2])

        return phi1_3d, Phi_3d, phi2_3d

    # ========== Visualization orchestration ==========
    
    def plot_gs_pvvox(self, alpha: float = 1.0, title: str = 'FM Steel 3D', **kwargs):
        """Visualize grain structure (PyVista voxels)."""
        from .viz.grain_structure_viz_3d import GrainStructureViz3D
        GrainStructureViz3D().plot_gs_pvvox(
            lgi=self.lgi, grain_locs=self.grain_locs, alpha=alpha, title=title, **kwargs)
    
    def plot_pag_map_pyvista(self, gids_to_plot=None, **kwargs):
        """Visualize PAG map."""
        from .viz.grain_structure_viz_3d import GrainStructureViz3D
        GrainStructureViz3D().plot_pag_map_pyvista(
            clusters_dict=self.clusters_dict, grain_locs=self.grain_locs,
            voxel_size=self._parent.voxel_size, **kwargs)
    
    def plot_ipf_map_pyvista_v2(self, gids_to_plot=None, **kwargs):
        """Visualize IPF map."""
        from .viz.orientation_viz_3d import OrientationViz3D
        if self.grain_orientations:
            OrientationViz3D().plot_ipf_map_pyvista_v2(
                grain_locs=self.grain_locs, grain_orientations=self.grain_orientations, **kwargs)
    
    def visualize_block_morphology(self, **kwargs):
        """Visualize all blocks with random colors."""
        from .viz.grain_structure_viz_3d import GrainStructureViz3D
        GrainStructureViz3D().visualize_block_morphology(
            blocks_dict=self.all_blocks, **kwargs)
    
    def visualize_block_morphology_v1(self, **kwargs):
        """Visualize all blocks with color bar."""
        from .viz.grain_structure_viz_3d import GrainStructureViz3D
        GrainStructureViz3D().visualize_block_morphology_v1(
            blocks_dict=self.all_blocks, **kwargs)
    
    def visualize_block_ipf_map(self, **kwargs):
        """Visualize blocks colored by IPF."""
        from .viz.orientation_viz_3d import OrientationViz3D
        OrientationViz3D().visualize_block_ipf_map(
            all_blocks=self.all_blocks, block_orientations=self.block_orientations, **kwargs)
    
    def plot_pag_ks_verification_v1(self, pag_id: int, **kwargs):
        """Verify KS relationship for a PAG."""
        from .viz.orientation_viz_3d import OrientationViz3D
        grain_ids = self.clusters_dict.get(pag_id, [])
        if grain_ids:
            OrientationViz3D().plot_pag_ks_verification_v1(
                pag_id=pag_id, pag_orientation=self.pag_orientations.get(pag_id, (0,0,0)),
                grain_ids=grain_ids, grain_orientations=self.grain_orientations, **kwargs)
    
    def plot_distribution(self, data: np.ndarray, title: str, xlabel: str, **kwargs):
        """Plot histogram."""
        from .viz.data_viz_3d import DataViz3D
        DataViz3D().plot_distribution(data=data, title=title, xlabel=xlabel, **kwargs)
    
    # ========== Pipeline continuation ==========

    def generate_subblocks(
        self,
        subblock_thickness_range_um: Tuple[float, float] = (0.5, 1.5),
        intrablock_ori_spread_deg: float = 2.0,
        thin_block_strategy: str = 'skip',
        random_seed: Optional[int] = None,
        subblock_slab_connectivity: int = 26,
    ) -> 'FMSteel3DWithSubBlocks':
        """
        Subdivide every block into sub-blocks (laths) with per-sub-block orientations.

        Sub-block slicing reuses each block's inherited slicing normal, so laths
        are parallel to the parent block's habit plane. Orientations are the parent
        block orientation plus a small random axis-angle perturbation within
        ±intrablock_ori_spread_deg.

        Parameters
        ----------
        subblock_thickness_range_um : tuple of (float, float)
            (min, max) sub-block thickness in the same physical units as LX/LY/LZ
            (typically µm). Converted to voxels using the stored voxel_size.
        intrablock_ori_spread_deg : float, optional
            Half-width of orientation spread around parent block (degrees). Default 2.0.
        thin_block_strategy : str, optional
            Policy for blocks too thin to subdivide: 'skip' keeps the block whole.
            Default 'skip'.
        random_seed : int, optional
            Seed for reproducibility.

        Returns
        -------
        FMSteel3DWithSubBlocks
        """
        from .subblock_generator_3d import SubBlockGenerator3D
        from .orientation_assigner_3d import OrientationAssigner3D
        from .with_subblocks_3d import FMSteel3DWithSubBlocks

        voxel_size = self._parent.voxel_size
        t_lo_vox = subblock_thickness_range_um[0] / voxel_size
        t_hi_vox = subblock_thickness_range_um[1] / voxel_size

        self._emit(
            1,
            f"Generating sub-blocks for {len(self.all_blocks)} blocks (thickness={subblock_thickness_range_um[0]:.3f}-{subblock_thickness_range_um[1]:.3f} um)",
        )
        self._emit(
            2,
            f"thickness in voxels={t_lo_vox:.3f}-{t_hi_vox:.3f}, spread={intrablock_ori_spread_deg:.3f} deg, strategy={thin_block_strategy}",
        )

        sb_gen = SubBlockGenerator3D()
        all_subblocks, block_to_subblocks_map, subblock_slicing_normals = \
            sb_gen.generate_subblocks_for_all_blocks(
                all_blocks=self.all_blocks,
                block_slicing_normals=self._parent.block_slicing_normals,
                subblock_thickness_range=(t_lo_vox, t_hi_vox),
                thin_block_strategy=thin_block_strategy,
                random_seed=random_seed,
                slab_connectivity=subblock_slab_connectivity,
            )

        ori_assigner = OrientationAssigner3D()
        subblock_orientations = ori_assigner.assign_subblock_orientations(
            all_subblocks=all_subblocks,
            block_orientations=self.block_orientations,
            intrablock_ori_spread_deg=intrablock_ori_spread_deg,
            random_seed=random_seed,
        )

        if all_subblocks:
            sizes = [len(v) for v in all_subblocks.values()]
            n_per_block = [len(v) for v in block_to_subblocks_map.values() if v]
            self._emit(1, f"Generated {len(all_subblocks)} sub-blocks")
            self._emit(
                2,
                f"voxels/sub-block min={min(sizes)}, max={max(sizes)}, mean={np.mean(sizes):.1f}; sub-blocks/block mean={np.mean(n_per_block) if n_per_block else 0.0:.1f}",
            )
        else:
            self._emit(1, "Generated 0 sub-blocks")

        return FMSteel3DWithSubBlocks(
            parent=self,
            all_subblocks=all_subblocks,
            block_to_subblocks_map=block_to_subblocks_map,
            subblock_orientations=subblock_orientations,
            subblock_slicing_normals=subblock_slicing_normals,
            random_seed=random_seed,
            verbosity=self._verbosity,
            log_sink=self._log_sink,
        )

    # ========== Export/serialization ==========
    
    def to_dict(self) -> Dict:
        """Serialize full hierarchy to dict."""
        return {
            'block_orientations': self.block_orientations,
            'pag_orientations': self.pag_orientations,
            'grain_orientations': self.grain_orientations,
            'n_grains': self.n_grains,
            'n_pags': self.n_pags,
            'n_blocks': self.n_blocks,
        }


__all__ = ['FMSteel3DWithOrientations']
