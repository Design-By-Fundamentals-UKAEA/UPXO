"""
FMSteel3DWithPAGs: State class representing grain structure partitioned into PAGs.

This module contains the class for FM steel structures after PAG clustering.
It adds hierarchical structure (PAGs and their contained grains) to the base
grain structure.

Classes:
    FMSteel3DWithPAGs: Base + PAGs state class.
"""

import numpy as np
from typing import Optional, Dict, List, Tuple



class FMSteel3DWithPAGs:
    """
    FM steel grain structure with PAG (Prior Austenite Grain) hierarchy.
    
    Holds the base grain structure plus the computed PAG clustering.
    PAGs are groups of grains that will be further subdivided into packets
    and then blocks.
    
    This is an intermediate state in the pipeline. It is created by calling
    generate_pag_clusters() on a FMSteel3DBase instance, and transitions to
    FMSteel3DWithBlocks by calling generate_blocks().
    
    Attributes
    ----------
    _parent : FMSteel3DBase
        Reference to parent grain structure (read-only base state).
    
    clusters_dict : dict[int, list[int]]
        Maps PAG ID → list of grain IDs contained in that PAG.
    
    neigh_clid : dict[int, list[int]]
        Maps PAG ID → list of neighboring PAG IDs.
    
    pag_orientations : dict[int, tuple[float, float, float]]
        Will store parent FCC (austenite) orientations for each PAG.
        Filled by assign_pag_orientations() or by downstream methods.
        Format: {pag_id: (phi1_deg, Phi_deg, phi2_deg)}.
    
    isolated_grains : set[int]
        Grain IDs that do NOT participate in PAG clustering
        (controlled by pag_grain_fraction parameter).

    grain_to_pag_id : dict[int, int]
        Reverse lookup: grain_id -> pag_id.
        Derived from clusters_dict at construction time.

    grain_to_local_pkt_idx : dict[int, int]
        Maps grain_id -> 1-based ordinal of that grain within its PAG's
        grain list (i.e. its local packet index).  Ordinal follows the
        insertion order of clusters_dict[pag_id], which is determined by
        the clustering algorithm and is stable across the pipeline.
        Used to construct canonical elset names es.pck.{pag}.{pkt}.

    _random_seed : int
        Random seed used for this PAG generation.
    """
    
    __slots__ = (
        '_parent',
        'clusters_dict',
        'neigh_clid',
        'pag_orientations',
        'isolated_grains',
        '_random_seed',
        'grain_to_pag_id',
        'grain_to_local_pkt_idx',
        '_verbosity',
        '_log_sink',
    )
    
    def __init__(self,
                 parent,
                 clusters_dict: Dict[int, List[int]],
                 neigh_clid: Dict[int, List[int]],
                 pag_orientations: Optional[Dict[int, Tuple[float, float, float]]] = None,
                 isolated_grains: Optional[set] = None,
                 random_seed: Optional[int] = None,
                 verbosity: Optional[int] = None,
                 log_sink=None):
        """
        Initialize FMSteel3DWithPAGs.
        
        Typically called internally by FMSteel3DBase.generate_pag_clusters().
        Direct instantiation allowed but not recommended.
        
        Parameters
        ----------
        parent : FMSteel3DBase
            Parent grain structure instance.
        clusters_dict : dict
            PAG clustering: {pag_id: [grain_id, grain_id, ...]}.
        neigh_clid : dict
            PAG adjacency: {pag_id: [neighbor_pag_id, ...]}.
        isolated_grains : set, optional
            Grain IDs not included in PAGs. Default empty set.
        pag_orientations : dict, optional
            Pre-computed PAG orientations. Default empty dict.
        random_seed : int, optional
            Random seed used for this instantiation.
        """
        self._parent = parent
        self.clusters_dict = clusters_dict
        self.neigh_clid = neigh_clid
        self.isolated_grains = isolated_grains or set()
        self.pag_orientations = pag_orientations or {}
        self._random_seed = random_seed
        self._verbosity = int(getattr(parent, '_verbosity', 0) if verbosity is None else verbosity)
        self._log_sink = getattr(parent, '_log_sink', None) if log_sink is None else log_sink

        self.grain_to_pag_id: Dict[int, int] = {
            gid: pag_id
            for pag_id, gids in clusters_dict.items()
            for gid in gids
        }
        self.grain_to_local_pkt_idx: Dict[int, int] = {
            gid: idx
            for pag_id, gids in clusters_dict.items()
            for idx, gid in enumerate(gids, start=1)
        }

    def _emit(self, level: int, msg: str, component: str = 'PAG') -> None:
        if self._verbosity < int(level):
            return
        text = f"[{component}][L{int(level)}] {msg}"
        if self._log_sink is not None:
            self._log_sink(text)
        else:
            print(text)
    
    # ========== Delegation properties (read-only access to parent) ==========
    
    @property
    def lgi(self) -> np.ndarray:
        """Labeled grain image from parent."""
        return self._parent.lgi
    
    @property
    def grain_locs(self) -> Dict[int, np.ndarray]:
        """Grain voxel coordinates from parent."""
        return self._parent.grain_locs
    
    @property
    def neigh_gid(self) -> Dict[int, List[int]]:
        """Grain neighbor relationships from parent."""
        return self._parent.neigh_gid
    
    @property
    def n_grains(self) -> int:
        """Total grains in parent structure."""
        return self._parent.n_grains
    
    @property
    def physical_dimensions(self):
        """Physical domain size from parent."""
        return self._parent.physical_dimensions
    
    @property
    def voxel_size(self) -> float:
        """Voxel size from parent."""
        return self._parent.voxel_size

    @property
    def units(self) -> str:
        """Physical unit string ('microns', 'mm', 'm') from base grain structure."""
        return self._parent.units
    
    # ========== Pipeline continuation ==========
    
    def generate_blocks(self,
                        block_thickness_range: Tuple[float, float] = (2.0, 5.0),
                        pag_ori_mode: str = 'random',
                        pag_ori_params: Optional[Dict] = None,
                        random_seed: Optional[int] = None) -> 'FMSteel3DWithBlocks':
        """
        Slice each PAG into martensitic blocks.

        For each PAG, uses crystallographic slicing planes to partition the PAG's
        grains into multiple blocks. Returns a new FMSteel3DWithBlocks instance.

        Parameters
        ----------
        block_thickness_range : tuple of (float, float), optional
            (lower, upper) bounds for block thickness in physical units.
            Each packet independently draws its thickness uniformly from this
            range, producing natural lath-width variation across the structure.
            Default (2.0, 5.0).

        pag_ori_mode : str, optional
            How to assign parent FCC orientations to PAGs if not already set.
            Options:

            * ``'random'`` — uniform random SO(3) orientations (default).
            * ``'hagb_constrained'`` — random SO(3) with HAGB constraint:
              each PAG orientation is rejected if its misorientation with any
              already-assigned neighbouring PAG is below ``hagb_threshold``
              (degrees, default 15°).  Pass extra keys in ``pag_ori_params``:
              ``orientation_pool`` (list of (phi1,Phi,phi2) tuples, or None for
              on-the-fly random draws), ``hagb_threshold`` (float, default 15.0),
              ``max_attempts`` (int, default 1000).
            * ``'fixed'`` — all PAGs get the same fixed orientation; pass
              ``euler_angles=(phi1,Phi,phi2)`` in ``pag_ori_params``.
            * ``'explicit'`` — orientations must be pre-loaded in
              ``self.pag_orientations`` before calling this method.

            Default ``'random'``.

        pag_ori_params : dict, optional
            Parameters for pag_ori_mode (e.g., texture weights).

        random_seed : int, optional
            Random seed for block slicing.

        Returns
        -------
        FMSteel3DWithBlocks
            New instance with computed block hierarchy.

        Notes
        -----
        Does not modify self; returns new instance.
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        self._emit(
            1,
            f"Generating blocks (thickness={block_thickness_range[0]:.3f}-{block_thickness_range[1]:.3f} vox)",
            component='BLOCK',
        )
        
        if not self.pag_orientations:
            if pag_ori_mode == 'random':
                # Phi sampled via arccos(1 - 2U) so cos(Phi) is uniform on [-1,1],
                # giving the correct sin(Phi) area element for SO(3).
                self.pag_orientations = {
                    pag_id: (np.random.uniform(0, 360),
                             float(np.degrees(np.arccos(1.0 - 2.0 * np.random.uniform()))),
                             np.random.uniform(0, 360))
                    for pag_id in self.clusters_dict.keys()
                }
            elif pag_ori_mode == 'hagb_constrained':
                from .orientation_assigner_3d import OrientationAssigner3D
                _ori = OrientationAssigner3D(verbosity=self._verbosity)
                _p = pag_ori_params or {}
                self.pag_orientations = _ori.assign_pag_orientations_with_hagb(
                    pag_ids=list(self.clusters_dict.keys()),
                    pag_neigh_map=self.neigh_clid,
                    orientation_pool=_p.get('orientation_pool', None),
                    hagb_threshold=float(_p.get('hagb_threshold', 15.0)),
                    random_seed=random_seed,
                    max_attempts=int(_p.get('max_attempts', 1000)),
                )
            elif pag_ori_mode == 'fixed':
                default_ori = (pag_ori_params.get('euler_angles', (0, 0, 0))
                               if pag_ori_params else (0, 0, 0))
                self.pag_orientations = {pid: default_ori for pid in self.clusters_dict.keys()}
        
        from .block_generator_3d import BlockGenerator3D
        from .with_blocks_3d import FMSteel3DWithBlocks
        
        block_gen = BlockGenerator3D()
        all_blocks, grain_to_blocks_map, grain_to_plane_idx, block_slicing_normals = \
            block_gen.generate_blocks_for_all_pags(
                clusters_dict=self.clusters_dict, grain_locs=self.grain_locs,
                pag_orientations=self.pag_orientations,
                block_thickness_range=block_thickness_range,
                random_seed=random_seed
            )

        if all_blocks:
            block_sizes = [len(v) for v in all_blocks.values()]
            self._emit(
                1,
                f"Generated {len(all_blocks)} blocks across {len(grain_to_blocks_map)} packets",
                component='BLOCK',
            )
            self._emit(
                2,
                f"voxels/block min={min(block_sizes)}, max={max(block_sizes)}, mean={np.mean(block_sizes):.1f}",
                component='BLOCK',
            )
        else:
            self._emit(1, "Generated 0 blocks", component='BLOCK')

        return FMSteel3DWithBlocks(parent=self, all_blocks=all_blocks,
                                   grain_to_blocks_map=grain_to_blocks_map,
                                   grain_to_plane_idx=grain_to_plane_idx,
                                   block_slicing_normals=block_slicing_normals,
                                   slicing_planes={}, random_seed=random_seed,
                                   verbosity=self._verbosity, log_sink=self._log_sink)
    
    def get_pag_statistics(self) -> Dict:
        """Compute statistics on PAG structure."""
        grains_per_pag = [len(g) for g in self.clusters_dict.values()]
        if grains_per_pag:
            return {
                'n_pags': len(self.clusters_dict),
                'min_grains_per_pag': min(grains_per_pag),
                'max_grains_per_pag': max(grains_per_pag),
                'mean_grains_per_pag': float(np.mean(grains_per_pag)),
                'n_isolated_grains': len(self.isolated_grains),
                'total_clustered_grains': sum(grains_per_pag)
            }
        return {'n_pags': 0, 'min_grains_per_pag': 0, 'max_grains_per_pag': 0,
                'mean_grains_per_pag': 0.0, 'n_isolated_grains': len(self.isolated_grains),
                'total_clustered_grains': 0}

    @property
    def n_pags(self) -> int:
        """Number of PAGs."""
        return len(self.clusters_dict)


__all__ = ['FMSteel3DWithPAGs']
