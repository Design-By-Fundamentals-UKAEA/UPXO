"""
FMSteel3DBase Implementation: Grain detection, neighbor graphs, PAG clustering.

Phase 1 working implementation of the base grain structure class.
Entry point for the microstructure generation pipeline.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import cc3d
from scipy import ndimage
import networkx as nx


@dataclass
class PhysicalDimensions:
    """Container for physical domain dimensions (Lx, Ly, Lz)."""
    Lx: float
    Ly: float
    Lz: float
    
    @classmethod
    def from_tuple(cls, dims_tuple: Tuple[float, float, float]):
        """Create from (Lx, Ly, Lz) tuple."""
        return cls(Lx=dims_tuple[0], Ly=dims_tuple[1], Lz=dims_tuple[2])
    
    def as_array(self) -> np.ndarray:
        """Return as (Lx, Ly, Lz) array."""
        return np.array([self.Lx, self.Ly, self.Lz])


class FMSteel3DBase:
    """
    Foundation FM steel grain structure from labeled grain image (LFI).
    
    Entry point to the pipeline. Detects grains, builds neighbor graph,
    optionally cleans small grains, then enables PAG clustering.
    """
    
    __slots__ = ('lgi', 'physical_dimensions', 'voxel_size', 'units',
                 'connectivity', 'grain_locs', 'n_grains', 'neigh_gid',
                 'min_grain_nvoxels', '_random_seed', '_verbosity', '_log_sink')

    _VALID_UNITS = {'microns', 'mm', 'm'}
    
    def __init__(self,
                 lgi: np.ndarray,
                 physical_dimensions: PhysicalDimensions,
                 voxel_size: float,
                 units: str = 'microns',
                 connectivity: int = 6,
                 grain_locs: Optional[Dict[int, np.ndarray]] = None,
                 n_grains: int = 0,
                 neigh_gid: Optional[Dict[int, List[int]]] = None,
                 min_grain_nvoxels: int = -1,
                 random_seed: Optional[int] = None,
                 verbosity: int = 0,
                 log_sink=None):
        """
        Initialize FMSteel3DBase from component data.
        
        Typically called internally by from_lfi classmethod. Direct instantiation
        is allowed but not recommended.
        
        Parameters
        ----------
        lgi : np.ndarray
            3D labeled grain image.
        physical_dimensions : PhysicalDimensions
            Physical domain dimensions (Lx, Ly, Lz).
        voxel_size : float
            Size of each voxel.
        connectivity : int, optional
            Grain connectivity (6, 18, 26). Default is 6 (face connectivity).
        grain_locs : dict, optional
            Pre-computed grain locations. If None, computed on demand.
        n_grains : int, optional
            Number of grains. If 0, computed from lgi.max().
        neigh_gid : dict, optional
            Pre-computed neighbor relationships. If None, computed on demand.
        min_grain_nvoxels : int, optional
            Minimum grain size. Default -1 (no cleanup).
        random_seed : int, optional
            Random seed for reproducibility.
        """
        if units not in self._VALID_UNITS:
            raise ValueError(
                f"Invalid units '{units}'. Valid: {self._VALID_UNITS}"
            )
        self.lgi = lgi.astype(np.int32) if lgi.dtype != np.int32 else lgi
        self.physical_dimensions = physical_dimensions
        self.voxel_size = float(voxel_size)
        self.units = units
        self.connectivity = int(connectivity)
        self.grain_locs = grain_locs or {}
        self.n_grains = int(n_grains)
        self.neigh_gid = neigh_gid or {}
        self.min_grain_nvoxels = int(min_grain_nvoxels)
        self._random_seed = random_seed
        self._verbosity = int(verbosity)
        self._log_sink = log_sink

    def _emit(self, level: int, msg: str, component: str = 'BASE') -> None:
        """Emit a structured message if verbosity allows."""
        if self._verbosity < int(level):
            return
        text = f"[{component}][L{int(level)}] {msg}"
        if self._log_sink is not None:
            self._log_sink(text)
        else:
            print(text)
    
    @classmethod
    def from_lfi(cls,
                 lfi: np.ndarray,
                 physical_dimensions: Tuple[float, float, float],
                 voxel_size: float = 1.0,
                 units: str = 'microns',
                 connectivity: int = 6,
                 min_grain_nvoxels: int = -1,
                 random_seed: Optional[int] = None,
                 verbosity: int = 0,
                 log_sink=None) -> 'FMSteel3DBase':
        """
        Create FMSteel3DBase instance from a labeled grain image.
        
        This is the primary factory method. It initializes grain locations,
        computes neighbor relationships, and optionally performs cleanup
        (dissolution of small grains) if min_grain_nvoxels >= 0.
        
        Parameters
        ----------
        lfi : np.ndarray
            3D labeled grain image. Shape (nx, ny, nz). Values are grain IDs
            (1, 2, ..., n_grains).
        physical_dimensions : tuple[float, float, float]
            Physical domain size as (Lx, Ly, Lz).
        voxel_size : float, optional
            Size of each voxel. Default 1.0. Must be positive.
        connectivity : int, optional
            Grain connectivity (6, 18, or 26). Default 6 (face connectivity only).
            Values: 6 = face, 18 = face+edge, 26 = face+edge+corner.
        min_grain_nvoxels : int, optional
            Minimum grain size in voxels. If >= 0, triggers cleanup via
            clean_gs_GMD_by_source_erosion_v1. Grains below threshold are
            dissolved into neighboring larger grains. Default -1 (no cleanup).
        random_seed : int, optional
            Random seed for reproducibility. If provided, numpy random state
            is set at initialization.
        
        Returns
        -------
        FMSteel3DBase
            Fully initialized grain structure instance.
        
        Raises
        ------
        ValueError
            If lfi is not 3D, connectivity is invalid, or physical_dimensions are invalid.
        
        Examples
        --------
        >>> lfi = np.random.randint(1, 100, size=(50, 50, 50))
        >>> fm = FMSteel3DBase.from_lfi(lfi, physical_dimensions=(100.0, 100.0, 100.0))
        >>> print(fm.n_grains)
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        lfi = np.asarray(lfi, dtype=np.int32)
        phys_dims = (PhysicalDimensions.from_tuple(physical_dimensions)
                     if isinstance(physical_dimensions, (tuple, list))
                     else physical_dimensions)
        
        instance = cls(lgi=lfi, physical_dimensions=phys_dims,
                      voxel_size=float(voxel_size), units=units,
                      connectivity=int(connectivity),
                      min_grain_nvoxels=int(min_grain_nvoxels), random_seed=random_seed,
                      verbosity=verbosity, log_sink=log_sink)

        instance._emit(1, f"Initializing from LFI shape={tuple(lfi.shape)}, connectivity={connectivity}")
        
        if min_grain_nvoxels >= 0:
            instance._emit(2, f"Cleaning grains below {min_grain_nvoxels} voxels")
            instance = instance.clean_small_grains(threshold=min_grain_nvoxels)
        
        instance.grain_locs = instance.compute_grain_locations()
        instance.n_grains = len(instance.grain_locs)
        instance.neigh_gid = instance.compute_neighbor_network(connectivity=connectivity)
        instance._emit(1, f"Initialized grain structure with {instance.n_grains} grains")
        
        return instance
    
    def compute_grain_locations(self) -> Dict[int, np.ndarray]:
        """
        Compute voxel coordinates for each grain from LGI.
        
        Uses efficient NumPy operations (np.argwhere) to extract all voxel
        coordinates for each grain ID.
        
        Returns
        -------
        dict[int, np.ndarray]
            Maps grain_id → (n_voxels, 3) array of voxel coordinates.
        
        Notes
        -----
        Results are cached in self.grain_locs after first call.
        """
        grain_locs = {}
        for gid in np.unique(self.lgi):
            if gid == 0:
                continue
            coords = np.argwhere(self.lgi == gid)
            grain_locs[int(gid)] = coords
        return grain_locs
    
    def compute_neighbor_network(self, connectivity: Optional[int] = None) -> Dict[int, List[int]]:
        """
        Compute grain neighbor relationships using cc3d.region_graph.
        
        Parameters
        ----------
        connectivity : int, optional
            Connectivity type (6, 18, 26). If None, uses self.connectivity.
        
        Returns
        -------
        dict[int, list[int]]
            Maps grain_id → list of neighbor grain_ids.
        
        Notes
        -----
        Results are cached in self.neigh_gid after first call.
        Uses cc3d.region_graph for efficient neighbor detection.
        """
        if connectivity is None:
            connectivity = self.connectivity
        
        # cc3d.region_graph returns a set of frozensets of adjacent label pairs,
        # not a NetworkX graph. Build the adjacency dict by iterating over edges.
        edges = cc3d.region_graph(self.lgi, connectivity=connectivity)
        neigh_dict: Dict[int, List[int]] = {}
        for edge in edges:
            pair = tuple(edge)
            a, b = int(pair[0]), int(pair[1])
            if a == 0 or b == 0:
                continue
            neigh_dict.setdefault(a, []).append(b)
            neigh_dict.setdefault(b, []).append(a)

        # Ensure every grain has an entry (even isolated ones with no touching neighbors)
        for gid in np.unique(self.lgi):
            if gid == 0:
                continue
            gid = int(gid)
            if gid not in neigh_dict:
                neigh_dict[gid] = []
            else:
                neigh_dict[gid] = sorted(list(set(neigh_dict[gid])))

        return neigh_dict
    
    def clean_small_grains(self, threshold: int, parameter_metric: str = 'mean') -> None:
        """
        Dissolve grains smaller than threshold into larger neighbors.
        
        Reimplements clean_gs_GMD_by_source_erosion_v1 from mcgs3_temporal_slice.py
        to work independently. Iteratively merges small grains with their
        largest neighboring grains until all remaining grains are >= threshold voxels.
        
        This method modifies self.lgi, self.n_grains, self.grain_locs, and
        self.neigh_gid in-place.
        
        Parameters
        ----------
        threshold : int
            Minimum grain size (voxels). Grains with fewer voxels are dissolved.
        parameter_metric : str, optional
            Metric for selecting sink grain ('mean', 'max', etc.). Default 'mean'.
        
        Notes
        -----
        This is called automatically by from_lfi if min_grain_nvoxels >= 0.
        """
        lgi_clean = self.lgi.copy()
        
        for _ in range(100):
            grain_sizes = {int(gid): np.sum(lgi_clean == gid)
                          for gid in np.unique(lgi_clean) if gid > 0}
            small_grains = [gid for gid, size in grain_sizes.items() if size < threshold]
            if not small_grains:
                break
            
            for small_gid in small_grains:
                mask = lgi_clean == small_gid
                coords = np.argwhere(mask)
                neighbor_gids = set()
                for x, y, z in coords:
                    for dx, dy, dz in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
                        nx_, ny_, nz_ = x+dx, y+dy, z+dz
                        if (0 <= nx_ < lgi_clean.shape[0] and 0 <= ny_ < lgi_clean.shape[1]
                            and 0 <= nz_ < lgi_clean.shape[2]):
                            ngid = lgi_clean[nx_, ny_, nz_]
                            if ngid != small_gid and ngid != 0:
                                neighbor_gids.add(int(ngid))
                if neighbor_gids:
                    best = max(neighbor_gids, key=lambda g: grain_sizes.get(g, 0))
                    lgi_clean[mask] = best
        
        new_inst = FMSteel3DBase(lgi=lgi_clean, physical_dimensions=self.physical_dimensions,
                                 voxel_size=self.voxel_size, connectivity=self.connectivity,
                                 min_grain_nvoxels=self.min_grain_nvoxels, random_seed=self._random_seed,
                                 verbosity=self._verbosity, log_sink=self._log_sink)
        new_inst.grain_locs = new_inst.compute_grain_locations()
        new_inst.n_grains = len(new_inst.grain_locs)
        new_inst.neigh_gid = new_inst.compute_neighbor_network()
        return new_inst
    
    def get_grain_statistics(self) -> Dict[str, any]:
        """
        Compute basic statistics on grain structure.
        
        Returns
        -------
        dict
            Keys: 'n_grains', 'min_voxels', 'max_voxels', 'mean_voxels',
            'median_voxels', 'total_voxels', 'domain_voxels'.
        """
        if not self.grain_locs:
            self.grain_locs = self.compute_grain_locations()
        sizes = [len(coords) for coords in self.grain_locs.values()]
        return {
            'n_grains': self.n_grains,
            'min_voxels': int(np.min(sizes)) if sizes else 0,
            'max_voxels': int(np.max(sizes)) if sizes else 0,
            'mean_voxels': float(np.mean(sizes)) if sizes else 0.0,
            'median_voxels': float(np.median(sizes)) if sizes else 0.0,
            'total_voxels': int(np.sum(sizes)),
            'domain_voxels': int(np.prod(self.lgi.shape))
        }
    
    def generate_pag_clusters(self,
                              pag_size_distribution: Dict,
                              pag_grain_fraction: float = 1.0,
                              use_non_neigh_pag: bool = False,
                              random_seed: Optional[int] = None) -> 'FMSteel3DWithPAGs':
        """
        Partition grains into PAGs (Prior Austenite Grains) via neighbor clustering.

        Uses stochastic breadth-first-search on the grain neighbor graph to form
        PAGs. Returns a new FMSteel3DWithPAGs instance.

        Parameters
        ----------
        pag_size_distribution : dict
            Dict with keys 'sizes' (list of target cluster sizes) and
            'probs' (list of corresponding probabilities, must sum to ~1.0).

            Example: {'sizes': [3, 4, 5, 6, 7],
                      'probs': [0.10, 0.30, 0.40, 0.15, 0.05]}

        pag_grain_fraction : float, optional
            Fraction of grains that participate in PAG clustering (0.0 to 1.0).
            Remaining grains become isolated (no blocks, no orientations).
            Default 1.0 (all grains participate).

        use_non_neigh_pag : bool, optional
            If True, each new PAG seed is chosen from grains that do not yet
            neighbour any formed PAG, keeping PAGs spatially separated.
            As pag_grain_fraction increases this becomes impossible; once no
            non-neighbour candidates remain the algorithm falls back to a
            random seed from any remaining unclustered grain.
            Default False (original random-seed behaviour).

        random_seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        FMSteel3DWithPAGs
            New instance with computed PAG hierarchy.
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        self._emit(
            1,
            f"Generating PAG clusters (fraction={pag_grain_fraction:.3f}, non_neigh={use_non_neigh_pag})",
            component='PAG',
        )

        sizes = np.array(pag_size_distribution['sizes'])
        probs = np.array(pag_size_distribution['probs'])
        probs = probs / probs.sum()

        n_to_cluster = int(np.ceil(pag_grain_fraction * self.n_grains))
        clusters_dict = {}
        clustered = set()
        pag_id = 1
        available = set(self.grain_locs.keys()) - {0}

        # pag_neighbor_grains: unclustered grains that touch at least one formed PAG.
        # Used only when use_non_neigh_pag=True to bias seed selection away from
        # the expanding PAG frontier.
        pag_neighbor_grains: set = set()

        while len(clustered) < n_to_cluster and available:

            # --- seed selection ---
            if use_non_neigh_pag:
                non_neigh = available - pag_neighbor_grains
                pool = non_neigh if non_neigh else available
            else:
                pool = available
            seed = int(np.random.choice(list(pool)))

            target = int(np.random.choice(sizes, p=probs))

            # --- BFS cluster growth ---
            cluster: set = set()
            queue = [seed]
            visited: set = set()
            while queue and len(cluster) < target:
                cur = queue.pop(0)
                if cur in visited or cur in clustered:
                    continue
                visited.add(cur)
                cluster.add(cur)
                for ngid in self.neigh_gid.get(cur, []):
                    if ngid not in visited and ngid not in clustered and len(cluster) < target:
                        queue.append(ngid)

            if cluster:
                clusters_dict[pag_id] = sorted(list(cluster))
                clustered.update(cluster)
                pag_id += 1

                if use_non_neigh_pag:
                    # Expand the frontier: add every unclustered grain that
                    # touches the newly formed PAG.
                    for gid in cluster:
                        for ngid in self.neigh_gid.get(gid, []):
                            if ngid not in clustered:
                                pag_neighbor_grains.add(ngid)
                    # Remove any just-clustered grains from the frontier.
                    pag_neighbor_grains -= clustered

            available -= clustered

        isolated = set(self.grain_locs.keys()) - clustered - {0}
        
        neigh_clid = {}
        for pid, glist in clusters_dict.items():
            pag_neigh = set()
            for gid in glist:
                for ngid in self.neigh_gid.get(gid, []):
                    if ngid not in glist:
                        for opid, olist in clusters_dict.items():
                            if ngid in olist:
                                pag_neigh.add(opid)
                                break
            neigh_clid[pid] = sorted(list(pag_neigh))
        
        from .with_pags_3d import FMSteel3DWithPAGs
        self._emit(
            1,
            f"Generated {len(clusters_dict)} PAGs; isolated grains={len(isolated)}",
            component='PAG',
        )
        if clusters_dict:
            grains_per_pag = [len(v) for v in clusters_dict.values()]
            self._emit(
                2,
                f"grains/PAG min={min(grains_per_pag)}, max={max(grains_per_pag)}, mean={np.mean(grains_per_pag):.1f}",
                component='PAG',
            )
        return FMSteel3DWithPAGs(parent=self, clusters_dict=clusters_dict, neigh_clid=neigh_clid,
                                 pag_orientations={}, isolated_grains=isolated, random_seed=random_seed,
                                 verbosity=self._verbosity, log_sink=self._log_sink)


__all__ = ['FMSteel3DBase', 'PhysicalDimensions']
