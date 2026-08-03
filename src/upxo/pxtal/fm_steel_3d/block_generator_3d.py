"""
BlockGenerator3D: Worker class for slicing PAGs into martensitic blocks.

This module contains the stateless block generation logic. It receives PAG data
and returns block partitions. No state is held between calls.

Classes:
    BlockGenerator3D: Block generation worker.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import cc3d
from upxo.geoEntities.plane import Plane
from upxo.xtalphy.crystal_orientation import euler_bunge_to_matrix


class BlockGenerator3D:
    """
    Worker class for generating martensitic blocks within PAGs.
    
    This is a stateless utility class. It receives PAG voxel data and orientation
    information, and returns partitioned blocks. Multiple calls produce different
    results based on random seeds (stochastic slicing plane selection).
    
    All methods are instance methods (not static) to allow storing session
    config (default parameters, verbosity, etc.) in future.
    """
    
    __slots__ = ('_verbosity', '_default_voxel_size')
    
    def __init__(self, verbosity: int = 0, default_voxel_size: float = 1.0):
        """
        Initialize BlockGenerator3D.
        
        Parameters
        ----------
        verbosity : int, optional
            Verbosity level (0=silent, 1=progress, 2=detailed). Default 0.
        default_voxel_size : float, optional
            Default voxel size for visualization. Default 1.0.
        """
        self._verbosity = int(verbosity)
        self._default_voxel_size = float(default_voxel_size)
    
    def generate_blocks_for_all_pags(self,
                                      clusters_dict: Dict[int, List[int]],
                                      grain_locs: Dict[int, np.ndarray],
                                      pag_orientations: Dict[int, Tuple[float, float, float]],
                                      block_thickness_range: Tuple[float, float] = (2.0, 5.0),
                                      random_seed: Optional[int] = None,
                                      slab_connectivity: int = 26) -> Tuple[Dict[str, np.ndarray], Dict[int, List[str]], Dict[int, int], Dict[str, np.ndarray]]:
        """
        Generate blocks for all PAGs.

        Parameters
        ----------
        clusters_dict : dict
            {pag_id: [grain_id, ...]}.
        grain_locs : dict
            {grain_id: (n_voxels, 3) array}.
        pag_orientations : dict
            {pag_id: (phi1, Phi, phi2)}.
        block_thickness_range : tuple of (float, float), optional
            (lower, upper) bounds for block thickness in physical units.
            Each packet draws its own thickness uniformly from this range,
            producing natural thickness variation across the microstructure.
            Default (2.0, 5.0).
        random_seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        tuple
            (all_blocks, grain_to_blocks_map, grain_to_plane_idx, block_slicing_normals).
            grain_to_blocks_map keys are grain_ids (= packet identifiers); values
            are the list of block_ids created from that packet.
            block_slicing_normals maps each block_id to the unit normal of the
            slicing plane used to create it — required by SubBlockGenerator3D.
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        t_lo, t_hi = float(block_thickness_range[0]), float(block_thickness_range[1])

        all_blocks: Dict[str, np.ndarray] = {}
        grain_to_blocks_map: Dict[int, List[str]] = {}
        grain_to_plane_idx: Dict[int, int] = {}
        block_slicing_normals: Dict[str, np.ndarray] = {}

        for pag_id, grain_ids in clusters_dict.items():
            # Each grain in the PAG is one packet — a spatially distinct region
            # that shares one {111}γ habit plane and one set of 6 KS variants.
            # The 4 {111}γ candidate planes are cycled across grains so that
            # adjacent grains in the same PAG tend to use different habit planes,
            # which is consistent with the physical packet structure.
            candidate_planes = self.get_slicing_planes_for_packet(
                pag_id=pag_id, pag_orientations=pag_orientations)
            n_planes = len(candidate_planes)

            for packet_idx, grain_id in enumerate(grain_ids):
                if grain_id not in grain_locs:
                    continue
                grain_voxels = grain_locs[grain_id]

                # Assign {111}γ plane by cycling: grain 0 → plane 0, grain 1 → plane 1, ...
                plane_idx = packet_idx % n_planes
                grain_to_plane_idx[grain_id] = plane_idx

                # Each packet gets its own randomly drawn block thickness so that
                # the microstructure shows natural lath-width variation.
                packet_thickness = float(np.random.uniform(t_lo, t_hi))

                # Anchor plane at grain centroid
                grain_centroid = grain_voxels.mean(axis=0)
                slicing_plane = Plane(
                    point=grain_centroid,
                    normal=candidate_planes[plane_idx].normal)

                # Capture the unit normal before slicing — all sub-blocks within
                # this packet inherit the same slicing direction.
                packet_unit_n = slicing_plane.unit_normal.copy()

                new_blocks = self.slice_packet_into_blocks(
                    packet_gid=grain_id,
                    cluster_id=pag_id,
                    voxel_coords=grain_voxels,
                    block_thickness=packet_thickness,
                    slicing_plane=slicing_plane,
                    slab_connectivity=slab_connectivity,
                )
                all_blocks.update(new_blocks)
                grain_to_blocks_map[grain_id] = list(new_blocks.keys())
                for block_name in new_blocks:
                    block_slicing_normals[block_name] = packet_unit_n

        return all_blocks, grain_to_blocks_map, grain_to_plane_idx, block_slicing_normals
    
    def get_slicing_planes_for_packet(self,
                                       pag_id: int,
                                       pag_orientations: Dict[int, Tuple[float, float, float]]
                                       ) -> List[Plane]:
        """
        Return the four crystallographic slicing-plane candidates for a PAG.

        The four {111}FCC plane normals (habit planes for martensite packets) are
        defined in the PAG crystal frame and rotated into the sample frame using
        the PAG's Bunge Euler angles. If no orientation is stored for the PAG
        the four normals are returned in the reference (lab) frame as a fallback.

        Parameters
        ----------
        pag_id : int
            PAG identifier.
        pag_orientations : dict
            {pag_id: (phi1, Phi, phi2)} in degrees (Bunge ZXZ convention).

        Returns
        -------
        list of Plane
            Four Plane objects with point=[0,0,0] (templates — point is set
            to the PAG centroid by the caller before slicing).
        """
        # Four {111}FCC crystal-frame normals (one per possible martensite packet)
        crystal_frame_normals = np.array([
            [ 1,  1,  1],
            [ 1, -1,  1],
            [-1,  1,  1],
            [-1, -1,  1],
        ], dtype=float) / np.sqrt(3)

        parent_fcc_ea = pag_orientations.get(pag_id)
        if parent_fcc_ea is None:
            # No orientation assigned — use crystal-frame normals unchanged (lab = crystal)
            return [Plane(point=[0, 0, 0], normal=n) for n in crystal_frame_normals]

        # Rotate crystal-frame normals into the sample frame using the PAG rotation matrix
        parent_fcc_R = euler_bunge_to_matrix(
            parent_fcc_ea[0], parent_fcc_ea[1], parent_fcc_ea[2], degrees=True)
        sample_frame_normals = (parent_fcc_R @ crystal_frame_normals.T).T

        return [Plane(point=[0, 0, 0], normal=n) for n in sample_frame_normals]
    
    def slice_packet_into_blocks(self,
                                  packet_gid: int,
                                  cluster_id: int,
                                  voxel_coords: np.ndarray,
                                  block_thickness: float,
                                  slicing_plane: Plane,
                                  slab_connectivity: int = 26) -> Dict[str, np.ndarray]:
        """
        Slice a PAG/packet into contiguous blocks using plane-normal projection.

        Algorithm:
        1. Project all voxels onto the slicing-plane unit normal via signed distance.
        2. Divide the projection range into parallel slabs of width block_thickness.
        3. Within each slab, rasterise voxels into a local 3D boolean grid and run
           connected-component labeling (cc3d, connectivity=slab_connectivity,
           default 26) to find spatially contiguous blocks.
        4. Return each component as a named block with its global voxel coordinates.

        Parameters
        ----------
        packet_gid : int
            Grain/packet identifier (used in block names).
        cluster_id : int
            Parent cluster (PAG) identifier (used in block names).
        voxel_coords : np.ndarray
            (n_voxels, 3) integer voxel coordinates.
        block_thickness : float
            Desired slab thickness along the slicing-plane normal.
        slicing_plane : Plane
            Plane whose unit_normal defines the slicing direction and whose
            point is the projection origin (typically the PAG centroid).

        Returns
        -------
        newly_created_blocks : dict
            {block_name: (n_voxels, 3) array of global voxel coordinates}.
            Names are already globally unique (cluster_id + packet_gid +
            a local per-packet counter), so no separate global id is needed.
        """
        if len(voxel_coords) < 1:
            return {}

        unit_n = slicing_plane.unit_normal
        # Signed distance of each voxel from the slicing plane
        distances = np.dot(voxel_coords - slicing_plane.point, unit_n)
        min_dist = float(distances.min())
        max_dist = float(distances.max())

        num_slices = max(1, int(np.ceil((max_dist - min_dist) / block_thickness)))

        newly_created_blocks: Dict[str, np.ndarray] = {}
        local_block_counter = 1

        for i in range(num_slices):
            slab_min = min_dist + i * block_thickness
            slab_max = slab_min + block_thickness
            # Inclusive upper bound on the last slab to capture voxels at max_dist
            if i == num_slices - 1:
                in_slab = (distances >= slab_min) & (distances <= slab_max + 1e-9)
            else:
                in_slab = (distances >= slab_min) & (distances < slab_max)

            slab_voxels = voxel_coords[in_slab]
            if len(slab_voxels) == 0:
                continue

            # Rasterise into a tight local bounding box for connected-component labeling
            min_coord = slab_voxels.min(axis=0)
            max_coord = slab_voxels.max(axis=0)
            local_shape = tuple((max_coord - min_coord + 1).astype(int))
            local_image = np.zeros(local_shape, dtype=np.uint8)
            local_coords = (slab_voxels - min_coord).astype(int)
            local_image[local_coords[:, 0], local_coords[:, 1], local_coords[:, 2]] = 1

            labeled, num_comp = cc3d.connected_components(
                local_image, connectivity=slab_connectivity, return_N=True)

            for comp_id in range(1, num_comp + 1):
                block_coords_local = np.argwhere(labeled == comp_id)
                block_coords_global = block_coords_local + min_coord
                block_name = f'B_{cluster_id}_{packet_gid}_{local_block_counter}'
                newly_created_blocks[block_name] = block_coords_global
                local_block_counter += 1

        return newly_created_blocks


__all__ = ['BlockGenerator3D']
