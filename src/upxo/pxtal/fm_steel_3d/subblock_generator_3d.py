"""
SubBlockGenerator3D: Worker class for slicing martensitic blocks into sub-blocks/laths.

Uses the same signed-distance slab projection as BlockGenerator3D, but the slicing
normal is inherited from the parent block (via block_slicing_normals) rather than
recomputed from crystallography.

Classes:
    SubBlockGenerator3D: Sub-block generation worker.
"""

import numpy as np
import cc3d
from typing import Dict, List, Tuple, Optional


class SubBlockGenerator3D:
    """
    Worker class for slicing martensitic blocks into sub-blocks (laths).

    Sub-blocks within a block share the same slicing direction as their parent
    block and carry a small orientation spread around the parent block orientation.

    Thin blocks whose projected thickness along the slicing normal is less than
    the minimum requested sub-block thickness are handled via thin_block_strategy:

        'skip'    — treat the entire block as a single sub-block (no subdivision).
        'upscale' — caller must rescale the grid before calling; treated as 'skip'
                    here if the block is still too thin after rescaling.

    All voxels in every block are exhausted: the final slab of each block absorbs
    any remainder so no voxels are left unassigned.
    """

    __slots__ = ('_verbosity',)

    def __init__(self, verbosity: int = 0):
        """
        Parameters
        ----------
        verbosity : int, optional
            0 = silent, 1 = per-block summary. Default 0.
        """
        self._verbosity = int(verbosity)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_subblocks_for_all_blocks(
        self,
        all_blocks: Dict[str, np.ndarray],
        block_slicing_normals: Dict[str, np.ndarray],
        subblock_thickness_range: Tuple[float, float] = (1.0, 2.0),
        thin_block_strategy: str = 'skip',
        random_seed: Optional[int] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, List[str]], Dict[str, np.ndarray]]:
        """
        Generate sub-blocks for every block in all_blocks.

        Parameters
        ----------
        all_blocks : dict
            {block_id: (n_voxels, 3) integer voxel coordinate array}.
        block_slicing_normals : dict
            {block_id: unit_normal (3,)} — inherited from FMSteel3DWithBlocks.
        subblock_thickness_range : tuple of (float, float)
            (min, max) sub-block thickness in voxels. Each block draws its own
            thickness uniformly from this range.
        thin_block_strategy : str
            Policy for blocks whose projected thickness < min thickness:
            'skip'    — the whole block becomes one sub-block.
            'upscale' — same as 'skip' at this stage; the caller should have
                        already rescaled the grid before calling.
        random_seed : int, optional
            Seed for reproducibility.

        Returns
        -------
        all_subblocks : dict
            {subblock_id: (n_voxels, 3) array}.
        block_to_subblocks_map : dict
            {block_id: [subblock_id, ...]}.
        subblock_slicing_normals : dict
            {subblock_id: unit_normal (3,)} — same normal as parent block,
            available for any further subdivision level.
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        t_lo = float(subblock_thickness_range[0])
        t_hi = float(subblock_thickness_range[1])

        all_subblocks: Dict[str, np.ndarray] = {}
        block_to_subblocks_map: Dict[str, List[str]] = {}
        subblock_slicing_normals: Dict[str, np.ndarray] = {}

        for block_id, voxel_coords in all_blocks.items():
            if len(voxel_coords) == 0:
                block_to_subblocks_map[block_id] = []
                continue

            unit_n = block_slicing_normals.get(block_id)
            if unit_n is None:
                # No normal stored — fall back to treating block as one sub-block
                sb_name = f'SB_{block_id}_1'
                all_subblocks[sb_name] = voxel_coords
                block_to_subblocks_map[block_id] = [sb_name]
                subblock_slicing_normals[sb_name] = np.array([0., 0., 1.])
                continue

            distances = np.dot(voxel_coords.astype(float), unit_n)
            available_thickness = float(distances.max() - distances.min())

            if available_thickness < t_lo:
                # Block is too thin — apply strategy (both 'skip' and 'upscale'
                # resolve to keeping the block whole at this point).
                sb_name = f'SB_{block_id}_1'
                all_subblocks[sb_name] = voxel_coords
                block_to_subblocks_map[block_id] = [sb_name]
                subblock_slicing_normals[sb_name] = unit_n
                if self._verbosity >= 1:
                    print(f"  [thin] {block_id}: thickness {available_thickness:.2f} < "
                          f"{t_lo:.2f} vox — strategy='{thin_block_strategy}'")
                continue

            sb_thickness = float(np.random.uniform(t_lo, t_hi))
            new_subblocks = self._slice_block(block_id, voxel_coords, unit_n, sb_thickness)

            all_subblocks.update(new_subblocks)
            block_to_subblocks_map[block_id] = list(new_subblocks.keys())
            for sb_name in new_subblocks:
                subblock_slicing_normals[sb_name] = unit_n

        return all_subblocks, block_to_subblocks_map, subblock_slicing_normals

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _slice_block(
        self,
        block_id: str,
        voxel_coords: np.ndarray,
        unit_n: np.ndarray,
        subblock_thickness: float,
    ) -> Dict[str, np.ndarray]:
        """
        Slice one block into sub-blocks along unit_n.

        Identical algorithm to BlockGenerator3D.slice_packet_into_blocks:
        signed-distance slabs + 6-connected cc3d labeling. All voxels are
        covered — the final slab absorbs any remainder.

        Parameters
        ----------
        block_id : str
            Parent block identifier (used to construct sub-block names).
        voxel_coords : np.ndarray
            (n_voxels, 3) integer voxel coordinates.
        unit_n : np.ndarray
            Unit normal (3,) defining the slicing direction.
        subblock_thickness : float
            Slab width along unit_n in voxels.

        Returns
        -------
        dict
            {subblock_name: (n_voxels, 3) global voxel coordinate array}
        """
        distances = np.dot(voxel_coords.astype(float), unit_n)
        min_dist = float(distances.min())
        max_dist = float(distances.max())

        num_slices = max(1, int(np.ceil((max_dist - min_dist) / subblock_thickness)))

        new_subblocks: Dict[str, np.ndarray] = {}
        local_sb_counter = 1

        for i in range(num_slices):
            slab_min = min_dist + i * subblock_thickness
            slab_max = slab_min + subblock_thickness
            # Final slab: inclusive upper bound captures voxels exactly at max_dist
            if i == num_slices - 1:
                in_slab = (distances >= slab_min) & (distances <= slab_max + 1e-9)
            else:
                in_slab = (distances >= slab_min) & (distances < slab_max)

            slab_voxels = voxel_coords[in_slab]
            if len(slab_voxels) == 0:
                continue

            min_coord = slab_voxels.min(axis=0)
            max_coord = slab_voxels.max(axis=0)
            local_shape = tuple((max_coord - min_coord + 1).astype(int))
            local_image = np.zeros(local_shape, dtype=np.uint8)
            local_coords = (slab_voxels - min_coord).astype(int)
            local_image[local_coords[:, 0], local_coords[:, 1], local_coords[:, 2]] = 1

            labeled, num_comp = cc3d.connected_components(
                local_image, connectivity=26, return_N=True)

            for comp_id in range(1, num_comp + 1):
                sb_local = np.argwhere(labeled == comp_id)
                sb_global = sb_local + min_coord
                sb_name = f'SB_{block_id}_{local_sb_counter}'
                new_subblocks[sb_name] = sb_global
                local_sb_counter += 1

        return new_subblocks


__all__ = ['SubBlockGenerator3D']
