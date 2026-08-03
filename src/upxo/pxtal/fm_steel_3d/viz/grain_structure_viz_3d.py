"""
GrainStructureViz3D: PyVista grain and block visualizations.

Worker class for grain and block-level voxel visualizations.
"""

import numpy as np
from typing import Optional, Dict, List
import pyvista as pv


class GrainStructureViz3D:
    """PyVista visualizations for grain and block structures."""
    
    __slots__ = ('_default_cmap', '_default_window_size', '_default_opacity')
    
    def __init__(self, default_cmap: str = 'viridis',
                 default_window_size: tuple = (1000, 1000),
                 default_opacity: float = 0.8):
        self._default_cmap = default_cmap
        self._default_window_size = default_window_size
        self._default_opacity = default_opacity
    
    def plot_gs_pvvox(self, lgi: np.ndarray, grain_locs: Dict[int, np.ndarray],
                      alpha: float = 1.0, title: str = 'Grain Structure',
                      cmap: Optional[str] = None, voxel_size: float = 1.0,
                      downsample: float = 1.0):
        """Visualize grain structure as PyVista voxels."""
        cmap = cmap or self._default_cmap
        plotter = pv.Plotter()
        coords, colors = [], []
        for gid in np.unique(lgi):
            if gid == 0:
                continue
            voxel_coords = np.argwhere(lgi == gid) * voxel_size
            if len(voxel_coords) > 0:
                coords.append(voxel_coords)
                colors.append(np.full(len(voxel_coords), gid))
        if coords:
            poly = pv.PolyData(np.vstack(coords))
            poly['grain_id'] = np.concatenate(colors)
            plotter.add_mesh(poly, scalars='grain_id', cmap=cmap, opacity=alpha, point_size=5)
        plotter.set_background('white')
        try:
            plotter.show(title=title)
        finally:
            # Explicit close (rather than leaving it to GC/interpreter shutdown)
            # so this native VTK render window can't leave the process hanging
            # after the main application window is closed.
            plotter.close()
    
    def plot_pag_map_pyvista(self, clusters_dict: Dict[int, List[int]],
                             grain_locs: Dict[int, np.ndarray],
                             voxel_size: float = 1.0, opacity: float = 1.0,
                             title: str = 'PAG Map', **kwargs):
        """Visualize grains colored by PAG."""
        plotter = pv.Plotter()
        coords_all, colors_all = [], []
        for pag_id, grain_ids in clusters_dict.items():
            for gid in grain_ids:
                if gid in grain_locs:
                    voxel_coords = grain_locs[gid] * voxel_size
                    coords_all.append(voxel_coords)
                    colors_all.append(np.full(len(voxel_coords), pag_id))
        if coords_all:
            poly = pv.PolyData(np.vstack(coords_all))
            poly['pag_id'] = np.concatenate(colors_all)
            plotter.add_mesh(poly, scalars='pag_id', cmap='tab20', opacity=opacity, point_size=5)
        plotter.set_background('white')
        try:
            plotter.show(title=title)
        finally:
            # Explicit close (rather than leaving it to GC/interpreter shutdown)
            # so this native VTK render window can't leave the process hanging
            # after the main application window is closed.
            plotter.close()
    
    def visualize_block_morphology(self, blocks_dict: Dict[str, np.ndarray],
                                   title: str = 'Block Morphology',
                                   voxel_size: float = 1.0, opacity: float = 1.0, **kwargs):
        """Visualize all blocks with random colors."""
        plotter = pv.Plotter()
        coords_all, colors_all = [], []
        for block_idx, (block_id, voxels) in enumerate(blocks_dict.items()):
            coords_all.append(voxels * voxel_size)
            colors_all.append(np.full(len(voxels), block_idx % 256))
        if coords_all:
            poly = pv.PolyData(np.vstack(coords_all))
            poly['block_id'] = np.concatenate(colors_all)
            plotter.add_mesh(poly, scalars='block_id', cmap='nipy_spectral', opacity=opacity, point_size=5)
        plotter.set_background('white')
        try:
            plotter.show(title=title)
        finally:
            # Explicit close (rather than leaving it to GC/interpreter shutdown)
            # so this native VTK render window can't leave the process hanging
            # after the main application window is closed.
            plotter.close()
    
    def visualize_block_morphology_v1(self, blocks_dict: Dict[str, np.ndarray],
                                      title: str = 'Block Morphology',
                                      voxel_size: float = 1.0, **kwargs):
        """Visualize blocks with scalar bar."""
        plotter = pv.Plotter()
        coords_all, colors_all = [], []
        for block_idx, (block_id, voxels) in enumerate(blocks_dict.items()):
            coords_all.append(voxels * voxel_size)
            colors_all.append(np.full(len(voxels), block_idx))
        if coords_all:
            poly = pv.PolyData(np.vstack(coords_all))
            poly['block_index'] = np.concatenate(colors_all)
            plotter.add_mesh(poly, scalars='block_index', cmap='viridis', point_size=5,
                           scalar_bar_args={'title': 'Block ID'})
        plotter.set_background('white')
        try:
            plotter.show(title=title)
        finally:
            # Explicit close (rather than leaving it to GC/interpreter shutdown)
            # so this native VTK render window can't leave the process hanging
            # after the main application window is closed.
            plotter.close()
    
    def visualize_blocks_in_packet(self, blocks_dict: Dict[str, np.ndarray],
                                   packet_gid=None, voxel_size: float = 1.0,
                                   opacity: float = 1.0, title: str = 'Blocks in Packet', **kwargs):
        """Visualize blocks in a packet."""
        self.visualize_block_morphology(blocks_dict, title=title, voxel_size=voxel_size, opacity=opacity)
    
    def viz_browse_grains(self, lgi, grain_locs, **kwargs):
        """Interactive grain browser (placeholder)."""
        print("viz_browse_grains: Not yet fully implemented — use plot_gs_pvvox instead.")
    
    def viz_clip_plane(self, lgi, grain_locs, **kwargs):
        """Clip plane visualization (placeholder)."""
        print("viz_clip_plane: Not yet fully implemented.")
    
    def viz_mesh_slice(self, lgi, grain_locs, **kwargs):
        """Mesh slice visualization (placeholder)."""
        print("viz_mesh_slice: Not yet fully implemented.")
    
    def viz_mesh_slice_ortho(self, lgi, grain_locs, **kwargs):
        """Orthogonal slice visualization (placeholder)."""
        print("viz_mesh_slice_ortho: Not yet fully implemented.")
    
    def plot_grains(self, lgi, grain_locs, gids, **kwargs):
        """Visualize specific grains."""
        subset_locs = {gid: grain_locs[gid] for gid in gids if gid in grain_locs}
        self.plot_gs_pvvox(lgi, subset_locs, **kwargs)
    
    def plot_grain_sets(self, lgi, grain_locs, data=None, **kwargs):
        """Visualize grain sets (placeholder)."""
        self.plot_gs_pvvox(lgi, grain_locs, **kwargs)
    
    def plot_single_voxel_grains(self, lgi, grain_locs, title='Single Voxel Grains', **kwargs):
        """Highlight single-voxel grains."""
        single_voxel = {gid: locs for gid, locs in grain_locs.items() if len(locs) == 1}
        print(f"Found {len(single_voxel)} single-voxel grains")
    
    def plot_scalar_field_slice_orthogonal(self, lgi, **kwargs):
        """Scalar field on orthogonal slices (placeholder)."""
        print("plot_scalar_field_slice_orthogonal: Not yet fully implemented.")
    
    def plot_scalar_field_slice(self, lgi, **kwargs):
        """Scalar field on a single slice (placeholder)."""
        print("plot_scalar_field_slice: Not yet fully implemented.")
    
    def plot_gs_pvvox_subset(self, lgi_subset, grain_locs_subset, **kwargs):
        """Visualize a grain structure subset."""
        self.plot_gs_pvvox(lgi_subset, grain_locs_subset, **kwargs)

    @staticmethod
    def _scatter_exploded_grid(feature_voxels: Dict[object, np.ndarray],
                               offsets: Dict[object, np.ndarray]):
        """Shared bounding-box + scatter step for the two exploded-grid builders
        below: rigidly shift each feature's voxels by its own offset, then lay
        every feature into one int32 label grid sized to fit all of them.
        Returns (grid, n_features)."""
        ids = list(feature_voxels.keys())
        shifted = {}
        all_coords = []
        for fid in ids:
            voxels = feature_voxels[fid]
            if voxels is None or len(voxels) == 0:
                continue
            sv = voxels + offsets.get(fid, np.zeros(3))
            shifted[fid] = sv
            all_coords.append(sv)

        if not all_coords:
            return np.zeros((1, 1, 1), dtype=np.int32), 0

        stacked = np.concatenate(all_coords, axis=0)
        mins = stacked.min(axis=0)
        maxs = stacked.max(axis=0)
        shape = (maxs - mins + 1).astype(int)
        grid = np.zeros(tuple(shape), dtype=np.int32)

        for idx, fid in enumerate(ids, start=1):
            if fid not in shifted:
                continue
            sv = (shifted[fid] - mins).astype(int)
            grid[sv[:, 0], sv[:, 1], sv[:, 2]] = idx

        return grid, len(ids)

    @staticmethod
    def build_exploded_block_grid(all_blocks: Dict[str, np.ndarray],
                                  clusters_dict: Dict[int, List[int]],
                                  grain_locs: Dict[int, np.ndarray],
                                  grain_to_pag_id: Dict[int, int],
                                  grain_to_blocks_map: Dict[int, List[str]],
                                  explosion_factor: float):
        """
        Build a 3D int32 label grid where each active voxel holds a 1-based
        block index, with every block's full voxel set rigidly translated by
        explosion_factor * (its own centroid minus its parent PAG's centroid).
        explosion_factor=0 reproduces the true, unexploded layout exactly --
        translation preserves each block's shape, it only relocates it.

        Returns (grid, n_blocks).
        """
        pag_centroid = {}
        for pag_id, gids in clusters_dict.items():
            parts = [grain_locs[g] for g in gids if g in grain_locs and len(grain_locs[g])]
            if parts:
                pag_centroid[pag_id] = np.concatenate(parts, axis=0).mean(axis=0)

        block_to_pag = {}
        for gid, blk_list in grain_to_blocks_map.items():
            pag_id = grain_to_pag_id.get(gid)
            if pag_id is None:
                continue
            for bid in blk_list:
                block_to_pag[bid] = pag_id

        offsets = {}
        for block_id, voxels in all_blocks.items():
            if voxels is None or len(voxels) == 0:
                continue
            pag_c = pag_centroid.get(block_to_pag.get(block_id))
            if pag_c is None:
                continue
            offsets[block_id] = np.round(
                (voxels.mean(axis=0) - pag_c) * explosion_factor).astype(int)

        return GrainStructureViz3D._scatter_exploded_grid(all_blocks, offsets)

    @staticmethod
    def build_exploded_subblock_grid(all_subblocks: Dict[str, np.ndarray],
                                     all_blocks: Dict[str, np.ndarray],
                                     block_to_subblocks_map: Dict[str, List[str]],
                                     explosion_factor: float):
        """
        Same idea one level down: each sub-block's full voxel set is rigidly
        translated by explosion_factor * (its own centroid minus its parent
        block's centroid) -- blocks/PAGs stay in their true positions.

        Returns (grid, n_subblocks).
        """
        subblock_to_block = {}
        for block_id, sb_list in block_to_subblocks_map.items():
            for sb_id in sb_list:
                subblock_to_block[sb_id] = block_id

        offsets = {}
        for sb_id, voxels in all_subblocks.items():
            if voxels is None or len(voxels) == 0:
                continue
            block_voxels = all_blocks.get(subblock_to_block.get(sb_id))
            if block_voxels is None or len(block_voxels) == 0:
                continue
            offsets[sb_id] = np.round(
                (voxels.mean(axis=0) - block_voxels.mean(axis=0)) * explosion_factor
            ).astype(int)

        return GrainStructureViz3D._scatter_exploded_grid(all_subblocks, offsets)

    def lfi_to_polydata(self, lfi, scalar_array, voxel_size=1.0, cmap='tab20',
                        title='', scalar_name='value', show_scalar_bar=True,
                        threshold_min=0.5, clim=None, below_color=None,
                        show_edges=True):
        """Show a voxel grid coloured by a per-voxel scalar array."""
        nx, ny, nz = lfi.shape
        grid = pv.ImageData()
        grid.dimensions = (nx + 1, ny + 1, nz + 1)
        grid.spacing = (voxel_size, voxel_size, voxel_size)
        grid.origin = (0.0, 0.0, 0.0)
        grid.cell_data[scalar_name] = scalar_array.flatten(order='F').astype(float)
        filled = grid.threshold(value=threshold_min, scalars=scalar_name).extract_surface()

        pl = pv.Plotter(window_size=(900, 750))
        mesh_kwargs = dict(
            scalars=scalar_name, cmap=cmap,
            show_edges=show_edges, edge_color='black', line_width=0.5,
            show_scalar_bar=show_scalar_bar,
            scalar_bar_args={'title': scalar_name, 'n_labels': 5},
        )
        if clim is not None:
            mesh_kwargs['clim'] = list(clim)
        if below_color is not None:
            mesh_kwargs['below_color'] = below_color

        pl.add_mesh(filled, **mesh_kwargs)
        pl.add_title(title, font_size=14)
        pl.set_background('white')
        pl.show_axes()
        try:
            pl.show()
        finally:
            # Explicit close (rather than leaving it to GC/interpreter shutdown)
            # so this native VTK render window can't leave the process hanging
            # after the main application window is closed.
            pl.close()

    def feature_fid_to_polydata(self, lgi_shape, iso_grain_map, block_fid_map,
                                voxel_size=1.0,
                                iso_cmap='cool', block_cmap='hot',
                                title='Feature ID Map', show_edges=True):
        """Show isolated grains and blocks in one window using two independent cmaps."""
        nx, ny, nz = lgi_shape
        grid = pv.ImageData()
        grid.dimensions = (nx + 1, ny + 1, nz + 1)
        grid.spacing = (voxel_size, voxel_size, voxel_size)
        grid.origin = (0.0, 0.0, 0.0)
        grid.cell_data['Grain FID'] = iso_grain_map.flatten(order='F').astype(float)
        grid.cell_data['Block FID'] = block_fid_map.flatten(order='F').astype(float)

        iso_mesh = grid.threshold(value=0.5, scalars='Grain FID').extract_surface()
        blk_mesh = grid.threshold(value=0.5, scalars='Block FID').extract_surface()

        pl = pv.Plotter(window_size=(900, 750))

        if iso_mesh.n_cells > 0:
            pl.add_mesh(
                iso_mesh, scalars='Grain FID', cmap=iso_cmap,
                show_edges=show_edges, edge_color='black', line_width=0.5,
                scalar_bar_args={
                    'title': 'Grain FID', 'n_labels': 3,
                    'position_x': 0.05, 'position_y': 0.05,
                    'width': 0.35, 'height': 0.10,
                },
            )

        if blk_mesh.n_cells > 0:
            pl.add_mesh(
                blk_mesh, scalars='Block FID', cmap=block_cmap,
                show_edges=show_edges, edge_color='black', line_width=0.5,
                scalar_bar_args={
                    'title': 'Block FID', 'n_labels': 5,
                    'position_x': 0.05, 'position_y': 0.22,
                    'width': 0.35, 'height': 0.10,
                },
            )

        pl.add_title(title, font_size=14)
        pl.set_background('white')
        pl.show_axes()
        try:
            pl.show()
        finally:
            # Explicit close (rather than leaving it to GC/interpreter shutdown)
            # so this native VTK render window can't leave the process hanging
            # after the main application window is closed.
            pl.close()

    def subblock_family_polydata(self, fid_map, fid_to_rgb, voxel_size=1.0,
                                 title='Sub-block Family Color Map', show_edges=True):
        """Display sub-blocks using an explicit FID->RGB lookup table."""
        nx, ny, nz = fid_map.shape
        flat_fids = fid_map.flatten(order='F').astype(int)

        safe_fids = np.clip(flat_fids, 0, len(fid_to_rgb) - 1)
        rgb = fid_to_rgb[safe_fids].astype(np.float32)
        occupied = (flat_fids > 0).astype(float)

        grid = pv.ImageData()
        grid.dimensions = (nx + 1, ny + 1, nz + 1)
        grid.spacing = (voxel_size, voxel_size, voxel_size)
        grid.origin = (0.0, 0.0, 0.0)
        grid.cell_data['occupied'] = occupied
        grid.cell_data['SubBlockRGB'] = rgb

        filled = grid.threshold(value=0.5, scalars='occupied').extract_surface()

        pl = pv.Plotter(window_size=(900, 750))
        pl.add_mesh(filled, scalars='SubBlockRGB', rgb=True,
                    show_edges=show_edges, edge_color='black', line_width=0.5)
        pl.add_title(title, font_size=14)
        pl.set_background('white')
        pl.show_axes()
        try:
            pl.show()
        finally:
            # Explicit close (rather than leaving it to GC/interpreter shutdown)
            # so this native VTK render window can't leave the process hanging
            # after the main application window is closed.
            pl.close()


__all__ = ['GrainStructureViz3D']
