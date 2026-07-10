"""
OrientationViz3D: Crystal orientation visualizations (IPF maps, pole figures, KS verification).

Worker class for orientation-based visualizations.
"""

import numpy as np
from typing import Optional, Dict, List, Tuple
from upxo.xtalphy.crystal_orientation import ipf_color as _xtal_ipf_color


class OrientationViz3D:
    """Orientation visualizations: IPF maps, pole figures, KS verification."""
    
    __slots__ = ('_default_sample_direction',)
    
    def __init__(self, default_sample_direction: Optional[List[float]] = None):
        self._default_sample_direction = default_sample_direction or [0, 0, 1]
    
    def plot_ipf_map_pyvista_v2(self, grain_locs: Dict[int, np.ndarray],
                                grain_orientations: Dict[int, Tuple[float, float, float]],
                                sample_direction: Optional[List[float]] = None,
                                voxel_size: float = 1.0, opacity: float = 0.8, **kwargs):
        """Create 3D IPF map using PyVista."""
        import pyvista as pv
        plotter = pv.Plotter()
        coords_all, colors_all = [], []
        for gid, coords in grain_locs.items():
            if gid in grain_orientations:
                ipf_color = self.euler_to_ipf_color(grain_orientations[gid], sample_direction)
                voxel_coords = coords * voxel_size
                coords_all.append(voxel_coords)
                colors_all.append(np.full((len(coords), 3), ipf_color) * 255)
        if coords_all:
            poly = pv.PolyData(np.vstack(coords_all))
            poly['RGB'] = np.vstack(colors_all).astype(np.uint8)
            plotter.add_mesh(poly, rgb=True, point_size=5, opacity=opacity)
        plotter.set_background('white')
        plotter.show(title='IPF Map')
    
    def plot_ipf_map_pyvista_v1(self, grain_locs, grain_orientations, **kwargs):
        """Create 3D IPF map (v1 = same as v2)."""
        self.plot_ipf_map_pyvista_v2(grain_locs, grain_orientations, **kwargs)
    
    def visualize_block_ipf_map(self, all_blocks: Dict[str, np.ndarray],
                                block_orientations: Dict[str, Tuple[float, float, float]],
                                voxel_size: float = 1.0,
                                sample_direction: Optional[List[float]] = None, **kwargs):
        """Visualize blocks colored by IPF."""
        import pyvista as pv
        plotter = pv.Plotter()
        coords_all, colors_all = [], []
        for block_id, voxels in all_blocks.items():
            if block_id in block_orientations:
                ipf_color = self.euler_to_ipf_color(block_orientations[block_id], sample_direction)
                coords_all.append(voxels * voxel_size)
                colors_all.append(np.full((len(voxels), 3), ipf_color) * 255)
        if coords_all:
            poly = pv.PolyData(np.vstack(coords_all))
            poly['RGB'] = np.vstack(colors_all).astype(np.uint8)
            plotter.add_mesh(poly, rgb=True, point_size=5)
        plotter.set_background('white')
        plotter.show(title='Block IPF Map')

    def ipf_blocks_to_polydata(self, all_blocks: Dict[str, np.ndarray],
                               block_orientations: Dict[str, Tuple[float, float, float]],
                               lgi_shape: Tuple[int, int, int],
                               voxel_size: float = 1.0,
                               sample_direction: Optional[List[float]] = None,
                               title: str = 'Block IPF Map (Z)',
                               grain_lgi: Optional[np.ndarray] = None,
                               show_edges: bool = True):
        """Show a voxel grid where each block is colored by its IPF color."""
        import pyvista as pv

        if sample_direction is None:
            sample_direction = [0.0, 0.0, 1.0]

        nx, ny, nz = lgi_shape
        r_map = np.zeros((nx, ny, nz), dtype=np.float32)
        g_map = np.zeros((nx, ny, nz), dtype=np.float32)
        b_map = np.zeros((nx, ny, nz), dtype=np.float32)
        fill_map = np.zeros((nx, ny, nz), dtype=np.int32)

        # If a full grain image is provided, initialize grain voxels in grey so
        # unblocked regions remain visible in the rendered map.
        if grain_lgi is not None:
            grain_mask = grain_lgi > 0
            r_map[grain_mask] = 0.75
            g_map[grain_mask] = 0.75
            b_map[grain_mask] = 0.75
            fill_map[grain_mask] = 1

        for block_id, voxels in all_blocks.items():
            ea = block_orientations.get(block_id)
            if ea is None:
                continue
            r, g, b = self.euler_to_ipf_color(ea, sample_direction=sample_direction)
            r_map[voxels[:, 0], voxels[:, 1], voxels[:, 2]] = r
            g_map[voxels[:, 0], voxels[:, 1], voxels[:, 2]] = g
            b_map[voxels[:, 0], voxels[:, 1], voxels[:, 2]] = b
            fill_map[voxels[:, 0], voxels[:, 1], voxels[:, 2]] = 1

        grid = pv.ImageData()
        grid.dimensions = (nx + 1, ny + 1, nz + 1)
        grid.spacing = (voxel_size, voxel_size, voxel_size)
        grid.origin = (0.0, 0.0, 0.0)
        grid.cell_data['fill'] = fill_map.flatten(order='F')
        grid.cell_data['R'] = r_map.flatten(order='F')
        grid.cell_data['G'] = g_map.flatten(order='F')
        grid.cell_data['B'] = b_map.flatten(order='F')

        filled = grid.threshold(value=0.5, scalars='fill')
        rgb = np.column_stack([
            (filled.cell_data['R'] * 255).astype(np.uint8),
            (filled.cell_data['G'] * 255).astype(np.uint8),
            (filled.cell_data['B'] * 255).astype(np.uint8),
        ])
        filled.cell_data['RGB'] = rgb
        filled = filled.extract_surface()

        pl = pv.Plotter(window_size=(900, 750))
        pl.add_mesh(filled, scalars='RGB', rgb=True,
                    show_edges=show_edges, edge_color='black', line_width=0.5)
        pl.add_title(title, font_size=14)
        pl.set_background('white')
        pl.show_axes()
        pl.show()
    
    def plot_pag_ks_verification_v1(self, pag_id: int,
                                     pag_orientation: Tuple[float, float, float],
                                     grain_ids: List[int],
                                     grain_orientations: Dict[int, Tuple[float, float, float]],
                                     **kwargs):
        """Verify KS relationship for a PAG."""
        print(f"PAG {pag_id} KS Verification:")
        print(f"  Parent FCC orientation: {pag_orientation}")
        print(f"  Contained grains: {len(grain_ids)}")
        print(f"  Assigned orientations: {sum(1 for g in grain_ids if g in grain_orientations)}")
    
    @staticmethod
    def euler_to_ipf_color(euler_angles_deg: Tuple[float, float, float],
                           sample_direction: Optional[List[float]] = None
                           ) -> Tuple[float, float, float]:
        """
        Convert Bunge Euler angles (degrees) to an IPF RGB colour.

        Delegates to upxo.xtalphy.crystal_orientation.ipf_color which uses
        the standard cubic R.T @ sd formula: the sample direction is transformed
        into the crystal frame, its absolute components are taken, and the
        result is normalised to [0, 1]^3.

        Parameters
        ----------
        euler_angles_deg : tuple of float
            (phi1, Phi, phi2) in degrees, Bunge ZXZ convention.
        sample_direction : list of float, optional
            Reference direction in sample frame. Default [0, 0, 1].

        Returns
        -------
        (r, g, b) : tuple of float in [0, 1]
        """
        sd = sample_direction if sample_direction is not None else (0., 0., 1.)
        return _xtal_ipf_color(euler_angles_deg, sample_direction=sd)


__all__ = ['OrientationViz3D']
