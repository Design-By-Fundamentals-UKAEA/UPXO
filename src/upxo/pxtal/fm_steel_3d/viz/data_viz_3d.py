"""
DataViz3D: Statistical data visualizations (histograms, correlations).

Worker class for data and property distributions.
"""

import numpy as np
from typing import Optional, Dict, List, Tuple


class DataViz3D:
    """Statistical data visualizations."""
    
    __slots__ = ('_default_figure_size',)
    
    def __init__(self, default_figure_size: tuple = (8, 5)):
        self._default_figure_size = default_figure_size
    
    def plot_distribution(self, data: np.ndarray, title: str, xlabel: str,
                          bins: int = 20, figsize: Optional[tuple] = None,
                          ylabel: str = 'Frequency', **kwargs):
        """Plot histogram."""
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=figsize or self._default_figure_size)
        ax.hist(data, bins=bins, edgecolor='black', alpha=0.7)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        plt.tight_layout()
        plt.show()
    
    def plot_mprop_correlations(self, mprop_dict: Dict, figsize: Optional[tuple] = None):
        """Plot correlation matrix using seaborn."""
        import matplotlib.pyplot as plt
        try:
            import seaborn as sns
            import pandas as pd
        except ImportError:
            print("seaborn/pandas required for correlation plots")
            return
        data = [{'grain_id': gid, **props} for gid, props in mprop_dict.items()]
        if not data:
            return
        df = pd.DataFrame(data)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            fig, ax = plt.subplots(figsize=figsize or self._default_figure_size)
            sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', center=0, ax=ax)
            ax.set_title('Morphological Property Correlations')
            plt.tight_layout()
            plt.show()
    
    def plot_pag_size_distribution(self, clusters_dict: Dict[int, List[int]], **kwargs):
        """Plot distribution of grains per PAG."""
        sizes = [len(grains) for grains in clusters_dict.values()]
        self.plot_distribution(np.array(sizes), 'PAG Size Distribution', 'Grains per PAG', **kwargs)
    
    def plot_blocks_per_packet_distribution(self, grain_to_blocks_map: Dict[int, List[str]], **kwargs):
        """Plot distribution of blocks per packet (keyed by grain_id = packet id)."""
        sizes = [len(blocks) for blocks in grain_to_blocks_map.values()]
        self.plot_distribution(np.array(sizes), 'Blocks per Packet', 'Blocks', **kwargs)
    
    def plot_block_voxel_distribution(self, all_blocks: Dict[str, np.ndarray], **kwargs):
        """Plot distribution of voxels per block."""
        sizes = [len(v) for v in all_blocks.values()]
        self.plot_distribution(np.array(sizes), 'Block Voxel Distribution', 'Voxels per Block', **kwargs)
    
    def plot_misorientation_distribution(self, grain_orientations: Dict, grain_neighbors: Dict, **kwargs):
        """Plot misorientation angle distribution (placeholder)."""
        print("plot_misorientation_distribution: Requires OrientationAssigner3D — use get_misorientation_statistics()")


__all__ = ['DataViz3D']
