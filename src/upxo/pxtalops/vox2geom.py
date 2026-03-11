import numpy as np
import cc3d
from upxo.viz import gsviz
import upxo.gsdataops.gid_ops as gidOps
import upxo.gsdataops.grid_ops as gridOps
import upxo.propOps.mpropOps as mpropOps
import upxo.uiOps.outputDisplay as opDisp
import upxo.gbops.grainBoundOps3d as gbOps
import upxo.flags_and_controls.flags as FLAGS

class geometryfi3d:
    """
    Import
    ------
    from upxo.pxtalops.vox2geom import geometryfi3d
    """
    __slots__ = ('lfi', 'fid', 'n', 'mprops', 'neigh_fid', 'spb', 'bboxes',
                 'scaleHistory'
                 )
    
    n_plotTypes = FLAGS.n_plotType_3D_voxels

    def __init__(self, lfi):
        self.lfi = lfi
        self.mprops = {}
        self.scaleHistory = {0: 1}
        self.bboxes = None

    def __repr__(self):
        return f"VOX.2.GEOM. MID.{id(self)}"
    
    # ================================================================================================

    # NEIGHBOUR OPERATIONS

    def find_neigh(self):
        neigh_fids = gidOps.find_neighs3d(self.lfi, 6)
        return neigh_fids

    def get_feature_sizes(self):
        return mpropOps.get_feature_volumes(self.lfi)
    
    def find_mprops(self):
        self.mprops['nvox'] = self.get_feature_sizes()

    # ================================================================================================

    # VOXEL GRID SCALING OPERATIONS

    def scale(self, scaleFactor, reindex=False, plotgs=True, alpha=1.0, cmap='nipy_spectral',
              quickPlot_kwargs={'scalar_name': 'lfi', 'alpha': 0.75, 'cmap': 'viridis'}):
        self.lfi = gridOps.rescale_grid_3d(self.lfi, scaleFactor, method='nearest')
        if reindex:
            self.reindex(findMprops=False, plotgs=False, quickPlot_kwargs=quickPlot_kwargs)
        self.quick_plot(**quickPlot_kwargs) if plotgs else None

    # ================================================================================================

    # COMMON INTERFACE BOUNDARY DETECTION

    def find_spb(self):
        from skimage.segmentation import find_boundaries
        spb = find_boundaries(self.lfi, connectivity=2, mode='subpixel', background=0)

    # ================================================================================================

    # VOXEL ID RENUMBERINGS & ASSOCIATED OPERATIONS

    def reindex(self, findMprops=True, plotgs=True,
                quickPlot_kwargs={'scalar_name': 'lfi', 'alpha': 0.75, 'cmap': 'viridis'}):
        """Reindex the feature IDs to be contiguous and start from 0."""
        self.lfi = gridOps.shuffle_feature_IDs(cc3d.connected_components(self.lfi,
                    connectivity=6, out_dtype=np.uint32).astype(np.int32))
        self.fid = np.unique(self.lfi)
        self.neigh_fid = self.find_neigh()
        if findMprops:
            self.find_mprops()
        self.quick_plot(**quickPlot_kwargs) if plotgs else None

    # ================================================================================================

    # VOXEL GRAIN BOUNDARY CLEANING OPERATIONS - GLOBAL MORPHOLOGICAL OPERATIONS

    def prepare(self, run_pre_cleaning_step=True):
        if run_pre_cleaning_step:
            self.prepare_pre_vox_cleaning(plotgs=False)
        self.cleanVoxMorph_DE_npass(niterations=2,
                DILfpSizes=[4, 4], ERSfpSizes=[4, 4],
                footprints=['ball', 'ball'], removeEndVox=[True, True], plotgs=False)
        self.scale(2, plotgs=False)

    def prepare_pre_vox_cleaning(self, plotgs=False,
                    quickPlot_kwargs={'scalar_name': 'lfi', 'alpha': 0.75, 'cmap': 'viridis'}):
        self.report(message="Pre-Initial cleaning report: \n")
        print("\nInitiating pre-cleaning")
        self.reindex(findMprops=False, plotgs=False)
        self.detect_and_merge_islands()
        self.reindex(findMprops=True, plotgs=False)
        self.quick_plot(**quickPlot_kwargs) if plotgs else None
        self.report(message="Post-Initial cleaning report: \n")

    # ================================================================================================

    def detect_and_merge_islands(self):
        islands = self.detect_islands(self.neigh_fid)
        if len(islands) > 0:
            self.lfi = self.merge_islands(self.lfi, islands, self.neigh_fid)
            self.reindex(plotgs=False)

    def detect_islands(self, neigh_fids):
        return gidOps.detect_islands(neigh_fids)

    def merge_islands(self, lfi, islands, neigh_fids):
        for island in islands:
            lfi[lfi==island] = neigh_fids[island]
        return lfi

    # ================================================================================================

    def filter_small_inconsistant_features(self):
        pass

    # ================================================================================================

    def voxel_smoother(self, addMajFilter=True, majority_filter_kwargs={'n': 2, 'sizes': [3, 3]},
                addDilErs=True, dilers_kwargs={'DILfpSizes': [4, 4], 'ERSfpSizes': [4, 4],
                            'footprints': ['ball', 'ball'], 'removeEndVox': [True, True]},
                reindex=False, plotgs=False, 
                quickPlot_kwargs={'scalar_name': 'lfi', 'alpha': 0.75, 'cmap': 'viridis'}):
        if addMajFilter:
            self.lfi = self.cleanVoxMorph_majority_filter_npass(lfi=self.lfi, plotgs=False,
                                    **majority_filter_kwargs)
            self.report(message="Post majority filter report: \n")
        if addDilErs:
            self.lfi = self.cleanVoxMorph_DE_npass(self.lfi,
                niterations=dilers_kwargs.get('niterations', 2),
                DILfpSizes=dilers_kwargs.get('DILfpSizes', [4, 4]),
                ERSfpSizes=dilers_kwargs.get('ERSfpSizes', [4, 4]),
                footprints=dilers_kwargs.get('footprints', ['ball', 'ball']),
                removeEndVox=dilers_kwargs.get('removeEndVox', [True, True]),
                plotgs=False)
        if reindex:
            self.reindex(findMprops=False, plotgs=False)
        self.quick_plot(**quickPlot_kwargs) if plotgs else None
        self.report(message="Post voxel smoother report: \n")

    def cleanVoxMorph_majority_filter_npass(self, lfi=None, plotgs=True,
                quickPlot_kwargs={'scalar_name': 'lfi', 'alpha': 0.75, 'cmap': 'viridis'},
                **majority_filter_kwargs):
        """ majority_filter_kwargs={'n': 2, 'sizes': [3, 3]} """
        self.lfi = gridOps.majority_filter_3d_npass(lfi=self.lfi,
                        **majority_filter_kwargs)
        self.quick_plot(**quickPlot_kwargs) if plotgs else None

    def cleanVoxMorph_DE_npass(self, niterations=2, DILfpSizes=[4, 4],
                ERSfpSizes=[4, 4], footprints=['ball', 'ball'],
                removeEndVox=[True, True], plotgs=True, alpha=1.0, cmap='nipy_spectral',
                quickPlot_kwargs={'scalar_name': 'lfi', 'alpha': 0.75, 'cmap': 'viridis'}):
        self.lfi = gridOps.smooth_voxMorph_npass(self.lfi, niterations=niterations,
            DILfpSizes=DILfpSizes, ERSfpSizes=ERSfpSizes, footprints=footprints,
            removeEndVox=removeEndVox)
        self.quick_plot(**quickPlot_kwargs) if plotgs else None

    # ================================================================================================

    # 
    # ================================================================================================
    # ================================================================================================
    # ================================================================================================
    # ================================================================================================
    # ================================================================================================
    # ================================================================================================
    # ================================================================================================
    # ================================================================================================
    # ================================================================================================
    # ================================================================================================
    # ================================================================================================
    # ================================================================================================
    # ================================================================================================

    # REPORTING

    def report(self, message='Report: \n'):
        opDisp.preManifold_clean_report(self.lfi, message=message)

    # ================================================================================================

    # PLOTTING

    def quick_plot(self, scalar_name='lfi', alpha=0.9, cmap='nipy_spectral', show_edges=False,
                   title='', xname='', yname='', zname=''):
        gsviz.plot_pvgrid(gsviz.make_pvgrid(self.lfi, scalar_name=scalar_name),
                            scalar_name=scalar_name, show_edges=show_edges, alpha=alpha, title=title,
                            cmap=cmap, _xname_=xname, _yname_=yname, _zname_=zname)

    def see(self, plotType, **kwargs):
        """
        plotType: 0 to n for different gsviz.vox2geom_plots. Value of n depends on the UPXO version.
        I will add more features and n will increase to allow you more ways to plot. For now, we have
        the following options for plotType, which are the options for gsviz.vox2geom_plots:
            0: quick plot of the voxel grid with default settings. This is the same as 
                geometryfi3d.quick_plot() with default settings.
            1: grain_viewer. -1 falls back to grain viewer.
            2: view_boundary_voxels
            3: see_clip_plane
            4: see_mesh_slice
            5: see_mesh_slice_ortho
        """
        if plotType not in np.arange(0, self.n_plotTypes+1):
            raise ValueError(f"Unknown plotType '{plotType}'. Available options: 0 to {self.n_plotTypes} for",
                             "different gsviz.vox2geom_plots")
        if plotType == 0:
            self.quick_plot(**kwargs)
            return

        gsviz.vox2geom_plots(plotType, lfi=self.lfi, **kwargs)